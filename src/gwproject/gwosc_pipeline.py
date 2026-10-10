from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .preprocessing import preprocess
from .provenance import write_manifest
from .segments import FiniteEventSegment, select_finite_event_segment


def _require_gw() -> tuple[Any, Any, Any]:
    try:
        from gwosc import datasets
        from gwosc.api import fetch_event_json
        from gwpy.timeseries import TimeSeries
    except ImportError as exc:
        raise RuntimeError("Instale as dependencias GW: pip install -e '.[gw]'") from exc
    return datasets, fetch_event_json, TimeSeries


def event_base_name(name: str) -> str:
    return name.rsplit("-v", 1)[0]


def event_version(name: str) -> int:
    parts = name.rsplit("-v", 1)
    return int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0


def list_latest_events(catalog_names: Iterable[str]) -> list[str]:
    datasets, _, _ = _require_gw()
    latest: dict[str, tuple[int, str]] = {}
    for catalog in catalog_names:
        for name in datasets.find_datasets(type="events", catalog=catalog):
            base, version = event_base_name(name), event_version(name)
            if base not in latest or version > latest[base][0]:
                latest[base] = (version, name)
    return sorted(item[1] for item in latest.values())


def _select_event_payload(event_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    events = payload.get("events", {})
    if event_name in events:
        return events[event_name]
    base = event_base_name(event_name)
    candidates = [
        (event_version(key), value)
        for key, value in events.items()
        if event_base_name(key) == base
    ]
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    if len(events) == 1:
        return next(iter(events.values()))
    raise KeyError(f"evento {event_name!r} nao encontrado no payload")


def fetch_event_parameters(
    event_name: str,
    retries: int = 4,
    backoff: float = 1.5,
) -> dict[str, Any]:
    _, fetch_event_json, _ = _require_gw()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            latest = _select_event_payload(event_name, fetch_event_json(event_name))
            return {
                "name": event_name,
                "base_name": event_base_name(event_name),
                "version": event_version(event_name),
                "gps": latest.get("GPS"),
                "chirp_mass": latest.get("chirp_mass"),
                "chirp_mass_source": latest.get("chirp_mass_source"),
                "mass_1": latest.get("mass_1"),
                "mass_2": latest.get("mass_2"),
                "mass_1_source": latest.get("mass_1_source"),
                "mass_2_source": latest.get("mass_2_source"),
                "redshift": latest.get("redshift"),
                "luminosity_distance": latest.get("luminosity_distance"),
                "far": latest.get("far"),
                "snr": latest.get("network_matched_filter_snr"),
                "catalog": latest.get("catalog.shortName"),
                "metadata_status": "ok",
            }
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(backoff * attempt)
    return {
        "name": event_name,
        "base_name": event_base_name(event_name),
        "version": event_version(event_name),
        "metadata_status": f"error: {last_error}",
    }


def build_catalog(catalog_names: Iterable[str], workers: int = 4) -> pd.DataFrame:
    events = list_latest_events(catalog_names)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        rows = list(executor.map(fetch_event_parameters, events))
    return (
        pd.DataFrame(rows)
        .sort_values(["gps", "name"], na_position="last")
        .reset_index(drop=True)
    )


def fetch_open_data(
    detector: str,
    gps: float,
    *,
    before: float = 16.0,
    after: float = 16.0,
    sample_rate: int = 4096,
    cache: bool = True,
):
    _, _, TimeSeries = _require_gw()
    return TimeSeries.fetch_open_data(
        detector,
        gps - before,
        gps + after,
        sample_rate=sample_rate,
        cache=cache,
    )


def save_event_strain(
    event_name: str,
    gps: float,
    output_root: str | Path,
    *,
    detectors: Iterable[str] = ("H1", "L1", "V1"),
    before: float = 16.0,
    after: float = 16.0,
    sample_rate: int = 4096,
) -> list[dict[str, Any]]:
    root = Path(output_root) / "raw" / event_name
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for detector in detectors:
        target = root / f"{detector}.hdf5"
        try:
            if not target.exists():
                strain = fetch_open_data(
                    detector,
                    gps,
                    before=before,
                    after=after,
                    sample_rate=sample_rate,
                )
                strain.write(target, format="hdf5", overwrite=True)
            rows.append(
                {
                    "event": event_name,
                    "detector": detector,
                    "raw_path": str(target),
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "event": event_name,
                    "detector": detector,
                    "raw_path": None,
                    "status": f"error: {exc}",
                }
            )
    return rows


def existing_event_strain_entries(
    event_name: str,
    output_root: str | Path,
    detectors: Iterable[str] = ("H1", "L1", "V1"),
) -> list[dict[str, Any]]:
    root = Path(output_root) / "raw" / event_name
    rows: list[dict[str, Any]] = []
    for detector in detectors:
        target = root / f"{detector}.hdf5"
        rows.append(
            {
                "event": event_name,
                "detector": detector,
                "raw_path": str(target) if target.exists() else None,
                "status": "ok" if target.exists() else "data_unavailable_local",
            }
        )
    return rows


def _preprocess_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "psd_segment_seconds": float(config["psd"]["segment_seconds"]),
        "psd_overlap_fraction": float(config["psd"]["overlap_fraction"]),
        "psd_average": str(config["psd"]["average"]),
        "whitening_fmin": float(config["whitening"]["fmin_hz"]),
        "whitening_fmax": float(config["whitening"]["fmax_hz"]),
        "whitening_tukey_alpha": float(config["whitening"]["tukey_alpha"]),
        "bandpass_fmin": float(config["bandpass"]["fmin_hz"]),
        "bandpass_fmax": float(config["bandpass"]["fmax_hz"]),
        "bandpass_order": int(config["bandpass"]["order"]),
        "edge_crop_seconds": float(config["signal"]["edge_crop_seconds"]),
        "white_std_min": float(config["gates"]["white_std_min"]),
        "white_std_max": float(config["gates"]["white_std_max"]),
        "max_zero_fraction": float(config["data_quality"]["max_zero_fraction"]),
        "max_zero_run_seconds": float(config["data_quality"]["max_zero_run_seconds"]),
        "return_diagnostics": True,
    }


def _segment_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    quality = config.get("data_quality", {})
    return {
        "allow_edge_trim": bool(quality.get("allow_edge_nonfinite_trim", False)),
        "reject_internal_nonfinite": bool(quality.get("reject_internal_nonfinite", True)),
        "min_valid_before_event_seconds": float(
            quality.get("min_valid_before_event_seconds", 8.0)
        ),
        "min_valid_after_event_seconds": float(
            quality.get("min_valid_after_event_seconds", 8.0)
        ),
        "min_contiguous_valid_seconds": float(
            quality.get("min_contiguous_valid_seconds", 20.0)
        ),
    }


def _prepare_event_timeseries(
    raw: Any,
    event_gps: float | None,
    config: dict[str, Any],
) -> tuple[Any, np.ndarray, FiniteEventSegment]:
    values = np.asarray(raw.value, dtype=float)
    sample_rate = float(raw.sample_rate.value)
    if event_gps is None:
        event_index = len(values) // 2
    else:
        times = np.asarray(raw.times.value, dtype=float)
        event_index = int(np.argmin(np.abs(times - float(event_gps))))
    selected, info = select_finite_event_segment(
        values,
        event_index,
        sample_rate,
        **_segment_kwargs(config),
    )
    selected_raw = raw[info.start_index : info.stop_index]
    return selected_raw, selected, info


def process_raw_file(
    raw_path: str | Path,
    output_path: str | Path,
    *,
    config: dict[str, Any],
    event_gps: float | None = None,
) -> dict[str, Any]:
    _, _, TimeSeries = _require_gw()
    raw = TimeSeries.read(raw_path, format="hdf5")
    try:
        _, values, segment = _prepare_event_timeseries(raw, event_gps, config)
        sample_rate = float(raw.sample_rate.value)
        processed, psd, diagnostics = preprocess(
            values,
            sample_rate,
            **_preprocess_kwargs(config),
        )
    except ValueError as exc:
        return {"status": "rejected", "reason": str(exc), "raw_path": str(raw_path)}

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, processed.astype(np.float32))
    np.savez(
        target.with_suffix(".psd.npz"),
        frequencies=psd.frequencies,
        values=psd.values,
    )
    details = {
        "status": "ok",
        "processed_path": str(target),
        "white_std_core": diagnostics.white_std_core,
        "zero_fraction": diagnostics.quality.zero_fraction,
        "longest_zero_run_seconds": diagnostics.quality.longest_zero_run_seconds,
        "sample_rate": sample_rate,
        "event_index_processed": segment.event_index_segment,
        "trimmed_nonfinite_edges": segment.trimmed,
        "finite_segment": segment.as_dict(),
    }
    write_manifest(
        target.with_suffix(".manifest.json"),
        {"raw_path": str(raw_path), "event_gps": event_gps, "config": config, **details},
    )
    return details


def gwpy_qtransform_consistent(
    raw_path: str | Path,
    gps: float,
    output_path: str | Path,
    *,
    qrange: tuple[float, float] = (4.0, 64.0),
    frange: tuple[float, float] = (20.0, 512.0),
    outseg_seconds: float = 4.0,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Gera somente Q-transform GWPy; falhas nunca viram STFT silenciosamente."""

    _, _, TimeSeries = _require_gw()
    raw = TimeSeries.read(raw_path, format="hdf5")
    if config is None:
        values = np.asarray(raw.value, dtype=float)
        if not np.all(np.isfinite(values)) or float(np.std(values)) <= 0:
            return {"status": "invalid_or_empty_data", "transform_type": "gwpy_qtransform"}
        selected_raw = raw
        segment_info: dict[str, Any] | None = None
    else:
        try:
            selected_raw, values, segment = _prepare_event_timeseries(raw, gps, config)
            if float(np.std(values)) <= 0:
                raise ValueError("segmento constante ou vazio")
            segment_info = segment.as_dict()
        except ValueError as exc:
            return {
                "status": "invalid_or_empty_data",
                "reason": str(exc),
                "transform_type": "gwpy_qtransform",
            }

    start, end = gps - outseg_seconds / 2.0, gps + outseg_seconds / 2.0
    selected_start = float(selected_raw.times.value[0])
    selected_end = float(selected_raw.times.value[-1])
    if start < selected_start or end > selected_end:
        return {
            "status": "error",
            "reason": "outseg do Q-transform fora do trecho finito selecionado",
            "transform_type": "gwpy_qtransform",
        }
    try:
        qgram = selected_raw.q_transform(
            qrange=qrange,
            frange=frange,
            outseg=(start, end),
            logf=True,
            norm="median",
            whiten=True,
        )
        array = np.asarray(qgram, dtype=float)
        if not np.all(np.isfinite(array)) or float(np.std(array)) <= 0:
            raise ValueError("Q-transform vazio ou invalido")
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, array.astype(np.float32))
        metadata = {
            "status": "ok",
            "transform_type": "gwpy_qtransform",
            "qrange": list(qrange),
            "frange": list(frange),
            "outseg_seconds": outseg_seconds,
            "whiten": True,
            "normalization": "median",
            "shape": list(array.shape),
            "finite_segment": segment_info,
        }
        target.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return {**metadata, "path": str(target)}
    except Exception as exc:
        return {
            "status": "error",
            "reason": str(exc),
            "transform_type": "gwpy_qtransform",
        }
