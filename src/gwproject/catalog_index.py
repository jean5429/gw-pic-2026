from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

DEFAULT_DETECTORS = ("H1", "L1", "V1")


def _value(mapping: Any, key: str, default: Any = None) -> Any:
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return default


def classify_channel(entry: dict[str, Any]) -> tuple[str, str | None]:
    download_status = str(entry.get("status", "missing"))
    processing = entry.get("processing") or {}
    qtransform = entry.get("qtransform") or {}
    processing_status = str(_value(processing, "status", "not_run"))
    q_status = str(_value(qtransform, "status", "not_run"))
    processing_reason = str(_value(processing, "reason", ""))
    q_reason = str(_value(qtransform, "reason", ""))

    if download_status != "ok":
        return "data_unavailable", download_status
    if processing_status == "ok" and q_status == "ok":
        return "approved", None
    if "NaN" in processing_reason or "infinito" in processing_reason:
        return "rejected_nonfinite", processing_reason
    if "whitening" in processing_reason.lower():
        return "rejected_whitening", processing_reason
    if processing_status not in {"ok", "not_run"}:
        return "rejected_processing", processing_reason or processing_status
    if q_status not in {"ok", "not_run", "skipped_upstream_rejected"}:
        return "rejected_qtransform", q_reason or q_status
    if q_status == "skipped_upstream_rejected":
        return "rejected_processing", q_reason or processing_reason or q_status
    return "rejected_other", processing_reason or q_reason or download_status


def build_channels_quality(
    catalog: pd.DataFrame,
    log_rows: Iterable[dict[str, Any]],
    *,
    detectors: Iterable[str] = DEFAULT_DETECTORS,
) -> pd.DataFrame:
    if "name" not in catalog.columns:
        raise ValueError("catalog.csv deve conter a coluna 'name'")
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in log_rows:
        event = str(row.get("event", ""))
        detector = str(row.get("detector", ""))
        if event and detector:
            by_key[(event, detector)] = row

    records: list[dict[str, Any]] = []
    for event in catalog["name"].astype(str):
        for detector in detectors:
            entry = by_key.get(
                (event, detector),
                {
                    "event": event,
                    "detector": detector,
                    "status": "missing_from_run_log",
                },
            )
            processing = entry.get("processing") or {}
            qtransform = entry.get("qtransform") or {}
            status, reason = classify_channel(entry)
            records.append(
                {
                    "event": event,
                    "detector": detector,
                    "download_status": entry.get("status", "missing"),
                    "processing_status": processing.get("status", "not_run"),
                    "qtransform_status": qtransform.get("status", "not_run"),
                    "scientific_status": status,
                    "approved": status == "approved",
                    "raw_path": entry.get("raw_path"),
                    "processed_path": processing.get("processed_path"),
                    "qgram_path": qtransform.get("path"),
                    "white_std_core": processing.get("white_std_core"),
                    "zero_fraction": processing.get("zero_fraction"),
                    "longest_zero_run_seconds": processing.get("longest_zero_run_seconds"),
                    "trimmed_nonfinite_edges": processing.get("trimmed_nonfinite_edges", False),
                    "event_index_processed": processing.get("event_index_processed"),
                    "rejection_reason": reason,
                }
            )
    return pd.DataFrame.from_records(records)


def build_event_quality(catalog: pd.DataFrame, channels: pd.DataFrame) -> pd.DataFrame:
    catalog_by_name = catalog.set_index("name", drop=False)
    rows: list[dict[str, Any]] = []
    for event, group in channels.groupby("event", sort=False):
        approved = group[group["scientific_status"] == "approved"]
        rejected = group[group["scientific_status"] != "approved"]
        n_approved = int(len(approved))
        if n_approved >= 2:
            event_status = "network_ready"
        elif n_approved == 1:
            event_status = "single_detector"
        else:
            event_status = "unusable"
        metadata = catalog_by_name.loc[event].to_dict() if event in catalog_by_name.index else {}
        row = {
            "event": event,
            "channels_total": int(len(group)),
            "channels_approved": n_approved,
            "channels_rejected": int(len(rejected)),
            "detectors_available": ",".join(sorted(group.loc[group["download_status"] == "ok", "detector"])),
            "detectors_approved": ",".join(sorted(approved["detector"])),
            "detectors_rejected": ",".join(sorted(rejected["detector"])),
            "network_ready": n_approved >= 2,
            "single_detector_usable": n_approved == 1,
            "event_status": event_status,
        }
        for key in (
            "base_name",
            "version",
            "catalog",
            "gps",
            "chirp_mass",
            "chirp_mass_source",
            "mass_1",
            "mass_2",
            "mass_1_source",
            "mass_2_source",
            "chirp_mass_detector",
            "mass_1_detector",
            "mass_2_detector",
            "chirp_mass_detector_origin",
            "component_mass_detector_origin",
            "redshift",
            "luminosity_distance",
            "snr",
            "far",
            "metadata_status",
            "metadata_api",
            "parameter_estimation_name",
            "parameter_waveform_family",
            "metadata_refreshed_at_utc",
        ):
            row[key] = metadata.get(key)
        rows.append(row)
    return pd.DataFrame.from_records(rows).sort_values("gps", na_position="last").reset_index(drop=True)


def write_catalog_indexes(
    catalog_path: str | Path,
    run_log_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    catalog_path = Path(catalog_path)
    run_log_path = Path(run_log_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv(catalog_path)
    log_rows = json.loads(run_log_path.read_text(encoding="utf-8"))
    channels = build_channels_quality(catalog, log_rows)
    events = build_event_quality(catalog, channels)

    channels.to_csv(output_dir / "channels_quality.csv", index=False)
    channels[channels["scientific_status"] == "approved"].to_csv(
        output_dir / "approved_channels.csv", index=False
    )
    channels[channels["scientific_status"] == "rejected_nonfinite"].to_csv(
        output_dir / "rejected_nonfinite_channels.csv", index=False
    )
    channels[channels["scientific_status"] == "data_unavailable"].to_csv(
        output_dir / "unavailable_channels.csv", index=False
    )
    events.to_csv(output_dir / "catalog_dataset.csv", index=False)
    events.to_csv(output_dir / "event_quality_summary.csv", index=False)
    events[events["event_status"] == "network_ready"].to_csv(
        output_dir / "network_ready_events.csv", index=False
    )
    events[events["event_status"] == "single_detector"].to_csv(
        output_dir / "single_detector_events.csv", index=False
    )
    events[events["event_status"] == "unusable"].to_csv(
        output_dir / "unusable_events.csv", index=False
    )

    counts = channels["scientific_status"].value_counts().to_dict()
    summary = {
        "events_total": int(len(events)),
        "channels_total": int(len(channels)),
        "channels_approved": int(counts.get("approved", 0)),
        "channels_rejected": int(len(channels) - counts.get("approved", 0)),
        "network_ready_events": int((events["event_status"] == "network_ready").sum()),
        "single_detector_events": int((events["event_status"] == "single_detector").sum()),
        "unusable_events": int((events["event_status"] == "unusable").sum()),
        "channel_status_counts": {str(k): int(v) for k, v in counts.items()},
    }
    (output_dir / "catalog_dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
