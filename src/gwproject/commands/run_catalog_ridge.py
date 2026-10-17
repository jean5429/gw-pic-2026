from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from gwproject.catalog_analysis import (
    analyze_ridge_channel,
    catalog_detector_chirp_mass,
    catalog_detector_component_masses,
    catalog_redshift,
    stable_seed,
)
from gwproject.config import load_toml
from gwproject.provenance import write_manifest


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


ALGORITHM_REVISION = "catalog-conditioned-ridge-no-monotonic-forcing-v2"


def _config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(
        {"algorithm_revision": ALGORITHM_REVISION, "config": config},
        sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _manifest_for_processed(path: Path) -> dict[str, Any]:
    manifest = path.with_suffix(".manifest.json")
    return _read_json(manifest) if manifest.exists() else {}


def _channel_slug(event: str, detector: str) -> str:
    return f"{event}__{detector}"


def _process_one(
    row: dict[str, Any],
    metadata: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
    save_arrays: bool,
    resume: bool,
) -> dict[str, Any]:
    event = str(row["event"])
    detector = str(row["detector"])
    slug = _channel_slug(event, detector)
    result_path = output_dir / "channels" / f"{slug}.json"
    arrays_path = output_dir / "ridge_arrays" / f"{slug}.npz"
    config_sha256 = _config_sha256(config)
    if resume and result_path.exists():
        saved = _read_json(result_path)
        if saved.get("config_sha256") == config_sha256:
            return saved

    base = {
        "config_sha256": config_sha256,
        "event": event,
        "detector": detector,
        "processed_path": str(row.get("processed_path", "")),
        "analysis_mode": str(
            config["catalog_ridge"].get(
                "analysis_mode", "catalog_conditioned_diagnostic"
            )
        ),
    }
    allow_source = bool(config["catalog_ridge"].get("allow_source_frame_fallback", False))
    mc, mc_frame = catalog_detector_chirp_mass(
        metadata, allow_source_fallback=allow_source
    )
    masses, mass_frame = catalog_detector_component_masses(
        metadata, allow_source_fallback=allow_source
    )
    base.update(
        {
            "reference_chirp_mass_solar": mc,
            "reference_chirp_mass_frame": mc_frame,
            "component_mass_frame": mass_frame,
            "catalog_mass_1_solar": masses[0] if masses else None,
            "catalog_mass_2_solar": masses[1] if masses else None,
            "catalog_redshift": catalog_redshift(metadata),
            "analysis_mass_frame_required": "detector",
        }
    )
    if mc is None:
        result = {
            **base,
            "status": "missing_detector_frame_metadata",
            "reason": (
                "massa chirp detector-frame indisponivel; execute "
                "gwproject catalog refresh-metadata"
            ),
        }
    else:
        try:
            processed_path = Path(str(row["processed_path"]))
            data = np.load(processed_path).astype(float)
            manifest = _manifest_for_processed(processed_path)
            sample_rate = float(manifest.get("sample_rate", 4096.0))
            event_index = manifest.get("event_index_processed")
            if event_index is None:
                event_index = len(data) // 2
            seed = stable_seed(event, detector, base_seed=int(config["catalog_ridge"]["base_seed"]))
            analysis, arrays = analyze_ridge_channel(
                data,
                sample_rate,
                int(event_index),
                float(mc),
                m1=masses[0] if masses else None,
                m2=masses[1] if masses else None,
                config=config,
                seed=seed,
            )
            result = {
                **base,
                **analysis,
                "sample_rate": sample_rate,
                "event_index_processed": int(event_index),
                "trimmed_nonfinite_edges": bool(manifest.get("trimmed_nonfinite_edges", False)),
            }
            if save_arrays and arrays:
                arrays_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(arrays_path, **arrays)
                result["ridge_arrays_path"] = str(arrays_path)
        except Exception as exc:
            result = {**base, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _aggregate_events(catalog: pd.DataFrame, channel_results: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_event = {name: group for name, group in channel_results.groupby("event")}
    disagreement_max = float(config["catalog_ridge"]["detector_disagreement_max"])
    for metadata in catalog.to_dict(orient="records"):
        event = str(metadata["name"])
        group = by_event.get(event, pd.DataFrame())
        accepted = group[group["status"] == "ok"] if not group.empty else pd.DataFrame()
        allow_source = bool(config["catalog_ridge"].get("allow_source_frame_fallback", False))
        mc, mc_frame = catalog_detector_chirp_mass(
            metadata, allow_source_fallback=allow_source
        )
        if accepted.empty:
            rows.append(
                {
                    "event": event,
                    "reference_chirp_mass_solar": mc,
                    "reference_chirp_mass_frame": mc_frame,
                    "status": "no_accepted_channel_fit",
                    "detectors_used": "",
                    "n_detectors": 0,
                }
            )
            continue
        estimates = accepted["estimated_chirp_mass_solar"].astype(float).to_numpy()
        combined = float(np.median(estimates))
        disagreement = (
            float((np.max(estimates) - np.min(estimates)) / combined)
            if len(estimates) >= 2 and combined > 0
            else None
        )
        relative_error = abs(combined - mc) / mc if mc and mc > 0 else None
        network_ready = len(estimates) >= 2
        rows.append(
            {
                "event": event,
                "reference_chirp_mass_solar": mc,
                "reference_chirp_mass_frame": mc_frame,
                "combined_chirp_mass_solar": combined,
                "combined_relative_error": relative_error,
                "detector_disagreement": disagreement,
                "detectors_used": ",".join(sorted(accepted["detector"].astype(str))),
                "n_detectors": int(len(accepted)),
                "network_ready": network_ready,
                "disagreement_passed": (
                    disagreement <= disagreement_max if disagreement is not None else None
                ),
                "status": "network" if network_ready else "single_detector",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnostico catalogo-condicionado de chirp mass por ridge"
    )
    parser.add_argument("--approved", type=Path, default=Path("outputs/gwtc4_218/approved_channels.csv"))
    parser.add_argument("--catalog", type=Path, default=Path("outputs/gwtc4_218/catalog.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/catalog_analysis.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research/gwtc4_218/ridge_diagnostic"))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--event", action="append", default=[])
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-save-arrays", action="store_true")
    args = parser.parse_args()

    config = load_toml(args.config)
    approved = pd.read_csv(args.approved)
    catalog = pd.read_csv(args.catalog)
    if args.event:
        approved = approved[approved["event"].isin(args.event)].copy()
        catalog = catalog[catalog["name"].isin(args.event)].copy()
    if args.limit > 0:
        selected_events = list(dict.fromkeys(approved["event"].astype(str)))[: args.limit]
        approved = approved[approved["event"].isin(selected_events)].copy()
        catalog = catalog[catalog["name"].isin(selected_events)].copy()
    metadata_map = {str(row["name"]): row for row in catalog.to_dict(orient="records")}
    allow_source = bool(config["catalog_ridge"].get("allow_source_frame_fallback", False))
    available_events = sum(
        catalog_detector_chirp_mass(row, allow_source_fallback=allow_source)[0] is not None
        for row in catalog.to_dict(orient="records")
    )
    if available_events == 0:
        raise SystemExit(
            "Nenhum evento possui massa chirp detector-frame. Execute primeiro: "
            "python .\\gwproject catalog refresh-metadata"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    workers = args.workers or int(config["catalog_ridge"]["workers"])
    tasks = approved.to_dict(orient="records")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _process_one,
                row,
                metadata_map.get(str(row["event"]), {}),
                config,
                args.output_dir,
                not args.no_save_arrays and bool(config["catalog_ridge"]["save_ridge_arrays"]),
                not args.no_resume,
            ): row
            for row in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="ridge catalog"):
            results.append(future.result())

    per_channel = pd.DataFrame(results).sort_values(["event", "detector"])
    per_channel.to_csv(args.output_dir / "per_channel.csv", index=False)
    failures = per_channel[per_channel["status"] != "ok"].copy()
    failures.to_csv(args.output_dir / "failures.csv", index=False)
    per_event = _aggregate_events(catalog, per_channel, config)
    per_event.to_csv(args.output_dir / "per_event.csv", index=False)

    summary = {
        "channels_requested": int(len(tasks)),
        "channels_with_accepted_fit": int((per_channel["status"] == "ok").sum()),
        "channels_failed_or_skipped": int((per_channel["status"] != "ok").sum()),
        "events_requested": int(len(catalog)),
        "events_with_network_estimate": int((per_event["status"] == "network").sum()),
        "events_with_single_detector_estimate": int((per_event["status"] == "single_detector").sum()),
        "events_without_estimate": int((per_event["status"] == "no_accepted_channel_fit").sum()),
        "events_with_detector_frame_metadata": int(available_events),
        "events_missing_detector_frame_metadata": int(len(catalog) - available_events),
        "analysis_mode": str(config["catalog_ridge"]["analysis_mode"]),
        "monotonic_ridge_forced": bool(
            config["catalog_ridge"]["enforce_monotonic_ridge"]
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_manifest(
        args.output_dir / "manifest.json",
        {
            "config": config,
            "arguments": vars(args),
            "summary": summary,
            "scope_warning": (
                "Diagnostico condicionado ao catalogo: a massa publicada define janela e "
                "banda antes do ajuste. Portanto, a proximidade entre estimativa e catalogo "
                "nao valida uma medicao cega ou independente. O ajuste newtoniano por ridge "
                "nao substitui inferencia Bayesiana."
            ),
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
