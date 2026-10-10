from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gwproject.catalog_index import write_catalog_indexes
from gwproject.config import load_toml
from gwproject.gwosc_pipeline import (
    build_catalog,
    existing_event_strain_entries,
    gwpy_qtransform_consistent,
    process_raw_file,
    save_event_strain,
)
from gwproject.preprocessing import q_transform_self_test
from gwproject.provenance import anonymize_paths, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Catalogo e strain GWTC em fluxo modular v8")
    parser.add_argument("--config", type=Path, default=Path("configs/processing.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gwtc"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-events", type=int, default=0, help="0 = todos")
    parser.add_argument("--download-strain", action="store_true")
    parser.add_argument("--process", action="store_true")
    parser.add_argument("--qgrams", action="store_true")
    parser.add_argument(
        "--diagnostic-qgrams-for-rejected",
        action="store_true",
        help="gera Q-gram mesmo quando o processamento reprova; nunca entra em approved_channels.csv",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="recria os CSVs de qualidade a partir de catalog.csv e run_log.json",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = args.output_dir / "catalog.csv"
    run_log_path = args.output_dir / "run_log.json"

    if args.rebuild_index and not any((args.download_strain, args.process, args.qgrams)):
        summary = write_catalog_indexes(catalog_path, run_log_path, args.output_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    cfg = load_toml(args.config)
    qcfg = cfg["qtransform"]
    q_ratio = q_transform_self_test(
        fs=float(cfg["signal"]["sample_rate"]),
        duration_seconds=float(qcfg["self_test_seconds"]),
        fmin=float(qcfg["fmin_hz"]),
        fmax=float(qcfg["fmax_hz"]),
        n_bins=int(qcfg["n_bins"]),
        q=float(qcfg["display_q"]),
        seed=int(qcfg["self_test_seed"]),
    )
    q_passed = q_ratio <= float(cfg["gates"]["q_flatness_max_ratio"])
    if not q_passed:
        raise SystemExit(
            f"Autoteste Q falhou: ratio={q_ratio:.3f} > "
            f"{float(cfg['gates']['q_flatness_max_ratio']):.3f}"
        )

    catalogs = ["GWTC-1-confident", "GWTC-2.1-confident", "GWTC-3-confident"]
    catalog = build_catalog(catalogs, workers=args.workers)
    if args.max_events > 0:
        catalog = catalog.head(args.max_events).copy()
    catalog.to_csv(catalog_path, index=False)

    log: list[dict] = []
    if any((args.download_strain, args.process, args.qgrams)):
        for row in catalog.itertuples(index=False):
            if pd.isna(row.gps):
                for detector in ("H1", "L1", "V1"):
                    log.append(
                        {
                            "event": row.name,
                            "detector": detector,
                            "status": "gps_missing",
                        }
                    )
                continue
            if args.download_strain:
                entries = save_event_strain(
                    row.name,
                    float(row.gps),
                    args.output_dir,
                    sample_rate=int(cfg["signal"]["sample_rate"]),
                )
            else:
                entries = existing_event_strain_entries(row.name, args.output_dir)
            log.extend(entries)
            for entry in entries:
                if entry.get("status") != "ok":
                    continue
                detector = entry["detector"]
                raw_path = Path(entry["raw_path"])
                processing_ok = True
                if args.process:
                    proc_path = (
                        args.output_dir
                        / "processed"
                        / row.name
                        / f"{detector}_processed.npy"
                    )
                    entry["processing"] = process_raw_file(
                        raw_path,
                        proc_path,
                        config=cfg,
                        event_gps=float(row.gps),
                    )
                    processing_ok = entry["processing"].get("status") == "ok"
                if args.qgrams:
                    if processing_ok or args.diagnostic_qgrams_for_rejected or not args.process:
                        q_path = (
                            args.output_dir
                            / "qgrams"
                            / row.name
                            / f"{detector}_qgram.npy"
                        )
                        entry["qtransform"] = gwpy_qtransform_consistent(
                            raw_path,
                            float(row.gps),
                            q_path,
                            frange=(float(qcfg["fmin_hz"]), float(qcfg["fmax_hz"])),
                            config=cfg,
                        )
                        if not processing_ok:
                            entry["qtransform"]["scientific_eligibility"] = False
                            entry["qtransform"]["upstream_processing_status"] = "rejected"
                    else:
                        entry["qtransform"] = {
                            "status": "skipped_upstream_rejected",
                            "reason": entry.get("processing", {}).get("reason"),
                            "scientific_eligibility": False,
                        }

    run_log_path.write_text(
        json.dumps(anonymize_paths(log), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = write_catalog_indexes(catalog_path, run_log_path, args.output_dir)
    write_manifest(
        args.output_dir / "manifest.json",
        {
            "catalogs": catalogs,
            "n_events": len(catalog),
            "config": cfg,
            "q_transform_self_test": {"ratio": q_ratio, "passed": q_passed},
            "catalog_quality_summary": summary,
            "arguments": vars(args)
            | {"output_dir": str(args.output_dir), "config": str(args.config)},
        },
    )
    print(f"Catalogo salvo em {catalog_path} ({len(catalog)} eventos)")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
