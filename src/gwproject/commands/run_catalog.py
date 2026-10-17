from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from gwproject.provenance import write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Catalogo e strain GWTC em fluxo modular auditavel")
    parser.add_argument("--config", type=Path, default=Path("configs/processing.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gwtc4_218"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--event-workers",
        type=int,
        default=4,
        help="eventos adquiridos/processados em paralelo",
    )
    parser.add_argument("--max-events", type=int, default=0, help="0 = todos")
    parser.add_argument(
        "--catalog-scope",
        choices=["gwtc4-218"],
        default="gwtc4-218",
        help="escopo canônico confirmatorio",
    )
    parser.add_argument("--download-strain", action="store_true")
    parser.add_argument(
        "--refresh-catalog",
        action="store_true",
        help="reconsulta a lista GWOSC; por padrao preserva metadados enriquecidos existentes",
    )
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

    catalogs = [
        "GWTC-1-confident",
        "GWTC-2.1-confident",
        "GWTC-3-confident",
        "GWTC-4.0",
    ]
    excluded_events = {
        "GW230630_070659": (
            "satisfazia o limiar inicial, mas o release GWTC-4 o marca como "
            "evidencia de origem instrumental"
        )
    }
    expected_event_count = 218
    reused_enriched_catalog = bool(catalog_path.is_file() and not args.refresh_catalog)
    if reused_enriched_catalog:
        catalog = pd.read_csv(catalog_path)
        bases = catalog["name"].astype(str).str.rsplit("-v", n=1).str[0]
        if (
            len(catalog) != expected_event_count
            or bases.nunique() != expected_event_count
            or bases.isin(excluded_events).any()
        ):
            raise RuntimeError(
                "catalogo local existente nao satisfaz o contrato canônico de 218 eventos"
            )
    else:
        catalog = build_catalog(
            catalogs,
            workers=args.workers,
            exclude_event_bases=excluded_events,
            expected_event_count=expected_event_count,
        )
    if args.max_events > 0:
        catalog = catalog.head(args.max_events).copy()
    catalog.to_csv(catalog_path, index=False)

    log: list[dict] = []
    if any((args.download_strain, args.process, args.qgrams)):
        def process_event(row: dict) -> list[dict]:
            event_log: list[dict] = []
            event_name = str(row["name"])
            gps_value = row.get("gps")
            if pd.isna(gps_value):
                for detector in ("H1", "L1", "V1"):
                    event_log.append(
                        {
                            "event": event_name,
                            "detector": detector,
                            "status": "gps_missing",
                        }
                    )
                return event_log
            if args.download_strain:
                entries = save_event_strain(
                    event_name,
                    float(gps_value),
                    args.output_dir,
                    sample_rate=int(cfg["signal"]["sample_rate"]),
                )
            else:
                entries = existing_event_strain_entries(event_name, args.output_dir)
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
                        / event_name
                        / f"{detector}_processed.npy"
                    )
                    entry["processing"] = process_raw_file(
                        raw_path,
                        proc_path,
                        config=cfg,
                        event_gps=float(gps_value),
                    )
                    processing_ok = entry["processing"].get("status") == "ok"
                if args.qgrams:
                    if processing_ok or args.diagnostic_qgrams_for_rejected or not args.process:
                        q_path = (
                            args.output_dir
                            / "qgrams"
                            / event_name
                            / f"{detector}_qgram.npy"
                        )
                        entry["qtransform"] = gwpy_qtransform_consistent(
                            raw_path,
                            float(gps_value),
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
            return entries

        rows = catalog.to_dict(orient="records")
        with ThreadPoolExecutor(max_workers=max(1, args.event_workers)) as executor:
            futures = {executor.submit(process_event, row): row["name"] for row in rows}
            for future in as_completed(futures):
                log.extend(future.result())
        log.sort(key=lambda item: (str(item.get("event")), str(item.get("detector"))))

    run_log_path.write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = write_catalog_indexes(catalog_path, run_log_path, args.output_dir)
    write_manifest(
        args.output_dir / "manifest.json",
        {
            "catalogs": catalogs,
            "catalog_scope": args.catalog_scope,
            "expected_event_count_before_debug_limit": expected_event_count,
            "excluded_events": excluded_events,
            "scope_sources": [
                "https://gwosc.org/GWTC-4.0/",
                "https://gwosc.org/eventapi/html/GWTC-4.0/",
                "https://arxiv.org/abs/2508.18082",
            ],
            "n_events": len(catalog),
            "reused_enriched_catalog": reused_enriched_catalog,
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
