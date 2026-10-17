from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil

import pandas as pd

from gwproject.catalog_index import write_catalog_indexes
from gwproject.catalog_metadata import detector_frame_availability, refresh_catalog_dataframe
from gwproject.provenance import write_manifest


def _atomic_csv(frame: pd.DataFrame, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Atualiza metadados GWOSC v2 sem baixar novamente strain ou Q-grams"
    )
    parser.add_argument("--catalog", type=Path, default=Path("outputs/gwtc4_218/catalog.csv"))
    parser.add_argument("--run-log", type=Path, default=Path("outputs/gwtc4_218/run_log.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gwtc4_218"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--no-rebuild-index", action="store_true")
    args = parser.parse_args()

    if not args.catalog.exists():
        raise FileNotFoundError(args.catalog)
    old = pd.read_csv(args.catalog)
    refreshed = refresh_catalog_dataframe(old, workers=args.workers)
    availability = detector_frame_availability(refreshed)
    if availability["events_with_detector_chirp_mass"] == 0:
        raise SystemExit(
            "A atualizacao nao recuperou nenhuma massa detector-frame; catalog.csv foi preservado."
        )

    backup = None
    if not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = args.catalog.with_name(f"{args.catalog.stem}.before_metadata_refresh_{stamp}.csv")
        shutil.copy2(args.catalog, backup)
    _atomic_csv(refreshed, args.catalog)

    index_summary = None
    if not args.no_rebuild_index:
        index_summary = write_catalog_indexes(args.catalog, args.run_log, args.output_dir)

    summary = {
        **availability,
        "catalog": str(args.catalog),
        "backup": str(backup) if backup else None,
        "index_rebuilt": not args.no_rebuild_index,
        "catalog_index_summary": index_summary,
    }
    summary_path = args.output_dir / "catalog_metadata_refresh_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_manifest(
        args.output_dir / "catalog_metadata_refresh_manifest.json",
        {
            "arguments": vars(args),
            "summary": summary,
            "scope": "Somente metadados; strain, processados e Q-grams nao sao alterados.",
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
