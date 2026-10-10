from __future__ import annotations

import argparse
import json
from pathlib import Path

from gwproject.catalog_index import write_catalog_indexes
from gwproject.provenance import write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Constroi os indices de qualidade do catalogo GWTC")
    parser.add_argument("--catalog", type=Path, default=Path("outputs/gwtc/catalog.csv"))
    parser.add_argument("--run-log", type=Path, default=Path("outputs/gwtc/run_log.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gwtc"))
    args = parser.parse_args()
    summary = write_catalog_indexes(args.catalog, args.run_log, args.output_dir)
    write_manifest(
        args.output_dir / "catalog_index_manifest.json",
        {
            "catalog": str(args.catalog),
            "run_log": str(args.run_log),
            "summary": summary,
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
