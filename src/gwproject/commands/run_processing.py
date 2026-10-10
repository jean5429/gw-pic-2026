from __future__ import annotations

import argparse
import json
from pathlib import Path

from gwproject.config import load_toml
from gwproject.gwosc_pipeline import process_raw_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Processa um HDF5 de strain com gates auditaveis")
    parser.add_argument("raw_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/processing.toml"))
    args = parser.parse_args()
    result = process_raw_file(args.raw_file, args.output_file, config=load_toml(args.config))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
