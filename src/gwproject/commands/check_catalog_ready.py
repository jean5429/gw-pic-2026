from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Verifica se os canais aprovados estao materialmente disponiveis")
    parser.add_argument("--approved", type=Path, default=Path("outputs/gwtc4_218/approved_channels.csv"))
    args = parser.parse_args()
    frame = pd.read_csv(args.approved)
    missing = []
    for row in frame.to_dict(orient="records"):
        for field in ("raw_path", "processed_path", "qgram_path"):
            value = row.get(field)
            if not value or not Path(str(value)).exists():
                missing.append(
                    {
                        "event": row.get("event"),
                        "detector": row.get("detector"),
                        "field": field,
                        "path": value,
                    }
                )
    summary = {
        "channels": int(len(frame)),
        "raw_ok": int(sum(Path(str(value)).exists() for value in frame["raw_path"])),
        "processed_ok": int(sum(Path(str(value)).exists() for value in frame["processed_path"])),
        "qgram_ok": int(sum(Path(str(value)).exists() for value in frame["qgram_path"])),
        "missing_count": len(missing),
        "missing": missing[:50],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
