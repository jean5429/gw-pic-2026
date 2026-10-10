from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gwproject.research.domain_shift import grouped_domain_audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audita se estatisticas simples reproduzem rotulos Gravity Spy transferidos."
    )
    parser.add_argument("statistics_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research/domain_shift_audit"))
    parser.add_argument("--target-column", default="predicted_class")
    parser.add_argument("--positive-label", default="Scattered_Light")
    parser.add_argument("--group-column", default="event")
    args = parser.parse_args()
    frame = pd.read_csv(args.statistics_csv)
    result = grouped_domain_audit(
        frame,
        target_column=args.target_column,
        positive_label=args.positive_label,
        group_column=args.group_column,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "domain_shift_audit.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = []
    for feature, values in result["univariate"].items():
        rows.append({"feature": feature, **{k: v for k, v in values.items() if k not in {"thresholds", "directions"}}})
    pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False).to_csv(
        args.output_dir / "univariate_audit.csv", index=False
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
