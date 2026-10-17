from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Agrega resultados de varias sementes")
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("models/gravity_spy_harmonized_v1d1"),
    )
    args = parser.parse_args()
    summaries = []
    for path in sorted(args.root.glob("seed_*/summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    if len(summaries) < 2:
        raise SystemExit("Sao necessarias pelo menos duas sementes concluídas")

    metrics = ["macro_f1", "balanced_accuracy", "kappa", "accuracy"]
    aggregated = {"n_seeds": len(summaries), "seeds": [s["seed"] for s in summaries], "metrics": {}}
    for metric in metrics:
        values = np.array([s["test_metrics"][metric] for s in summaries], dtype=float)
        aggregated["metrics"][metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
            "values": values.tolist(),
        }
    target = args.root / "aggregate_summary.json"
    target.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
    print(json.dumps(aggregated, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
