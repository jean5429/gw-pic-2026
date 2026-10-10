from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gwproject.plotting import FIGURE_DPI, plot_confusion_matrix


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenera confusion_matrix.png a partir de confusion_matrix.npy, sem retreinar"
    )
    parser.add_argument(
        "roots",
        type=Path,
        nargs="*",
        default=[Path("outputs/gravity_spy_harmonized"), Path("outputs/gravity_spy_60epochs")],
        help="pastas com subpastas seed_* (padrao: harmonized e 60epochs)",
    )
    args = parser.parse_args()

    runs = sorted(path.parent for root in args.roots for path in root.glob("*/confusion_matrix.npy"))
    if not runs:
        raise SystemExit("nenhum confusion_matrix.npy encontrado")
    for run_dir in runs:
        cm = np.load(run_dir / "confusion_matrix.npy")
        classes = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["classes"]
        if cm.shape != (len(classes), len(classes)):
            raise SystemExit(f"{run_dir}: matriz {cm.shape} incompativel com {len(classes)} classes")
        plot_confusion_matrix(cm, classes, run_dir / "confusion_matrix.png")
        print(f"Salvo: {run_dir / 'confusion_matrix.png'} ({FIGURE_DPI} DPI)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
