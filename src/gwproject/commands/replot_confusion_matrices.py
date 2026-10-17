"""Regenera confusion_matrix.png a partir de confusion_matrix.npy, sem retreinar."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gwproject.plotting import FIGURE_DPI, plot_confusion_matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        type=Path,
        nargs="*",
        default=[Path("models/gravity_spy_harmonized_v1d1")],
        help="pastas com subpastas seed_* contendo confusion_matrix.npy e summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "grava <output-dir>/<raiz>/<seed>/confusion_matrix.png; sem esta opcao, "
            "substitui o PNG ao lado de cada .npy"
        ),
    )
    args = parser.parse_args()

    runs = sorted({path.parent for root in args.roots for path in root.glob("*/confusion_matrix.npy")})
    if not runs:
        raise SystemExit("nenhum confusion_matrix.npy encontrado")
    for run_dir in runs:
        cm = np.load(run_dir / "confusion_matrix.npy", allow_pickle=False)
        classes = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["classes"]
        if cm.shape != (len(classes), len(classes)):
            raise SystemExit(f"{run_dir}: matriz {cm.shape} incompativel com {len(classes)} classes")
        target_dir = run_dir if args.output_dir is None else args.output_dir / run_dir.parent.name / run_dir.name
        target = target_dir / "confusion_matrix.png"
        plot_confusion_matrix(cm, classes, target)
        print(f"Salvo: {target} ({FIGURE_DPI} DPI)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
