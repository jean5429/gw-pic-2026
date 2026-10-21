"""Figuras do Relatorio de Resultados (linha gwproject 2.0).

Executar na raiz do projeto, com o ambiente gwproject e os modelos em
models/gravity_spy_harmonized_v1d1/:

    python docs/relatorios/fontes/resultados/make_figures.py

Le v9_per_class_f1.csv (medias das 5 seeds da linha v9, extraidas da branch v9)
e os summary.json/confusion_matrix.npy dos modelos da linha 2.0.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gwproject.plotting import FIGURE_DPI

HERE = Path(__file__).resolve().parent
MODELS = Path("models/gravity_spy_harmonized_v1d1")
V9, V20 = "#898781", "#2a78d6"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 9.5, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def per_class_f1_20() -> pd.DataFrame:
    rows = {}
    for path in sorted(MODELS.glob("seed_*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        for name, value in summary["per_class_f1"].items():
            rows.setdefault(name, []).append(value)
    return pd.DataFrame(
        {"classe": list(rows), "f1_20": [np.mean(v) for v in rows.values()],
         "min_20": [np.min(v) for v in rows.values()], "max_20": [np.max(v) for v in rows.values()]}
    )


def f1_comparison() -> None:
    v9 = pd.read_csv(HERE / "v9_per_class_f1.csv")
    data = per_class_f1_20().merge(v9, on="classe").sort_values("f1_20")
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for i, row in enumerate(data.itertuples()):
        ax.plot([row.f1_v9, row.f1_20], [i, i], color=GRID, lw=2, zorder=1)
    ax.scatter(data.f1_v9, y, s=34, color=V9, zorder=2, label="linha v9 (Relatório V2)", edgecolor="white", linewidth=0.8)
    ax.scatter(data.f1_20, y, s=40, color=V20, zorder=3, label="linha 2.0 (atual)", edgecolor="white", linewidth=0.8)
    ax.set_yticks(y, labels=data.classe)
    ax.set_xlabel("F1 no conjunto de teste (média das 5 seeds)")
    ax.set_xlim(0.4, 1.02)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "f1_por_classe_v9_vs_20.png", dpi=FIGURE_DPI)
    plt.close(fig)


def confusion_matrix() -> None:
    from gwproject.plotting import plot_confusion_matrix

    run = MODELS / "seed_42_cuda_balanced"
    classes = json.loads((run / "summary.json").read_text(encoding="utf-8"))["classes"]
    cm = np.load(run / "confusion_matrix.npy")
    plot_confusion_matrix(cm, classes, HERE / "matriz_confusao_seed42.png")


def main() -> None:
    f1_comparison()
    confusion_matrix()
    print(f"Figuras em {HERE}")


if __name__ == "__main__":
    main()
