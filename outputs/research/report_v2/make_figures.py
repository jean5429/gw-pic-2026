"""Gera os graficos do Relatorio V2 a partir das saidas dos ensembles.

Uso (na raiz do projeto): python outputs/research/report_v2/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gwproject.plotting import FIGURE_DPI

OUT = Path(__file__).resolve().parent
RESEARCH = OUT.parent
RUNS = {
    "v1 (transposto)": "gravity_spy_ensemble",
    "Transposição corrigida": "gravity_spy_ensemble_fixed",
    "Escala Gravity Spy": "gravity_spy_ensemble_gsrender",
}
MAHALANOBIS_Q99 = 637.8
SERIES = "#2a78d6"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.family": "Segoe UI",
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _clean_axes(ax, grid_axis="x", which="major"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.8, which=which)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False)


def class_counts(frames):
    counts = frames["Escala Gravity Spy"].predicted_class.value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(counts.index, counts.values, color=SERIES, height=0.62)
    for y, value in enumerate(counts.values):
        ax.text(value + 0.5, y, str(value), va="center", color=INK, fontsize=9)
    ax.set_xlabel("Canais (de 150)")
    ax.set_xlim(0, counts.max() + 5)
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_classes_gsrender.png", dpi=FIGURE_DPI)
    plt.close(fig)


def mahalanobis(frames):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, 3.0))
    for i, (label, frame) in enumerate(frames.items()):
        jitter = i + rng.uniform(-0.18, 0.18, len(frame))
        ax.scatter(frame.mahalanobis_distance, jitter, s=14, color=SERIES, alpha=0.55,
                   edgecolor="white", linewidth=0.4)
        median = frame.mahalanobis_distance.median()
        ax.plot([median, median], [i - 0.3, i + 0.3], color=INK, lw=2)
        ax.text(median, i + 0.36, f"mediana {median:,.0f}".replace(",", "."),
                ha="center", fontsize=8.5, color=INK)
    ax.axvline(MAHALANOBIS_Q99, color=INK_2, ls="--", lw=1.2)
    ax.text(MAHALANOBIS_Q99 * 1.06, 2.45, "limiar q99 = 638\n(dados de treino)",
            fontsize=8.5, color=INK_2, va="top")
    ax.set_xscale("log")
    ax.set_yticks(range(len(frames)), labels=list(frames))
    ax.set_ylim(-0.5, 2.6)
    ax.set_xlabel("Distância de Mahalanobis (escala log)")
    _clean_axes(ax, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig_mahalanobis.png", dpi=FIGURE_DPI)
    plt.close(fig)


def main():
    frames = {label: pd.read_csv(RESEARCH / folder / "per_channel.csv") for label, folder in RUNS.items()}
    class_counts(frames)
    mahalanobis(frames)
    print(f"Figuras salvas em {OUT} ({FIGURE_DPI} DPI)")


if __name__ == "__main__":
    main()
