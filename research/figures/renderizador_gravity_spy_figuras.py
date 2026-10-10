"""Figuras do experimento exploratorio de renderizacao (10/10/2026).

Executar na raiz do projeto, depois de classify-ensemble nos dois modos:

    python research/figures/renderizador_gravity_spy_figuras.py

Le apenas saidas existentes e grava dois PNGs em research/figures/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image

from gwproject.config import load_toml
from gwproject.plotting import FIGURE_DPI
from gwproject.qgram_inference import catalog_network_input, qgram_to_uint8, render_settings

RUN = Path("outputs/research/renderer_gravity_spy_20261010")
OUT = Path("research/figures")
MODES = ("percentile", "gravity_spy")
LABEL = {"percentile": "percentis (canônico)", "gravity_spy": "Gravity Spy (exploratório)"}
COLOR = {"percentile": "#2a78d6", "gravity_spy": "#eb6834"}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CHANNELS = [
    ("GW150914-v4", "L1"),
    ("GW170814-v4", "L1"),
    ("GW190917_114630-v1", "H1"),
    ("GW170817-v3", "H1"),
]
PANELS = [
    ("Chirp", "H1_BTGpb9EV7L"),
    ("Chirp", "H1_Nsd6qtc4FJ"),
    ("No_Glitch", "L1_DaU5LQcHtr"),
    ("No_Glitch", "H1_jx05R09EhV"),
]
STATUS = [
    ("reject_ood", "rejeitado (OOD)"),
    ("review", "revisão"),
    ("in_domain_candidate", "candidato\nem domínio"),
]


def _style(axis) -> None:
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(AXIS)
    axis.tick_params(colors=INK_2, labelsize=8)
    axis.yaxis.label.set_color(INK_2)
    axis.xaxis.label.set_color(INK_2)
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)


def _runs() -> dict[str, pd.DataFrame]:
    return {mode: pd.read_csv(RUN / f"ensemble_{mode}_cpu" / "per_channel.csv") for mode in MODES}


def inputs_figure(runs: dict[str, pd.DataFrame]) -> Path:
    cfg = load_toml("configs/research_program.toml")["harmonized_inference"]
    approved = pd.read_csv("outputs/gwtc4_218/approved_channels.csv").set_index(["event", "detector"])
    figure = Figure(figsize=(10.5, 7.2), facecolor=SURFACE)
    axes = figure.subplots(3, 4)
    for column, (event, detector) in enumerate(CHANNELS):
        qgram = np.load(approved.loc[(event, detector), "qgram_path"], allow_pickle=False)
        for row, mode in enumerate(MODES):
            pixels = catalog_network_input(
                qgram_to_uint8(qgram, **render_settings(cfg, mode)),
                height=int(cfg["resize_height"]),
                width=int(cfg["resize_width"]),
            )
            record = runs[mode].set_index(["event", "detector"]).loc[(event, detector)]
            axis = axes[row, column]
            axis.imshow(pixels, cmap="gray", vmin=0, vmax=255, aspect="auto")
            axis.set_title(
                f"{event.split('-')[0]} {detector}\n{record['forced_closed_set_class']} | {record['open_world_status']}",
                fontsize=7.5,
                color=INK,
            )
    metadata = Path("data/gravity_spy_harmonized/TrainingSet")
    for column, (label, stem) in enumerate(PANELS):
        with Image.open(metadata / label / f"{stem}_spectrogram_1.0.png") as image:
            pixels = np.asarray(image.convert("L"))
        axis = axes[2, column]
        axis.imshow(pixels, cmap="gray", vmin=0, vmax=255, aspect="auto")
        axis.set_title(f"Gravity Spy (validação)\n{label}", fontsize=7.5, color=INK)
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(AXIS)
    for row, text in enumerate([LABEL["percentile"], LABEL["gravity_spy"], "painéis de treino"]):
        axes[row, 0].set_ylabel(text, fontsize=8, color=INK_2)
    figure.suptitle(
        "Entrada da CNN (140 × 170, frequência alta no topo) | classe forçada | status open-world",
        fontsize=9.5,
        color=INK,
    )
    figure.tight_layout()
    path = OUT / "renderizador_gravity_spy_entradas.png"
    figure.savefig(path, dpi=FIGURE_DPI, facecolor=SURFACE)
    return path


def summary_figure(runs: dict[str, pd.DataFrame]) -> Path:
    catalog = pd.read_csv("outputs/gwtc4_218/catalog.csv")[["name", "snr"]].rename(columns={"name": "event"})
    figure = Figure(figsize=(12.5, 4.0), facecolor=SURFACE)
    left, middle, right = figure.subplots(1, 3)

    excess = {mode: runs[mode]["mahalanobis_log_excess_99"].to_numpy() for mode in MODES}
    edges = np.linspace(min(v.min() for v in excess.values()), max(v.max() for v in excess.values()), 46)
    for mode in MODES:
        counts, _ = np.histogram(excess[mode], bins=edges)
        left.stairs(counts, edges, color=COLOR[mode], linewidth=2, label=LABEL[mode])
    left.axvline(0.0, color=INK_2, linestyle="--", linewidth=1)
    left.text(0.03, left.get_ylim()[1] * 0.55, "limiar q99\n(validação)", color=INK_2, fontsize=7.5, va="top")
    left.set_xlabel("log(1 + d) − log(1 + q99), mediana dos 5 modelos")
    left.set_ylabel("canais")
    left.set_title("Distância de Mahalanobis ao domínio", fontsize=9.5, color=INK)
    left.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc="upper right")
    _style(left)

    positions = np.arange(len(STATUS))
    width = 0.38
    for offset, mode in zip((-width / 2, width / 2), MODES):
        counts = runs[mode]["open_world_status"].value_counts()
        values = [int(counts.get(key, 0)) for key, _ in STATUS]
        bars = middle.bar(positions + offset, values, width=width - 0.03, color=COLOR[mode], label=LABEL[mode])
        middle.bar_label(bars, labels=[str(v) for v in values], fontsize=7.5, color=INK, padding=2)
    middle.set_xticks(positions, [text for _, text in STATUS])
    middle.set_ylabel("canais (de 475)")
    middle.set_title("Decisão open-world do ensemble", fontsize=9.5, color=INK)
    middle.legend(frameon=False, fontsize=7.5, labelcolor=INK_2)
    _style(middle)

    terciles = None
    for mode in MODES:
        frame = runs[mode].assign(chirp=lambda d: (d["forced_closed_set_class"] == "Chirp").astype(float))
        events = frame.groupby("event")["chirp"].mean().to_frame().join(catalog.set_index("event"))
        if terciles is None:
            _, terciles = pd.qcut(events["snr"], 3, retbins=True)
        events["tercile"] = pd.cut(events["snr"], terciles, labels=False, include_lowest=True)
        fraction = events.groupby("tercile")["chirp"].mean().reindex(range(3)).to_numpy()
        right.plot(range(3), fraction, color=COLOR[mode], linewidth=2, marker="o", markersize=7,
                   label=LABEL[mode])
        right.annotate(f"{fraction[-1]:.2f}", (2, fraction[-1]), xytext=(6, 0), textcoords="offset points",
                       va="center", fontsize=7.5, color=INK)
    names = [f"{terciles[i]:.1f}-{terciles[i + 1]:.1f}" for i in range(3)]
    right.set_xticks(range(3), names)
    right.set_xlim(-0.3, 2.45)
    right.set_xlabel("tercil da SNR de rede do catálogo")
    right.set_ylabel("fração de canais Chirp (média por evento)")
    right.set_title("Classe forçada Chirp versus SNR", fontsize=9.5, color=INK)
    right.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc="upper left")
    _style(right)

    figure.suptitle(
        "Mesmos 475 canais, 5 checkpoints e referência OOD; muda apenas a renderização da intensidade",
        fontsize=9.5,
        color=INK,
    )
    figure.tight_layout()
    path = OUT / "renderizador_gravity_spy_ood.png"
    figure.savefig(path, dpi=FIGURE_DPI, facecolor=SURFACE)
    return path


def main() -> None:
    runs = _runs()
    for path in (inputs_figure(runs), summary_figure(runs)):
        print(f"Salvo: {path}")


if __name__ == "__main__":
    main()
