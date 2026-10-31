"""Figuras do artigo de revisao didatico (docs/relatorios/Artigo_Revisao_Didatica_GW.docx).

Executar na raiz do projeto, com o ambiente gwproject:

    python docs/relatorios/fontes/artigo_final/make_figures.py

Usa os dados locais do GW150914 (outputs/gwtc4_218), o historico de treino das 5 seeds
(models/gravity_spy_harmonized_v1d1) e os relatorios dos testes de fechamento
(outputs/research/plano1 e plano1_v1_1). Grava os PNGs nesta pasta e numeros.json com
os valores calculados aqui e citados no texto.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy.stats import theilslopes

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))

from gwproject.alignment import best_lag_alignment  # noqa: E402
from gwproject.preprocessing import bandpass_butter  # noqa: E402

EVENT = "GW150914-v4"
EVENT_GPS = 1126259462.4
RAW = ROOT / "outputs/gwtc4_218/raw" / EVENT
PROC = ROOT / "outputs/gwtc4_218/processed" / EVENT
QGRAMS = ROOT / "outputs/gwtc4_218/qgrams" / EVENT
RIDGE = ROOT / "outputs/research/gwtc4_218/ridge_diagnostic"
MODELS = ROOT / "models/gravity_spy_harmonized_v1d1"
PLANO1 = ROOT / "outputs/research/plano1/condicoes"
PLANO11 = ROOT / "outputs/research/plano1_v1_1/condicoes"
DPI = 300

# Paleta de referencia (skill dataviz): tinta, grade e slots categoricos em ordem fixa.
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

G_SI, C_SI, MSUN_SI = 6.67430e-11, 2.99792458e8, 1.98847e30

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2,
    "axes.titlesize": 9,
    "axes.titleweight": "semibold",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
})

NUMBERS: dict[str, object] = {}


def comma(fmt: str = "{:.1f}") -> FuncFormatter:
    return FuncFormatter(lambda value, _pos: fmt.format(value).replace(".", ",").replace("-", "−"))


def br(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",").replace("-", "−")


def wash(color: str, alpha: float) -> tuple[float, float, float]:
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(alpha * rgb + (1.0 - alpha))


def save(fig, name: str) -> None:
    fig.savefig(HERE / name, dpi=DPI)
    plt.close(fig)
    print("ok", name)


def load_raw(detector: str) -> tuple[np.ndarray, np.ndarray, float]:
    with h5py.File(RAW / f"{detector}.hdf5", "r") as handle:
        dataset = handle[f"{detector}:Strain"]
        values = np.asarray(dataset[()], dtype=float)
        x0, dx = float(dataset.attrs["x0"]), float(dataset.attrs["dx"])
    times = x0 + dx * np.arange(len(values)) - EVENT_GPS
    return times, values, 1.0 / dx


def load_qgram(detector: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    meta = json.loads((QGRAMS / f"{detector}_qgram.json").read_text(encoding="utf-8"))
    energy = np.load(QGRAMS / f"{detector}_qgram.npy")  # (tempo, frequencia), layout gwpy
    n_time, n_freq = energy.shape
    span = float(meta["outseg_seconds"])
    times = -span / 2 + span * (np.arange(n_time) + 0.5) / n_time
    frequencies = np.geomspace(meta["frange"][0], meta["frange"][1], n_freq)
    return times, frequencies, energy


# ----------------------------------------------------------------------------- fluxo
def box(ax, x, y, w, h, title, body, color, title_size=7.2, body_size=6.3):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
        linewidth=0.9, edgecolor=color, facecolor=wash(color, 0.10),
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.11, title, ha="center", va="top", fontsize=title_size,
            fontweight="semibold", color=INK)
    if body:
        ax.text(x + w / 2, y + h - 0.29, body, ha="center", va="top", fontsize=body_size,
                color=INK_2, linespacing=1.25)


def arrow(ax, start, end, color=INK_2):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8,
                                 linewidth=0.9, color=color, shrinkA=0, shrinkB=0))


def fig_fluxo() -> None:
    width, height = 6.3, 2.95
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    blue, orange, aqua = SLOTS[0], SLOTS[1], SLOTS[2]
    margin, gap, n = 0.05, 0.12, 6
    w = (width - 2 * margin - (n - 1) * gap) / n
    h, top_y = 0.80, 1.58
    xs = [margin + i * (w + gap) for i in range(n)]
    main = [
        ("Dados", "strain h(t)\nGWOSC, 4.096 Hz\n218 eventos GWTC-4", blue),
        ("Condicionamento", "ruído (PSD)\nbranqueamento\npassa-banda", blue),
        ("Transformada Q", "Q-gram 20–512 Hz\nimagem 140 × 170\nem tons de cinza", blue),
        ("Rede neural", "ensemble de\n5 ResNet-18\n22 classes", aqua),
        ("“Conheço isto?”", "teste OOD:\nenergia e\nMahalanobis", aqua),
        ("Resposta", "classe e decisão:\nrejeitar, revisar\nou candidato", aqua),
    ]
    for x, (title, body, color) in zip(xs, main):
        box(ax, x, top_y, w, h, title, body, color)
    for i in range(n - 1):
        arrow(ax, (xs[i] + w, top_y + h / 2), (xs[i + 1], top_y + h / 2))
    # A "camera": as tres primeiras etapas.
    bracket_y = top_y + h + 0.12
    ax.plot([xs[0], xs[0], xs[2] + w, xs[2] + w], [bracket_y - 0.05, bracket_y, bracket_y, bracket_y - 0.05],
            color=INK_2, lw=0.8)
    ax.text((xs[0] + xs[2] + w) / 2, bracket_y + 0.06, "a nossa “câmera”: do sinal à imagem",
            ha="center", va="bottom", fontsize=7.6, color=INK, fontstyle="italic")
    # Ramos inferiores.
    low_y, low_h = 0.45, 0.66
    bx0, bw = xs[1], xs[2] + w - xs[1]
    box(ax, bx0, low_y, bw, low_h, "Validação física (GW150914)",
        "alinhamento H1 × L1 · ridge (massa de chirp)\nfiltro casado (SNR)", orange)
    arrow(ax, (xs[1] + w / 2, top_y), (xs[1] + w / 2, low_y + low_h))
    tx0, tw = xs[3], xs[4] + w - xs[3]
    box(ax, tx0, low_y, tw, low_h, "Treino supervisionado",
        "Gravity Spy: 7.807 imagens rotuladas\n5 seeds · AdamW · parada antecipada", aqua)
    arrow(ax, (xs[3] + w / 2, low_y + low_h), (xs[3] + w / 2, top_y))
    gx = xs[5]
    box(ax, gx, low_y, w, low_h, "Explicação", "Grad-CAM:\nonde a rede olhou", aqua)
    arrow(ax, (xs[5] + w / 2, top_y), (xs[5] + w / 2, low_y + low_h))
    # Legenda das tres partes do estudo.
    legend = [("Parte I – dados e condicionamento", blue), ("Parte II – validação física", orange),
              ("Parte III – aprendizado de máquina", aqua)]
    lx = margin
    for label, color in legend:
        ax.add_patch(Rectangle((lx, 0.1), 0.12, 0.12, facecolor=wash(color, 0.25), edgecolor=color, lw=0.9))
        ax.text(lx + 0.18, 0.16, label, va="center", fontsize=7.0, color=INK_2)
        lx += 2.12
    save(fig, "fluxo_estudo.png")


# ----------------------------------------------------------------------------- ruido
def fig_ruido() -> None:
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    curves = {}
    for detector, color in (("H1", SLOTS[0]), ("L1", SLOTS[1])):
        psd = np.load(PROC / f"{detector}_processed.psd.npz")
        f, p = psd["frequencies"], psd["values"]
        keep = (f >= 10) & (f <= 2000)
        asd = np.sqrt(p[keep])
        curves[detector] = (f[keep], asd)
        ax.loglog(f[keep], asd, color=color, lw=1.1, label=detector)
    f_l1, a_l1 = curves["L1"]
    f_h1, a_h1 = curves["H1"]

    def asd_at(freq, f, a, window=3.0):
        sel = (f > freq - window) & (f < freq + window)
        return float(a[sel].max())

    notes = [
        (13.0, "parede sísmica (< 20 Hz)", 24.0, 40.0),
        (36.0, "linhas de calibração", 36.0, 12.0),
        (60.0, "rede elétrica (60 Hz)", 80.0, 8.0),
        (505.0, "modos de violino dos espelhos (~500 Hz)", 215.0, 3.0),
    ]
    for freq, label, text_x, up in notes:
        y = max(asd_at(freq, f_l1, a_l1), asd_at(freq, f_h1, a_h1))
        ax.annotate(label, xy=(freq, y), xytext=(text_x, y * up), fontsize=7.2, color=INK_2,
                    ha="center", va="bottom",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=1, shrinkB=2))
    floor_sel = (f_l1 > 80) & (f_l1 < 300)
    floor = float(np.median(a_l1[floor_sel]))
    ax.annotate("região mais sensível\n(~80–300 Hz)", xy=(160, floor), xytext=(160, floor / 6),
                fontsize=7.2, color=INK_2, ha="center", va="top",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=1, shrinkB=2))
    ax.set_xlim(10, 2000)
    ax.set_ylim(floor / 30, max(a_l1.max(), a_h1.max()) * 40)
    ax.set_xlabel("frequência (Hz)")
    ax.set_ylabel("ASD do ruído (1/√Hz)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}".replace(".", ",")))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([10, 20, 50, 100, 200, 500, 1000, 2000])
    ax.grid(True, which="major", axis="both")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncols=2)
    fig.tight_layout()
    NUMBERS["asd_floor_L1_80_300Hz"] = floor
    save(fig, "ruido_asd_gw150914.png")


# ----------------------------------------------------------------------------- do bruto ao Q-gram
def fig_sinal_ao_qgram() -> None:
    times, raw, fs = load_raw("L1")
    processed = np.load(PROC / "L1_processed.npy").astype(float)
    q_t, q_f, energy = load_qgram("L1")

    fig = plt.figure(figsize=(6.3, 6.6))
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 0.025], height_ratios=[1, 1, 1.35],
                            hspace=0.55, wspace=0.04)
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.plot(times, raw / 1e-19, color=SLOTS[0], lw=0.7)
    ax1.set_title("(a) Strain bruto h(t) do L1, 32 s: o sinal não aparece")
    ax1.set_xlabel("tempo em relação ao evento (s)")
    ax1.set_ylabel("h(t)  (× 10⁻¹⁹)")
    ax1.set_xlim(times[0], times[-1])
    ax1.yaxis.set_major_formatter(comma("{:.1f}"))
    ax1.grid(True, axis="y")
    ax1.set_axisbelow(True)

    ax2 = fig.add_subplot(grid[1, 0])
    sel = (times > -0.25) & (times < 0.08)
    ax2.plot(times[sel], processed[sel], color=SLOTS[0], lw=1.0)
    ax2.set_title("(b) Depois do branqueamento e do passa-banda (20–1.024 Hz)")
    ax2.set_xlabel("tempo em relação ao evento (s)")
    ax2.set_ylabel("amplitude branqueada (σ)")
    ax2.set_xlim(-0.25, 0.08)
    ax2.xaxis.set_major_formatter(comma("{:.2f}"))
    ax2.grid(True, axis="y")
    ax2.set_axisbelow(True)
    peak = float(times[sel][np.argmax(np.abs(processed[sel]))])
    ax2.annotate("chirp: a frequência e a amplitude\ncrescem até a fusão", xy=(peak, processed[sel].max()),
                 xytext=(-0.21, processed[sel].max() * 0.95), fontsize=7.2, color=INK_2, va="top",
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkB=3))

    ax3 = fig.add_subplot(grid[2, 0])
    tsel = (q_t >= -0.5) & (q_t <= 0.5)
    mesh = ax3.pcolormesh(q_t[tsel], q_f, energy[tsel].T, shading="auto", cmap="viridis",
                          vmin=0.0, vmax=25.5, rasterized=True)
    ax3.set_yscale("log")
    ax3.set_yticks([20, 50, 100, 200, 500])
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax3.yaxis.set_minor_formatter(NullFormatter())
    ax3.set_title("(c) Q-gram: energia normalizada no plano tempo × frequência (1 s central)")
    ax3.set_xlabel("tempo em relação ao evento (s)")
    ax3.set_ylabel("frequência (Hz)")
    ax3.xaxis.set_major_formatter(comma("{:.1f}"))
    for spine in ax3.spines.values():
        spine.set_visible(False)
    cax = fig.add_subplot(grid[2, 1])
    bar = fig.colorbar(mesh, cax=cax)
    bar.set_label("energia normalizada Z", color=INK_2)
    bar.outline.set_visible(False)
    bar.ax.yaxis.set_major_formatter(comma("{:.0f}"))
    fig.subplots_adjust(left=0.1, right=0.93, top=0.96, bottom=0.07)
    NUMBERS["qgram_L1_max_energy"] = float(energy.max())
    NUMBERS["processed_L1_core_std"] = float(np.std(processed[int(2 * fs):-int(2 * fs)]))
    save(fig, "gw150914_bruto_branqueado_qgram.png")


# ----------------------------------------------------------------------------- alinhamento
def fig_alinhamento() -> None:
    series = {}
    for detector in ("H1", "L1"):
        times, _raw, fs = load_raw(detector)
        white = np.load(PROC / f"{detector}_processed.npy").astype(float)
        series[detector] = bandpass_butter(white, fs, 35.0, 350.0, 4)
    window = (times > -0.12) & (times < 0.04)
    result = best_lag_alignment(series["H1"][window], series["L1"][window], fs,
                                max_lag_seconds=0.012, allow_sign_flip=True)
    lag_ms = 1000.0 * result.lag_seconds
    NUMBERS["alinhamento"] = {"lag_ms": lag_ms, "sign": result.sign, "correlation": result.correlation,
                              "janela_s": [-0.12, 0.04], "banda_hz": [35, 350], "dados": "branqueados"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.3, 4.3), sharex=True)
    show = (times > -0.15) & (times < 0.06)
    ax1.plot(times[show], series["H1"][show], color=SLOTS[0], lw=1.3, label="H1 (Hanford)")
    ax1.plot(times[show], series["L1"][show], color=SLOTS[1], lw=1.3, label="L1 (Livingston)")
    ax1.set_title("(a) Sinais branqueados e filtrados em 35–350 Hz, como chegaram")
    ax1.set_ylabel("amplitude (σ)")
    ax1.legend(loc="upper left", ncols=2)
    ax2.plot(times[show], series["H1"][show], color=SLOTS[0], lw=1.3, label="H1")
    ax2.plot(times[show] + result.lag_seconds, result.sign * series["L1"][show], color=SLOTS[1],
             lw=1.3, label=f"L1 deslocado {br(lag_ms)} ms e invertido")
    ax2.set_title(f"(b) Depois do alinhamento: correlação de {br(result.correlation, 2)}")
    ax2.set_ylabel("amplitude (σ)")
    ax2.set_xlabel("tempo em relação ao evento (s)")
    ax2.legend(loc="upper left", ncols=2)
    for ax in (ax1, ax2):
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(comma("{:.0f}"))
        ax.margins(y=0.25)
    ax2.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax2.xaxis.set_major_formatter(comma("{:.2f}"))
    ax2.set_xlim(-0.15, 0.06)
    fig.tight_layout()
    save(fig, "alinhamento_h1_l1.png")


# ----------------------------------------------------------------------------- ridge
def slope_to_mass(slope: float) -> float:
    return float((C_SI**3 / G_SI) * (5.0 * abs(slope) / 256.0) ** 0.6 * np.pi ** (-1.6) / MSUN_SI)


def mass_to_slope(mass: float) -> float:
    return float(-(256.0 / 5.0) * (np.pi ** 1.6 * mass * MSUN_SI * G_SI / C_SI**3) ** (5.0 / 3.0))


def fig_ridge() -> None:
    q_t, q_f, energy = load_qgram("H1")
    arrays = {d: np.load(RIDGE / "ridge_arrays" / f"{EVENT}__{d}.npz") for d in ("H1", "L1")}
    info = {d: json.loads((RIDGE / "channels" / f"{EVENT}__{d}.json").read_text(encoding="utf-8"))
            for d in ("H1", "L1")}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    tsel = (q_t >= -0.6) & (q_t <= 0.1)
    ax1.pcolormesh(q_t[tsel], q_f, energy[tsel].T, shading="auto", cmap="viridis", vmin=0, vmax=25.5,
                   rasterized=True)
    ax1.set_yscale("log")
    ax1.set_ylim(20, 512)
    ax1.set_yticks([20, 50, 100, 200, 500])
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax1.yaxis.set_minor_formatter(NullFormatter())
    plan = info["H1"]["plan"]
    ax1.add_patch(Rectangle((plan["window_start_seconds"], plan["fmin_hz"]),
                            plan["window_stop_seconds"] - plan["window_start_seconds"],
                            plan["fmax_hz"] - plan["fmin_hz"], fill=False, edgecolor="white", lw=0.9))
    ax1.scatter(arrays["H1"]["fit_times"], arrays["H1"]["fit_frequencies"], s=9, color=SLOTS[1],
                edgecolor="white", linewidth=0.4, zorder=3)
    ax1.text(plan["window_start_seconds"] + 0.005, plan["fmax_hz"] * 1.08, "janela do ridge",
             fontsize=6.8, color="white", va="bottom")
    ax1.set_title("(a) Trilha do sinal no Q-gram (H1)")
    ax1.set_xlabel("tempo (s)")
    ax1.set_ylabel("frequência (Hz)")
    ax1.xaxis.set_major_formatter(comma("{:.1f}"))
    for spine in ax1.spines.values():
        spine.set_visible(False)

    fits = {}
    for detector, color in (("H1", SLOTS[0]), ("L1", SLOTS[1])):
        t = np.asarray(arrays[detector]["fit_times"], dtype=float)
        f = np.asarray(arrays[detector]["fit_frequencies"], dtype=float)
        y = f ** (-8.0 / 3.0)
        slope, intercept, _lo, _hi = theilslopes(y, t)
        mass = slope_to_mass(slope)
        fits[detector] = {"slope": float(slope), "mass_solar": mass,
                          "pipeline_mass_solar": info[detector]["estimated_chirp_mass_solar"]}
        ax2.scatter(t, y * 1e4, s=10, color=color, edgecolor="white", linewidth=0.4, zorder=3,
                    label=f"{detector}: " + r"$\mathcal{M}$" + f" ≈ {br(mass)} " + r"$M_\odot$")
        grid_t = np.linspace(t.min(), t.max(), 50)
        ax2.plot(grid_t, (intercept + slope * grid_t) * 1e4, color=color, lw=1.4)
    reference = float(info["H1"]["reference_chirp_mass_solar"])
    t_h1 = np.asarray(arrays["H1"]["fit_times"], dtype=float)
    y_h1 = np.asarray(arrays["H1"]["fit_frequencies"], dtype=float) ** (-8.0 / 3.0)
    anchor_t, anchor_y = float(np.median(t_h1)), float(np.median(y_h1))
    grid_t = np.linspace(t_h1.min(), t_h1.max(), 50)
    ax2.plot(grid_t, (anchor_y + mass_to_slope(reference) * (grid_t - anchor_t)) * 1e4, color=MUTED,
             lw=1.0, label=f"inclinação do catálogo ({br(reference)} " + r"$M_\odot$)")
    ax2.set_title(r"(b) $f^{-8/3}$ contra $t$: a inclinação dá $\mathcal{M}$")
    ax2.set_xlabel("tempo (s)")
    ax2.set_ylabel(r"$f^{-8/3}$  (× 10$^{-4}$ s$^{8/3}$)")
    ax2.xaxis.set_major_formatter(comma("{:.1f}"))
    ax2.yaxis.set_major_formatter(comma("{:.1f}"))
    ax2.grid(True, axis="y")
    ax2.set_axisbelow(True)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, fontsize=7.2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    NUMBERS["ridge_gw150914"] = {"reference_solar": reference, **fits}
    save(fig, "ridge_gw150914.png")


# ----------------------------------------------------------------------------- arquitetura
def fig_arquitetura() -> None:
    try:
        from gwproject.gravity_spy import build_resnet18_grayscale

        model = build_resnet18_grayscale(22, pretrained=False, conv1_stride=2, use_maxpool=False)
        n_params = int(sum(p.numel() for p in model.parameters()))
        import torch

        shapes = {}
        hooks = []
        for name in ("conv1", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"):
            module = getattr(model, name)
            hooks.append(module.register_forward_hook(
                lambda _m, _i, out, name=name: shapes.__setitem__(name, tuple(out.shape[1:]))))
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, 1, 140, 170))
        for hook in hooks:
            hook.remove()
        NUMBERS["resnet18"] = {"n_params": n_params, "shapes": {k: list(v) for k, v in shapes.items()}}
    except Exception as error:  # noqa: BLE001
        print("aviso: sem torch para contar parametros:", error)
        shapes = {"conv1": (64, 70, 85), "layer1": (64, 70, 85), "layer2": (128, 35, 43),
                  "layer3": (256, 18, 22), "layer4": (512, 9, 11)}

    width, height = 6.3, 3.95
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    blue = SLOTS[0]
    axis_y = 2.62
    stages = [
        ("Entrada", "1 × 140 × 170", 140, 1, "imagem\nem cinza"),
        ("conv1", "64 × {} × {}".format(*shapes["conv1"][1:]), shapes["conv1"][1], 64, "7×7, passo 2\nBN + ReLU"),
        ("layer1", "64 × {} × {}".format(*shapes["layer1"][1:]), shapes["layer1"][1], 64, "2 blocos\nresiduais"),
        ("layer2", "128 × {} × {}".format(*shapes["layer2"][1:]), shapes["layer2"][1], 128, "2 blocos\npasso 2"),
        ("layer3", "256 × {} × {}".format(*shapes["layer3"][1:]), shapes["layer3"][1], 256, "2 blocos\npasso 2"),
        ("layer4", "512 × {} × {}".format(*shapes["layer4"][1:]), shapes["layer4"][1], 512,
         "2 blocos, passo 2\n(lida pelo Grad-CAM)"),
        ("GAP", "512", None, 512, "média\nespacial"),
        ("FC + softmax", "22", None, 22, "probabilidade\nde cada classe"),
    ]
    slot, x0 = 0.76, 0.1
    spans = []
    for i, (name, shape, spatial, channels, note) in enumerate(stages):
        if spatial is None:
            block_h, block_w = (0.95, 0.12) if name == "GAP" else (0.55, 0.12)
        else:
            block_h = 0.3 + 1.2 * spatial / 140.0
            block_w = 0.12 + 0.045 * np.log2(max(channels, 1))
        cx = x0 + slot * (i + 0.5)
        face = wash(blue, 0.12 + 0.05 * min(i, 5))
        ax.add_patch(FancyBboxPatch((cx - block_w / 2, axis_y - block_h / 2), block_w, block_h,
                                    boxstyle="round,pad=0,rounding_size=0.03", linewidth=0.9,
                                    edgecolor=blue, facecolor=face))
        spans.append((cx - block_w / 2, cx + block_w / 2))
        ax.text(cx, axis_y + 0.86, name, ha="center", va="bottom", fontsize=7.3, fontweight="semibold", color=INK)
        ax.text(cx, axis_y - 0.86, shape, ha="center", va="top", fontsize=6.6, color=INK)
        ax.text(cx, axis_y - 1.07, note, ha="center", va="top", fontsize=6.1, color=INK_2, linespacing=1.2)
    for (_l0, r0), (l1, _r1) in zip(spans[:-1], spans[1:]):
        arrow(ax, (r0 + 0.03, axis_y), (l1 - 0.03, axis_y))
    # Bloco residual.
    by, bh = 0.12, 0.62
    ax.text(0.12, 1.06, "Bloco residual (BasicBlock):  y = ReLU( F(x) + x )", fontsize=7.4,
            fontweight="semibold", color=INK, va="bottom")
    parts = ["x", "conv 3×3", "BN", "ReLU", "conv 3×3", "BN", "+", "ReLU", "y"]
    px = 0.12
    spans = []
    for label in parts:
        w = 0.24 if label in ("x", "y", "+") else (0.62 if label.startswith("conv") else 0.42)
        if label in ("x", "y"):
            ax.text(px + w / 2, by + bh / 2, label, ha="center", va="center", fontsize=8.4, color=INK,
                    fontstyle="italic")
        elif label == "+":
            ax.add_patch(matplotlib.patches.Circle((px + w / 2, by + bh / 2), 0.11, facecolor="white",
                                                   edgecolor=blue, lw=0.9))
            ax.text(px + w / 2, by + bh / 2, "+", ha="center", va="center", fontsize=9, color=INK)
        else:
            ax.add_patch(FancyBboxPatch((px, by + 0.14), w, bh - 0.28, boxstyle="round,pad=0,rounding_size=0.03",
                                        linewidth=0.8, edgecolor=blue, facecolor=wash(blue, 0.10)))
            ax.text(px + w / 2, by + bh / 2, label, ha="center", va="center", fontsize=6.9, color=INK)
        spans.append((px, px + w))
        px += w + 0.13
    for (_l0, r0), (l1, _r1) in zip(spans[:-1], spans[1:]):
        arrow(ax, (r0 + 0.02, by + bh / 2), (l1 - 0.02, by + bh / 2))
    xs, xp = (spans[0][0] + spans[0][1]) / 2, (spans[6][0] + spans[6][1]) / 2
    top = by + bh + 0.1
    ax.plot([xs, xs, xp], [by + bh / 2 + 0.12, top, top], color=MUTED, lw=0.9)
    arrow(ax, (xp, top), (xp, by + bh / 2 + 0.11), color=MUTED)
    ax.text((xs + xp) / 2, top + 0.03, "atalho (identidade): a entrada x é somada à saída", ha="center",
            va="bottom", fontsize=6.4, color=INK_2)
    save(fig, "arquitetura_resnet18.png")


# ----------------------------------------------------------------------------- treino
def fig_treino() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.75))
    best = {}
    for slot, seed in enumerate((11, 23, 42, 71, 101)):
        history = pd.read_csv(MODELS / f"seed_{seed}_cuda_balanced" / "training_history.csv")
        color = SLOTS[slot]
        ax1.plot(history["epoch"], history["val_macro_f1"], color=color, lw=1.4, label=f"seed {seed}")
        row = history.loc[history["val_macro_f1"].idxmax()]
        best[seed] = {"epoch": int(row["epoch"]), "val_macro_f1": float(row["val_macro_f1"]),
                      "epochs_run": int(history["epoch"].max())}
        ax1.scatter([row["epoch"]], [row["val_macro_f1"]], s=26, color=color, edgecolor="white",
                    linewidth=1.0, zorder=3)
    ax1.axvspan(0.5, 5.5, color=NEUTRAL, zorder=0)
    ax1.text(3.0, 0.985, "fase 1:\nbase\ncongelada", ha="center", va="top", fontsize=6.4, color=INK_2)
    ax1.set_title("(a) Macro-F1 na validação, 5 seeds")
    ax1.set_xlabel("época")
    ax1.set_ylabel("macro-F1 (validação)")
    ax1.set_ylim(0.4, 1.0)
    ax1.set_xlim(0.5, 30.5)
    ax1.yaxis.set_major_formatter(comma("{:.1f}"))
    ax1.grid(True, axis="y")
    ax1.set_axisbelow(True)
    ax1.legend(loc="lower right", fontsize=6.8, ncols=2)

    history = pd.read_csv(MODELS / "seed_42_cuda_balanced" / "training_history.csv")
    ax2.plot(history["epoch"], history["train_loss"], color=SLOTS[0], lw=1.4, label="treino")
    ax2.plot(history["epoch"], history["val_loss"], color=SLOTS[1], lw=1.4, label="validação")
    ax2.axvspan(0.5, 5.5, color=NEUTRAL, zorder=0)
    best_epoch = best[42]["epoch"]
    ax2.axvline(best_epoch, color=MUTED, lw=0.9)
    ax2.text(best_epoch - 0.4, 1.0, f"melhor época\n({best_epoch})", fontsize=6.6, color=INK_2,
             va="center", ha="right")
    ax2.set_title("(b) Função de perda, seed 42")
    ax2.set_xlabel("época")
    ax2.set_ylabel("entropia cruzada ponderada")
    ax2.set_xlim(0.5, history["epoch"].max() + 0.5)
    ax2.yaxis.set_major_formatter(comma("{:.1f}"))
    ax2.grid(True, axis="y")
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper right")
    fig.tight_layout()
    temperatures = {}
    for seed in (11, 23, 42, 71, 101):
        summary = json.loads((MODELS / f"seed_{seed}_cuda_balanced" / "summary.json").read_text(encoding="utf-8"))
        temperatures[seed] = summary["calibration"]["temperature"]
    NUMBERS["treino"] = {"melhor_epoca": best, "temperaturas": temperatures}
    save(fig, "curvas_treino.png")


# ----------------------------------------------------------------------------- fechamento por classe
def fig_f1_fechamento() -> None:
    old = pd.read_csv(PLANO1 / "B1_banda_gravity_spy/fechamento/per_class.csv")
    new = pd.read_csv(PLANO11 / "B1v11_banda_gravity_spy/fechamento/per_class.csv")
    data = old[["label", "n", "f1"]].merge(new[["label", "n", "f1"]], on="label", suffixes=("_v10", "_v11"))
    data = data[data["n_v11"] >= 10].sort_values("f1_v11")
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(6.3, 4.6))
    for i, row in enumerate(data.itertuples()):
        ax.plot([row.f1_v10, row.f1_v11], [i, i], color=GRID, lw=2.2, zorder=1)
    ax.scatter(data.f1_v10, y, s=30, color=MUTED, edgecolor="white", linewidth=1.0, zorder=2,
               label="v1.0 (receita da linha 2.0)")
    ax.scatter(data.f1_v11, y, s=36, color=SLOTS[0], edgecolor="white", linewidth=1.0, zorder=3,
               label="v1.1 (receita do Gravity Spy)")
    ax.axvline(0.85, color=AXIS, lw=0.9, zorder=0)
    ax.text(0.85, len(data) - 0.35, "0,85", ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_yticks(y, labels=[f"{row.label} ({row.n_v11})" for row in data.itertuples()])
    ax.set_xlabel("F1 no conjunto de fechamento (condição B1, ensemble de 5 modelos)")
    ax.set_xlim(-0.02, 1.04)
    ax.set_ylim(-0.7, len(data) - 0.1)
    ax.xaxis.set_major_formatter(comma("{:.1f}"))
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.legend(loc="lower right", fontsize=7.4)
    fig.tight_layout()
    NUMBERS["f1_fechamento"] = data.to_dict(orient="records")
    save(fig, "f1_fechamento_v10_v11.png")


# ----------------------------------------------------------------------------- evolucao
def report(path: Path) -> dict:
    return json.loads((path / "fechamento/report.json").read_text(encoding="utf-8"))


def fig_evolucao() -> None:
    reports = {
        "A1": report(PLANO1 / "A1_atual_percentile"),
        "A2": report(PLANO1 / "A2_atual_gravity_spy"),
        "B1": report(PLANO1 / "B1_banda_gravity_spy"),
        "B2": report(PLANO1 / "B2_banda_percentile"),
        "A2v11": report(PLANO11 / "A2v11_atual_gravity_spy"),
        "B1v11": report(PLANO11 / "B1v11_banda_gravity_spy"),
    }
    NUMBERS["fechamento"] = {
        name: {
            "n": r["evaluation"]["n"],
            "accuracy": r["evaluation"]["accuracy"],
            "macro_f1": r["evaluation"]["macro_f1"],
            "fractions": r["evaluation"]["open_world_fractions"],
            "bootstrap_95": r.get("bootstrap_95"),
            "passed": r["criteria"]["all_passed"],
        }
        for name, r in reports.items()
    }
    f1_rows = [
        ("Imagens do Gravity Spy", "linha v9 (Relatório V2)", 0.856),
        ("Imagens do Gravity Spy", "linha 2.0", 0.945),
        ("Nossa câmera (fechamento)", "A1 · modelos atuais + percentis", reports["A1"]["evaluation"]["macro_f1"]),
        ("Nossa câmera (fechamento)", "A2 · modelos atuais + escala GS", reports["A2"]["evaluation"]["macro_f1"]),
        ("Nossa câmera (fechamento)", "B1 · 20–512 Hz + escala GS (v1.0)", reports["B1"]["evaluation"]["macro_f1"]),
        ("Nossa câmera (fechamento)", "B1 · receita do Gravity Spy (v1.1)", reports["B1v11"]["evaluation"]["macro_f1"]),
    ]
    reject_rows = [
        ("Eventos GWTC-4 (475 canais)", "renderização por percentis", 1.0),
        ("Eventos GWTC-4 (475 canais)", "escala Gravity Spy", 253 / 475),
        ("Nossa câmera (fechamento)", "A1 · modelos atuais + percentis",
         reports["A1"]["evaluation"]["open_world_fractions"]["reject_ood"]),
        ("Nossa câmera (fechamento)", "A2 · modelos atuais + escala GS",
         reports["A2"]["evaluation"]["open_world_fractions"]["reject_ood"]),
        ("Nossa câmera (fechamento)", "B1 · 20–512 Hz + escala GS (v1.0)",
         reports["B1"]["evaluation"]["open_world_fractions"]["reject_ood"]),
        ("Nossa câmera (fechamento)", "B1 · receita do Gravity Spy (v1.1)",
         reports["B1v11"]["evaluation"]["open_world_fractions"]["reject_ood"]),
    ]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.3, 5.6))
    for ax, rows, goal, goal_label, title, xlabel in (
        (ax1, f1_rows, 0.85, "meta 0,85", "(a) Acerto equilibrado entre classes (macro-F1)", "macro-F1"),
        (ax2, reject_rows, 0.05, "meta 0,05", "(b) Fração rejeitada como “fora do domínio”", "fração rejeitada"),
    ):
        # Uma linha de cabecalho (sem barra) antes de cada grupo.
        entries: list[tuple[str, float | None]] = []
        last_group = None
        for group, label, value in rows:
            if group != last_group:
                entries.append((group, None))
                last_group = group
            entries.append((label, value))
        entries = entries[::-1]
        y = np.arange(len(entries))
        for yi, (label, value) in zip(y, entries):
            if value is None:
                continue
            ax.barh(yi, value, height=0.5, color=SLOTS[0], zorder=2)
            x_text = value + 0.012 if abs(value - goal) > 0.03 else max(value, goal) + 0.015
            ax.text(x_text, yi, f"{value:.3f}".replace(".", ","), va="center", fontsize=7.2, color=INK)
        ax.axvline(goal, color=INK_2, lw=0.9, zorder=3)
        ax.text(goal + 0.01, len(entries) - 0.55, goal_label, fontsize=7, color=INK_2, va="bottom")
        ax.set_yticks(y, labels=[label for label, _v in entries], fontsize=7.4)
        for tick, (_label, value) in zip(ax.get_yticklabels(), entries):
            if value is None:
                tick.set_fontstyle("italic")
                tick.set_fontweight("semibold")
                tick.set_color(INK)
        ax.set_xlim(0, 1.12)
        ax.set_ylim(-0.6, len(entries) - 0.1)
        ax.xaxis.set_major_formatter(comma("{:.1f}"))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(True, axis="x")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
    fig.tight_layout(h_pad=1.6)
    save(fig, "evolucao_resultados.png")


def main() -> int:
    fig_fluxo()
    fig_ruido()
    fig_sinal_ao_qgram()
    fig_alinhamento()
    fig_ridge()
    fig_arquitetura()
    fig_treino()
    fig_f1_fechamento()
    fig_evolucao()
    (HERE / "numeros.json").write_text(json.dumps(NUMBERS, indent=2, ensure_ascii=False, default=float),
                                       encoding="utf-8")
    print(json.dumps({k: v for k, v in NUMBERS.items() if k not in ("f1_fechamento",)}, indent=1,
                     ensure_ascii=False, default=float)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
