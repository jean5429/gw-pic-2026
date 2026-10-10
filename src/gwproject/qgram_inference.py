from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

UInt8Array = NDArray[np.uint8]

# Escala de cor dos PNGs do Gravity Spy: energia normalizada de 0 a 25,5 em viridis.
GRAVITY_SPY_ENERGY_MAX = 25.5


def qgram_to_uint8(
    qgram: ArrayLike,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.5,
    source_duration_seconds: float | None = None,
    target_duration_seconds: float | None = None,
    baseline_mode: str = "none",
    flip_frequency: bool = False,
    layout: str = "image",
    render_mode: str = "percentile",
) -> UInt8Array:
    """Converte um Q-gram em imagem uint8 (linhas = frequencia, colunas = tempo).

    ``layout="image"`` espera a matriz ja como (frequencia, tempo).
    ``layout="gwpy"`` espera a saida crua de ``gwpy.Spectrogram`` salva pelo
    pipeline GWOSC, que tem forma (tempo, frequencia); ela e transposta antes
    do recorte temporal. Em ambos os casos a linha 0 e a menor frequencia, e
    ``flip_frequency=True`` coloca as frequencias altas no topo (Gravity Spy).

    ``render_mode="percentile"`` aplica baseline + log1p + estiramento por
    percentis. ``render_mode="gravity_spy"`` reproduz a escala dos PNGs de
    treino: energia normalizada (gwpy ``norm="median"``) recortada em
    [0, 25,5], colorida em viridis e convertida para cinza (ITU-R 601, como o
    ``convert("L")``/``Grayscale`` do treino); baseline e percentis sao ignorados.
    """
    array = np.asarray(qgram, dtype=float)
    if array.ndim != 2:
        raise ValueError("qgram deve ser uma matriz 2D")
    layout = str(layout).lower()
    if layout == "gwpy":
        array = array.T
    elif layout != "image":
        raise ValueError("layout deve ser image ou gwpy")
    if not np.all(np.isfinite(array)):
        raise ValueError("qgram contem NaN ou infinito")
    if source_duration_seconds is not None or target_duration_seconds is not None:
        if source_duration_seconds is None or target_duration_seconds is None:
            raise ValueError("source e target duration devem ser fornecidos juntos")
        if not 0 < target_duration_seconds <= source_duration_seconds:
            raise ValueError("duracoes de Q-gram invalidas")
        width = array.shape[1]
        target_width = max(1, int(round(width * target_duration_seconds / source_duration_seconds)))
        start = max(0, (width - target_width) // 2)
        array = array[:, start : start + target_width]
    render_mode = str(render_mode).lower()
    if render_mode == "gravity_spy":
        if flip_frequency:
            array = array[::-1, :]
        return _gravity_spy_gray(array)
    if render_mode != "percentile":
        raise ValueError("render_mode deve ser percentile ou gravity_spy")
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("percentis invalidos")
    baseline_mode = str(baseline_mode).lower()
    if baseline_mode == "none":
        excess = np.maximum(array, 0.0)
    elif baseline_mode == "subtract_one":
        excess = np.maximum(array - 1.0, 0.0)
    elif baseline_mode == "global_median":
        baseline = float(np.median(array))
        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError("baseline global invalido")
        excess = np.maximum(array / baseline - 1.0, 0.0)
    elif baseline_mode == "per_frequency_median":
        baseline = np.median(array, axis=1, keepdims=True)
        if not np.all(np.isfinite(baseline)) or np.any(baseline <= 0):
            raise ValueError("baseline por frequencia invalido")
        excess = np.maximum(array / baseline - 1.0, 0.0)
    else:
        raise ValueError(
            "baseline_mode deve ser none, subtract_one, global_median ou per_frequency_median"
        )
    transformed = np.log1p(excess)
    if flip_frequency:
        transformed = transformed[::-1, :]
    low, high = np.percentile(transformed, [lower_percentile, upper_percentile])
    if not np.isfinite(high) or high <= low:
        raise ValueError("qgram sem contraste")
    normalized = np.clip((transformed - low) / (high - low), 0.0, 1.0)
    return np.round(255.0 * normalized).astype(np.uint8)


def _gravity_spy_gray(energy: NDArray[np.float64]) -> UInt8Array:
    from matplotlib import colormaps

    scaled = np.clip(energy / GRAVITY_SPY_ENERGY_MAX, 0.0, 1.0)
    rgb = np.round(255.0 * colormaps["viridis"](scaled)[..., :3])
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    return np.clip(np.round(gray), 0, 255).astype(np.uint8)


def load_gravity_spy_checkpoint(path: str | Path, device: Any):
    import torch

    from .gravity_spy import build_resnet18_grayscale

    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    classes = list(checkpoint["classes"])
    profile = checkpoint.get("profile") or {}
    model = build_resnet18_grayscale(
        len(classes),
        pretrained=False,
        conv1_stride=int(profile.get("conv1_stride", 2)),
        use_maxpool=bool(profile.get("use_maxpool", False)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, classes, profile


def probability_entropy(probabilities: ArrayLike) -> float:
    p = np.asarray(probabilities, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) == 0:
        return float("nan")
    entropy = -float(np.sum(p * np.log(p)))
    return entropy / float(np.log(max(2, len(probabilities))))


class CatalogQgramDataset:
    """Dataset picklable de Q-grams para inferencia com workers no Windows."""

    def __init__(
        self,
        frame,
        transform,
        *,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.5,
        source_duration_seconds: float | None = None,
        target_duration_seconds: float | None = None,
        baseline_mode: str = "none",
        flip_frequency: bool = False,
        layout: str = "gwpy",
        render_mode: str = "percentile",
    ):
        self.records = frame[["event", "detector", "qgram_path"]].to_dict(orient="records")
        self.transform = transform
        self.lower_percentile = float(lower_percentile)
        self.upper_percentile = float(upper_percentile)
        self.source_duration_seconds = source_duration_seconds
        self.target_duration_seconds = target_duration_seconds
        self.baseline_mode = str(baseline_mode)
        self.flip_frequency = bool(flip_frequency)
        self.layout = str(layout)
        self.render_mode = str(render_mode)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image

        row = self.records[index]
        array = np.load(row["qgram_path"])
        image_array = qgram_to_uint8(
            array,
            lower_percentile=self.lower_percentile,
            upper_percentile=self.upper_percentile,
            source_duration_seconds=self.source_duration_seconds,
            target_duration_seconds=self.target_duration_seconds,
            baseline_mode=self.baseline_mode,
            flip_frequency=self.flip_frequency,
            layout=self.layout,
            render_mode=self.render_mode,
        )
        image = Image.fromarray(image_array)
        return self.transform(image), index
