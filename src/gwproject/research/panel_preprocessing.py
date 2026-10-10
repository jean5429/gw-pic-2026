from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from PIL import Image

GrayArray = NDArray[np.uint8]


@dataclass(frozen=True)
class PanelCropConfig:
    """Configuracao explicita para harmonizar PNGs Gravity Spy.

    O detector automatico busca a maior regiao escura contigua. Os limites de
    frequencia assumem que o eixo vertical do painel original e logaritmico,
    com frequencias altas no topo.
    """

    source_fmin_hz: float = 10.0
    source_fmax_hz: float = 2048.0
    target_fmin_hz: float = 20.0
    target_fmax_hz: float = 512.0
    dark_threshold: int = 105
    row_dark_fraction: float = 0.35
    col_dark_fraction: float = 0.35
    padding_pixels: int = 1
    output_height: int = 140
    output_width: int = 170
    use_log_frequency: bool = True


def _largest_contiguous_run(indices: Iterable[int]) -> tuple[int, int]:
    values = list(int(v) for v in indices)
    if not values:
        raise ValueError("nenhuma regiao candidata encontrada")
    best_start = best_end = run_start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            if previous - run_start > best_end - best_start:
                best_start, best_end = run_start, previous
            run_start = value
        previous = value
    if previous - run_start > best_end - best_start:
        best_start, best_end = run_start, previous
    return best_start, best_end + 1


def detect_spectrogram_panel(
    image: Image.Image | np.ndarray,
    *,
    dark_threshold: int = 105,
    row_dark_fraction: float = 0.35,
    col_dark_fraction: float = 0.35,
    padding_pixels: int = 1,
) -> tuple[int, int, int, int]:
    """Retorna ``(left, top, right, bottom)`` da area espectral escura.

    A deteccao usa densidade de pixels escuros por linha e coluna. E adequada
    aos PNGs Gravity Spy, que possuem painel escuro em fundo branco. Se a
    deteccao falhar, a funcao interrompe explicitamente em vez de inventar um
    recorte silencioso.
    """

    gray = np.asarray(Image.fromarray(image).convert("L") if isinstance(image, np.ndarray) else image.convert("L"))
    if gray.ndim != 2 or min(gray.shape) < 8:
        raise ValueError("imagem pequena ou invalida para detectar painel")
    dark = gray <= int(dark_threshold)
    rows = np.flatnonzero(dark.mean(axis=1) >= float(row_dark_fraction))
    cols = np.flatnonzero(dark.mean(axis=0) >= float(col_dark_fraction))
    top, bottom = _largest_contiguous_run(rows)
    left, right = _largest_contiguous_run(cols)
    pad = max(0, int(padding_pixels))
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(gray.shape[1], right + pad)
    bottom = min(gray.shape[0], bottom + pad)
    if right - left < 8 or bottom - top < 8:
        raise ValueError("painel detectado e pequeno demais")
    return left, top, right, bottom


def frequency_row_bounds(
    height: int,
    *,
    source_fmin_hz: float,
    source_fmax_hz: float,
    target_fmin_hz: float,
    target_fmax_hz: float,
    log_frequency: bool = True,
) -> tuple[int, int]:
    """Mapeia uma banda fisica a linhas de imagem, com frequencia alta no topo."""

    if height < 2:
        raise ValueError("height invalido")
    values = [source_fmin_hz, source_fmax_hz, target_fmin_hz, target_fmax_hz]
    if any((not np.isfinite(v)) or v <= 0 for v in values):
        raise ValueError("frequencias devem ser positivas e finitas")
    if not source_fmin_hz < source_fmax_hz:
        raise ValueError("banda fonte invalida")
    lo = max(source_fmin_hz, target_fmin_hz)
    hi = min(source_fmax_hz, target_fmax_hz)
    if not lo < hi:
        raise ValueError("bandas de frequencia nao se sobrepoem")

    if log_frequency:
        denom = np.log(source_fmax_hz / source_fmin_hz)
        top_fraction = np.log(source_fmax_hz / hi) / denom
        bottom_fraction = np.log(source_fmax_hz / lo) / denom
    else:
        denom = source_fmax_hz - source_fmin_hz
        top_fraction = (source_fmax_hz - hi) / denom
        bottom_fraction = (source_fmax_hz - lo) / denom

    top = int(np.floor(np.clip(top_fraction, 0.0, 1.0) * height))
    bottom = int(np.ceil(np.clip(bottom_fraction, 0.0, 1.0) * height))
    top = max(0, min(height - 1, top))
    bottom = max(top + 1, min(height, bottom))
    return top, bottom


def crop_frequency_band(
    panel: Image.Image,
    *,
    source_fmin_hz: float = 10.0,
    source_fmax_hz: float = 2048.0,
    target_fmin_hz: float = 20.0,
    target_fmax_hz: float = 512.0,
    log_frequency: bool = True,
) -> Image.Image:
    top, bottom = frequency_row_bounds(
        panel.height,
        source_fmin_hz=source_fmin_hz,
        source_fmax_hz=source_fmax_hz,
        target_fmin_hz=target_fmin_hz,
        target_fmax_hz=target_fmax_hz,
        log_frequency=log_frequency,
    )
    return panel.crop((0, top, panel.width, bottom))


def robust_uint8(
    image: Image.Image | np.ndarray,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.5,
    invert: bool = False,
) -> GrayArray:
    array = np.asarray(Image.fromarray(image).convert("L") if isinstance(image, np.ndarray) else image.convert("L"), dtype=float)
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("percentis invalidos")
    low, high = np.percentile(array, [lower_percentile, upper_percentile])
    if not np.isfinite(high) or high <= low:
        raise ValueError("imagem sem contraste")
    normalized = np.clip((array - low) / (high - low), 0.0, 1.0)
    if invert:
        normalized = 1.0 - normalized
    return np.round(255.0 * normalized).astype(np.uint8)


def prepare_gravity_spy_panel(
    source: str | Path | Image.Image,
    *,
    config: PanelCropConfig = PanelCropConfig(),
    normalize: bool = False,
) -> Image.Image:
    """Remove elementos graficos e harmoniza a banda fisica do PNG Gravity Spy."""

    if isinstance(source, Image.Image):
        image = source.convert("L")
    else:
        with Image.open(source) as handle:
            image = handle.convert("L").copy()
    bbox = detect_spectrogram_panel(
        image,
        dark_threshold=config.dark_threshold,
        row_dark_fraction=config.row_dark_fraction,
        col_dark_fraction=config.col_dark_fraction,
        padding_pixels=config.padding_pixels,
    )
    panel = image.crop(bbox)
    panel = crop_frequency_band(
        panel,
        source_fmin_hz=config.source_fmin_hz,
        source_fmax_hz=config.source_fmax_hz,
        target_fmin_hz=config.target_fmin_hz,
        target_fmax_hz=config.target_fmax_hz,
        log_frequency=config.use_log_frequency,
    )
    if normalize:
        panel = Image.fromarray(robust_uint8(panel))
    return panel.resize(
        (config.output_width, config.output_height),
        Image.Resampling.BILINEAR,
    )
