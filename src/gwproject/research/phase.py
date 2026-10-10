from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def randomize_fourier_phase(
    strain: ArrayLike,
    *,
    fraction: float = 1.0,
    seed: int | None = None,
    preserve_mean: bool = True,
) -> FloatArray:
    """Randomiza a fase mantendo exatamente o modulo do espectro real.

    ``fraction=0`` devolve o sinal original (salvo erro de arredondamento) e
    ``fraction=1`` substitui inteiramente as fases nao triviais. Valores
    intermediarios interpolam circularmente entre fase original e aleatoria.
    """

    x = np.asarray(strain, dtype=float)
    if x.ndim != 1 or x.size < 4:
        raise ValueError("strain deve ser vetor 1D com pelo menos quatro amostras")
    if not np.all(np.isfinite(x)):
        raise ValueError("strain contem NaN ou infinito")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction deve pertencer a [0, 1]")
    spectrum = np.fft.rfft(x)
    magnitude = np.abs(spectrum)
    original_phase = np.angle(spectrum)
    rng = np.random.default_rng(seed)
    random_phase = rng.uniform(-np.pi, np.pi, size=spectrum.size)
    random_phase[0] = original_phase[0]
    if x.size % 2 == 0:
        random_phase[-1] = original_phase[-1]
    phase_delta = np.angle(np.exp(1j * (random_phase - original_phase)))
    new_phase = original_phase + fraction * phase_delta
    randomized = np.fft.irfft(magnitude * np.exp(1j * new_phase), n=x.size)
    if preserve_mean:
        randomized += x.mean() - randomized.mean()
    return np.asarray(randomized, dtype=float)
