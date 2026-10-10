from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AlignmentResult:
    correlation: float
    lag_samples: int
    lag_seconds: float
    sign: float
    a_aligned: FloatArray
    b_aligned: FloatArray


def _overlap(a: FloatArray, b: FloatArray, lag: int) -> tuple[FloatArray, FloatArray]:
    """Retorna trechos alinhados; lag > 0 significa que b e deslocada para a direita."""
    if lag > 0:
        return a[lag:], b[:-lag]
    if lag < 0:
        return a[:lag], b[-lag:]
    return a, b


def best_lag_alignment(
    a: ArrayLike,
    b: ArrayLike,
    fs: float,
    max_lag_seconds: float = 0.05,
    allow_sign_flip: bool = True,
) -> AlignmentResult:
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    n = min(len(x), len(y))
    if n < 4 or fs <= 0:
        raise ValueError("series ou fs invalidos")
    x, y = x[:n], y[:n]
    max_lag = min(n - 3, int(round(max_lag_seconds * fs)))

    best: tuple[float, int, float, FloatArray, FloatArray] | None = None
    for lag in range(-max_lag, max_lag + 1):
        xa, ya = _overlap(x, y, lag)
        if len(xa) < 3:
            continue
        sx, sy = float(np.std(xa)), float(np.std(ya))
        if sx <= 0 or sy <= 0:
            continue
        r = float(np.corrcoef(xa, ya)[0, 1])
        score = abs(r) if allow_sign_flip else r
        if best is None or score > best[0]:
            sign = -1.0 if allow_sign_flip and r < 0 else 1.0
            best = (score, lag, sign, xa.copy(), sign * ya.copy())

    if best is None:
        raise ValueError("nao foi possivel calcular correlacao")
    score, lag, sign, xa, ya = best
    return AlignmentResult(float(score), lag, lag / fs, sign, xa, ya)
