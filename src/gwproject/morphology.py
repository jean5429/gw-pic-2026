from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .alignment import AlignmentResult, best_lag_alignment
from .preprocessing import apply_notches, bandpass_butter

FloatArray = NDArray[np.float64]


def physical_morphology_series(
    data: ArrayLike,
    fs: float,
    *,
    fmin: float = 35.0,
    fmax: float = 350.0,
    order: int = 4,
    notches_hz: ArrayLike = (),
    notch_q: float = 30.0,
) -> FloatArray:
    """Condicionamento em unidades fisicas: passa-banda e notches, sem whitening."""
    series = bandpass_butter(data, fs, fmin, fmax, order)
    frequencies = np.asarray(notches_hz, dtype=float).ravel()
    if len(frequencies):
        series = apply_notches(series, fs, frequencies, quality_factor=notch_q)
    return np.asarray(series, dtype=float)


def strict_interpolate_reference(times: ArrayLike, reference_table: ArrayLike) -> FloatArray:
    t = np.asarray(times, dtype=float).ravel()
    table = np.asarray(reference_table, dtype=float)
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError("reference_table deve ter ao menos duas colunas")
    rt, rv = table[:, 0], table[:, 1]
    if np.any(np.diff(rt) <= 0):
        raise ValueError("eixo temporal da referencia nao e estritamente crescente")
    tolerance = 0.51 * float(np.median(np.diff(rt)))
    if t.min() < rt.min() - tolerance or t.max() > rt.max() + tolerance:
        raise ValueError(
            f"janela [{t.min():.6f}, {t.max():.6f}] fora do dominio "
            f"[{rt.min():.6f}, {rt.max():.6f}]"
        )
    return np.interp(t, rt, rv)


def validate_morphology(
    ours: ArrayLike,
    reference: ArrayLike,
    fs: float,
    *,
    max_lag_seconds: float = 0.001,
    allow_sign_flip: bool = False,
) -> AlignmentResult:
    return best_lag_alignment(
        ours,
        reference,
        fs,
        max_lag_seconds=max_lag_seconds,
        allow_sign_flip=allow_sign_flip,
    )
