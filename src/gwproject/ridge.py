from __future__ import annotations

from dataclasses import dataclass
from warnings import catch_warnings, simplefilter

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import theilslopes

G_SI = 6.67430e-11
C_SI = 2.99792458e8
MSUN_SI = 1.98847e30

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ChirpMassEstimate:
    mass_solar: float
    dispersion_low_solar: float
    dispersion_high_solar: float
    slope: float
    n_points: int
    n_bootstrap: int
    fit_method: str = "theil_sen"

    @property
    def ci_low_solar(self) -> float:
        """Alias legado; o intervalo nao inclui o vies sistematico da extracao."""
        return self.dispersion_low_solar

    @property
    def ci_high_solar(self) -> float:
        return self.dispersion_high_solar


def _mass_from_slope(slope: float) -> float:
    if slope >= 0 or not np.isfinite(slope):
        raise ValueError("a inclinacao fisica deve ser negativa")
    mass_si = (C_SI**3 / G_SI) * (5.0 * abs(slope) / 256.0) ** 0.6 * np.pi ** (-1.6)
    return float(mass_si / MSUN_SI)


def _least_squares_slope(x: FloatArray, y: FloatArray) -> float:
    x_centered = x - np.mean(x)
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 0 or not np.isfinite(denominator):
        return float("nan")
    y_centered = y - np.mean(y)
    return float(np.dot(x_centered, y_centered) / denominator)


def _negative_slope(x: FloatArray, y: FloatArray) -> tuple[float, str]:
    """Retorna uma inclinacao negativa robusta.

    Theil-Sen pode devolver exatamente zero em ridges quantizados em poucos bins,
    porque a mediana das inclinacoes par-a-par fica dominada por empates. Nessa
    situacao usamos regressao linear apenas como fallback, sem inverter sinal ou
    fabricar um chirp quando o caminho e realmente plano.
    """

    with catch_warnings():
        simplefilter("ignore", RuntimeWarning)
        slope, _, _, _ = theilslopes(y, x, 0.95)
    slope = float(slope)
    if np.isfinite(slope) and slope < 0:
        return slope, "theil_sen"

    fallback = _least_squares_slope(x, y)
    if np.isfinite(fallback) and fallback < 0:
        return fallback, "ols_fallback_quantized_ridge"

    raise ValueError("ridge sem inclinacao fisica negativa")


def estimate_chirp_mass_from_ridge_points(
    times: ArrayLike,
    frequencies: ArrayLike,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
    enforce_monotonic: bool = True,
) -> ChirpMassEstimate:
    t = np.asarray(times, dtype=float).ravel()
    f = np.asarray(frequencies, dtype=float).ravel()
    mask = np.isfinite(t) & np.isfinite(f) & (f > 0)
    t, f = t[mask], f[mask]
    if len(t) < 10:
        raise ValueError("sao necessarios pelo menos 10 pontos")
    order = np.argsort(t)
    t, f = t[order], f[order]
    if np.ptp(t) <= 0:
        raise ValueError("ridge sem extensao temporal")
    if enforce_monotonic:
        f = np.maximum.accumulate(f)
    if np.ptp(f) <= 0:
        raise ValueError("ridge com frequencia constante")

    y = f ** (-8.0 / 3.0)
    slope, fit_method = _negative_slope(t, y)
    mass = _mass_from_slope(slope)

    rng = np.random.default_rng(seed)
    boot: list[float] = []
    n = len(t)
    block = max(2, int(round(np.sqrt(n))))
    for _ in range(n_bootstrap):
        starts = rng.integers(0, max(1, n - block + 1), size=int(np.ceil(n / block)))
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        tb, yb = t[idx], y[idx]
        order_b = np.argsort(tb)
        tb, yb = tb[order_b], yb[order_b]
        if len(np.unique(tb)) < 3:
            continue
        try:
            sb, _ = _negative_slope(tb, yb)
        except ValueError:
            continue
        boot.append(_mass_from_slope(sb))

    if len(boot) < max(30, n_bootstrap // 10):
        raise RuntimeError("bootstrap insuficiente para ridge quantizado")
    low, high = np.percentile(boot, [2.5, 97.5])
    return ChirpMassEstimate(
        mass,
        float(low),
        float(high),
        float(slope),
        n,
        len(boot),
        fit_method,
    )


def ridge_from_energy(
    energy: ArrayLike,
    frequencies: ArrayLike,
    times: ArrayLike,
    *,
    min_quantile: float = 0.70,
    max_points: int = 100,
) -> tuple[FloatArray, FloatArray]:
    e = np.asarray(energy, dtype=float)
    f = np.asarray(frequencies, dtype=float).ravel()
    t = np.asarray(times, dtype=float).ravel()
    if e.shape != (len(f), len(t)):
        raise ValueError("energy deve ter shape (n_frequencias, n_tempos)")
    if not 0 <= min_quantile < 1:
        raise ValueError("min_quantile deve pertencer a [0,1)")
    peak_idx = np.argmax(e, axis=0)
    peak_f = f[peak_idx]
    peak_e = e[peak_idx, np.arange(len(t))]
    keep = peak_e >= np.quantile(peak_e, min_quantile)
    t_keep, f_keep = t[keep], peak_f[keep]
    if len(t_keep) > max_points:
        select = np.linspace(0, len(t_keep) - 1, max_points).round().astype(int)
        t_keep, f_keep = t_keep[select], f_keep[select]
    return t_keep.astype(float), f_keep.astype(float)
