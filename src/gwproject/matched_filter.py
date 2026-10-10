from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import get_window

G_SI = 6.67430e-11
C_SI = 2.99792458e8
MSUN_SI = 1.98847e30

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class TemplateBand:
    f_low: float
    f_isco: float
    f_high_inspiral: float
    f_high_extended: float
    bandwidth_inspiral: float
    bandwidth_extended: float


def chirp_mass(m1: float, m2: float) -> float:
    if m1 <= 0 or m2 <= 0:
        raise ValueError("massas devem ser positivas")
    return float((m1 * m2) ** 0.6 / (m1 + m2) ** 0.2)


def f_isco(m1: float, m2: float) -> float:
    total_mass = (m1 + m2) * MSUN_SI
    return float(C_SI**3 / (6.0**1.5 * np.pi * G_SI * total_mass))


def f_low_for_duration(mc_solar: float, seconds: float, floor_hz: float = 25.0) -> float:
    if mc_solar <= 0 or seconds <= 0:
        raise ValueError("parametros invalidos")
    gmc = G_SI * mc_solar * MSUN_SI / C_SI**3
    f = ((5.0 / 256.0) * np.pi ** (-8.0 / 3.0) * gmc ** (-5.0 / 3.0) / seconds) ** (3.0 / 8.0)
    return float(max(floor_hz, f))


def template_band(
    m1: float,
    m2: float,
    *,
    max_template_seconds: float,
    floor_hz: float,
    analysis_f_high_hz: float,
    extension_factor: float = 2.5,
) -> TemplateBand:
    mc = chirp_mass(m1, m2)
    low = f_low_for_duration(mc, max_template_seconds, floor_hz)
    isco = f_isco(m1, m2)
    high_inspiral = min(isco, analysis_f_high_hz)
    high_extended = min(max(isco, extension_factor * isco), analysis_f_high_hz)
    return TemplateBand(
        low,
        isco,
        high_inspiral,
        high_extended,
        max(0.0, high_inspiral - low),
        max(0.0, high_extended - low),
    )


def taylorf2_template(
    m1: float,
    m2: float,
    frequencies: ArrayLike,
    f_low: float,
    f_high: float,
) -> ComplexArray:
    freqs = np.asarray(frequencies, dtype=float)
    if freqs.ndim != 1 or not (0 < f_low < f_high):
        raise ValueError("grade ou banda invalida")
    total_mass = (m1 + m2) * MSUN_SI
    eta = (m1 * m2) / (m1 + m2) ** 2
    if not 0 < eta <= 0.25:
        raise ValueError("eta invalido")
    out = np.zeros(len(freqs), dtype=np.complex128)
    band = (freqs >= f_low) & (freqs <= f_high)
    f = freqs[band]
    if len(f) < 3:
        raise ValueError("banda sem bins suficientes")
    v = (np.pi * G_SI * total_mass * f / C_SI**3) ** (1.0 / 3.0)
    v2 = v * v
    phase = (
        -np.pi / 4.0
        + 3.0 / (128.0 * eta * v**5)
        * (
            1.0
            + (3715.0 / 756.0 + 55.0 * eta / 9.0) * v2
            - 16.0 * np.pi * v2 * v
            + (15293365.0 / 508032.0 + 27145.0 * eta / 504.0 + 3085.0 * eta**2 / 72.0) * v2**2
        )
    )
    out[band] = f ** (-7.0 / 6.0) * np.exp(1j * phase)
    return out


def taylorf2_tapered_extension(
    m1: float,
    m2: float,
    frequencies: ArrayLike,
    f_low: float,
    f_high: float,
    *,
    isco_frequency: float | None = None,
) -> ComplexArray:
    """Extrapolacao pedagogica suave acima de f_ISCO.

    Nao substitui um modelo IMR calibrado. Serve para mostrar quanto o corte duro
    em f_ISCO altera o ranking no banco didatico.
    """
    freqs = np.asarray(frequencies, dtype=float)
    template = taylorf2_template(m1, m2, freqs, f_low, f_high)
    isco = f_isco(m1, m2) if isco_frequency is None else float(isco_frequency)
    if f_high > isco:
        taper = (freqs > isco) & (freqs <= f_high)
        template[taper] *= 0.5 * (1.0 + np.cos(np.pi * (freqs[taper] - isco) / (f_high - isco)))
    return template


def matched_filter_snr(
    data: ArrayLike,
    fs: float,
    template_frequency_domain: ArrayLike,
    psd_frequencies: ArrayLike,
    psd_values: ArrayLike,
    *,
    f_low: float,
    f_high: float,
) -> FloatArray:
    x = np.asarray(data, dtype=float).ravel()
    h = np.asarray(template_frequency_domain, dtype=np.complex128).ravel()
    pf = np.asarray(psd_frequencies, dtype=float).ravel()
    pv = np.asarray(psd_values, dtype=float).ravel()
    n = len(x)
    if len(h) != n // 2 + 1:
        raise ValueError("template deve estar na grade rfft dos dados")
    if len(pf) != len(pv):
        raise ValueError("grade e PSD com tamanhos distintos")
    finite_positive = pv[np.isfinite(pv) & (pv > 0)]
    if len(finite_positive) == 0:
        raise ValueError("PSD invalida")
    dt = 1.0 / fs
    df = 1.0 / (n * dt)
    window = get_window(("tukey", 0.125), n)
    dtilde = np.fft.rfft((x - np.mean(x)) * window) * dt
    freqs = np.fft.rfftfreq(n, dt)
    sn = np.interp(freqs, pf, pv)
    floor = max(float(np.median(finite_positive)) * 1e-12, np.finfo(float).tiny)
    sn = np.maximum(sn, floor)
    sn[(freqs < f_low) | (freqs > f_high)] = np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = 4.0 * dtilde * np.conj(h) / sn
        sigma2 = 4.0 * np.sum(np.nan_to_num(np.abs(h) ** 2 / sn)) * df
    integrand[~np.isfinite(integrand)] = 0.0
    if not np.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError("norma do template invalida")
    full = np.zeros(n, dtype=np.complex128)
    full[: len(integrand)] = integrand
    z = np.fft.ifft(full) * n * df
    return (np.abs(z) / np.sqrt(sigma2)).astype(float)


def top2_ratio(rows: list[dict[str, float | str]], key: str = "network_snr") -> float:
    if len(rows) < 2:
        return float("inf")
    ordered = sorted(rows, key=lambda row: float(row[key]), reverse=True)
    return float(ordered[0][key]) / max(float(ordered[1][key]), np.finfo(float).tiny)
