from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import butter, filtfilt, get_window, iirnotch, sosfiltfilt, welch

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DataQualityReport:
    zero_fraction: float
    longest_zero_run_seconds: float
    standard_deviation: float


@dataclass(frozen=True)
class PSDResult:
    frequencies: FloatArray
    values: FloatArray
    segment_seconds: float
    overlap_fraction: float
    average: str


@dataclass(frozen=True)
class QTransformResult:
    energy: FloatArray
    frequencies: FloatArray
    q: float
    filter_norms: FloatArray


@dataclass(frozen=True)
class PreprocessDiagnostics:
    white_std_core: float
    edge_crop_seconds: float
    quality: DataQualityReport


def _as_finite_1d(data: ArrayLike) -> FloatArray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("data deve ser unidimensional")
    if len(arr) < 16:
        raise ValueError("serie curta demais")
    if not np.all(np.isfinite(arr)):
        raise ValueError("data contem NaN ou infinito")
    return arr


def _longest_true_run(mask: NDArray[np.bool_]) -> int:
    if not np.any(mask):
        return 0
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return int(np.max(stops - starts))


def data_quality_report(data: ArrayLike, fs: float) -> DataQualityReport:
    x = _as_finite_1d(data)
    if fs <= 0:
        raise ValueError("fs deve ser positivo")
    zeros = x == 0.0
    return DataQualityReport(
        zero_fraction=float(np.mean(zeros)),
        longest_zero_run_seconds=_longest_true_run(zeros) / float(fs),
        standard_deviation=float(np.std(x)),
    )


def validate_data_segment(
    data: ArrayLike,
    fs: float,
    *,
    max_zero_fraction: float = 0.01,
    max_zero_run_seconds: float = 0.25,
) -> DataQualityReport:
    report = data_quality_report(data, fs)
    if report.standard_deviation <= 0:
        raise ValueError("segmento constante ou vazio")
    if report.zero_fraction > max_zero_fraction:
        raise ValueError(
            f"segmento com {100*report.zero_fraction:.2f}% de zeros; limite = "
            f"{100*max_zero_fraction:.2f}%"
        )
    if report.longest_zero_run_seconds > max_zero_run_seconds:
        raise ValueError(
            f"data gap de {report.longest_zero_run_seconds:.3f} s; limite = "
            f"{max_zero_run_seconds:.3f} s"
        )
    return report


def estimate_psd(
    data: ArrayLike,
    fs: float,
    segment_seconds: float = 4.0,
    overlap_fraction: float = 0.5,
    average: str = "median",
    *,
    validate_segment: bool = True,
    max_zero_fraction: float = 0.01,
    max_zero_run_seconds: float = 0.25,
) -> PSDResult:
    x = _as_finite_1d(data)
    if fs <= 0:
        raise ValueError("fs deve ser positivo")
    if validate_segment:
        validate_data_segment(
            x,
            fs,
            max_zero_fraction=max_zero_fraction,
            max_zero_run_seconds=max_zero_run_seconds,
        )
    if not 0 <= overlap_fraction < 1:
        raise ValueError("overlap_fraction deve pertencer a [0,1)")
    if average not in {"mean", "median"}:
        raise ValueError("average deve ser 'mean' ou 'median'")
    nperseg = min(len(x), max(16, int(round(segment_seconds * fs))))
    noverlap = min(nperseg - 1, int(round(nperseg * overlap_fraction)))
    f, sn = welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        average=average,
        scaling="density",
    )
    positive = sn[np.isfinite(sn) & (sn > 0)]
    if len(positive) == 0:
        raise ValueError("PSD invalida")
    median = float(np.median(positive))
    floor = max(median * 1e-12, np.finfo(float).tiny)
    sn = np.maximum(np.nan_to_num(sn, nan=floor, posinf=floor, neginf=floor), floor)
    dynamic = float(np.max(sn) / max(np.min(sn), floor))
    if not np.isfinite(dynamic) or dynamic > 1e30:
        raise ValueError("PSD numericamente degenerada")
    return PSDResult(f.astype(float), sn.astype(float), segment_seconds, overlap_fraction, average)


def _interpolate_psd(frequencies: FloatArray, psd: PSDResult, fmin: float, fmax: float) -> FloatArray:
    if not 0 <= fmin < fmax:
        raise ValueError("banda de whitening invalida")
    sn = np.interp(frequencies, psd.frequencies, psd.values)
    positive = psd.values[np.isfinite(psd.values) & (psd.values > 0)]
    floor = max(float(np.median(positive)) * 1e-12, np.finfo(float).tiny)
    sn = np.maximum(sn, floor)
    sn[(frequencies < fmin) | (frequencies > fmax)] = np.inf
    return sn


def whiten(
    data: ArrayLike,
    fs: float,
    psd: PSDResult,
    fmin: float = 10.0,
    fmax: float | None = None,
    tukey_alpha: float = 0.125,
) -> FloatArray:
    """Whitening unilateral para PSD em unidades data^2/Hz."""
    x = _as_finite_1d(data)
    if fs <= 0:
        raise ValueError("fs deve ser positivo")
    if fmax is None:
        fmax = 0.98 * fs / 2.0
    fmax = min(float(fmax), np.nextafter(fs / 2.0, 0.0))
    n = len(x)
    dt = 1.0 / fs
    window = get_window(("tukey", tukey_alpha), n, fftbins=True)
    htilde = np.fft.rfft((x - np.mean(x)) * window) * dt
    frequencies = np.fft.rfftfreq(n, dt)
    sn = _interpolate_psd(frequencies, psd, fmin, fmax)
    with np.errstate(divide="ignore", invalid="ignore"):
        wtilde = htilde / np.sqrt(sn / 2.0)
    wtilde[~np.isfinite(wtilde)] = 0.0
    return (np.fft.irfft(wtilde, n=n) / dt * np.sqrt(dt)).astype(float)


def core_standard_deviation(data: ArrayLike, fs: float, edge_crop_seconds: float = 2.0) -> float:
    x = _as_finite_1d(data)
    crop = int(round(edge_crop_seconds * fs))
    if 2 * crop >= len(x):
        raise ValueError("recorte de borda maior que a serie")
    core = x[crop:-crop] if crop else x
    return float(np.std(core))


def bandpass_butter(data: ArrayLike, fs: float, fmin: float, fmax: float, order: int = 4) -> FloatArray:
    x = _as_finite_1d(data)
    nyquist = fs / 2.0
    if not (0 < fmin < fmax < nyquist):
        raise ValueError(f"banda invalida: {fmin}-{fmax} Hz para fs={fs}")
    if order < 1:
        raise ValueError("order deve ser >= 1")
    sos = butter(order, [fmin / nyquist, fmax / nyquist], btype="bandpass", output="sos")
    return np.asarray(sosfiltfilt(sos, x), dtype=float)


def apply_notches(
    data: ArrayLike,
    fs: float,
    frequencies_hz: ArrayLike,
    *,
    quality_factor: float = 30.0,
) -> FloatArray:
    x = _as_finite_1d(data).copy()
    if quality_factor <= 0:
        raise ValueError("quality_factor deve ser positivo")
    for frequency in np.asarray(frequencies_hz, dtype=float).ravel():
        if frequency <= 0 or frequency >= fs / 2:
            continue
        b, a = iirnotch(float(frequency), quality_factor, fs=fs)
        x = filtfilt(b, a, x)
    return np.asarray(x, dtype=float)


def q_transform_constant_q(
    data: ArrayLike,
    fs: float,
    fmin: float = 20.0,
    fmax: float = 512.0,
    n_bins: int = 64,
    q: float = 12.0,
) -> QTransformResult:
    """Banco gaussiano de Q constante, normalizado em L2 por frequencia."""
    x = _as_finite_1d(data)
    nyquist = fs / 2.0
    if not (0 < fmin < fmax < nyquist):
        raise ValueError("faixa de frequencia invalida")
    if n_bins < 2 or q <= 0:
        raise ValueError("n_bins >= 2 e q > 0")

    n = len(x)
    spectrum = np.fft.fft(x - np.mean(x))
    frequencies = np.fft.fftfreq(n, 1.0 / fs)
    positive = frequencies > 0
    centers = np.geomspace(fmin, fmax, n_bins)
    energy = np.empty((n_bins, n), dtype=np.float64)
    filter_norms = np.empty(n_bins, dtype=np.float64)

    for k, center in enumerate(centers):
        sigma_f = center / q
        window = np.zeros(n, dtype=np.float64)
        window[positive] = np.exp(-0.5 * ((frequencies[positive] - center) / sigma_f) ** 2)
        norm = float(np.sqrt(np.sum(window * window)))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"filtro degenerado em {center:g} Hz")
        window /= norm
        analytic_band = np.fft.ifft(2.0 * spectrum * window)
        energy[k] = np.abs(analytic_band) ** 2
        filter_norms[k] = norm
    return QTransformResult(energy, centers.astype(float), q, filter_norms)


def q_flatness_ratio(
    result: QTransformResult,
    start: int | None = None,
    stop: int | None = None,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
) -> float:
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("percentis invalidos")
    medians = np.median(result.energy[:, slice(start, stop)], axis=1)
    positive = medians[np.isfinite(medians) & (medians > 0)]
    if len(positive) == 0:
        return float("inf")
    low, high = np.percentile(positive, [lower_percentile, upper_percentile])
    return float(high / low)


def q_transform_self_test(
    *,
    fs: float,
    duration_seconds: float,
    fmin: float,
    fmax: float,
    n_bins: int,
    q: float,
    seed: int = 2026,
) -> float:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=int(round(fs * duration_seconds)))
    result = q_transform_constant_q(x, fs, fmin=fmin, fmax=fmax, n_bins=n_bins, q=q)
    crop = int(round(fs))
    return q_flatness_ratio(result, crop, -crop)


def preprocess(
    data: ArrayLike,
    fs: float,
    *,
    psd_segment_seconds: float = 4.0,
    psd_overlap_fraction: float = 0.5,
    psd_average: str = "median",
    whitening_fmin: float = 10.0,
    whitening_fmax: float | None = None,
    whitening_tukey_alpha: float = 0.125,
    bandpass_fmin: float = 20.0,
    bandpass_fmax: float = 1024.0,
    bandpass_order: int = 4,
    edge_crop_seconds: float = 2.0,
    white_std_min: float | None = None,
    white_std_max: float | None = None,
    max_zero_fraction: float = 0.01,
    max_zero_run_seconds: float = 0.25,
    return_diagnostics: bool = False,
):
    quality = validate_data_segment(
        data,
        fs,
        max_zero_fraction=max_zero_fraction,
        max_zero_run_seconds=max_zero_run_seconds,
    )
    psd = estimate_psd(
        data,
        fs,
        psd_segment_seconds,
        psd_overlap_fraction,
        psd_average,
        validate_segment=False,
    )
    white = whiten(data, fs, psd, whitening_fmin, whitening_fmax, whitening_tukey_alpha)
    white_std = core_standard_deviation(white, fs, edge_crop_seconds)
    if white_std_min is not None and white_std < white_std_min:
        raise ValueError(f"whitening frio: sigma={white_std:.3f} < {white_std_min:.3f}")
    if white_std_max is not None and white_std > white_std_max:
        raise ValueError(f"whitening quente: sigma={white_std:.3f} > {white_std_max:.3f}")
    filtered = bandpass_butter(white, fs, bandpass_fmin, bandpass_fmax, bandpass_order)
    diagnostics = PreprocessDiagnostics(white_std, edge_crop_seconds, quality)
    if return_diagnostics:
        return filtered, psd, diagnostics
    return filtered, psd
