from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .matched_filter import chirp_mass, f_isco
from .preprocessing import q_transform_constant_q
from .ridge import estimate_chirp_mass_from_ridge_points

G_SI = 6.67430e-11
C_SI = 2.99792458e8
MSUN_SI = 1.98847e30
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RidgePlan:
    fmin_hz: float
    fmax_hz: float
    window_start_seconds: float
    window_stop_seconds: float
    segment_start_seconds: float
    segment_stop_seconds: float
    available_before_seconds: float
    available_after_seconds: float
    status: str
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class LocalTemplate:
    name: str
    m1: float
    m2: float
    central: bool = False


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def catalog_redshift(row: dict[str, Any]) -> float | None:
    value = finite_float(row.get("redshift"))
    return value if value is not None and value >= 0 else None


def catalog_chirp_mass(row: dict[str, Any]) -> tuple[float | None, str]:
    """Retorna a massa chirp como publicada, preservando o frame original.

    Para analises no sinal observado, prefira ``catalog_detector_chirp_mass``.
    """
    for key, frame in (("chirp_mass", "detector_catalog"), ("chirp_mass_source", "source")):
        value = finite_float(row.get(key))
        if value is not None and value > 0:
            return value, frame
    masses, frame = catalog_component_masses(row)
    if masses is not None:
        return chirp_mass(*masses), frame
    return None, "missing"


def catalog_component_masses(row: dict[str, Any]) -> tuple[tuple[float, float] | None, str]:
    """Retorna massas de componentes preservando o frame disponibilizado."""
    candidates = (
        (("mass_1", "mass_2"), "detector_catalog"),
        (("mass_1_source", "mass_2_source"), "source"),
    )
    for keys, frame in candidates:
        m1, m2 = finite_float(row.get(keys[0])), finite_float(row.get(keys[1]))
        if m1 is not None and m2 is not None and m1 > 0 and m2 > 0:
            return (max(m1, m2), min(m1, m2)), frame
    return None, "missing"


def catalog_detector_component_masses(
    row: dict[str, Any],
    *,
    allow_source_fallback: bool = False,
) -> tuple[tuple[float, float] | None, str]:
    """Retorna massas no frame do detector para modelar o sinal observado.

    Usa primeiro massas detector-frame publicadas. Quando apenas massas
    source-frame estao presentes, aplica ``m_det=(1+z)m_source`` se o redshift
    estiver disponivel. O fallback source-frame e opt-in e sempre rotulado.
    """
    derived_m1 = finite_float(row.get("mass_1_detector"))
    derived_m2 = finite_float(row.get("mass_2_detector"))
    if derived_m1 is not None and derived_m2 is not None and derived_m1 > 0 and derived_m2 > 0:
        origin = str(row.get("component_mass_detector_origin") or "detector_explicit")
        return (max(derived_m1, derived_m2), min(derived_m1, derived_m2)), origin

    m1 = finite_float(row.get("mass_1"))
    m2 = finite_float(row.get("mass_2"))
    if m1 is not None and m2 is not None and m1 > 0 and m2 > 0:
        return (max(m1, m2), min(m1, m2)), "detector_catalog"

    s1 = finite_float(row.get("mass_1_source"))
    s2 = finite_float(row.get("mass_2_source"))
    if s1 is not None and s2 is not None and s1 > 0 and s2 > 0:
        redshift = catalog_redshift(row)
        if redshift is not None:
            factor = 1.0 + redshift
            return (max(s1, s2) * factor, min(s1, s2) * factor), "detector_from_source_redshift"
        if allow_source_fallback:
            return (max(s1, s2), min(s1, s2)), "source_unconverted"
    return None, "missing_detector_frame"


def catalog_detector_chirp_mass(
    row: dict[str, Any],
    *,
    allow_source_fallback: bool = False,
) -> tuple[float | None, str]:
    """Retorna massa chirp no frame do detector para o sinal observado."""
    derived = finite_float(row.get("chirp_mass_detector"))
    if derived is not None and derived > 0:
        origin = str(row.get("chirp_mass_detector_origin") or "detector_explicit")
        return derived, origin

    value = finite_float(row.get("chirp_mass"))
    if value is not None and value > 0:
        return value, "detector_catalog"

    source = finite_float(row.get("chirp_mass_source"))
    if source is not None and source > 0:
        redshift = catalog_redshift(row)
        if redshift is not None:
            return source * (1.0 + redshift), "detector_from_source_redshift"
        if allow_source_fallback:
            return source, "source_unconverted"

    masses, frame = catalog_detector_component_masses(
        row,
        allow_source_fallback=allow_source_fallback,
    )
    if masses is not None:
        return chirp_mass(*masses), frame
    return None, "missing_detector_frame"


def chirp_time_seconds(mc_solar: float, frequency_hz: float) -> float:
    if mc_solar <= 0 or frequency_hz <= 0:
        raise ValueError("chirp mass e frequencia devem ser positivas")
    gmc = G_SI * mc_solar * MSUN_SI / C_SI**3
    return float((5.0 / 256.0) * gmc ** (-5.0 / 3.0) * (np.pi * frequency_hz) ** (-8.0 / 3.0))


def frequency_for_chirp_time(mc_solar: float, seconds: float) -> float:
    if mc_solar <= 0 or seconds <= 0:
        raise ValueError("chirp mass e tempo devem ser positivos")
    gmc = G_SI * mc_solar * MSUN_SI / C_SI**3
    return float(((5.0 / 256.0) * gmc ** (-5.0 / 3.0) / seconds) ** (3.0 / 8.0) / np.pi)


def adaptive_ridge_plan(
    mc_solar: float,
    *,
    available_before_seconds: float,
    available_after_seconds: float,
    m1: float | None,
    m2: float | None,
    config: dict[str, Any],
) -> RidgePlan:
    ridge_cfg = config["catalog_ridge"]
    base_fmin = float(ridge_cfg["fmin_hz"])
    base_fmax = float(ridge_cfg["fmax_hz"])
    max_before = min(float(ridge_cfg["max_window_before_seconds"]), available_before_seconds)
    padding = float(ridge_cfg["padding_seconds"])
    min_bandwidth = float(ridge_cfg["minimum_bandwidth_hz"])
    stop_floor = float(ridge_cfg["minimum_seconds_before_merger"])

    if max_before <= stop_floor + 0.05:
        return RidgePlan(
            base_fmin,
            base_fmax,
            -max_before,
            -stop_floor,
            -max_before,
            min(available_after_seconds, padding),
            available_before_seconds,
            available_after_seconds,
            "insufficient_time",
            "janela temporal anterior ao evento curta demais",
        )

    fmin = max(base_fmin, frequency_for_chirp_time(mc_solar, max_before))
    if m1 is not None and m2 is not None:
        fmax_physical = float(ridge_cfg["isco_fraction"]) * f_isco(m1, m2)
        fmax = min(base_fmax, fmax_physical)
    else:
        fmax = base_fmax

    if fmax - fmin < min_bandwidth:
        return RidgePlan(
            fmin,
            fmax,
            -max_before,
            -stop_floor,
            -max_before,
            min(available_after_seconds, padding),
            available_before_seconds,
            available_after_seconds,
            "insufficient_inspiral_band",
            f"banda inspiral util = {max(0.0, fmax-fmin):.3f} Hz",
        )

    time_at_fmin = chirp_time_seconds(mc_solar, fmin)
    time_at_fmax = chirp_time_seconds(mc_solar, fmax)
    start_before = min(max_before, max(time_at_fmin * 1.10, stop_floor + 0.05))
    stop_before = max(stop_floor, time_at_fmax * 0.70)
    if start_before <= stop_before:
        start_before = min(max_before, stop_before + 0.05)
    segment_start = -min(available_before_seconds, start_before + padding)
    segment_stop = min(available_after_seconds, padding)
    return RidgePlan(
        fmin_hz=fmin,
        fmax_hz=fmax,
        window_start_seconds=-start_before,
        window_stop_seconds=-stop_before,
        segment_start_seconds=segment_start,
        segment_stop_seconds=segment_stop,
        available_before_seconds=available_before_seconds,
        available_after_seconds=available_after_seconds,
        status="ok",
    )


def downsample_energy_max(
    energy: ArrayLike,
    times: ArrayLike,
    max_time_bins: int,
) -> tuple[FloatArray, FloatArray]:
    e = np.asarray(energy, dtype=float)
    t = np.asarray(times, dtype=float).ravel()
    if e.ndim != 2 or e.shape[1] != len(t):
        raise ValueError("shape incompatível entre energy e times")
    if max_time_bins < 10:
        raise ValueError("max_time_bins deve ser >= 10")
    if len(t) <= max_time_bins:
        return e, t
    edges = np.linspace(0, len(t), max_time_bins + 1).round().astype(int)
    output = np.empty((e.shape[0], max_time_bins), dtype=float)
    out_times = np.empty(max_time_bins, dtype=float)
    for index in range(max_time_bins):
        start, stop = int(edges[index]), int(edges[index + 1])
        stop = max(stop, start + 1)
        output[:, index] = np.max(e[:, start:stop], axis=1)
        out_times[index] = float(np.mean(t[start:stop]))
    return output, out_times


def ridge_dynamic_programming(
    energy: ArrayLike,
    frequencies: ArrayLike,
    times: ArrayLike,
    *,
    maximum_jump_bins: int = 5,
    downward_tolerance_bins: int = 1,
    jump_penalty: float = 0.12,
    downward_penalty: float = 0.50,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Extrai um caminho quase monotono de alta energia por programacao dinamica."""

    e = np.asarray(energy, dtype=float)
    f = np.asarray(frequencies, dtype=float).ravel()
    t = np.asarray(times, dtype=float).ravel()
    if e.shape != (len(f), len(t)):
        raise ValueError("energy deve ter shape (n_frequencias, n_tempos)")
    if len(f) < 3 or len(t) < 10:
        raise ValueError("mapa tempo-frequencia curto demais")
    if maximum_jump_bins < 1 or downward_tolerance_bins < 0:
        raise ValueError("parametros de transicao invalidos")

    positive = e[np.isfinite(e) & (e > 0)]
    floor = float(np.median(positive)) * 1e-9 if len(positive) else np.finfo(float).tiny
    e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
    baseline = np.median(e, axis=1, keepdims=True)
    baseline = np.maximum(baseline, floor)
    score = np.log1p(e / baseline)
    score -= np.median(score, axis=0, keepdims=True)

    n_f, n_t = score.shape
    previous = score[:, 0].copy()
    back = np.zeros((n_t, n_f), dtype=np.int16 if n_f < 32767 else np.int32)
    indices = np.arange(n_f)
    for time_index in range(1, n_t):
        current = np.full(n_f, -np.inf, dtype=float)
        for freq_index in range(n_f):
            low = max(0, freq_index - maximum_jump_bins)
            high = min(n_f, freq_index + downward_tolerance_bins + 1)
            predecessors = indices[low:high]
            delta = freq_index - predecessors
            penalties = jump_penalty * delta.astype(float) ** 2
            penalties += downward_penalty * np.maximum(-delta, 0).astype(float) ** 2
            candidates = previous[predecessors] - penalties
            best_local = int(np.argmax(candidates))
            best_predecessor = int(predecessors[best_local])
            current[freq_index] = score[freq_index, time_index] + candidates[best_local]
            back[time_index, freq_index] = best_predecessor
        previous = current

    path = np.empty(n_t, dtype=int)
    path[-1] = int(np.argmax(previous))
    for time_index in range(n_t - 1, 0, -1):
        path[time_index - 1] = int(back[time_index, path[time_index]])
    path_energy = e[path, np.arange(n_t)]
    return t.astype(float), f[path].astype(float), path_energy.astype(float)


def select_ridge_fit_points(
    ridge_times: ArrayLike,
    ridge_frequencies: ArrayLike,
    ridge_energy: ArrayLike,
    *,
    config: dict[str, Any],
) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
    """Seleciona pontos de ajuste preservando cobertura temporal e frequencial.

    O corte puramente global por energia pode concentrar os pontos em um trecho
    quase horizontal do ridge. A selecao tenta quantis progressivamente menos
    restritivos e so aceita um conjunto com cobertura suficiente.
    """

    t = np.asarray(ridge_times, dtype=float).ravel()
    f = np.asarray(ridge_frequencies, dtype=float).ravel()
    e = np.asarray(ridge_energy, dtype=float).ravel()
    if not (len(t) == len(f) == len(e)):
        raise ValueError("ridge_times, ridge_frequencies e ridge_energy incompatíveis")
    if len(t) == 0:
        return t, f, {"status": "empty_ridge"}

    qcfg = config["catalog_ridge"]
    minimum_points = int(qcfg["minimum_fit_points"])
    maximum_points = int(qcfg["max_fit_points"])
    minimum_time_fraction = float(qcfg.get("minimum_fit_time_fraction", 0.55))
    minimum_unique = int(qcfg.get("minimum_unique_frequencies", 3))
    minimum_frequency_span = float(qcfg.get("minimum_frequency_span_hz", 3.0))

    requested = float(qcfg["path_energy_quantile"])
    fallbacks = list(qcfg.get("path_energy_quantile_fallback", [requested, 0.40, 0.25, 0.0]))
    quantiles: list[float] = []
    for value in [requested, *fallbacks, 0.0]:
        value = min(0.999999, max(0.0, float(value)))
        if value not in quantiles:
            quantiles.append(value)

    full_time_span = float(np.ptp(t))
    best: tuple[FloatArray, FloatArray, dict[str, Any]] | None = None
    for quantile in quantiles:
        threshold = float(np.quantile(e, quantile))
        keep = np.isfinite(e) & (e >= threshold)
        tk = t[keep]
        fk = np.maximum.accumulate(f[keep]) if np.any(keep) else f[keep]
        diagnostics = {
            "selection_quantile": quantile,
            "selection_threshold": threshold,
            "n_selected_before_cap": int(len(tk)),
            "time_span_seconds": float(np.ptp(tk)) if len(tk) else 0.0,
            "time_span_fraction": (float(np.ptp(tk)) / full_time_span if full_time_span > 0 and len(tk) else 0.0),
            "frequency_span_hz": float(np.ptp(fk)) if len(fk) else 0.0,
            "unique_frequencies": int(len(np.unique(fk))),
        }
        best = (tk, fk, diagnostics)
        if len(tk) < minimum_points:
            continue
        if diagnostics["time_span_fraction"] < minimum_time_fraction:
            continue
        if diagnostics["unique_frequencies"] < minimum_unique:
            continue
        if diagnostics["frequency_span_hz"] < minimum_frequency_span:
            continue
        if len(tk) > maximum_points:
            selected = np.linspace(0, len(tk) - 1, maximum_points).round().astype(int)
            tk, fk = tk[selected], fk[selected]
        diagnostics["n_selected"] = int(len(tk))
        diagnostics["status"] = "ok"
        return tk.astype(float), fk.astype(float), diagnostics

    assert best is not None
    tk, fk, diagnostics = best
    diagnostics["n_selected"] = int(len(tk))
    if len(tk) < minimum_points:
        diagnostics["status"] = "insufficient_ridge_points"
    elif diagnostics["time_span_fraction"] < minimum_time_fraction:
        diagnostics["status"] = "insufficient_time_coverage"
    elif diagnostics["unique_frequencies"] < minimum_unique:
        diagnostics["status"] = "flat_ridge"
    elif diagnostics["frequency_span_hz"] < minimum_frequency_span:
        diagnostics["status"] = "insufficient_frequency_span"
    else:
        diagnostics["status"] = "ridge_selection_failed"
    return tk.astype(float), fk.astype(float), diagnostics


def analyze_ridge_channel(
    data: ArrayLike,
    sample_rate: float,
    event_index: int,
    reference_chirp_mass: float,
    *,
    m1: float | None,
    m2: float | None,
    config: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    x = np.asarray(data, dtype=float).ravel()
    if not np.all(np.isfinite(x)):
        raise ValueError("processed data contem NaN ou infinito")
    if not 0 <= event_index < len(x):
        raise ValueError("event_index fora do processed data")
    before = event_index / sample_rate
    after = (len(x) - event_index - 1) / sample_rate
    plan = adaptive_ridge_plan(
        reference_chirp_mass,
        available_before_seconds=before,
        available_after_seconds=after,
        m1=m1,
        m2=m2,
        config=config,
    )
    if plan.status != "ok":
        return {"status": plan.status, "reason": plan.reason, "plan": plan.as_dict()}, {}

    start = max(0, event_index + int(round(plan.segment_start_seconds * sample_rate)))
    stop = min(len(x), event_index + int(round(plan.segment_stop_seconds * sample_rate)))
    if stop - start < 64:
        return {"status": "segment_too_short", "plan": plan.as_dict()}, {}
    segment = x[start:stop]
    qcfg = config["catalog_ridge"]
    q_result = q_transform_constant_q(
        segment,
        sample_rate,
        fmin=plan.fmin_hz,
        fmax=plan.fmax_hz,
        n_bins=int(qcfg["n_frequency_bins"]),
        q=float(qcfg["q"]),
    )
    relative_times = (np.arange(len(segment)) + start - event_index) / sample_rate
    mask = (
        (relative_times >= plan.window_start_seconds)
        & (relative_times <= plan.window_stop_seconds)
    )
    if int(np.sum(mask)) < 20:
        return {"status": "analysis_window_too_short", "plan": plan.as_dict()}, {}
    energy = q_result.energy[:, mask]
    times = relative_times[mask]
    energy, times = downsample_energy_max(
        energy,
        times,
        int(qcfg["max_time_bins"]),
    )
    ridge_t, ridge_f, ridge_e = ridge_dynamic_programming(
        energy,
        q_result.frequencies,
        times,
        maximum_jump_bins=int(qcfg["maximum_jump_bins"]),
        downward_tolerance_bins=int(qcfg["downward_tolerance_bins"]),
        jump_penalty=float(qcfg["jump_penalty"]),
        downward_penalty=float(qcfg["downward_penalty"]),
    )
    ridge_t_fit, ridge_f_fit, selection = select_ridge_fit_points(
        ridge_t,
        ridge_f,
        ridge_e,
        config=config,
    )
    arrays = {
        "ridge_times": ridge_t,
        "ridge_frequencies": ridge_f,
        "ridge_energy": ridge_e,
        "fit_times": ridge_t_fit,
        "fit_frequencies": ridge_f_fit,
    }
    if selection["status"] != "ok":
        return {
            "status": selection["status"],
            "n_points": int(len(ridge_t_fit)),
            "selection": selection,
            "plan": plan.as_dict(),
        }, arrays

    try:
        estimate = estimate_chirp_mass_from_ridge_points(
            ridge_t_fit,
            ridge_f_fit,
            n_bootstrap=int(qcfg["bootstrap_samples"]),
            seed=seed,
            enforce_monotonic=True,
        )
    except ValueError as exc:
        message = str(exc)
        status = "nonphysical_ridge_slope"
        if "frequencia constante" in message:
            status = "flat_ridge"
        return {
            "status": status,
            "reason": message,
            "n_points": int(len(ridge_t_fit)),
            "selection": selection,
            "plan": plan.as_dict(),
        }, arrays
    except RuntimeError as exc:
        return {
            "status": "bootstrap_insufficient",
            "reason": str(exc),
            "n_points": int(len(ridge_t_fit)),
            "selection": selection,
            "plan": plan.as_dict(),
        }, arrays
    relative_error = abs(estimate.mass_solar - reference_chirp_mass) / reference_chirp_mass
    result = {
        "status": "ok",
        "reference_chirp_mass_solar": reference_chirp_mass,
        "estimated_chirp_mass_solar": estimate.mass_solar,
        "relative_error": relative_error,
        "ridge_bootstrap_dispersion_low_solar": estimate.dispersion_low_solar,
        "ridge_bootstrap_dispersion_high_solar": estimate.dispersion_high_solar,
        "slope": estimate.slope,
        "n_points": estimate.n_points,
        "n_bootstrap": estimate.n_bootstrap,
        "fit_method": estimate.fit_method,
        "q": float(qcfg["q"]),
        "selection": selection,
        "plan": plan.as_dict(),
        "passed_relative_error": relative_error <= float(qcfg["relative_error_max"]),
    }
    return result, arrays


def stable_seed(*parts: str, base_seed: int = 2026) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int((base_seed + int.from_bytes(digest[:4], "little")) % (2**32 - 1))


def _masses_from_total_q(total: float, q: float) -> tuple[float, float]:
    q = min(1.0, max(0.02, q))
    m1 = total / (1.0 + q)
    m2 = total - m1
    return max(m1, m2), min(m1, m2)


def build_local_template_bank(
    m1: float,
    m2: float,
    *,
    scale_factors: Iterable[float] = (0.90, 1.10),
    minimum_component_mass: float = 0.5,
) -> list[LocalTemplate]:
    if m1 <= 0 or m2 <= 0:
        raise ValueError("massas devem ser positivas")
    m1, m2 = max(m1, m2), min(m1, m2)
    total = m1 + m2
    q = m2 / m1
    candidates: list[LocalTemplate] = [LocalTemplate("catalog_central", m1, m2, True)]
    for factor in scale_factors:
        candidates.append(LocalTemplate(f"scale_{factor:.3f}", factor * m1, factor * m2))
    more_equal = _masses_from_total_q(total, min(1.0, np.sqrt(q)))
    more_asymmetric = _masses_from_total_q(total, max(0.02, q**1.5))
    candidates.extend(
        [
            LocalTemplate("same_total_more_equal", *more_equal),
            LocalTemplate("same_total_more_asymmetric", *more_asymmetric),
        ]
    )
    output: list[LocalTemplate] = []
    seen: set[tuple[int, int]] = set()
    for candidate in candidates:
        if candidate.m2 < minimum_component_mass:
            continue
        key = (round(candidate.m1 * 1000), round(candidate.m2 * 1000))
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
    return output
