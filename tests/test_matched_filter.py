import numpy as np

from gwproject.config import load_toml
from gwproject.matched_filter import (
    estimate_offsource_psd,
    matched_filter_snr,
    taylorf2_tapered_extension,
    taylorf2_template,
    template_band,
    top2_ratio,
)
from gwproject.preprocessing import estimate_psd

FS = 4096.0
DURATION = 32.0
N = int(FS * DURATION)
TC = 16.4


def aligo_psd(f, s0=1e-49, f0=215.0, floor_hz=9.0):
    f = np.atleast_1d(np.asarray(f, dtype=float))
    sn = np.full_like(f, np.inf)
    mask = f >= floor_hz
    x = f[mask] / f0
    sn[mask] = s0 * (
        x**-4.14 - 5.0 * x**-2
        + 111.0 * (1.0 - x**2 + 0.5 * x**4) / (1.0 + 0.5 * x**2)
    )
    sn[~mask] = sn[mask].max() * 1e6
    return sn


def colored_noise(seed):
    rng = np.random.default_rng(seed)
    frequencies = np.fft.rfftfreq(N, 1.0 / FS)
    sn = np.where(np.isfinite(aligo_psd(frequencies)), aligo_psd(frequencies), 0.0)
    sigma = np.sqrt(sn * N * FS / 4.0)
    spectrum = rng.normal(size=len(frequencies)) * sigma + 1j * rng.normal(size=len(frequencies)) * sigma
    spectrum[0] = 0.0
    return np.fft.irfft(spectrum, n=N)


def test_template_bank_has_no_inert_member():
    cfg = load_toml("configs/matched_filter.toml")
    for name, masses in cfg["templates"].items():
        band = template_band(
            *masses,
            max_template_seconds=cfg["signal"]["max_template_seconds"],
            floor_hz=cfg["signal"]["f_low_hz"],
            analysis_f_high_hz=cfg["signal"]["analysis_f_high_hz"],
            extension_factor=cfg["signal"]["extension_factor"],
        )
        assert band.bandwidth_inspiral >= cfg["gate"]["minimum_template_bandwidth_hz"], name


def test_offsource_psd_excludes_central_transient():
    rng = np.random.default_rng(2026)
    clean = rng.normal(size=int(FS * 12))
    contaminated = clean.copy()
    event_index = len(clean) // 2
    contaminated[event_index - 50 : event_index + 51] += 1e5
    clean_psd = estimate_offsource_psd(
        clean, FS, event_index, exclusion_seconds=1.0, segment_seconds=2.0
    )
    contaminated_psd = estimate_offsource_psd(
        contaminated, FS, event_index, exclusion_seconds=1.0, segment_seconds=2.0
    )
    assert np.allclose(clean_psd.values, contaminated_psd.values)


def test_extended_ranking_has_required_margin():
    cfg = load_toml("configs/matched_filter.toml")
    frequencies = np.fft.rfftfreq(N, 1.0 / FS)
    sn = np.where(np.isfinite(aligo_psd(frequencies)), aligo_psd(frequencies), np.inf)
    df = 1.0 / DURATION
    target = taylorf2_template(36.0, 29.0, frequencies, 25.0, 250.0)
    target *= np.exp(-2j * np.pi * frequencies * TC)
    band_mask = (frequencies >= 25.0) & (frequencies <= 250.0)
    target *= 17.0 / np.sqrt(4 * np.sum(np.abs(target[band_mask]) ** 2 / sn[band_mask]) * df)
    data = colored_noise(42) + np.fft.irfft(target, n=N) * FS
    psd = estimate_psd(data, FS, segment_seconds=4.0)
    center = int(TC * FS)
    half = int(0.10 * FS)
    rows = []
    for name, masses in cfg["templates"].items():
        template_info = template_band(
            *masses,
            max_template_seconds=cfg["signal"]["max_template_seconds"],
            floor_hz=cfg["signal"]["f_low_hz"],
            analysis_f_high_hz=cfg["signal"]["analysis_f_high_hz"],
            extension_factor=cfg["signal"]["extension_factor"],
        )
        template = taylorf2_tapered_extension(
            *masses,
            frequencies,
            template_info.f_low,
            template_info.f_high_extended,
            isco_frequency=template_info.f_isco,
        )
        snr = matched_filter_snr(
            data,
            FS,
            template,
            psd.frequencies,
            psd.values,
            f_low=template_info.f_low,
            f_high=template_info.f_high_extended,
        )
        rows.append({
            "template": name,
            "network_snr": float(np.max(snr[center-half:center+half+1])),
        })
    rows.sort(key=lambda row: row["network_snr"], reverse=True)
    assert rows[0]["template"] == cfg["gate"]["expected_template"]
    assert top2_ratio(rows) >= cfg["gate"]["minimum_top2_snr_ratio"]
