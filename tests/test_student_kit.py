"""Contratos numericos do kit didatico gw_student_kit."""
from __future__ import annotations

import numpy as np
import pytest

from gw_student_kit import (
    align_to_merger,
    bandpass_filter,
    compute_matched_filter,
    compute_psd,
    generate_detector_noise,
    generate_newtonian_chirp,
    inject_signal,
)
from gw_student_kit.synthetic import SOLAR_MASS_SECONDS

FS = 4096.0


def test_matched_filter_noise_statistic_has_mean_two():
    """Em ruido gaussiano, rho^2 segue chi^2 com 2 graus de liberdade."""
    rng = np.random.default_rng(7)
    _, template = generate_newtonian_chirp(fs=FS, duration=8.0, chirp_mass_solar=30.0, t_merger=6.0)
    values = []
    for _ in range(12):
        noise = generate_detector_noise(fs=FS, duration=8.0, rng=rng)
        snr, _, _ = compute_matched_filter(noise, template, FS)
        values.append(np.mean(snr ** 2))
    assert np.mean(values) == pytest.approx(2.0, rel=0.05)


def test_injection_is_recovered_at_target_snr_and_merger_time():
    rng = np.random.default_rng(11)
    merger = 6.0
    _, template = generate_newtonian_chirp(fs=FS, duration=8.0, chirp_mass_solar=30.0, t_merger=merger)
    data = inject_signal(generate_detector_noise(fs=FS, duration=8.0, rng=rng), template, target_snr=20.0)
    snr, peak, _ = compute_matched_filter(data, template, FS)
    aligned = align_to_merger(snr, int(merger * FS))
    assert peak == pytest.approx(20.0, abs=2.5)
    assert np.argmax(aligned) / FS == pytest.approx(merger, abs=2e-3)


def test_filtered_template_and_band_keep_snr_finite_and_located():
    rng = np.random.default_rng(3)
    merger = 5.0
    _, template = generate_newtonian_chirp(fs=FS, duration=8.0, chirp_mass_solar=25.0, t_merger=merger)
    data = inject_signal(generate_detector_noise(fs=FS, duration=8.0, rng=rng), template, target_snr=15.0)
    filtered = bandpass_filter(data, FS, 30.0, 400.0)
    template_filtered = bandpass_filter(template, FS, 30.0, 400.0)
    _, psd = compute_psd(filtered, FS)
    snr, peak, _ = compute_matched_filter(filtered, template_filtered, FS, psd=psd, f_low=30.0, f_high=400.0)
    aligned = align_to_merger(snr, int(merger * FS))
    assert 8.0 < peak < 16.0
    assert np.argmax(aligned) / FS == pytest.approx(merger, abs=2e-3)


def test_empty_analysis_band_is_rejected():
    with pytest.raises(ValueError, match="banda"):
        compute_matched_filter(np.ones(64), np.ones(64), FS, f_low=3000.0, f_high=4000.0)


def test_newtonian_chirp_follows_leading_order_frequency_law():
    chirp_mass = 30.0
    merger = 3.0
    time, strain = generate_newtonian_chirp(fs=FS, duration=4.0, chirp_mass_solar=chirp_mass,
                                            t_merger=merger, f_min=30.0, f_max=300.0)
    support = np.flatnonzero(strain)
    script_m = chirp_mass * SOLAR_MASS_SECONDS
    tau_30 = (5.0 / 256.0) * (np.pi * 30.0) ** (-8.0 / 3.0) * script_m ** (-5.0 / 3.0)
    # O sinal comeca quando f = 30 Hz, cerca de 0,26 s antes da fusao para 30 M_sol.
    assert merger - time[support[0]] == pytest.approx(tau_30, abs=2.0 / FS)
    assert time[support[-1]] < merger
    with pytest.raises(ValueError):
        generate_newtonian_chirp(chirp_mass_solar=-1.0)


def test_inject_signal_uses_optimal_white_noise_snr():
    rng = np.random.default_rng(5)
    noise = generate_detector_noise(fs=FS, duration=4.0, rng=rng)
    _, signal = generate_newtonian_chirp(fs=FS, duration=4.0, chirp_mass_solar=20.0, t_merger=3.0)
    injected = inject_signal(noise, signal, target_snr=12.0) - noise
    assert np.sqrt(np.sum(injected ** 2)) / np.std(noise) == pytest.approx(12.0)
