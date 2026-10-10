import numpy as np

from gwproject.preprocessing import preprocess, q_transform_constant_q
from gwproject.ridge import estimate_chirp_mass_from_ridge_points, ridge_from_energy

FS = 2048.0
DURATION = 16.0
N = int(FS * DURATION)
TC = 8.4
MC = 30.7


def chirp_newtonian(mc_solar=MC, f_start=25.0, f_end=250.0, peak=1.0):
    tsun = 6.67430e-11 * 1.98847e30 / 2.99792458e8**3
    t = np.arange(N) / FS
    k = (5.0 / 256.0) * np.pi ** (-8.0 / 3.0) * (mc_solar * tsun) ** (-5.0 / 3.0)
    tau = TC - t
    inside = (tau > k * f_end ** (-8 / 3)) & (tau < k * f_start ** (-8 / 3))
    h = np.zeros(N)
    phase = -2 * np.pi * (8 / 5) * k ** (3 / 8) * tau[inside] ** (5 / 8)
    h[inside] = ((tau[inside] / k) ** (-3 / 8)) ** (2 / 3) * np.cos(phase)
    return h * (peak / np.max(np.abs(h)))


def test_complete_q_ridge_chain_with_q5_is_accurate():
    rng = np.random.default_rng(5)
    data = chirp_newtonian(peak=8.0) + rng.normal(size=N)
    wide, _ = preprocess(
        data,
        FS,
        whitening_fmax=900.0,
        bandpass_fmin=30.0,
        bandpass_fmax=400.0,
    )
    t_rel = np.arange(N) / FS - TC
    mask = (t_rel >= -0.14) & (t_rel <= -0.012)
    qresult = q_transform_constant_q(wide, FS, fmin=35.0, fmax=300.0, n_bins=96, q=5.0)
    ridge_t, ridge_f = ridge_from_energy(
        qresult.energy[:, mask], qresult.frequencies, t_rel[mask], min_quantile=0.70, max_points=80
    )
    estimate = estimate_chirp_mass_from_ridge_points(ridge_t, ridge_f, n_bootstrap=200, seed=42)
    assert abs(estimate.mass_solar - MC) / MC <= 0.15
    relative_width = (
        estimate.dispersion_high_solar - estimate.dispersion_low_solar
    ) / estimate.mass_solar
    assert relative_width <= 0.50


def test_pure_noise_does_not_pass_chirp_mass_gate():
    rng = np.random.default_rng(1001)
    data = rng.normal(size=N)
    wide, _ = preprocess(
        data,
        FS,
        whitening_fmax=900.0,
        bandpass_fmin=30.0,
        bandpass_fmax=400.0,
    )
    t_rel = np.arange(N) / FS - TC
    mask = (t_rel >= -0.14) & (t_rel <= -0.012)
    qresult = q_transform_constant_q(wide, FS, fmin=35.0, fmax=300.0, n_bins=96, q=5.0)
    ridge_t, ridge_f = ridge_from_energy(qresult.energy[:, mask], qresult.frequencies, t_rel[mask])
    try:
        estimate = estimate_chirp_mass_from_ridge_points(ridge_t, ridge_f, n_bootstrap=100, seed=7)
    except (ValueError, RuntimeError):
        return
    assert abs(estimate.mass_solar - MC) / MC > 0.15


def test_quantized_staircase_uses_ols_fallback_instead_of_zero_theil_slope():
    times = np.linspace(-1.0, -0.02, 40)
    frequencies = np.concatenate([np.full(30, 45.0), np.full(10, 90.0)])
    estimate = estimate_chirp_mass_from_ridge_points(
        times,
        frequencies,
        n_bootstrap=120,
        seed=123,
    )
    assert estimate.slope < 0
    assert estimate.fit_method == "ols_fallback_quantized_ridge"
    assert np.isfinite(estimate.mass_solar)


def test_flat_ridge_is_rejected_without_sign_flip():
    times = np.linspace(-1.0, -0.02, 40)
    frequencies = np.full(40, 60.0)
    try:
        estimate_chirp_mass_from_ridge_points(
            times,
            frequencies,
            n_bootstrap=50,
            seed=123,
        )
    except ValueError as exc:
        assert "frequencia constante" in str(exc)
        return
    raise AssertionError("ridge plano deveria ser rejeitado")
