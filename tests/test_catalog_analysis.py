import numpy as np

from gwproject.catalog_analysis import (
    adaptive_ridge_plan,
    build_local_template_bank,
    catalog_detector_chirp_mass,
    catalog_detector_component_masses,
    chirp_time_seconds,
    frequency_for_chirp_time,
    ridge_dynamic_programming,
)


def _config():
    return {
        "catalog_ridge": {
            "fmin_hz": 25.0,
            "fmax_hz": 300.0,
            "max_window_before_seconds": 14.0,
            "padding_seconds": 0.5,
            "minimum_bandwidth_hz": 12.0,
            "minimum_seconds_before_merger": 0.012,
            "isco_fraction": 0.95,
        }
    }


def test_chirp_time_and_frequency_are_inverse():
    mc = 20.0
    frequency = 35.0
    time = chirp_time_seconds(mc, frequency)
    recovered = frequency_for_chirp_time(mc, time)
    assert np.isclose(recovered, frequency, rtol=1e-10)


def test_adaptive_plan_is_inside_available_segment():
    plan = adaptive_ridge_plan(
        20.0,
        available_before_seconds=16.0,
        available_after_seconds=16.0,
        m1=30.0,
        m2=20.0,
        config=_config(),
    )
    assert plan.status == "ok"
    assert -16.0 <= plan.segment_start_seconds < plan.window_start_seconds < 0
    assert plan.window_start_seconds < plan.window_stop_seconds < 0
    assert plan.fmax_hz > plan.fmin_hz


def test_dynamic_programming_tracks_increasing_ridge():
    rng = np.random.default_rng(4)
    n_f, n_t = 40, 120
    energy = rng.exponential(0.2, size=(n_f, n_t))
    true_idx = np.round(np.linspace(5, 31, n_t)).astype(int)
    energy[true_idx, np.arange(n_t)] += 10.0
    frequencies = np.linspace(30, 250, n_f)
    times = np.linspace(-1.0, -0.02, n_t)
    _, path_frequency, _ = ridge_dynamic_programming(
        energy,
        frequencies,
        times,
        maximum_jump_bins=3,
        jump_penalty=0.1,
    )
    expected = frequencies[true_idx]
    assert np.median(np.abs(path_frequency - expected)) < 8.0


def test_local_template_bank_contains_central_and_variations():
    bank = build_local_template_bank(36.0, 29.0)
    assert bank[0].central
    assert bank[0].name == "catalog_central"
    assert len(bank) >= 4


def test_detector_frame_masses_are_reconstructed_from_source_and_redshift():
    row = {
        "mass_1_source": 30.0,
        "mass_2_source": 20.0,
        "chirp_mass_source": 21.0,
        "redshift": 0.10,
    }
    masses, frame = catalog_detector_component_masses(row)
    mc, mc_frame = catalog_detector_chirp_mass(row)
    assert frame == "detector_from_source_redshift"
    assert mc_frame == "detector_from_source_redshift"
    assert np.allclose(masses, (33.0, 22.0))
    assert np.isclose(mc, 23.1)


def test_detector_frame_mass_requires_redshift_without_explicit_fallback():
    row = {"mass_1_source": 30.0, "mass_2_source": 20.0}
    masses, frame = catalog_detector_component_masses(row)
    assert masses is None
    assert frame == "missing_detector_frame"

    masses, frame = catalog_detector_component_masses(
        row, allow_source_fallback=True
    )
    assert masses == (30.0, 20.0)
    assert frame == "source_unconverted"


def test_explicit_detector_frame_columns_take_precedence():
    row = {
        "mass_1_detector": 44.0,
        "mass_2_detector": 22.0,
        "chirp_mass_detector": 26.0,
        "mass_1_source": 30.0,
        "mass_2_source": 20.0,
        "chirp_mass_source": 21.0,
        "redshift": 0.10,
        "component_mass_detector_origin": "derived_from_source_redshift",
        "chirp_mass_detector_origin": "derived_from_source_redshift",
    }
    masses, frame = catalog_detector_component_masses(row)
    mc, mc_frame = catalog_detector_chirp_mass(row)
    assert masses == (44.0, 22.0)
    assert mc == 26.0
    assert frame == "derived_from_source_redshift"
    assert mc_frame == "derived_from_source_redshift"
