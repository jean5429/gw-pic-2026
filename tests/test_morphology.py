import numpy as np
import pytest

from gwproject.morphology import (
    physical_morphology_series,
    strict_interpolate_reference,
    validate_morphology,
)
from gwproject.preprocessing import bandpass_butter


def test_morphology_series_is_physical_not_whitened():
    rng = np.random.default_rng(10)
    fs = 1024.0
    x = rng.normal(size=int(8 * fs))
    expected = bandpass_butter(x, fs, 35.0, 350.0)
    actual = physical_morphology_series(x, fs, fmin=35.0, fmax=350.0)
    assert np.allclose(actual, expected)


def test_published_gate_rejects_global_sign_inversion():
    rng = np.random.default_rng(11)
    fs = 1024.0
    series = rng.normal(size=4096)
    result = validate_morphology(
        -series,
        series,
        fs,
        max_lag_seconds=0.001,
        allow_sign_flip=False,
    )
    assert result.correlation < 0.95
    assert result.sign == 1.0


def test_strict_interpolation_rejects_extrapolation():
    reference = np.column_stack([np.linspace(0.25, 0.45, 100), np.ones(100)])
    with pytest.raises(ValueError, match="fora do dominio"):
        strict_interpolate_reference(np.array([0.24, 0.30]), reference)
