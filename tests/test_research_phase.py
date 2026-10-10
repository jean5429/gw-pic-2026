from __future__ import annotations

import numpy as np
import pytest

from gwproject.research.phase import randomize_fourier_phase


def test_phase_randomization_preserves_magnitude_spectrum():
    rng = np.random.default_rng(2)
    signal = rng.normal(size=1024)
    randomized = randomize_fourier_phase(signal, fraction=1.0, seed=7)
    np.testing.assert_allclose(
        np.abs(np.fft.rfft(signal)),
        np.abs(np.fft.rfft(randomized)),
        rtol=1e-10,
        atol=1e-10,
    )
    assert not np.allclose(signal, randomized)


def test_zero_fraction_recovers_signal():
    signal = np.linspace(-1, 1, 101)
    recovered = randomize_fourier_phase(signal, fraction=0.0, seed=3)
    np.testing.assert_allclose(signal, recovered, atol=1e-12)


def test_invalid_fraction_rejected():
    with pytest.raises(ValueError):
        randomize_fourier_phase(np.arange(10), fraction=1.2)
