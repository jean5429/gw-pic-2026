import numpy as np

from gwproject.alignment import best_lag_alignment


def test_h1_l1_diagnostic_can_apply_lag_and_sign():
    rng = np.random.default_rng(4)
    a = rng.normal(size=4096)
    delay = 17
    b = np.concatenate([np.zeros(delay), -a[:-delay]])
    result = best_lag_alignment(a, b, fs=1024.0, max_lag_seconds=0.05, allow_sign_flip=True)
    assert abs(result.correlation - 1.0) < 1e-10
    assert result.sign == -1.0
    assert abs(abs(result.lag_samples) - delay) <= 1
