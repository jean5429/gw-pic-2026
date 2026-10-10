import numpy as np
import pytest

from gwproject.qgram_inference import probability_entropy, qgram_to_uint8


def test_qgram_normalization_returns_uint8():
    qgram = np.arange(100, dtype=float).reshape(10, 10)
    image = qgram_to_uint8(qgram)
    assert image.dtype == np.uint8
    assert image.shape == qgram.shape
    assert image.min() == 0
    assert image.max() == 255


def test_constant_qgram_is_rejected():
    with pytest.raises(ValueError, match="sem contraste"):
        qgram_to_uint8(np.ones((10, 10)))


def test_probability_entropy_bounds():
    assert 0 <= probability_entropy([0.9, 0.1]) <= 1


def test_qgram_center_crop_matches_duration_ratio():
    qgram = np.arange(8 * 40, dtype=float).reshape(8, 40)
    image = qgram_to_uint8(
        qgram,
        source_duration_seconds=4.0,
        target_duration_seconds=1.0,
    )
    assert image.shape == (8, 10)
