from __future__ import annotations

import numpy as np

from gwproject.qgram_inference import qgram_to_uint8


def test_qgram_per_frequency_baseline_and_flip():
    qgram = np.ones((4, 8), dtype=float)
    qgram[0, 3:5] = 10.0
    qgram[3, 3:5] = 2.0
    result = qgram_to_uint8(
        qgram,
        lower_percentile=0,
        upper_percentile=100,
        baseline_mode="per_frequency_median",
        flip_frequency=True,
    )
    assert result.shape == qgram.shape
    # A linha de maior excesso originalmente no topo deve ir para baixo.
    assert result[-1].max() == 255


def test_qgram_subtract_one_suppresses_unit_floor():
    qgram = np.ones((10, 10), dtype=float)
    qgram[5, 5] = 3.0
    result = qgram_to_uint8(
        qgram,
        lower_percentile=0,
        upper_percentile=100,
        baseline_mode="subtract_one",
    )
    assert np.count_nonzero(result) == 1
