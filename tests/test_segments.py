import numpy as np
import pytest

from gwproject.segments import select_finite_event_segment


def test_finite_segment_without_trim():
    x = np.linspace(-1, 1, 320)
    selected, info = select_finite_event_segment(x, 160, 10.0)
    assert np.array_equal(selected, x)
    assert not info.trimmed
    assert info.event_index_segment == 160


def test_leading_nonfinite_edge_can_be_trimmed():
    x = np.ones(320)
    x[:40] = np.nan
    selected, info = select_finite_event_segment(
        x,
        160,
        10.0,
        min_valid_before_event_seconds=10.0,
        min_valid_after_event_seconds=10.0,
        min_contiguous_valid_seconds=20.0,
    )
    assert info.trimmed
    assert info.start_index == 40
    assert info.event_index_segment == 120
    assert np.all(np.isfinite(selected))


def test_event_inside_nonfinite_region_is_rejected():
    x = np.ones(320)
    x[150:170] = np.nan
    with pytest.raises(ValueError, match="instante do evento"):
        select_finite_event_segment(x, 160, 10.0)


def test_internal_gap_is_rejected():
    x = np.ones(320)
    x[20:30] = np.nan
    with pytest.raises(ValueError, match="gap nao finito interno"):
        select_finite_event_segment(
            x,
            160,
            10.0,
            min_valid_before_event_seconds=8.0,
            min_valid_after_event_seconds=8.0,
            min_contiguous_valid_seconds=20.0,
        )
