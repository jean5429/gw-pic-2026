from __future__ import annotations

import numpy as np
from PIL import Image

from gwproject.research.panel_preprocessing import (
    PanelCropConfig,
    crop_frequency_band,
    detect_spectrogram_panel,
    frequency_row_bounds,
    prepare_gravity_spy_panel,
)


def test_detect_spectrogram_panel_on_synthetic_canvas():
    canvas = np.full((120, 180), 255, dtype=np.uint8)
    canvas[20:100, 30:145] = 30
    canvas[30:90, 150:160] = 50  # barra de cores separada
    bbox = detect_spectrogram_panel(Image.fromarray(canvas), padding_pixels=0)
    left, top, right, bottom = bbox
    assert 28 <= left <= 32
    assert 18 <= top <= 22
    assert 143 <= right <= 147
    assert 98 <= bottom <= 102


def test_frequency_row_bounds_log_scale():
    top, bottom = frequency_row_bounds(
        100,
        source_fmin_hz=10,
        source_fmax_hz=2048,
        target_fmin_hz=20,
        target_fmax_hz=512,
        log_frequency=True,
    )
    assert 20 <= top <= 30
    assert 80 <= bottom <= 95
    assert top < bottom


def test_prepare_panel_returns_requested_size():
    canvas = np.full((120, 180), 255, dtype=np.uint8)
    canvas[20:100, 30:145] = np.tile(np.linspace(0, 100, 115, dtype=np.uint8), (80, 1))
    image = prepare_gravity_spy_panel(
        Image.fromarray(canvas),
        config=PanelCropConfig(crop_mode="auto", output_height=64, output_width=72, padding_pixels=0),
    )
    assert image.size == (72, 64)


def test_prepare_panel_uses_official_fixed_crop():
    canvas = np.full((700, 800), 255, dtype=np.uint8)
    canvas[66:532, 105:671] = 40
    image = prepare_gravity_spy_panel(
        Image.fromarray(canvas),
        config=PanelCropConfig(output_height=140, output_width=170),
    )
    assert image.size == (170, 140)
    assert np.asarray(image).mean() < 50
