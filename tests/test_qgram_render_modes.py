from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from gwproject.qgram_inference import (
    GRAVITY_SPY_ENERGY_MAX,
    CatalogQgramDataset,
    catalog_network_input,
    gravity_spy_gray,
    qgram_to_uint8,
    render_settings,
)


def test_gravity_spy_gray_matches_png_round_trip(tmp_path):
    """A cadeia deve igualar um PNG viridis (0 a 25,5) relido com ``convert("L")``.

    ``matplotlib.image.imsave`` grava um pixel por elemento, como o PNG de treino seria
    lido antes do recorte, e o caminho de normalizacao e independente do implementado.
    """
    from matplotlib.image import imsave

    energy = np.linspace(-2.0, 40.0, 420).reshape(12, 35)
    path = tmp_path / "energy.png"
    imsave(path, energy, cmap="viridis", vmin=0.0, vmax=GRAVITY_SPY_ENERGY_MAX)
    with Image.open(path) as image:
        expected = np.asarray(image.convert("L"))
    np.testing.assert_array_equal(gravity_spy_gray(energy), expected)


def test_gravity_spy_gray_is_monotonic_and_saturates():
    ramp = gravity_spy_gray(np.linspace(0.0, 2 * GRAVITY_SPY_ENERGY_MAX, 2001)[None, :])[0]
    assert np.all(np.diff(ramp.astype(int)) >= 0)
    assert ramp[0] < ramp[-1]
    top = gravity_spy_gray(np.array([[GRAVITY_SPY_ENERGY_MAX, 1e6]]))[0]
    assert top[0] == top[1]
    bottom = gravity_spy_gray(np.array([[0.0, -5.0]]))[0]
    assert bottom[0] == bottom[1]


def test_gravity_spy_mode_keeps_axis_contract():
    """Entrada ``(tempo, frequencia)``; saida ``(frequencia, tempo recortado)``."""
    n_time, n_freq = 40, 20
    qgram = np.ones((n_time, n_freq))
    qgram[n_time // 2, n_freq - 1] = 20.0
    image = qgram_to_uint8(
        qgram,
        source_duration_seconds=4.0,
        target_duration_seconds=1.0,
        flip_frequency=True,
        render_mode="gravity_spy",
    )
    assert image.shape == (n_freq, 10)
    row, col = np.unravel_index(np.argmax(image), image.shape)
    assert row == 0
    assert col == n_time // 2 - 15


def test_gravity_spy_mode_ignores_baseline_and_percentiles():
    qgram = np.random.default_rng(0).gamma(2.0, 1.0, size=(30, 12))
    reference = qgram_to_uint8(qgram, render_mode="gravity_spy")
    other = qgram_to_uint8(
        qgram,
        lower_percentile=10,
        upper_percentile=90,
        baseline_mode="per_frequency_median",
        render_mode="gravity_spy",
    )
    np.testing.assert_array_equal(reference, other)
    np.testing.assert_array_equal(reference, gravity_spy_gray(qgram.T))


def test_default_render_mode_is_unchanged_percentile():
    qgram = np.random.default_rng(1).gamma(2.0, 1.0, size=(30, 12))
    kwargs = dict(baseline_mode="per_frequency_median", flip_frequency=True)
    np.testing.assert_array_equal(
        qgram_to_uint8(qgram, **kwargs),
        qgram_to_uint8(qgram, render_mode="percentile", **kwargs),
    )


def test_invalid_render_mode_is_rejected():
    with pytest.raises(ValueError, match="render_mode"):
        qgram_to_uint8(np.ones((4, 4)), render_mode="viridis")


def test_dataset_forwards_render_mode(tmp_path):
    qgram = np.random.default_rng(2).gamma(2.0, 1.0, size=(40, 16))
    path = tmp_path / "H1_qgram.npy"
    np.save(path, qgram)
    frame = pd.DataFrame([{"event": "GWX", "detector": "H1", "qgram_path": str(path)}])
    dataset = CatalogQgramDataset(
        frame,
        lambda image: np.asarray(image),
        source_duration_seconds=4.0,
        target_duration_seconds=1.0,
        flip_frequency=True,
        render_mode="gravity_spy",
    )
    pixels, index = dataset[0]
    assert index == 0
    expected = qgram_to_uint8(
        qgram,
        source_duration_seconds=4.0,
        target_duration_seconds=1.0,
        flip_frequency=True,
        render_mode="gravity_spy",
    )
    np.testing.assert_array_equal(pixels, expected)


def test_render_settings_reads_config_and_override():
    cfg = {
        "lower_percentile": 1.0,
        "upper_percentile": 99.5,
        "source_outseg_seconds": 4.0,
        "target_duration_seconds": 1.0,
        "baseline_mode": "per_frequency_median",
        "flip_frequency": True,
    }
    assert render_settings(cfg)["render_mode"] == "percentile"
    assert render_settings({**cfg, "render_mode": "gravity_spy"})["render_mode"] == "gravity_spy"
    assert render_settings(cfg, "gravity_spy")["source_duration_seconds"] == 4.0
    assert render_settings(cfg, "gravity_spy")["render_mode"] == "gravity_spy"


def test_network_input_matches_inference_transform():
    """Grad-CAM e render-qgram devem ver o mesmo tensor do ensemble."""
    import torch
    from torchvision import transforms

    image = qgram_to_uint8(
        np.random.default_rng(3).gamma(2.0, 1.0, size=(1000, 500)),
        source_duration_seconds=4.0,
        target_duration_seconds=1.0,
        flip_frequency=True,
        render_mode="gravity_spy",
    )
    canonical = transforms.Compose([
        transforms.Resize((140, 170)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])(Image.fromarray(image))
    pixels = catalog_network_input(image, height=140, width=170)
    tensor = (torch.from_numpy(pixels.astype(np.float32) / 255.0) - 0.5) / 0.5
    assert pixels.shape == (140, 170)
    torch.testing.assert_close(tensor[None], canonical, rtol=0, atol=1e-6)
