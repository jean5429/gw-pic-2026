from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gwproject.qgram_inference import qgram_to_uint8

GW150914_L1 = Path("outputs/gwtc/qgrams/GW150914-v4/L1_qgram.npy")


def test_gwpy_layout_is_transposed_before_time_crop():
    # gwpy.Spectrogram: eixo 0 = tempo (40 amostras), eixo 1 = frequencia (8).
    qgram = np.ones((40, 8), dtype=float)
    qgram[20, 6] = 50.0  # instante central, frequencia alta
    image = qgram_to_uint8(
        qgram,
        lower_percentile=0,
        upper_percentile=100,
        source_duration_seconds=4.0,
        target_duration_seconds=1.0,
        flip_frequency=True,
        layout="gwpy",
    )
    # Linhas = frequencia (8), colunas = tempo recortado (40 / 4 = 10).
    assert image.shape == (8, 10)
    row, col = np.unravel_index(int(image.argmax()), image.shape)
    assert row == 8 - 1 - 6  # frequencia alta perto do topo
    assert col == 20 - 15  # recorte central [15, 25)


def test_invalid_layout_is_rejected():
    with pytest.raises(ValueError, match="layout"):
        qgram_to_uint8(np.arange(16.0).reshape(4, 4), layout="freq_time")


@pytest.mark.skipif(not GW150914_L1.exists(), reason="Q-gram do GW150914 nao baixado")
def test_gw150914_chirp_lands_at_center_time_and_upper_band():
    raw = np.load(GW150914_L1)
    n_times, n_freqs = raw.shape
    assert (n_times, n_freqs) == (1000, 500)  # 4 s x 500 frequencias log (20-512 Hz)

    image = qgram_to_uint8(
        raw,
        source_duration_seconds=4.0,
        target_duration_seconds=1.0,
        baseline_mode="per_frequency_median",
        flip_frequency=True,
        layout="gwpy",
    )
    assert image.shape == (500, 250)

    # Centroide dos pixels saturados = assinatura do chirp.
    rows, cols = np.nonzero(image >= np.percentile(image, 99.9))
    center_col = float(np.median(cols))
    center_row = float(np.median(rows))
    # A fusao ocorre ~0,02 s apos o GPS do catalogo, no centro do recorte de 1 s.
    assert abs(center_col - 125) < 0.1 * 250
    # Pico em ~150 Hz: acima da media geometrica de 20-512 Hz (~101 Hz),
    # portanto na metade superior apos o flip.
    frequency = 20.0 * (512.0 / 20.0) ** ((n_freqs - 1 - center_row) / (n_freqs - 1))
    assert 100.0 < frequency < 300.0


def _viridis_gray(value: float) -> int:
    from matplotlib import colormaps

    rgb = np.round(255.0 * np.asarray(colormaps["viridis"](value)[:3]))
    return int(round(float(rgb @ np.array([0.299, 0.587, 0.114]))))


def test_gravity_spy_render_uses_fixed_energy_scale():
    # Energia 0 -> base do viridis; >= 25,5 -> topo (amarelo); sem estiramento.
    qgram = np.array([[0.0, 12.75, 25.5, 100.0]])
    image = qgram_to_uint8(qgram, render_mode="gravity_spy")
    assert image.tolist() == [[_viridis_gray(0.0), _viridis_gray(0.5), _viridis_gray(1.0), _viridis_gray(1.0)]]
    # A escala e absoluta: multiplicar um Q-gram fraco nao e o mesmo que normaliza-lo.
    weak = qgram_to_uint8(np.full((2, 2), 1.0), render_mode="gravity_spy")
    assert int(weak.max()) < _viridis_gray(0.2)


def test_invalid_render_mode_is_rejected():
    with pytest.raises(ValueError, match="render_mode"):
        qgram_to_uint8(np.arange(16.0).reshape(4, 4), render_mode="jet")


@pytest.mark.skipif(not GW150914_L1.exists(), reason="Q-gram do GW150914 nao baixado")
def test_gw150914_gravity_spy_render_background_matches_training_range():
    image = qgram_to_uint8(
        np.load(GW150914_L1),
        source_duration_seconds=4.0,
        target_duration_seconds=1.0,
        flip_frequency=True,
        layout="gwpy",
        render_mode="gravity_spy",
    )
    # PNGs de treino harmonizados tem fundo cinza-escuro (mediana ~40), nao preto.
    assert 30 <= float(np.median(image)) <= 60
    assert image.max() > 150  # o chirp ainda satura em direcao ao amarelo
