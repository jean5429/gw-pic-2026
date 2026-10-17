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


def test_qgram_duration_crop_uses_time_axis_and_reorients_to_image_layout():
    """Contrato: entrada ``(n_tempo, n_frequencia)``; saida ``(n_frequencia, n_tempo)``.

    Ate 26/07/2026 o recorte de duracao operava sobre o eixo 1 (frequencia) em vez do eixo 0
    (tempo), e nenhum teste comparava contra um artefato real -- ver a memoria de projeto
    "bug_eixo_qgram" e a nota de correcao em ``PREREGISTRO.md`` (26/07/2026). Este teste marca
    um pixel num tempo/frequencia conhecidos e verifica que ele sobrevive ao recorte central
    do TEMPO e aparece no eixo correto da imagem de saida.
    """
    n_time, n_freq = 40, 20
    qgram = np.zeros((n_time, n_freq))
    marked_time, marked_freq = n_time // 2, n_freq - 1  # frequencia mais alta = ultima coluna
    qgram[marked_time, marked_freq] = 100.0

    image = qgram_to_uint8(
        qgram,
        source_duration_seconds=4.0,
        target_duration_seconds=1.0,
        flip_frequency=True,
    )

    # saida transposta: eixo 0 vira frequencia (tamanho preservado, sem recorte fisico aqui),
    # eixo 1 vira tempo, recortado de 40 para round(40 * 1/4) = 10 bins centrais.
    assert image.shape == (n_freq, 10)
    # o pixel marcado estava no bin de tempo central (dentro do recorte) e na frequencia mais
    # alta; com flip_frequency=True a frequencia mais alta vai para a linha 0 (topo da imagem).
    row, col = np.unravel_index(np.argmax(image), image.shape)
    assert row == 0
    assert image[row, col] == image.max()


def test_qgram_duration_crop_shape_matches_real_gw150914_artifact_ratio():
    """Regressao de contrato usando as dimensoes reais medidas em 26/07/2026.

    ``outputs/gwtc4_218/qgrams/GW150914-v4/H1_qgram.json`` (pipeline real, nao
    sintetico) registra
    ``shape == [1000, 500]`` para ``outseg_seconds == 4.0``. Recortando para 1 s central e
    aplicando a orientacao de imagem, a saida deve ser ``(500, 250)``.
    """
    n_time, n_freq = 1000, 500
    qgram = np.random.default_rng(0).normal(loc=1.0, scale=0.05, size=(n_time, n_freq))
    qgram = np.clip(qgram, 1e-3, None)
    image = qgram_to_uint8(
        qgram,
        source_duration_seconds=4.0,
        target_duration_seconds=1.0,
    )
    assert image.shape == (n_freq, 250)
