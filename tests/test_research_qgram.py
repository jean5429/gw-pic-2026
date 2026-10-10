from __future__ import annotations

import numpy as np

from gwproject.qgram_inference import qgram_to_uint8


def test_qgram_per_frequency_baseline_and_flip():
    """Contrato de entrada ``(n_tempo, n_frequencia)``; saida ``(n_frequencia, n_tempo)``.

    ``baseline_mode="per_frequency_median"`` deve reduzir sobre o eixo do TEMPO (entrada
    eixo 0) para produzir um baseline por bin de FREQUENCIA (entrada eixo 1).
    ``flip_frequency`` opera sobre o eixo de frequencia da saida (pos-transposicao), nao
    sobre o eixo 0 de entrada. Ate 26/07/2026 esta funcao reduzia sobre o eixo errado (ver
    ``bug_eixo_qgram`` na memoria de projeto) e este teste, com uma fixture ``(4, 8)`` que ja
    pressupunha eixo 0 = frequencia, não pegava o erro.
    """
    qgram = np.ones((4, 8), dtype=float)  # (tempo=4, frequencia=8)
    qgram[0, 3:5] = 10.0  # excesso forte nas frequencias 3-4, no tempo 0
    qgram[3, 3:5] = 2.0  # excesso mais fraco nas mesmas frequencias, no tempo 3
    result = qgram_to_uint8(
        qgram,
        lower_percentile=0,
        upper_percentile=100,
        baseline_mode="per_frequency_median",
        flip_frequency=True,
    )
    assert result.shape == (8, 4)  # transposto: (frequencia, tempo)
    assert result.max() == 255
    row, col = np.unravel_index(np.argmax(result), result.shape)
    # o excesso mais forte ocorreu no tempo 0 -> coluna 0 apos a transposicao.
    assert col == 0
    # frequencias 3 e 4 (de 8, zero-indexed) ficam nas linhas 8-1-3=4 e 8-1-4=3 apos o flip.
    assert row in (3, 4)


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
