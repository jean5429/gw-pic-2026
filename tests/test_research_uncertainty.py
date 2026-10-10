from __future__ import annotations

import numpy as np

from gwproject.research.uncertainty import (
    energy_score,
    ensemble_statistics,
    expected_calibration_error,
    fit_mahalanobis_reference,
    mahalanobis_score,
)


def test_ensemble_statistics_detect_disagreement():
    logits = np.array([
        [[5.0, 0.0], [5.0, 0.0]],
        [[5.0, 0.0], [0.0, 5.0]],
    ])
    stats = ensemble_statistics(logits)
    assert stats["vote_fraction"][0] == 1.0
    assert stats["vote_fraction"][1] == 0.5
    assert stats["mutual_information"][1] > stats["mutual_information"][0]


def test_energy_more_negative_for_larger_logits():
    result = energy_score([[1.0, 0.0], [10.0, 0.0]])
    assert result[1] < result[0]


def test_ece_perfect_predictions_near_zero():
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    assert expected_calibration_error(probabilities, labels) < 1e-10


def test_mahalanobis_reference_scores_far_point_higher():
    rng = np.random.default_rng(4)
    x0 = rng.normal(0, 0.2, size=(30, 4))
    x1 = rng.normal(2, 0.2, size=(30, 4))
    x = np.vstack([x0, x1])
    y = np.array([0] * 30 + [1] * 30)
    reference = fit_mahalanobis_reference(x, y)
    scores = mahalanobis_score([[0, 0, 0, 0], [10, 10, 10, 10]], reference)
    assert scores[1] > scores[0]


def test_empirical_cdf_is_monotonic_and_bounded():
    from gwproject.research.uncertainty import empirical_cdf

    reference = np.array([0.0, 1.0, 2.0, 3.0])
    result = empirical_cdf([-1.0, 1.5, 10.0], reference)
    assert np.all(np.diff(result) >= 0)
    assert np.all((result > 0.0) & (result <= 1.0))
    assert result[-1] == 1.0


def test_calibration_metrics_and_temperature_scaling():
    from gwproject.research.uncertainty import (
        fit_temperature,
        multiclass_brier_score,
        negative_log_likelihood,
        softmax_probabilities,
        temperature_scale_logits,
    )

    logits = np.array([[8.0, 0.0], [0.0, 8.0], [3.0, 0.0], [0.0, 3.0]])
    labels = np.array([0, 1, 0, 1])
    probabilities = softmax_probabilities(logits)
    assert negative_log_likelihood(probabilities, labels) < 0.1
    assert multiclass_brier_score(probabilities, labels) < 0.02

    temperature = fit_temperature(logits, labels)
    assert temperature > 0
    scaled = temperature_scale_logits(logits, temperature)
    assert np.array_equal(np.argmax(scaled, axis=1), np.argmax(logits, axis=1))
