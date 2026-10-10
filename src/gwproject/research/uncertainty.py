from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def energy_score(logits: ArrayLike, temperature: float = 1.0) -> FloatArray:
    values = np.asarray(logits, dtype=float)
    if temperature <= 0:
        raise ValueError("temperature deve ser positivo")
    scaled = values / temperature
    maximum = np.max(scaled, axis=-1, keepdims=True)
    logsumexp = maximum[..., 0] + np.log(np.exp(scaled - maximum).sum(axis=-1))
    return np.asarray(-temperature * logsumexp, dtype=float)


def normalized_entropy(probabilities: ArrayLike, axis: int = -1) -> FloatArray:
    p = np.asarray(probabilities, dtype=float)
    safe = np.clip(p, 1e-12, 1.0)
    entropy = -np.sum(safe * np.log(safe), axis=axis)
    classes = p.shape[axis]
    return np.asarray(entropy / np.log(max(2, classes)), dtype=float)


def ensemble_statistics(logits_by_model: ArrayLike) -> dict[str, FloatArray]:
    """Resume incerteza epistemica de um ensemble.

    Entrada: ``(n_modelos, n_amostras, n_classes)``.
    """

    logits = np.asarray(logits_by_model, dtype=float)
    if logits.ndim != 3 or logits.shape[0] < 1 or logits.shape[2] < 2:
        raise ValueError("logits devem ter shape (modelos, amostras, classes)")
    probabilities = _softmax(logits, axis=2)
    mean_probability = probabilities.mean(axis=0)
    predictive_entropy = normalized_entropy(mean_probability, axis=1)
    expected_entropy = normalized_entropy(probabilities, axis=2).mean(axis=0)
    mutual_information = predictive_entropy - expected_entropy
    votes = np.argmax(probabilities, axis=2)
    winner = np.argmax(mean_probability, axis=1)
    vote_fraction = np.array(
        [np.mean(votes[:, i] == winner[i]) for i in range(votes.shape[1])],
        dtype=float,
    )
    sorted_probability = np.sort(mean_probability, axis=1)
    margin = sorted_probability[:, -1] - sorted_probability[:, -2]
    return {
        "mean_probability": np.asarray(mean_probability, dtype=float),
        "predictive_entropy": np.asarray(predictive_entropy, dtype=float),
        "expected_entropy": np.asarray(expected_entropy, dtype=float),
        "mutual_information": np.asarray(mutual_information, dtype=float),
        "vote_fraction": vote_fraction,
        "probability_margin": np.asarray(margin, dtype=float),
        "energy_mean": np.asarray(energy_score(logits).mean(axis=0), dtype=float),
        "energy_std": np.asarray(energy_score(logits).std(axis=0), dtype=float),
    }


def expected_calibration_error(
    probabilities: ArrayLike,
    labels: ArrayLike,
    *,
    n_bins: int = 15,
) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    if p.ndim != 2 or len(p) != len(y):
        raise ValueError("probabilidades/labels incompatíveis")
    if n_bins < 2:
        raise ValueError("n_bins deve ser >= 2")
    confidence = np.max(p, axis=1)
    prediction = np.argmax(p, axis=1)
    correct = prediction == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = max(1, len(y))
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidence > left) & (confidence <= right)
        if not np.any(mask):
            continue
        ece += (mask.sum() / total) * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


@dataclass(frozen=True)
class MahalanobisReference:
    class_means: FloatArray
    precision: FloatArray
    class_labels: NDArray[np.int64]


def fit_mahalanobis_reference(
    embeddings: ArrayLike,
    labels: ArrayLike,
    *,
    shrinkage: bool = True,
) -> MahalanobisReference:
    x = np.asarray(embeddings, dtype=float)
    y = np.asarray(labels, dtype=int)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("embeddings/labels incompatíveis")
    class_labels = np.unique(y)
    means = np.vstack([x[y == label].mean(axis=0) for label in class_labels])
    centered = np.vstack([x[y == label] - means[i] for i, label in enumerate(class_labels)])
    if shrinkage:
        try:
            from sklearn.covariance import LedoitWolf

            precision = LedoitWolf().fit(centered).precision_
        except Exception:
            covariance = np.cov(centered, rowvar=False)
            precision = np.linalg.pinv(covariance)
    else:
        covariance = np.cov(centered, rowvar=False)
        precision = np.linalg.pinv(covariance)
    return MahalanobisReference(
        class_means=np.asarray(means, dtype=float),
        precision=np.asarray(precision, dtype=float),
        class_labels=np.asarray(class_labels, dtype=np.int64),
    )


def mahalanobis_score(
    embeddings: ArrayLike,
    reference: MahalanobisReference,
) -> FloatArray:
    x = np.asarray(embeddings, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    differences = x[:, None, :] - reference.class_means[None, :, :]
    distances = np.einsum("nkd,df,nkf->nk", differences, reference.precision, differences)
    return np.asarray(np.min(distances, axis=1), dtype=float)
