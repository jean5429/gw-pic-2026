"""Exploratory, paired 2x2 interventions on the final grayscale CNN input.

These are image interventions, not strain injections or equivalent Q transforms.
Condition order is part of the analysis contract; do not reorder it.
"""
from __future__ import annotations

import numpy as np

CONDITIONS = ("identity", "gamma", "contrast", "contrast_gamma")


def intensity_variant(image, condition: str, *, gamma=0.7, lower=1.0, upper=99.5):
    """Return uint8 pixels and a flag for a degenerate contrast operation.

    Contrast (when enabled) precedes gamma. A collapsed percentile interval
    leaves contrast unchanged and is reported, never used to exclude a sample.
    Identity is exactly byte-preserving. No spatial coordinate is changed.
    """
    pixels = np.asarray(image)
    if pixels.ndim != 2 or pixels.dtype != np.uint8 or not pixels.size:
        raise ValueError("expected a nonempty 2D uint8 image")
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    if not np.isfinite(gamma) or gamma <= 0 or not 0 <= lower < upper <= 100:
        raise ValueError("invalid intensity parameters")
    if condition == "identity":
        return pixels.copy(), False
    values = pixels.astype(np.float64) / 255.0
    degenerate = False
    if condition in {"contrast", "contrast_gamma"}:
        low, high = np.percentile(values, [lower, upper])
        degenerate = bool(high <= low)
        if not degenerate:
            values = np.clip((values - low) / (high - low), 0.0, 1.0)
    if condition in {"gamma", "contrast_gamma"}:
        values = values ** gamma
    return np.rint(255.0 * values).astype(np.uint8), degenerate


def fast_mahalanobis(embeddings, reference):
    """Same quadratic form as the canonical scorer, evaluated class by class.

    Matrix multiplication avoids an unoptimized three-operand einsum and the
    (sample, class, feature) temporary. Raw scales remain checkpoint-specific.
    """
    x = np.atleast_2d(np.asarray(embeddings, dtype=np.float64))
    means = np.asarray(reference.class_means, dtype=np.float64)
    precision = np.asarray(reference.precision, dtype=np.float64)
    if x.ndim != 2 or means.ndim != 2 or x.shape[1] != means.shape[1]:
        raise ValueError("incompatible embeddings and class means")
    if precision.shape != (x.shape[1], x.shape[1]) or not len(means):
        raise ValueError("invalid precision or empty class means")
    result = np.full(len(x), np.inf)
    for mean in means:
        delta = x - mean
        distance = np.sum((delta @ precision) * delta, axis=1)
        result = np.minimum(result, distance)
    if not np.isfinite(result).all() or np.any(result < -1e-7):
        raise ValueError("nonfinite or negative Mahalanobis quadratic form")
    return np.maximum(result, 0.0)


def factorial_contrasts(values):
    """Per-block effects for columns identity, gamma, contrast, both."""
    y = np.asarray(values, dtype=float)
    if y.ndim != 2 or y.shape[1] != 4 or not np.isfinite(y).all():
        raise ValueError("expected finite (blocks, 4) values")
    base, gamma, contrast, both = y.T
    return {
        "gamma_minus_identity": gamma - base,
        "contrast_minus_identity": contrast - base,
        "both_minus_identity": both - base,
        "gamma_main_effect": 0.5 * ((gamma - base) + (both - contrast)),
        "contrast_main_effect": 0.5 * ((contrast - base) + (both - gamma)),
        "interaction": both - gamma - contrast + base,
    }


def paired_bootstrap(values, *, seed=20260918, replicates=2000):
    """Resample already-aggregated independent blocks, not detector rows/seeds.

    Intervals are exploratory percentile intervals conditional on this fixed
    ensemble and catalog. They are not simultaneous confidence intervals.
    """
    delta = np.asarray(values, dtype=float)
    if delta.ndim != 1 or len(delta) < 2 or not np.isfinite(delta).all():
        raise ValueError("expected at least two finite block contrasts")
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates required")
    rng = np.random.default_rng(seed)
    # Bounded memory even if a future dataset has many more blocks.
    means = np.empty(replicates)
    for start in range(0, replicates, 100):
        count = min(100, replicates - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        means[start:start + count] = delta[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"estimate": float(delta.mean()), "ci_low": float(low),
            "ci_high": float(high), "blocks": int(len(delta))}
