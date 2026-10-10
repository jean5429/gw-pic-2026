from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

DEFAULT_PIXEL_FEATURES = [
    "minimum",
    "p01",
    "p10",
    "median",
    "p90",
    "p99",
    "maximum",
    "mean",
    "std",
]


def pixel_statistics(array: ArrayLike) -> dict[str, float | int | str]:
    qgram = np.asarray(array, dtype=float)
    if qgram.ndim != 2:
        raise ValueError("qgram deve ser matriz 2D")
    finite = qgram[np.isfinite(qgram)]
    if finite.size == 0:
        raise ValueError("qgram sem valores finitos")
    percentiles = np.percentile(finite, [1, 10, 50, 90, 99])
    return {
        "shape": str(tuple(qgram.shape)),
        "size": int(qgram.size),
        "finite_fraction": float(finite.size / qgram.size),
        "minimum": float(finite.min()),
        "p01": float(percentiles[0]),
        "p10": float(percentiles[1]),
        "median": float(percentiles[2]),
        "p90": float(percentiles[3]),
        "p99": float(percentiles[4]),
        "maximum": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
    }


def grouped_domain_audit(
    frame: pd.DataFrame,
    *,
    target_column: str = "predicted_class",
    positive_label: str = "Scattered_Light",
    group_column: str = "event",
    feature_columns: Sequence[str] = DEFAULT_PIXEL_FEATURES,
    n_splits: int = 5,
    random_state: int = 2026,
) -> dict:
    """Testa se estatisticas de baixo nivel reproduzem os rotulos da CNN.

    O split e agrupado por evento para impedir que detectores do mesmo evento
    aparecam simultaneamente em treino e teste.
    """

    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    missing = (set(feature_columns) | {target_column, group_column}) - set(frame.columns)
    if missing:
        raise ValueError(f"colunas ausentes: {sorted(missing)}")
    y = (frame[target_column].astype(str) == positive_label).astype(int).to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError("auditoria binaria requer duas classes presentes")
    groups = frame[group_column].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    folds = min(int(n_splits), len(unique_groups))
    if folds < 2:
        raise ValueError("grupos insuficientes para validacao")
    cv = GroupKFold(n_splits=folds)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=int(random_state),
        ),
    )
    probabilities = cross_val_predict(
        model,
        frame[list(feature_columns)],
        y,
        groups=groups,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    univariate: dict[str, dict] = {}
    for feature in feature_columns:
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float)
        fill = np.nanmedian(values)
        values = np.where(np.isfinite(values), values, fill)
        held_predictions = np.zeros_like(y)
        held_scores = np.zeros_like(values)
        thresholds: list[float] = []
        directions: list[int] = []
        for train_index, test_index in cv.split(values, y, groups):
            unique_values = np.unique(values[train_index])
            candidates = (
                (unique_values[:-1] + unique_values[1:]) / 2.0
                if len(unique_values) > 1
                else unique_values
            )
            best = (-np.inf, float(candidates[0]), 1)
            for direction in (1, -1):
                for threshold in candidates:
                    train_prediction = (
                        values[train_index] >= threshold
                        if direction == 1
                        else values[train_index] <= threshold
                    ).astype(int)
                    score = balanced_accuracy_score(y[train_index], train_prediction)
                    if score > best[0]:
                        best = (float(score), float(threshold), int(direction))
            _, threshold, direction = best
            thresholds.append(threshold)
            directions.append(direction)
            held_predictions[test_index] = (
                values[test_index] >= threshold
                if direction == 1
                else values[test_index] <= threshold
            ).astype(int)
            held_scores[test_index] = direction * values[test_index]
        group_a = values[y == 1]
        group_b = values[y == 0]
        pooled = np.sqrt(
            ((len(group_a) - 1) * np.var(group_a, ddof=1) + (len(group_b) - 1) * np.var(group_b, ddof=1))
            / max(1, len(group_a) + len(group_b) - 2)
        )
        effect = float((group_a.mean() - group_b.mean()) / pooled) if pooled > 0 else float("nan")
        univariate[feature] = {
            "balanced_accuracy": float(balanced_accuracy_score(y, held_predictions)),
            "roc_auc_oriented": float(roc_auc_score(y, held_scores)),
            "cohen_d": effect,
            "thresholds": thresholds,
            "directions": directions,
        }

    return {
        "target": positive_label,
        "n_samples": int(len(frame)),
        "n_groups": int(len(unique_groups)),
        "class_counts": frame[target_column].astype(str).value_counts().to_dict(),
        "features": list(feature_columns),
        "grouped_logistic_regression": {
            "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
            "roc_auc": float(roc_auc_score(y, probabilities)),
            "confusion_matrix": confusion_matrix(y, predictions).tolist(),
        },
        "univariate": univariate,
    }


def audit_csv(
    input_csv: str | Path,
    output_json: str | Path,
    **kwargs,
) -> dict:
    import json

    result = grouped_domain_audit(pd.read_csv(input_csv), **kwargs)
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
