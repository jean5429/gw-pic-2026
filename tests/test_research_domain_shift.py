from __future__ import annotations

import numpy as np
import pandas as pd

from gwproject.research.domain_shift import (
    grouped_domain_audit,
    rendered_qgram_statistics,
)


def test_grouped_domain_audit_finds_trivial_feature():
    rng = np.random.default_rng(8)
    rows = []
    for event in range(30):
        label = "Scattered_Light" if event % 2 else "None_of_the_Above"
        center = -2.0 if label == "Scattered_Light" else 2.0
        for detector in ("H1", "L1"):
            values = {name: rng.normal() for name in ["minimum", "p10", "median", "p90", "p99", "maximum", "mean", "std"]}
            values["p01"] = rng.normal(center, 0.2)
            rows.append({"event": str(event), "detector": detector, "predicted_class": label, **values})
    result = grouped_domain_audit(pd.DataFrame(rows), n_splits=5)
    assert result["grouped_logistic_regression"]["balanced_accuracy"] > 0.9
    assert result["univariate"]["p01"]["balanced_accuracy"] > 0.9


def test_rendered_qgram_statistics_preserves_catalog_metadata(tmp_path):
    qgram_path = tmp_path / "qgram.npy"
    time = np.linspace(0.8, 2.0, 12)[:, None]
    frequency = np.linspace(0.9, 1.3, 8)[None, :]
    np.save(qgram_path, time * frequency)
    frame = pd.DataFrame(
        {
            "event": ["GW-test"],
            "detector": ["H1"],
            "qgram_path": [str(qgram_path)],
            "forced_closed_set_class": ["None_of_the_Above"],
        }
    )

    result = rendered_qgram_statistics(
        frame,
        lower_percentile=1.0,
        upper_percentile=99.5,
        source_duration_seconds=4.0,
        target_duration_seconds=1.0,
        baseline_mode="global_median",
        flip_frequency=True,
    )

    assert len(result) == 1
    assert result.loc[0, "event"] == "GW-test"
    assert result.loc[0, "forced_closed_set_class"] == "None_of_the_Above"
    assert result.loc[0, "shape"] == "(8, 3)"
    assert result.loc[0, "finite_fraction"] == 1.0
