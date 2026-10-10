from __future__ import annotations

import numpy as np
import pandas as pd

from gwproject.research.domain_shift import grouped_domain_audit


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
