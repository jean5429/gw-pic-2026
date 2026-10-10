from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from gwproject.commands import classify_catalog_qgrams_ensemble as command

ROOT = Path(__file__).resolve().parents[1]


def test_mahalanobis_log_excess_is_zero_at_validation_quantile():
    validation = np.arange(1.0, 101.0)
    threshold = np.quantile(validation, 0.99)
    excess = command.mahalanobis_log_excess(np.array([threshold, 0.0, 1000.0]), validation, 0.99)
    assert excess[0] == pytest.approx(0.0)
    assert excess[1] < 0 < excess[2]
    assert excess[2] == pytest.approx(np.log1p(1000.0) - np.log1p(threshold))


def test_render_mode_override_cannot_overwrite_canonical_output(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best_model.pt"
    checkpoint.write_bytes(b"not loaded before the guard")
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "argv", [
        "classify-ensemble",
        str(checkpoint),
        "--ood-reference", str(tmp_path / "ood_reference.npz"),
        "--render-mode", "gravity_spy",
    ])
    with pytest.raises(ValueError, match="saida canonica"):
        command.main()
