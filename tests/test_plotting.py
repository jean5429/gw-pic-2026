from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from PIL import Image

from gwproject.plotting import FIGURE_DPI, plot_confusion_matrix


def test_confusion_matrix_png_has_project_dpi_and_tolerates_empty_rows(tmp_path):
    cm = np.array([[3, 1, 0], [0, 0, 0], [1, 0, 4]])
    path = tmp_path / "sub" / "confusion_matrix.png"
    plot_confusion_matrix(cm, ["A", "B", "C"], path)
    with Image.open(path) as image:
        assert image.info["dpi"][0] == pytest.approx(FIGURE_DPI, abs=0.5)


def test_confusion_matrix_rejects_inconsistent_classes(tmp_path):
    with pytest.raises(ValueError, match="classes"):
        plot_confusion_matrix(np.eye(3), ["A", "B"], tmp_path / "x.png")
    with pytest.raises(ValueError, match="quadrada"):
        plot_confusion_matrix(np.ones((2, 3)), ["A", "B"], tmp_path / "x.png")


def test_replot_command_writes_to_separate_output(tmp_path, monkeypatch):
    from gwproject.commands import replot_confusion_matrices as command

    run = tmp_path / "models" / "seed_42"
    run.mkdir(parents=True)
    np.save(run / "confusion_matrix.npy", np.array([[5, 1], [2, 7]]))
    (run / "summary.json").write_text(json.dumps({"classes": ["Blip", "Chirp"]}), encoding="utf-8")
    output = tmp_path / "figures"
    monkeypatch.setattr(sys, "argv", ["replot", str(tmp_path / "models"), "--output-dir", str(output)])
    assert command.main() == 0
    assert (output / "models" / "seed_42" / "confusion_matrix.png").is_file()
    assert not (run / "confusion_matrix.png").exists()
