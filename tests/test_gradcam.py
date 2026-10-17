from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from gwproject.gravity_spy import build_resnet18_grayscale
from gwproject.research.gradcam import grad_cam, upsample_heatmap

ROOT = Path(__file__).resolve().parents[1]


def _model(n_classes: int = 4):
    torch.manual_seed(0)
    model = build_resnet18_grayscale(n_classes, pretrained=False, conv1_stride=2, use_maxpool=False)
    model.eval()
    return model


def test_grad_cam_has_layer4_resolution_and_unit_range():
    model = _model()
    image = torch.randn(1, 1, 140, 170)
    result = grad_cam(model, image)
    with torch.no_grad():
        logits = model(image)[0].numpy()
    # 140 x 170 -> conv1 (stride 2) 70 x 85 -> layer2 35 x 43 -> layer3 18 x 22 -> layer4 9 x 11.
    assert result.heatmap.shape == (9, 11)
    assert np.all(np.isfinite(result.heatmap))
    assert result.heatmap.min() >= 0.0
    assert result.heatmap.max() == pytest.approx(1.0) or result.heatmap.max() == 0.0
    assert result.class_index == int(np.argmax(logits))
    np.testing.assert_allclose(result.logits, logits, rtol=1e-5, atol=1e-5)
    assert len(model.layer4[-1]._forward_hooks) == 0


def test_grad_cam_respects_target_class_and_rejects_unknown():
    model = _model()
    image = torch.randn(1, 1, 140, 170)
    assert grad_cam(model, image, class_index=2).class_index == 2
    with pytest.raises(ValueError, match="classe"):
        grad_cam(model, image, class_index=9)
    with pytest.raises(ValueError, match="shape"):
        grad_cam(model, torch.randn(2, 1, 140, 170))


def test_grad_cam_without_contribution_returns_zeros_not_nan():
    model = _model()
    with torch.no_grad():
        model.fc.weight.zero_()
    result = grad_cam(model, torch.randn(1, 1, 140, 170))
    assert np.array_equal(result.heatmap, np.zeros_like(result.heatmap))


def test_grad_cam_works_with_frozen_parameters():
    model = _model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    result = grad_cam(model, torch.randn(1, 1, 140, 170))
    assert np.all(np.isfinite(result.heatmap))


def test_upsample_heatmap_shape_and_range():
    heatmap = np.zeros((9, 11))
    heatmap[4, 5] = 1.0
    upsampled = upsample_heatmap(heatmap, (140, 170))
    assert upsampled.shape == (140, 170)
    assert 0.0 <= upsampled.min() <= upsampled.max() <= 1.0


def test_gradcam_command_on_qgram_writes_figure_and_provenance(tmp_path, monkeypatch):
    from gwproject.commands import gradcam as command

    classes = [f"Classe_{index}" for index in range(4)]
    checkpoint = tmp_path / "best_model.pt"
    torch.save(
        {
            "classes": classes,
            "state_dict": _model(len(classes)).state_dict(),
            "profile": {"conv1_stride": 2, "use_maxpool": False},
            "temperature": 1.5,
        },
        checkpoint,
    )
    qgram = np.random.default_rng(0).gamma(2.0, 1.0, size=(1000, 500))
    qgram_path = tmp_path / "H1_qgram.npy"
    np.save(qgram_path, qgram)
    output = tmp_path / "gradcam"
    monkeypatch.setattr(sys, "argv", [
        "gradcam",
        "--checkpoint", str(checkpoint),
        "--qgram", str(qgram_path),
        "--config", str(ROOT / "configs" / "research_program.toml"),
        "--render-mode", "gravity_spy",
        "--target-class", "Classe_1",
        "--output-dir", str(output),
        "--device", "cpu",
    ])
    assert command.main() == 0
    figures = list(output.glob("gradcam_H1_qgram_gravity_spy__Classe_1.png"))
    assert len(figures) == 1
    record = json.loads(figures[0].with_suffix(".json").read_text(encoding="utf-8"))
    assert record["explained_class"] == "Classe_1"
    assert record["input"]["render_settings"]["render_mode"] == "gravity_spy"
    assert record["temperature"] == pytest.approx(1.5)
    assert sum(record["top_probabilities"].values()) <= 1.0 + 1e-9
    assert "interpretation_warning" in record
