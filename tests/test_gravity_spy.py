import io
import tarfile

import pandas as pd
import pytest

from gwproject.config import load_toml
from gwproject.downloads import safe_extract_tar
from gwproject.gravity_spy import (
    adapt_conv1_to_grayscale,
    build_resnet18_grayscale,
    choose_profile,
    freeze_backbone,
    attach_image_paths,
    index_spectrograms,
    official_splits,
)


def test_official_splits_do_not_mix_ids():
    frame = pd.DataFrame({
        "gravityspy_id": ["a", "b", "c"],
        "ifo": ["H1", "L1", "H1"],
        "label": ["Blip", "Blip", "Koi_Fish"],
        "sample_type": ["train", "validation", "test"],
    })
    splits = official_splits(frame)
    assert set(splits) == {"train", "val", "test"}


def test_image_index_uses_ifo_and_id_as_composite_key(tmp_path):
    first = tmp_path / "H1_nLQFR3I7EJ_spectrogram_1.0.png"
    second = tmp_path / "L1_nLQFR3I7EJ_spectrogram_1.0.png"
    first.write_bytes(b"h1")
    second.write_bytes(b"l1")
    index = index_spectrograms(tmp_path)
    assert len(index) == 2
    frame = pd.DataFrame({
        "ifo": ["H1", "L1"],
        "gravityspy_id": ["nLQFR3I7EJ", "nLQFR3I7EJ"],
    })
    attached = attach_image_paths(frame, index)
    assert attached["path"].tolist() == [str(first), str(second)]


def test_gravity_spy_config_selects_cuda_profile():
    cfg = load_toml("configs/gravity_spy.toml")
    profile = choose_profile(cfg, "cuda", requested="auto")
    assert profile.name == "cuda_balanced"
    assert profile.amp is True
    assert profile.batch_size <= 48


def test_conv1_adaptation_preserves_replicated_rgb_response():
    torch = pytest.importorskip("torch")
    weight = torch.randn(64, 3, 7, 7)
    gray = adapt_conv1_to_grayscale(weight)
    assert gray.shape == (64, 1, 7, 7)
    assert torch.allclose(gray, weight.sum(dim=1, keepdim=True))


def test_frozen_stage_trains_conv1_bn1_and_head():
    pytest.importorskip("torch")
    model = build_resnet18_grayscale(22, pretrained=False, conv1_stride=2, use_maxpool=False)
    freeze_backbone(model, True)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert any(name.startswith("conv1.") for name in trainable)
    assert any(name.startswith("bn1.") for name in trainable)
    assert any(name.startswith("fc.") for name in trainable)
    assert not any(name.startswith("layer4.") for name in trainable)


def test_safe_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../escape.txt")
        payload = b"bad"
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="insegura"):
        safe_extract_tar(archive, tmp_path / "out")
