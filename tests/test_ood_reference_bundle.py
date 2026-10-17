from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gwproject.commands.classify_catalog_qgrams_ensemble import _load_ood_bundle
from gwproject.provenance import sha256_file


def _write_bundle(
    path: Path,
    checkpoint_paths: list[Path],
    *,
    reference_fit_split: str = "train",
    calibration_split: str = "val",
) -> None:
    hashes = [sha256_file(item) for item in checkpoint_paths]
    np.savez_compressed(
        path,
        format_version=np.asarray("4"),
        reference_fit_split=np.asarray(reference_fit_split),
        calibration_split=np.asarray(calibration_split),
        checkpoint_paths=np.asarray([str(item) for item in checkpoint_paths], dtype=str),
        checkpoint_sha256=np.asarray(hashes, dtype=str),
        classes=np.asarray(["a", "b"], dtype=str),
        class_labels=np.asarray([0, 1], dtype=np.int64),
        class_means=np.zeros((2, 2, 3), dtype=float),
        precision=np.stack([np.eye(3), 2.0 * np.eye(3)], axis=0),
        validation_energy=np.asarray([[0.0, 1.0], [10.0, 11.0]], dtype=float),
        validation_mahalanobis=np.asarray([[2.0, 3.0], [20.0, 30.0]], dtype=float),
        temperatures=np.asarray([1.2, 0.9], dtype=float),
    )


def test_ood_bundle_matches_references_by_checkpoint_hash(tmp_path: Path):
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    first.write_bytes(b"first-model")
    second.write_bytes(b"second-model")
    bundle = tmp_path / "reference.npz"
    _write_bundle(bundle, [first, second])

    references = _load_ood_bundle(bundle, [second, first])

    assert references[0]["checkpoint_sha256"] == sha256_file(second)
    assert np.allclose(references[0]["validation_energy"], [10.0, 11.0])
    assert references[0]["temperature"] == 0.9
    assert references[1]["checkpoint_sha256"] == sha256_file(first)
    assert np.allclose(references[1]["validation_energy"], [0.0, 1.0])


def test_ood_bundle_rejects_checkpoint_without_reference(tmp_path: Path):
    first = tmp_path / "first.pt"
    unknown = tmp_path / "unknown.pt"
    first.write_bytes(b"first-model")
    unknown.write_bytes(b"unknown-model")
    bundle = tmp_path / "reference.npz"

    second = tmp_path / "second.pt"
    second.write_bytes(b"second-model")
    _write_bundle(bundle, [first, second])

    with pytest.raises(ValueError, match="sem referencia OOD"):
        _load_ood_bundle(bundle, [unknown])


def test_ood_bundle_rejects_leaky_split_contract(tmp_path: Path):
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    first.write_bytes(b"first-model")
    second.write_bytes(b"second-model")
    bundle = tmp_path / "reference.npz"
    _write_bundle(
        bundle,
        [first, second],
        reference_fit_split="val",
        calibration_split="val",
    )

    with pytest.raises(ValueError, match="separacao treino-ajuste"):
        _load_ood_bundle(bundle, [first, second])
