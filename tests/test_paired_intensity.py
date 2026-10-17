"""Prepared regression tests; these must be executed by the project owner."""
import json

import numpy as np
import pandas as pd
import pytest

from gwproject.research.paired_intensity import (
    CONDITIONS, factorial_contrasts, fast_mahalanobis, intensity_variant,
    paired_bootstrap,
)
from gwproject.research.uncertainty import MahalanobisReference, mahalanobis_score


def test_identity_is_byte_exact_and_independent():
    source = np.arange(256, dtype=np.uint8).reshape(16, 16)
    result, degenerate = intensity_variant(source, "identity")
    np.testing.assert_array_equal(result, source)
    assert not degenerate and not np.shares_memory(result, source)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_variants_preserve_shape_type_and_intensity_order(condition):
    source = np.arange(256, dtype=np.uint8).reshape(16, 16)
    result, _ = intensity_variant(source, condition)
    assert result.shape == source.shape and result.dtype == np.uint8
    assert np.all(np.diff(result.ravel().astype(int)) >= 0)


def test_gamma_has_prespecified_direction():
    result, _ = intensity_variant(np.array([[0, 64, 128, 255]], dtype=np.uint8), "gamma")
    assert result[0, 0] == 0 and result[0, 3] == 255
    assert result[0, 1] > 64 and result[0, 2] > 128


def test_constant_contrast_is_reported_not_dropped():
    source = np.full((4, 4), 63, dtype=np.uint8)
    result, degenerate = intensity_variant(source, "contrast")
    np.testing.assert_array_equal(result, source)
    assert degenerate


def test_combination_is_contrast_then_gamma_without_intermediate_rounding():
    source = np.array([[12, 70, 130, 190]], dtype=np.uint8)
    result, _ = intensity_variant(source, "contrast_gamma", lower=0, upper=100)
    expected = np.rint(255 * ((source.astype(float) - 12) / 178) ** 0.7).astype(np.uint8)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("kwargs", [{"gamma": 0}, {"gamma": np.nan}, {"lower": 100}, {"upper": -1}])
def test_invalid_parameters_fail(kwargs):
    with pytest.raises(ValueError):
        intensity_variant(np.zeros((2, 2), dtype=np.uint8), "identity", **kwargs)


def test_invalid_pixels_and_condition_fail():
    with pytest.raises(ValueError):
        intensity_variant(np.zeros((2, 2)), "identity")
    with pytest.raises(ValueError):
        intensity_variant(np.zeros((2, 2), dtype=np.uint8), "unregistered")


def test_quadratic_form_matches_canonical_scorer():
    rng = np.random.default_rng(17)
    matrix = rng.normal(size=(7, 7))
    reference = MahalanobisReference(rng.normal(size=(3, 7)), matrix.T @ matrix, np.arange(3))
    samples = rng.normal(size=(11, 7))
    np.testing.assert_allclose(fast_mahalanobis(samples, reference),
                               mahalanobis_score(samples, reference), rtol=1e-12, atol=1e-12)


def test_factorial_effects_and_interaction():
    # y = baseline + 2*gamma + 3*contrast + 4*gamma*contrast.
    values = np.array([[10, 12, 13, 19], [20, 22, 23, 29]], dtype=float)
    effects = factorial_contrasts(values)
    np.testing.assert_array_equal(effects["gamma_main_effect"], [4, 4])
    np.testing.assert_array_equal(effects["contrast_main_effect"], [5, 5])
    np.testing.assert_array_equal(effects["interaction"], [4, 4])


def test_bootstrap_is_reproducible_and_paired():
    first = paired_bootstrap(np.array([1.0, -1.0, 2.0, 0.0]), replicates=100)
    assert first == paired_bootstrap(np.array([1.0, -1.0, 2.0, 0.0]), replicates=100)
    constant = paired_bootstrap(np.full(218, 0.5), replicates=100)
    assert constant["estimate"] == constant["ci_low"] == constant["ci_high"] == 0.5
    assert constant["blocks"] == 218


def test_input_manifest_preserves_all_pairs_and_validation_paths(tmp_path, monkeypatch):
    import gwproject.gravity_spy as gs
    from gwproject.commands.paired_intensity import _input_rows
    approved = tmp_path / "approved.csv"
    metadata = tmp_path / "metadata.csv"
    channel = tmp_path / "channel.npy"
    panel = tmp_path / "panel.png"
    channel.write_bytes(b"fixture only; this test hashes but does not load the array")
    panel.write_bytes(b"fixture only; this test hashes but does not load the image")
    pd.DataFrame([dict(event="GW_A", detector="H1", qgram_path=str(channel), approved=True)]).to_csv(approved, index=False)
    pd.DataFrame([dict(ifo="H1", gravityspy_id="abcdefghij", label="A", sample_type="validation")]).to_csv(metadata, index=False)
    monkeypatch.setattr(gs, "validate_harmonized_dataset", lambda *a: {})
    monkeypatch.setattr(gs, "index_spectrograms", lambda *a: {("H1", "abcdefghij"): str(panel)})
    paths = dict(approved=approved, metadata=metadata, images_root=tmp_path)
    expected = dict(channels=1, events=1, validation_images=1)
    result = _input_rows(paths, expected)
    assert list(result.domain) == ["catalog", "gravity_spy_val"]
    assert result.iloc[1].path == str(panel.resolve())
    assert result.source_sha256.str.len().eq(64).all()
    with pytest.raises(ValueError, match="coverage"):
        _input_rows(paths, dict(channels=2, events=1, validation_images=1))
    pd.DataFrame([dict(event="GW_A", detector="H1", qgram_path=str(channel), approved=False)]).to_csv(approved, index=False)
    with pytest.raises(ValueError, match="not approved"):
        _input_rows(paths, expected)


def test_summary_uses_events_not_detector_rows(tmp_path):
    from gwproject.commands.paired_intensity import _summarize
    frame = pd.DataFrame([
        dict(sample_id="aH", domain="catalog", group="A", event="A", detector="H1", label="", split="external", path="", source_sha256=""),
        dict(sample_id="aL", domain="catalog", group="A", event="A", detector="L1", label="", split="external", path="", source_sha256=""),
        dict(sample_id="bH", domain="catalog", group="B", event="B", detector="H1", label="", split="external", path="", source_sha256=""),
        dict(sample_id="g1", domain="gravity_spy_val", group="g1", event="", detector="H1", label="X", split="val", path="", source_sha256=""),
        dict(sample_id="g2", domain="gravity_spy_val", group="g2", event="", detector="L1", label="Y", split="val", path="", source_sha256=""),
    ])
    ranks = np.ones((5, 4))
    ranks[:2, 1] = 0.5  # Only event A stops being rejected under gamma.
    result = dict(logits=np.zeros((5, 4, 2)), mahalanobis=np.ones((5, 4)),
                  energy_percentile=np.zeros((5, 4)), mahalanobis_percentile=ranks)
    cfg = dict(coverage=dict(events=2, channels=3, validation_images=2),
               experiment=dict(seed=1, gamma=0.7, contrast_lower_percentile=1.0,
                               contrast_upper_percentile=99.5, bootstrap_replicates=100))
    _summarize(frame, np.zeros((5, 4, 4), dtype=np.uint8), ["X", "Y"], [result],
               [dict(validation_mahalanobis=np.ones(2))], cfg,
               {"open_world": {"minimum_ood_model_fraction": 0.6}}, tmp_path)
    contrasts = pd.read_csv(tmp_path / "paired_contrasts.csv")
    selected = contrasts[(contrasts.domain == "catalog") &
                         (contrasts.metric == "mahalanobis_reject_99") &
                         (contrasts.contrast == "gamma_minus_identity")].iloc[0]
    assert selected.estimate == -0.5  # NOT -2/3: two detectors are not two events.
    assert selected.blocks == 2
    assert len(pd.read_csv(tmp_path / "per_event_condition.csv")) == 8
    detail = pd.read_csv(tmp_path / "per_sample_condition.csv")
    assert detail.loc[detail.domain.eq("catalog"), "correct_validation_only"].isna().all()
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "completed_exploratory_not_confirmatory"


def test_baseline_cache_matches_canonical_final_pixels_and_rejects_corruption(tmp_path):
    from PIL import Image
    from torchvision import transforms
    from gwproject.commands.paired_intensity import _cache_images
    from gwproject.provenance import sha256_file
    from gwproject.qgram_inference import qgram_to_uint8
    source = tmp_path / "qgram.npy"
    raw = np.random.default_rng(5).uniform(1, 20, size=(40, 20))
    np.save(source, raw)
    frame = pd.DataFrame([dict(domain="catalog", path=str(source), source_sha256=sha256_file(source))])
    renderer = dict(resize_height=14, resize_width=17, lower_percentile=1.0,
                    upper_percentile=99.5, source_outseg_seconds=4.0,
                    target_duration_seconds=1.0, baseline_mode="per_frequency_median", flip_frequency=True)
    actual = _cache_images(frame, renderer, tmp_path, "fixture", False)
    expected = transforms.Resize((14, 17))(Image.fromarray(qgram_to_uint8(
        raw, lower_percentile=1.0, upper_percentile=99.5, source_duration_seconds=4,
        target_duration_seconds=1, baseline_mode="per_frequency_median", flip_frequency=True)))
    np.testing.assert_array_equal(actual[0], np.asarray(expected))
    resumed = _cache_images(frame, renderer, tmp_path, "fixture", True)
    np.testing.assert_array_equal(resumed, actual)
    # Close the mapped file before deliberately corrupting a temporary test fixture on Windows.
    del resumed
    (tmp_path / "baseline_images.npy").write_bytes(b"corrupt fixture")
    with pytest.raises(ValueError, match="integrity"):
        _cache_images(frame, renderer, tmp_path, "fixture", True)


def test_cpu_inference_and_verified_model_resume(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import torch
    import gwproject.qgram_inference as qi
    from gwproject.commands.paired_intensity import _evaluate

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(16, 2, bias=False)
            torch.nn.init.constant_(self.fc.weight, 0.1)
            self.gw_temperature = 2.0

        def forward(self, images):
            return self.fc(images.flatten(1))

    loaded = []

    def load_model(path, device):
        loaded.append(path)
        return TinyModel().to(device).eval(), ["X", "Y"], {}

    monkeypatch.setattr(qi, "load_gravity_spy_checkpoint", load_model)
    reference = dict(reference=MahalanobisReference(np.zeros((2, 16)), np.eye(16), np.arange(2)),
                     validation_energy=np.array([-4.0, -2.0, 0.0]),
                     validation_mahalanobis=np.array([1.0, 8.0, 20.0]),
                     temperature=2.0, checkpoint_sha256="fixture")
    cfg = dict(experiment=dict(seed=5, gamma=0.7, contrast_lower_percentile=1.0,
                               contrast_upper_percentile=99.5), runtime=dict(mixed_precision=False))
    pixels = np.arange(48, dtype=np.uint8).reshape(3, 4, 4)
    args = SimpleNamespace(resume=False, batch_size=3)
    checkpoint = tmp_path / "fixture.pt"
    first = _evaluate(pixels, ["X", "Y"], [checkpoint], [reference], cfg, args,
                      tmp_path, "contract-fixture", torch.device("cpu"))
    assert first[0]["logits"].shape == (3, 4, 2)
    assert all(np.isfinite(array).all() for array in first[0].values())
    schedule = pd.read_csv(tmp_path / "inference_schedule.csv")
    assert len(schedule.drop_duplicates()) == 12
    args.resume = True
    second = _evaluate(pixels, ["X", "Y"], [checkpoint], [reference], cfg, args,
                       tmp_path, "contract-fixture", torch.device("cpu"))
    assert len(loaded) == 1
    np.testing.assert_array_equal(first[0]["logits"], second[0]["logits"])
    (tmp_path / "model_00.npz").write_bytes(b"corrupt fixture")
    with pytest.raises(ValueError, match="invalid resume"):
        _evaluate(pixels, ["X", "Y"], [checkpoint], [reference], cfg, args,
                  tmp_path, "contract-fixture", torch.device("cpu"))
