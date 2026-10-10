import numpy as np
import pytest

from gwproject.config import load_toml
from gwproject.preprocessing import (
    estimate_psd,
    preprocess,
    q_transform_self_test,
    whiten,
)


def test_whitening_unit_variance_in_core():
    rng = np.random.default_rng(1234)
    fs = 1024.0
    x = rng.normal(size=int(32 * fs))
    psd = estimate_psd(x, fs, segment_seconds=4.0)
    white = whiten(x, fs, psd, fmin=10.0, fmax=500.0)
    core = white[int(2 * fs):-int(2 * fs)]
    assert 0.92 < np.std(core) < 1.08


def test_partial_data_gap_is_rejected_before_psd():
    rng = np.random.default_rng(3)
    fs = 1024.0
    x = rng.normal(size=int(32 * fs))
    x[: int(10 * fs)] = 0.0
    with pytest.raises(ValueError, match="zeros|data gap"):
        estimate_psd(x, fs)


def test_preprocess_enforces_whitening_gate():
    rng = np.random.default_rng(8)
    fs = 1024.0
    x = rng.normal(size=int(32 * fs))
    filtered, _, diagnostics = preprocess(
        x,
        fs,
        whitening_fmax=500.0,
        bandpass_fmin=20.0,
        bandpass_fmax=400.0,
        white_std_min=0.90,
        white_std_max=1.10,
        return_diagnostics=True,
    )
    assert len(filtered) == len(x)
    assert 0.90 <= diagnostics.white_std_core <= 1.10


def test_q_self_test_uses_production_configuration():
    cfg = load_toml("configs/processing.toml")
    qcfg = cfg["qtransform"]
    ratio = q_transform_self_test(
        fs=cfg["signal"]["sample_rate"],
        duration_seconds=qcfg["self_test_seconds"],
        fmin=qcfg["fmin_hz"],
        fmax=qcfg["fmax_hz"],
        n_bins=qcfg["n_bins"],
        q=qcfg["display_q"],
        seed=qcfg["self_test_seed"],
    )
    assert ratio <= cfg["gates"]["q_flatness_max_ratio"]
