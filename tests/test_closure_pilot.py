from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gwproject.commands.closure_pilot import choose_variant
from gwproject.config import load_toml

VARIANTS = [
    {"name": "barata", "sample_rate": 4096, "seconds": 32, "recipe": "catalog"},
    {"name": "media", "sample_rate": 4096, "seconds": 64, "recipe": "gravity_spy"},
    {"name": "cara", "sample_rate": 16384, "seconds": 64, "recipe": "gravity_spy"},
]


def _scores(values: dict[str, list[float]]) -> pd.DataFrame:
    rows = [
        {"variant": name, "label": f"c{index}", "median_corr": value}
        for name, column in values.items()
        for index, value in enumerate(column)
    ]
    return pd.DataFrame(rows)


def test_choose_variant_prefers_cheaper_when_gain_is_small():
    scores = _scores({"barata": [0.60, 0.60], "media": [0.70, 0.70], "cara": [0.71, 0.71]})
    choice = choose_variant(scores, VARIANTS, min_gain=0.02)
    assert choice["name"] == "media"  # "cara" ganha so 0,01


def test_choose_variant_pays_for_a_real_gain():
    scores = _scores({"barata": [0.60, 0.60], "media": [0.70, 0.70], "cara": [0.80, 0.80]})
    assert choose_variant(scores, VARIANTS, min_gain=0.02)["name"] == "cara"


def test_gravity_spy_recipe_keeps_the_array_contract(tmp_path):
    gwpy = pytest.importorskip("gwpy.timeseries")
    from gwproject.gwosc_pipeline import gwpy_qtransform_consistent

    rate, gps0 = 4096, 1126400000.0
    data = np.random.default_rng(3).normal(size=32 * rate) * 1e-21
    raw = tmp_path / "raw.hdf5"
    gwpy.TimeSeries(data, t0=gps0, sample_rate=rate, name="H1:Strain").write(raw, format="hdf5")
    processing = load_toml("configs/processing.toml")
    result = gwpy_qtransform_consistent(
        raw, gps0 + 16, tmp_path / "q.npy", frange=(20.0, 512.0), config=processing,
        recipe="gravity_spy",
    )
    assert result["status"] == "ok" and result["recipe"] == "gravity_spy"
    array = np.load(tmp_path / "q.npy")
    assert array.shape[1] == 500 and array.shape[0] > array.shape[1]  # (tempo, frequencia)
    assert np.isfinite(array).all()
    meta = json.loads((tmp_path / "q.json").read_text(encoding="utf-8"))
    assert 4 <= meta["selected_q"] <= 64


def test_catalog_recipe_metadata_is_unchanged(tmp_path):
    gwpy = pytest.importorskip("gwpy.timeseries")
    from gwproject.gwosc_pipeline import gwpy_qtransform_consistent

    rate, gps0 = 4096, 1126400000.0
    data = np.random.default_rng(4).normal(size=32 * rate) * 1e-21
    raw = tmp_path / "raw.hdf5"
    gwpy.TimeSeries(data, t0=gps0, sample_rate=rate, name="H1:Strain").write(raw, format="hdf5")
    result = gwpy_qtransform_consistent(
        raw, gps0 + 16, tmp_path / "q.npy", frange=(20.0, 512.0),
        config=load_toml("configs/processing.toml"),
    )
    assert result["status"] == "ok"
    assert "recipe" not in result and "selected_q" not in result
