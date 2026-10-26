from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from gwproject.commands import build_closure_set, closure_report
from gwproject.research.closure import (
    annotate_files,
    classification_metrics,
    evaluate_criteria,
    grouped_bootstrap,
    select_files,
)


def _metadata(rows: list[tuple[float, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"event_time": gps, "ifo": ifo, "label": label, "sample_type": split,
             "gravityspy_id": f"id{index}"}
            for index, (gps, ifo, label, split) in enumerate(rows)
        ]
    )


def _annotate(frame: pd.DataFrame, file_seconds: int = 100) -> pd.DataFrame:
    return annotate_files(
        frame, run_start_gps=0, run_end_gps=10_000, file_seconds=file_seconds,
        before_seconds=16, after_seconds=16,
    )


def test_annotate_marks_windows_crossing_file_edges():
    frame = _annotate(_metadata([(150, "H1", "Blip", "test"), (205, "H1", "Blip", "test")]))
    assert frame["file_start"].tolist() == [100, 200]
    assert frame["window_inside_file"].tolist() == [True, False]


def test_annotate_requires_window_inside_public_segments():
    frame = annotate_files(
        _metadata([(150, "H1", "Blip", "test"), (170, "H1", "Blip", "test"), (150, "L1", "Blip", "test")]),
        run_start_gps=0, run_end_gps=10_000, file_seconds=100, before_seconds=16, after_seconds=16,
        public_segments={"H1": [(120.0, 180.0)], "L1": []},
    )
    assert frame["window_public"].tolist() == [True, False, False]
    assert frame["usable"].tolist() == [True, False, False]


def test_select_files_prefers_classes_still_below_target():
    rows = [(150 + i, "H1", "Blip", "test") for i in range(5)]  # arquivo cheio de Blip
    rows += [(250, "L1", "Chirp", "test")]  # arquivo com a classe rara
    rows += [(350, "H1", "Blip", "train")]  # treino nao conta na selecao
    plan = select_files(
        _annotate(_metadata(rows)), evaluation_splits=["test"], max_files=5, per_class_target=1,
    )
    assert set(zip(plan["ifo"], plan["file_start"])) == {("H1", 100), ("L1", 200)}
    assert plan["new_samples"].tolist() == [1, 1]


def test_select_files_rewards_first_sample_of_a_rare_class():
    # Arquivo 1: 3 Blips novos. Arquivo 2: 1 Tomte (classe ainda sem amostra).
    rows = [(120 + i, "H1", "Blip", "test") for i in range(30)]  # cobre Blip antes
    rows += [(250 + i, "H1", "Blip", "test") for i in range(3)]
    rows += [(350, "L1", "Tomte", "test")]
    plan = select_files(
        _annotate(_metadata(rows)), evaluation_splits=["test"], max_files=2, per_class_target=40,
    )
    assert plan["file_start"].tolist() == [100, 300]


def test_metrics_restrict_macro_f1_to_classes_with_enough_samples():
    labels = np.array(["A"] * 10 + ["B"] * 10 + ["C"])
    predictions = np.array(["A"] * 10 + ["B"] * 5 + ["A"] * 5 + ["A"])
    metrics = classification_metrics(labels, predictions, min_class_samples=10)
    assert metrics["decisive_classes"] == ["A", "B"]
    assert metrics["accuracy"] == 15 / 21
    criteria = evaluate_criteria(
        metrics, 0.01, {"macro_f1_min": 0.85, "accuracy_min": 0.5, "reject_ood_max": 0.05}
    )
    assert not criteria["C1_macro_f1"]["passed"]
    assert criteria["C1_accuracy"]["passed"] and criteria["C2_reject_ood"]["passed"]
    assert not criteria["all_passed"]


def test_grouped_bootstrap_resamples_whole_groups():
    frame = pd.DataFrame({"g": [1, 1, 2, 2], "x": [1.0, 1.0, 0.0, 0.0]})
    low, high = grouped_bootstrap(
        frame, lambda sample: float(sample["x"].mean()), group_column="g", replicates=200, seed=1,
    )
    assert 0.0 <= low <= 0.5 <= high <= 1.0


def _write_fake_gwosc(path: Path, start: int, seconds: int, rate: int = 4096) -> None:
    rng = np.random.default_rng(7)
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("strain/Strain", data=rng.normal(size=seconds * rate) * 1e-21)
        dataset.attrs["Xstart"] = start
        dataset.attrs["Xspacing"] = 1.0 / rate
        dataset.attrs["Xunits"] = "second"
        dataset.attrs["Yunits"] = ""
        handle["meta/Detector"] = b"H1"  # arquivos O1 nao tem GWOSCmeta


def test_closure_set_and_report_end_to_end(tmp_path, monkeypatch):
    start = 1126400000
    metadata = _metadata([
        (start + 30.0, "H1", "Blip", "test"),
        (start + 40.0, "H1", "Koi_Fish", "validation"),
        (start + 2.0, "H1", "Blip", "test"),  # janela sai do arquivo: excluido
    ])
    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    config = tmp_path / "plano1.toml"
    repo_config = Path("configs/plano1.toml").read_text(encoding="utf-8")
    config.write_text(
        repo_config.replace("file_seconds = 4096", "file_seconds = 64")
        .replace("run_start_gps = 1126051217", f"run_start_gps = {start}")
        .replace("min_class_samples = 10", "min_class_samples = 1")
        .replace("replicates = 2000", "replicates = 20"),
        encoding="utf-8",
    )

    def fake_locate(ifo, file_start, seconds, rate):
        return f"https://example.invalid/{ifo}-{file_start}-{seconds}.hdf5"

    def fake_download(url, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_fake_gwosc(destination, start, 64)
        return destination

    monkeypatch.setattr(build_closure_set, "public_segments", lambda *args: None)
    monkeypatch.setattr(build_closure_set, "locate_url", fake_locate)
    monkeypatch.setattr(build_closure_set, "download", fake_download)
    output = tmp_path / "fechamento"
    monkeypatch.setattr(sys, "argv", [
        "closure-set", "--metadata", str(metadata_path), "--config", str(config),
        "--output-root", str(output),
    ])
    assert build_closure_set.main() == 0

    channels = pd.read_csv(output / "closure_channels.csv")
    assert sorted(channels["event"]) == ["id0", "id1"]
    assert (channels["status"] == "ok").all()
    qgram = np.load(channels.loc[0, "qgram_path"])
    assert qgram.ndim == 2 and np.isfinite(qgram).all()
    assert not list((output / "_download").glob("*.hdf5"))  # arquivo baixado apagado

    # classify-ensemble simulado: acerta o Blip, erra o Koi_Fish.
    ensemble = tmp_path / "ensemble"
    ensemble.mkdir()
    pd.DataFrame({
        "event": ["id0", "id1"], "detector": ["H1", "H1"],
        "forced_closed_set_class": ["Blip", "Blip"],
        "open_world_status": ["in_domain_candidate", "reject_ood"],
        "ood_mahalanobis_99": [False, True], "ood_energy_99": [False, False],
    }).to_csv(ensemble / "per_channel.csv", index=False)
    (ensemble / "summary.json").write_text(
        json.dumps({"render_mode": "gravity_spy", "n_models": 5, "checkpoints": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", [
        "closure-report", str(ensemble), "--closure", str(output / "closure_channels.csv"),
        "--config", str(config),
    ])
    assert closure_report.main() == 0
    report = json.loads((ensemble / "fechamento" / "report.json").read_text(encoding="utf-8"))
    assert report["evaluation"]["accuracy"] == 0.5
    assert report["evaluation"]["open_world_fractions"]["reject_ood"] == 0.5
    assert report["criteria"]["all_passed"] is False
    assert (ensemble / "fechamento" / "confusion_matrix.png").exists()


def test_closure_set_records_files_without_public_data(tmp_path, monkeypatch):
    start = 1126400000
    metadata_path = tmp_path / "metadata.csv"
    _metadata([(start + 30.0, "H1", "Blip", "test")]).to_csv(metadata_path, index=False)
    config = tmp_path / "plano1.toml"
    config.write_text(
        Path("configs/plano1.toml").read_text(encoding="utf-8")
        .replace("file_seconds = 4096", "file_seconds = 64")
        .replace("run_start_gps = 1126051217", f"run_start_gps = {start}"),
        encoding="utf-8",
    )

    def no_data(ifo, file_start, seconds, rate):
        raise build_closure_set.NoPublicData("Cannot find a GWOSC dataset")

    monkeypatch.setattr(build_closure_set, "public_segments", lambda *args: None)
    monkeypatch.setattr(build_closure_set, "locate_url", no_data)
    output = tmp_path / "fechamento"
    monkeypatch.setattr(sys, "argv", [
        "closure-set", "--metadata", str(metadata_path), "--config", str(config),
        "--output-root", str(output),
    ])
    assert build_closure_set.main() == 0  # permanente: nao deixa o lote incompleto
    channels = pd.read_csv(output / "closure_channels.csv")
    assert channels["status"].tolist() == ["no_public_data"]


def test_mark_render_invalid_excludes_qgrams_any_renderer_rejects(tmp_path):
    from gwproject.config import load_toml

    good = np.ones((40, 8), dtype=np.float32) + np.random.default_rng(1).random((40, 8))
    bad = good.copy()
    bad[:, 3] = -5.0  # mediana negativa num bin de frequencia
    paths = []
    for name, array in (("good", good), ("bad", bad)):
        path = tmp_path / f"{name}.npy"
        np.save(path, array)
        paths.append(str(path))
    channels = pd.DataFrame({"event": ["g", "b"], "status": ["ok", "ok"], "reason": [None, None],
                             "qgram_path": paths})
    cfg = load_toml(Path("configs/research_program.toml"))["harmonized_inference"]
    marked = build_closure_set.mark_render_invalid(channels, cfg)
    assert marked["status"].tolist() == ["ok", "render_invalid"]
    assert pd.isna(marked.loc[1, "qgram_path"]) and "percentile" in marked.loc[1, "reason"]
