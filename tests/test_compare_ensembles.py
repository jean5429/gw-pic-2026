from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from gwproject.commands import compare_ensembles as command


def _rows(statuses: dict[tuple[str, str], tuple[str, str, float]]) -> pd.DataFrame:
    rows = []
    for (event, detector), (status, forced, excess) in sorted(statuses.items()):
        reject = status == "reject_ood"
        rows.append({
            "event": event,
            "detector": detector,
            "qgram_path": f"outputs/qgrams/{event}/{detector}_qgram.npy",
            "forced_closed_set_class": forced,
            "closed_set_ensemble_confidence": 0.6,
            "vote_fraction": 1.0,
            "mutual_information": 0.01,
            "energy_percentile_median": 0.5,
            "mahalanobis_percentile_median": 1.0 if reject else 0.5,
            "mahalanobis_log_excess_99": excess,
            "ood_energy_95": False,
            "ood_energy_99": False,
            "ood_mahalanobis_95": reject,
            "ood_mahalanobis_99": reject,
            "open_world_status": status,
        })
    return pd.DataFrame(rows)


def _summary(render_mode: str, *, checkpoint: str = "a", reference: str = "r") -> dict:
    return {
        "checkpoints": [{"path": "m.pt", "sha256": checkpoint}],
        "ood_calibration": {
            "reference_sha256": reference,
            "minimum_ood_model_fraction": 0.6,
            "energy_quantiles": [0.95, 0.99],
            "mahalanobis_quantiles": [0.95, 0.99],
            "maximum_mutual_information": 0.1,
        },
        "preprocessing": {"render_mode": render_mode, "flip_frequency": True},
        "open_world_counts": {},
        "forced_closed_set_class_counts": {},
    }


def _write(directory: Path, frame: pd.DataFrame, summary: dict) -> Path:
    directory.mkdir(parents=True)
    frame.to_csv(directory / "per_channel.csv", index=False)
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return directory


BASE = {
    ("GW1", "H1"): ("reject_ood", "None_of_the_Above", 2.0),
    ("GW1", "L1"): ("reject_ood", "None_of_the_Above", 2.0),
    ("GW2", "H1"): ("reject_ood", "Extremely_Loud", 1.0),
    ("GW2", "L1"): ("reject_ood", "Extremely_Loud", 1.0),
    ("GW3", "H1"): ("reject_ood", "Blip", 0.5),
}
VARIANT = {
    ("GW1", "H1"): ("review", "Chirp", 0.0),
    ("GW1", "L1"): ("review", "Chirp", 0.0),
    ("GW2", "H1"): ("review", "Extremely_Loud", 0.0),
    ("GW2", "L1"): ("reject_ood", "Extremely_Loud", 1.0),
    ("GW3", "H1"): ("reject_ood", "Blip", 0.5),
}


def test_events_are_averaged_before_paired_bootstrap():
    result = command.compare(_rows(BASE), _rows(VARIANT), seed=1, replicates=200)
    contrasts = result["contrasts"].set_index("metric")
    # Delta por evento em reject_ood: GW1 -1, GW2 -0,5, GW3 0 -> media -0,5.
    assert contrasts.loc["reject_ood", "delta_estimate"] == pytest.approx(-0.5)
    assert contrasts.loc["reject_ood", "baseline_channel_sum"] == 5
    assert contrasts.loc["reject_ood", "variant_channel_sum"] == 2
    assert contrasts.loc["reject_ood", "delta_blocks"] == 3
    # Excesso continuo: GW1 -2, GW2 -0,5, GW3 0 -> media -0,8333.
    assert contrasts.loc["mahalanobis_log_excess_99", "delta_estimate"] == pytest.approx(-2.5 / 3)
    # Classe forcada trocou em 2 dos 5 canais; por evento: 1, 0, 0.
    assert contrasts.loc["forced_class_changed", "variant_channel_sum"] == 2
    assert contrasts.loc["forced_class_changed", "delta_estimate"] == pytest.approx(1 / 3)
    transitions = result["status_transitions"]
    assert transitions.loc["reject_ood", "review"] == 3
    per_event = result["per_event"].set_index("event")
    assert per_event.loc["GW2", "n_channels"] == 2


def test_pairing_requires_same_channels_models_and_reference():
    base = (_rows(BASE), _summary("percentile"))
    validation = command.validate_pairing(base, (_rows(VARIANT), _summary("gravity_spy")))
    assert validation["events"] == 3
    assert validation["preprocessing_differences"] == {
        "render_mode": {"baseline": "percentile", "variant": "gravity_spy"}
    }
    with pytest.raises(ValueError, match="canais"):
        command.validate_pairing(base, (_rows(VARIANT).iloc[:-1], _summary("gravity_spy")))
    with pytest.raises(ValueError, match="checkpoints"):
        command.validate_pairing(base, (_rows(VARIANT), _summary("gravity_spy", checkpoint="b")))
    with pytest.raises(ValueError, match="reference_sha256"):
        command.validate_pairing(base, (_rows(VARIANT), _summary("gravity_spy", reference="s")))


def test_command_writes_report_and_refuses_nonempty_output(tmp_path, monkeypatch):
    base = _write(tmp_path / "base", _rows(BASE), _summary("percentile"))
    variant = _write(tmp_path / "variant", _rows(VARIANT), _summary("gravity_spy"))
    output = tmp_path / "comparison"
    argv = ["compare-ensembles", str(base), str(variant), "--output-dir", str(output),
            "--bootstrap-replicates", "200"]
    monkeypatch.setattr(sys, "argv", argv)
    assert command.main() == 0
    for name in ("per_channel_comparison.csv", "per_event_comparison.csv", "contrasts.csv",
                 "status_transitions.csv", "forced_class_transitions.csv", "summary.json",
                 "REPORT.md", "manifest.json"):
        assert (output / name).is_file(), name
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["variant"]["render_mode"] == "gravity_spy"
    assert "render_mode" in (output / "REPORT.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        command.main()
