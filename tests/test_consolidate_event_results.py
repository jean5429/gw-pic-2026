from __future__ import annotations

import pandas as pd
import pytest

from gwproject.commands.consolidate_event_results import consolidate_event_frames


def test_consolidate_event_frames_is_one_to_one_and_prefixed():
    frames = {
        "qc": pd.DataFrame({"event": ["B", "A"], "status": ["ok", "ok"]}),
        "cnn": pd.DataFrame({"event": ["A", "B"], "status": ["ood", "ood"]}),
    }

    result = consolidate_event_frames(frames)

    assert result["event"].tolist() == ["A", "B"]
    assert {"qc_status", "cnn_status"} <= set(result.columns)


def test_consolidate_event_frames_rejects_scope_mismatch():
    frames = {
        "qc": pd.DataFrame({"event": ["A", "B"]}),
        "cnn": pd.DataFrame({"event": ["A", "C"]}),
    }

    with pytest.raises(ValueError, match="escopo de eventos divergente"):
        consolidate_event_frames(frames)
