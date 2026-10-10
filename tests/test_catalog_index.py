import pandas as pd

from gwproject.catalog_index import build_channels_quality, build_event_quality


def test_catalog_index_distinguishes_unavailable_and_rejected():
    catalog = pd.DataFrame([
        {"name": "GWTEST-v1", "gps": 100.0, "chirp_mass": 20.0},
    ])
    log = [
        {
            "event": "GWTEST-v1",
            "detector": "H1",
            "status": "ok",
            "raw_path": "H1.hdf5",
            "processing": {"status": "ok", "processed_path": "H1.npy"},
            "qtransform": {"status": "ok", "path": "H1_q.npy"},
        },
        {
            "event": "GWTEST-v1",
            "detector": "L1",
            "status": "ok",
            "processing": {"status": "rejected", "reason": "data contem NaN ou infinito"},
        },
        {
            "event": "GWTEST-v1",
            "detector": "V1",
            "status": "error: failed to get data",
        },
    ]
    channels = build_channels_quality(catalog, log)
    counts = channels["scientific_status"].value_counts().to_dict()
    assert counts == {"approved": 1, "rejected_nonfinite": 1, "data_unavailable": 1}
    events = build_event_quality(catalog, channels)
    assert events.iloc[0]["event_status"] == "single_detector"
