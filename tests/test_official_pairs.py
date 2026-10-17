import numpy as np
import pandas as pd
import pytest

from gwproject.commands.audit_official_pairs import index_hdf5, pair_metadata, pixel_metrics


def test_pairing_requires_globally_unique_id_and_matching_split():
    metadata = pd.DataFrame([dict(gravityspy_id="abc", ifo="H1", label="A", sample_type="validation")])
    index = pd.DataFrame([dict(gravityspy_id="abc", label="A", split="val", hdf5_path="/A/validation/abc/1.0.png", readable_image=True)])
    assert pair_metadata(metadata, index, metadata).iloc[0].status == "matched"
    ambiguous = pd.concat([metadata, metadata.assign(ifo="L1")], ignore_index=True)
    assert pair_metadata(metadata, index, ambiguous).iloc[0].status == "missing_or_ambiguous_official_id"
    assert pair_metadata(metadata, index.assign(split="test"), metadata).iloc[0].status == "hdf5_label_or_split_disagreement"


def test_fixed_pixel_encodings_do_not_fit_each_image():
    local = np.arange(140 * 170, dtype=np.int64).reshape(140, 170).astype(np.uint8)
    result = pixel_metrics((local.astype(float) / 255)[None], local)
    assert result["mae_unit"] == 0
    assert result["mae_u8"] > 1 and result["mae_signed"] > 0
    with pytest.raises(ValueError):
        pixel_metrics(np.full((1, 140, 170), np.nan), local)


def test_complete_index_does_not_read_external_links(tmp_path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "fixture.h5"
    with h5py.File(source, "w") as handle:
        for label in ("A", "B"):
            handle.create_dataset(f"{label}/validation/{label}id/1.0.png", shape=(1, 140, 170), dtype="float32")
        handle["external"] = h5py.ExternalLink("missing.h5", "/unknown")
    with h5py.File(source, "r") as handle:
        indexed, counts, issues = index_hdf5(handle)
    assert len(indexed) == 2 and indexed.readable_image.all()
    assert counts["datasets"] == 2
    assert issues == [dict(path="/external", reason="non_hard_link_not_followed")]
