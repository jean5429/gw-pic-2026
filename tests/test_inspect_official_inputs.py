import numpy as np
import pytest

from gwproject.commands.inspect_official_inputs import hdf5_schema


def test_schema_does_not_follow_external_links(tmp_path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "source.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("images", shape=(100, 140, 170), dtype="uint8")
        handle["external"] = h5py.ExternalLink("missing.h5", "/private")
        handle["soft"] = h5py.SoftLink("/images")
        handle["alias"] = handle["images"]
    schema = hdf5_schema(source)
    assert schema["dataset_values_read"] is False
    assert schema["truncated"] is False
    assert any(item["kind"] == "ExternalLink" and not item["followed"] for item in schema["objects"])
    assert any(item["kind"] == "SoftLink" and not item["followed"] for item in schema["objects"])
    datasets = [item for item in schema["objects"] if item["kind"] == "dataset"]
    assert len(datasets) == 1 and datasets[0]["shape"] == (100, 140, 170)


def test_schema_limits_inventory_size(tmp_path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "many.h5"
    with h5py.File(source, "w") as handle:
        for index in range(4):
            handle.create_dataset(str(index), data=np.array([index]))
    schema = hdf5_schema(source, maximum_objects=2)
    assert len(schema["objects"]) == 2 and schema["truncated"]


def test_schema_refuses_excluded_branches(tmp_path):
    with pytest.raises(ValueError, match="excluded"):
        hdf5_schema(tmp_path / "ridge_atenuacao" / "ignored.h5")
