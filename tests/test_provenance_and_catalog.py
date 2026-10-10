from gwproject.gwosc_pipeline import _select_event_payload, event_base_name, event_version
from gwproject.provenance import runtime_manifest


def test_event_version_selection_is_explicit():
    payload = {"events": {
        "GW150914-v2": {"GPS": 1},
        "GW150914-v4": {"GPS": 2},
    }}
    assert event_base_name("GW150914-v4") == "GW150914"
    assert event_version("GW150914-v4") == 4
    assert _select_event_payload("GW150914-v4", payload)["GPS"] == 2


def test_manifest_records_package_versions():
    manifest = runtime_manifest()
    assert "packages" in manifest
    assert "numpy" in manifest["packages"]
    assert "scipy" in manifest["packages"]
    assert "accelerator" in manifest


def test_anonymize_paths_replaces_home_in_nested_values():
    from gwproject.provenance import anonymize_paths

    data = {
        "python_executable": r"c:\Users\Alice\miniforge3\python.exe",
        "nested": [r"Permission denied: 'C:\Users\alice\.astropy\cache'", "C:/Users/Alice/x", 3],
    }
    result = anonymize_paths(data, home=r"C:\Users\Alice")
    assert result["python_executable"] == r"~\miniforge3\python.exe"
    assert result["nested"] == [r"Permission denied: '~\.astropy\cache'", "~/x", 3]
