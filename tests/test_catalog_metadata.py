import pandas as pd

from gwproject.catalog_metadata import (
    add_detector_frame_columns,
    detector_frame_availability,
    flatten_parameter_estimation,
    refresh_catalog_dataframe,
    select_parameter_estimation,
)


def _preferred_payload():
    return {
        "results": [
            {
                "name": "search",
                "pipeline_type": "search",
                "is_preferred": False,
                "parameters": [{"name": "network_matched_filter_snr", "best": 9.0}],
            },
            {
                "name": "preferred-pe",
                "pipeline_type": "pe",
                "is_preferred": True,
                "waveform_family": "C01:Mixed",
                "parameters": [
                    {"name": "chirp_mass_source", "best": 27.9},
                    {"name": "mass_1_source", "best": 34.6},
                    {"name": "mass_2_source", "best": 30.0},
                    {"name": "redshift", "best": 0.10},
                ],
            },
        ]
    }


def test_selects_preferred_pe_and_flattens_parameters():
    result = select_parameter_estimation(_preferred_payload())
    values = flatten_parameter_estimation(result)
    assert result["name"] == "preferred-pe"
    assert values["redshift"] == 0.10
    assert values["mass_1_source"] == 34.6


def test_detector_frame_columns_are_derived_with_provenance():
    row = add_detector_frame_columns(
        {
            "chirp_mass_source": 27.9,
            "mass_1_source": 34.6,
            "mass_2_source": 30.0,
            "redshift": 0.10,
        }
    )
    assert abs(row["chirp_mass_detector"] - 30.69) < 1e-12
    assert abs(row["mass_1_detector"] - 38.06) < 1e-12
    assert row["component_mass_detector_origin"] == "derived_from_source_redshift"


def test_refresh_preserves_catalog_order_and_reports_availability():
    catalog = pd.DataFrame(
        [
            {"name": "A-v1", "gps": 1.0, "mass_1_source": 10.0, "mass_2_source": 8.0},
            {"name": "B-v1", "gps": 2.0, "mass_1_source": 20.0, "mass_2_source": 10.0},
        ]
    )

    def fetcher(name):
        return {
            "name": name,
            "redshift": 0.1 if name == "A-v1" else 0.2,
            "chirp_mass_source": 7.0 if name == "A-v1" else 12.0,
            "metadata_status": "ok",
        }

    refreshed = refresh_catalog_dataframe(catalog, workers=2, fetcher=fetcher)
    assert refreshed["name"].tolist() == ["A-v1", "B-v1"]
    summary = detector_frame_availability(refreshed)
    assert summary["events_with_detector_component_masses"] == 2
    assert summary["events_with_detector_chirp_mass"] == 2
