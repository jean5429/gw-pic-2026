from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Any, Iterable
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

GWOSC_API_ROOT = "https://gwosc.org/api/v2"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _request_json(
    session: requests.Session,
    url: str,
    *,
    timeout: float,
    retries: int,
    backoff: float,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout, headers={"Accept": "application/json"})
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("resposta JSON nao e um objeto")
            return payload
        except Exception as exc:  # requests and JSON errors
            last_error = exc
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise RuntimeError(f"falha ao consultar {url}: {last_error}")


def _parameter_score(result: dict[str, Any]) -> tuple[int, int, int]:
    parameters = {
        str(item.get("name")): item.get("best")
        for item in result.get("parameters", [])
        if isinstance(item, dict)
    }
    useful = sum(
        parameters.get(name) is not None
        for name in ("redshift", "chirp_mass_source", "mass_1_source", "mass_2_source")
    )
    return (
        int(bool(result.get("is_preferred"))),
        int(str(result.get("pipeline_type", "")).lower() == "pe"),
        useful,
    )


def select_parameter_estimation(payload: dict[str, Any]) -> dict[str, Any]:
    results = [item for item in payload.get("results", []) if isinstance(item, dict)]
    if not results:
        return {}
    return max(results, key=_parameter_score)


def flatten_parameter_estimation(result: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in result.get("parameters", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if name:
            values[str(name)] = item.get("best")
    return values


def add_detector_frame_columns(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    redshift = _finite(output.get("redshift"))
    factor = 1.0 + redshift if redshift is not None and redshift >= 0 else None

    source_chirp = _finite(output.get("chirp_mass_source"))
    source_m1 = _finite(output.get("mass_1_source"))
    source_m2 = _finite(output.get("mass_2_source"))

    explicit_chirp = _finite(output.get("chirp_mass"))
    explicit_m1 = _finite(output.get("mass_1"))
    explicit_m2 = _finite(output.get("mass_2"))

    if explicit_chirp is not None and explicit_chirp > 0:
        output["chirp_mass_detector"] = explicit_chirp
        chirp_origin = "gwosc_detector_catalog"
    elif factor is not None and source_chirp is not None and source_chirp > 0:
        output["chirp_mass_detector"] = source_chirp * factor
        chirp_origin = "derived_from_source_redshift"
    else:
        output["chirp_mass_detector"] = None
        chirp_origin = "missing"

    if explicit_m1 is not None and explicit_m2 is not None and explicit_m1 > 0 and explicit_m2 > 0:
        output["mass_1_detector"] = max(explicit_m1, explicit_m2)
        output["mass_2_detector"] = min(explicit_m1, explicit_m2)
        component_origin = "gwosc_detector_catalog"
    elif (
        factor is not None
        and source_m1 is not None
        and source_m2 is not None
        and source_m1 > 0
        and source_m2 > 0
    ):
        output["mass_1_detector"] = max(source_m1, source_m2) * factor
        output["mass_2_detector"] = min(source_m1, source_m2) * factor
        component_origin = "derived_from_source_redshift"
    else:
        output["mass_1_detector"] = None
        output["mass_2_detector"] = None
        component_origin = "missing"

    output["chirp_mass_detector_origin"] = chirp_origin
    output["component_mass_detector_origin"] = component_origin
    return output


def fetch_event_metadata_v2(
    event_name: str,
    *,
    timeout: float = 30.0,
    retries: int = 4,
    backoff: float = 1.5,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    own_session = session is None
    session = session or requests.Session()
    try:
        encoded = quote(event_name, safe="")
        detail_url = f"{GWOSC_API_ROOT}/event-versions/{encoded}?format=json"
        detail = _request_json(
            session, detail_url, timeout=timeout, retries=retries, backoff=backoff
        )
        parameters_url = detail.get("parameters_url")
        if not parameters_url:
            parameters_url = f"{GWOSC_API_ROOT}/event-versions/{encoded}/parameters?format=json"
        parameters = _request_json(
            session, str(parameters_url), timeout=timeout, retries=retries, backoff=backoff
        )
        estimation = select_parameter_estimation(parameters)
        values = flatten_parameter_estimation(estimation)
        row = {
            "name": event_name,
            "gps": detail.get("gps"),
            "catalog": detail.get("catalog"),
            "chirp_mass": values.get("chirp_mass"),
            "chirp_mass_source": values.get("chirp_mass_source"),
            "mass_1": values.get("mass_1"),
            "mass_2": values.get("mass_2"),
            "mass_1_source": values.get("mass_1_source"),
            "mass_2_source": values.get("mass_2_source"),
            "redshift": values.get("redshift"),
            "luminosity_distance": values.get("luminosity_distance"),
            "snr": values.get("network_matched_filter_snr"),
            "far": values.get("far"),
            "metadata_status": "ok",
            "metadata_api": "gwosc_api_v2",
            "parameter_estimation_name": estimation.get("name"),
            "parameter_waveform_family": estimation.get("waveform_family"),
            "parameter_is_preferred": estimation.get("is_preferred"),
            "metadata_refreshed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        return add_detector_frame_columns(row)
    except Exception as exc:
        return {
            "name": event_name,
            "metadata_status": f"error: {type(exc).__name__}: {exc}",
            "metadata_api": "gwosc_api_v2",
            "metadata_refreshed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    finally:
        if own_session:
            session.close()


def merge_metadata_row(existing: dict[str, Any], refreshed: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in refreshed.items():
        if key == "name" or value is None:
            continue
        if isinstance(value, float) and not np.isfinite(value):
            continue
        merged[key] = value
    return add_detector_frame_columns(merged)


def refresh_catalog_dataframe(
    catalog: pd.DataFrame,
    *,
    workers: int = 4,
    fetcher=fetch_event_metadata_v2,
) -> pd.DataFrame:
    if "name" not in catalog.columns:
        raise ValueError("catalog.csv deve conter a coluna 'name'")
    names = catalog["name"].astype(str).tolist()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        refreshed_rows = list(executor.map(fetcher, names))
    refreshed_map = {str(row.get("name")): row for row in refreshed_rows}
    merged = [
        merge_metadata_row(row, refreshed_map.get(str(row.get("name")), {}))
        for row in catalog.to_dict(orient="records")
    ]
    return pd.DataFrame.from_records(merged)


def detector_frame_availability(catalog: pd.DataFrame) -> dict[str, int]:
    chirp = pd.to_numeric(catalog.get("chirp_mass_detector"), errors="coerce")
    m1 = pd.to_numeric(catalog.get("mass_1_detector"), errors="coerce")
    m2 = pd.to_numeric(catalog.get("mass_2_detector"), errors="coerce")
    redshift = pd.to_numeric(catalog.get("redshift"), errors="coerce")
    return {
        "events_total": int(len(catalog)),
        "events_with_redshift": int(np.isfinite(redshift).sum()),
        "events_with_detector_chirp_mass": int(np.isfinite(chirp).sum()),
        "events_with_detector_component_masses": int((np.isfinite(m1) & np.isfinite(m2)).sum()),
        "metadata_errors": int(
            catalog.get("metadata_status", pd.Series(index=catalog.index, dtype=object))
            .astype(str)
            .str.startswith("error:")
            .sum()
        ),
    }
