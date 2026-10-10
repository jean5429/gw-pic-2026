from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from gwproject.catalog_analysis import (
    build_local_template_bank,
    catalog_detector_component_masses,
    catalog_redshift,
)
from gwproject.config import load_toml
from gwproject.matched_filter import (
    estimate_offsource_psd,
    matched_filter_snr,
    taylorf2_tapered_extension,
    taylorf2_template,
    template_band,
    top2_ratio,
)
from gwproject.provenance import write_manifest


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


ALGORITHM_REVISION = "catalog-matched-filter-offsource-psd-coincidence-diagnostic-v2"


def _config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(
        {"algorithm_revision": ALGORITHM_REVISION, "config": config},
        sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _analyse_event(
    event: str,
    channels: list[dict[str, Any]],
    metadata: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    result_path = output_dir / "events" / f"{event}.json"
    config_sha256 = _config_sha256(config)
    if resume and result_path.exists():
        saved = _read_json(result_path)
        if saved.get("config_sha256") == config_sha256:
            return saved
    try:
        from gwpy.timeseries import TimeSeries
    except ImportError as exc:
        raise RuntimeError("Instale o extra GW: pip install -e '.[gw]'") from exc

    cfg = config["catalog_matched_filter"]
    allow_source = bool(cfg.get("allow_source_frame_fallback", False))
    masses, frame = catalog_detector_component_masses(
        metadata, allow_source_fallback=allow_source
    )
    if masses is None:
        result = {
            "config_sha256": config_sha256,
            "event": event,
            "status": "missing_detector_frame_masses",
            "status_reason": "massas detector-frame indisponiveis e fallback source-frame desativado",
            "catalog_redshift": catalog_redshift(metadata),
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    bank = build_local_template_bank(
        *masses,
        scale_factors=[float(x) for x in cfg["scale_factors"]],
        minimum_component_mass=float(cfg["minimum_component_mass_solar"]),
    )
    template_rows: dict[str, dict[str, Any]] = {}
    for template in bank:
        band = template_band(
            template.m1,
            template.m2,
            max_template_seconds=float(cfg["max_template_seconds"]),
            floor_hz=float(cfg["f_low_hz"]),
            analysis_f_high_hz=float(cfg["analysis_f_high_hz"]),
            extension_factor=float(cfg["extension_factor"]),
        )
        template_rows[template.name] = {
            "template": template.name,
            "m1": template.m1,
            "m2": template.m2,
            "central": template.central,
            **band.__dict__,
            "detectors": {},
            "detector_peak_times_seconds": {},
            "active": band.bandwidth_inspiral >= float(cfg["minimum_template_bandwidth_hz"]),
        }

    for channel in channels:
        detector = str(channel["detector"])
        raw_path = Path(str(channel["raw_path"]))
        raw = TimeSeries.read(raw_path, format="hdf5")
        values = np.asarray(raw.value, dtype=float)
        if not np.all(np.isfinite(values)):
            continue
        sample_rate = float(raw.sample_rate.value)
        frequencies = np.fft.rfftfreq(len(values), 1.0 / sample_rate)
        gps = float(metadata["gps"])
        relative_times = np.asarray(raw.times.value, dtype=float) - gps
        event_index = int(np.argmin(np.abs(relative_times)))
        psd = estimate_offsource_psd(
            values,
            sample_rate,
            event_index,
            exclusion_seconds=float(cfg["psd_event_exclusion_seconds"]),
            segment_seconds=float(cfg["psd_segment_seconds"]),
            overlap_fraction=float(cfg["psd_overlap_fraction"]),
            average=str(cfg["psd_average"]),
        )
        half_window = int(round(float(cfg["peak_window_seconds"]) * sample_rate))

        for template in bank:
            row = template_rows[template.name]
            if not row["active"]:
                continue
            series_name = str(cfg["series"])
            if series_name == "inspiral":
                waveform = taylorf2_template(
                    template.m1,
                    template.m2,
                    frequencies,
                    float(row["f_low"]),
                    float(row["f_high_inspiral"]),
                )
                f_high = float(row["f_high_inspiral"])
            elif series_name == "extended":
                waveform = taylorf2_tapered_extension(
                    template.m1,
                    template.m2,
                    frequencies,
                    float(row["f_low"]),
                    float(row["f_high_extended"]),
                    isco_frequency=float(row["f_isco"]),
                )
                f_high = float(row["f_high_extended"])
            else:
                raise ValueError("series deve ser inspiral ou extended")
            snr = matched_filter_snr(
                values,
                sample_rate,
                waveform,
                psd.frequencies,
                psd.values,
                f_low=float(row["f_low"]),
                f_high=f_high,
            )
            local_start = max(0, event_index - half_window)
            local = snr[local_start : event_index + half_window + 1]
            peak = float(np.max(local))
            peak_index = local_start + int(np.argmax(local))
            row["detectors"][detector] = peak
            row["detector_peak_times_seconds"][detector] = float(
                relative_times[peak_index]
            )
            if bool(cfg["save_snr_series"]):
                target = output_dir / "snr_series" / event
                target.mkdir(parents=True, exist_ok=True)
                np.save(target / f"{detector}__{template.name}.npy", snr.astype(np.float32))

    ranking: list[dict[str, Any]] = []
    for row in template_rows.values():
        detector_values = {key: float(value) for key, value in row["detectors"].items()}
        network_snr = float(np.sqrt(sum(value * value for value in detector_values.values())))
        ranking.append({**row, "network_snr": network_snr, **{f"snr_{k}": v for k, v in detector_values.items()}})
    ranking.sort(key=lambda item: item["network_snr"], reverse=True)
    active_ranking = [row for row in ranking if row["active"]]
    central_rank = next(
        (index + 1 for index, row in enumerate(active_ranking) if row["central"]),
        None,
    )
    ratio = top2_ratio(active_ranking) if active_ranking else float("nan")
    winner = active_ranking[0] if active_ranking else None
    winner_peak_times = (
        list(winner["detector_peak_times_seconds"].values()) if winner else []
    )
    coincidence_span = (
        float(max(winner_peak_times) - min(winner_peak_times))
        if len(winner_peak_times) >= 2
        else None
    )
    gate = {
        "central_rank": central_rank,
        "central_rank_passed": (
            central_rank == 1 if bool(cfg["require_central_rank_1"]) else True
        ),
        "network_snr": winner["network_snr"] if winner else None,
        "network_snr_passed": (
            winner["network_snr"] >= float(cfg["network_snr_min"]) if winner else False
        ),
        "top2_snr_ratio": ratio,
        "top2_ratio_passed": (
            ratio >= float(cfg["minimum_top2_snr_ratio"]) if np.isfinite(ratio) else False
        ),
        "coincidence_span_seconds": coincidence_span,
        "coincidence_passed": bool(
            coincidence_span is not None
            and coincidence_span <= float(cfg["coincidence_window_seconds"])
        ),
    }
    gate["local_diagnostic_gate_passed"] = bool(
        gate["network_snr_passed"]
        and gate["top2_ratio_passed"]
        and gate["coincidence_passed"]
    )
    gate["catalog_consistency_passed"] = bool(central_rank == 1)
    gate["passed"] = bool(
        gate["local_diagnostic_gate_passed"]
        and (gate["catalog_consistency_passed"] if bool(cfg["require_central_rank_1"]) else True)
    )
    result = {
        "config_sha256": config_sha256,
        "event": event,
        "status": "ok" if active_ranking else "no_active_template",
        "status_reason": (
            None if active_ranking else
            "nenhum template manteve a largura minima da banda inspiral acima de f_low"
        ),
        "catalog_m1_solar": masses[0],
        "catalog_m2_solar": masses[1],
        "mass_frame": frame,
        "catalog_redshift": catalog_redshift(metadata),
        "analysis_mass_frame_required": "detector",
        "detectors": sorted(str(channel["detector"]) for channel in channels),
        "series": str(cfg["series"]),
        "psd_method": "offsource_two_sided_median",
        "ranking": active_ranking,
        "inactive_templates": [row for row in ranking if not row["active"]],
        "gate": gate,
        "scope_warning": (
            "Banco TaylorF2 local e diagnostico, construido em detector-frame. "
            "Nao ha FAR, chi2 ou busca cega: o resultado nao e um teste de deteccao. "
            "O gate local exige SNR, margem, coincidencia entre detectores e, por "
            "configuracao canônica, consistencia com o template central."
        ),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Matched filtering local para todo o catalogo aprovado")
    parser.add_argument("--approved", type=Path, default=Path("outputs/gwtc4_218/approved_channels.csv"))
    parser.add_argument("--catalog", type=Path, default=Path("outputs/gwtc4_218/catalog.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/catalog_analysis.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research/gwtc4_218/matched_filter_diagnostic"))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--event", action="append", default=[])
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    approved = pd.read_csv(args.approved)
    catalog = pd.read_csv(args.catalog)
    config = load_toml(args.config)
    if args.event:
        approved = approved[approved["event"].isin(args.event)].copy()
        catalog = catalog[catalog["name"].isin(args.event)].copy()
    events = list(dict.fromkeys(approved["event"].astype(str)))
    if args.limit > 0:
        events = events[: args.limit]
        approved = approved[approved["event"].isin(events)].copy()
        catalog = catalog[catalog["name"].isin(events)].copy()
    metadata_map = {str(row["name"]): row for row in catalog.to_dict(orient="records")}
    allow_source = bool(config["catalog_matched_filter"].get("allow_source_frame_fallback", False))
    available_events = sum(
        catalog_detector_component_masses(row, allow_source_fallback=allow_source)[0] is not None
        for row in catalog.to_dict(orient="records")
        if str(row.get("name")) in events
    )
    if available_events == 0:
        raise SystemExit(
            "Nenhum evento possui massas de componentes no frame do detector. Execute primeiro: "
            "gwproject catalog refresh-metadata"
        )
    channel_map = {
        event: group.to_dict(orient="records")
        for event, group in approved.groupby("event")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workers = args.workers or int(config["catalog_matched_filter"]["workers"])
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _analyse_event,
                event,
                channel_map[event],
                metadata_map.get(event, {}),
                config,
                args.output_dir,
                not args.no_resume,
            ): event
            for event in events
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="matched filter catalog"):
            results.append(future.result())

    event_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for result in results:
        gate = result.get("gate") or {}
        event_rows.append(
            {
                "event": result.get("event"),
                "status": result.get("status"),
                "catalog_m1_solar": result.get("catalog_m1_solar"),
                "catalog_m2_solar": result.get("catalog_m2_solar"),
                "mass_frame": result.get("mass_frame"),
                "catalog_redshift": result.get("catalog_redshift"),
                "detectors": ",".join(result.get("detectors", [])),
                "winner": (result.get("ranking") or [{}])[0].get("template"),
                "network_snr": gate.get("network_snr"),
                "central_rank": gate.get("central_rank"),
                "top2_snr_ratio": gate.get("top2_snr_ratio"),
                "central_rank_passed": gate.get("central_rank_passed"),
                "network_snr_passed": gate.get("network_snr_passed"),
                "top2_ratio_passed": gate.get("top2_ratio_passed"),
                "coincidence_span_seconds": gate.get("coincidence_span_seconds"),
                "coincidence_passed": gate.get("coincidence_passed"),
                "local_diagnostic_gate_passed": gate.get("local_diagnostic_gate_passed"),
                "catalog_consistency_passed": gate.get("catalog_consistency_passed"),
                "gate_requires_central_rank_1": bool(
                    config["catalog_matched_filter"]["require_central_rank_1"]
                ),
                "gate_passed": gate.get("passed"),
                "status_reason": result.get("status_reason"),
            }
        )
        for rank, row in enumerate(result.get("ranking", []), start=1):
            ranking_rows.append(
                {
                    "event": result.get("event"),
                    "rank": rank,
                    "template": row.get("template"),
                    "m1": row.get("m1"),
                    "m2": row.get("m2"),
                    "central": row.get("central"),
                    "mass_frame": result.get("mass_frame"),
                    "network_snr": row.get("network_snr"),
                    "snr_H1": row.get("snr_H1"),
                    "snr_L1": row.get("snr_L1"),
                    "snr_V1": row.get("snr_V1"),
                    "f_low": row.get("f_low"),
                    "f_high_inspiral": row.get("f_high_inspiral"),
                    "bandwidth_inspiral": row.get("bandwidth_inspiral"),
                }
            )
    per_event = pd.DataFrame(event_rows).sort_values("event")
    if "central_rank" in per_event:
        per_event["central_rank"] = pd.to_numeric(
            per_event["central_rank"], errors="coerce"
        ).astype("Int64")
    rankings = pd.DataFrame(ranking_rows).sort_values(["event", "rank"])
    per_event.to_csv(args.output_dir / "per_event.csv", index=False)
    rankings.to_csv(args.output_dir / "rankings.csv", index=False)
    completed = per_event[per_event["status"] == "ok"]

    def bool_sum(column: str) -> int:
        values = pd.to_numeric(completed[column], errors="coerce").fillna(0.0)
        return int(values.astype(bool).sum())

    summary = {
        "events_requested": len(events),
        "events_completed": int(len(completed)),
        "events_without_active_template": int((per_event["status"] == "no_active_template").sum()),
        "events_missing_detector_frame_masses": int(
            (per_event["status"] == "missing_detector_frame_masses").sum()
        ),
        "central_template_winners": int(
            bool_sum("catalog_consistency_passed")
        ),
        "network_snr_passed": int(
            bool_sum("network_snr_passed")
        ),
        "top2_ratio_passed": int(
            bool_sum("top2_ratio_passed")
        ),
        "coincidence_passed": int(
            bool_sum("coincidence_passed")
        ),
        "local_diagnostic_gates_passed": int(
            bool_sum("local_diagnostic_gate_passed")
        ),
        "full_gates_passed": int(
            bool_sum("gate_passed")
        ),
        "gate_requires_central_rank_1": bool(
            config["catalog_matched_filter"]["require_central_rank_1"]
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_manifest(
        args.output_dir / "manifest.json",
        {"config": config, "arguments": vars(args), "summary": summary},
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
