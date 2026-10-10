from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from gwproject.config import load_toml
from gwproject.matched_filter import (
    matched_filter_snr,
    taylorf2_tapered_extension,
    taylorf2_template,
    template_band,
    top2_ratio,
)
from gwproject.preprocessing import estimate_psd
from gwproject.provenance import write_manifest

GPS_MERGER = 1126259462.4


def main() -> int:
    parser = argparse.ArgumentParser(description="Matched filtering pedagogico v8 em GW150914")
    parser.add_argument("--config", type=Path, default=Path("configs/matched_filter.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/matched_filter_gw150914"))
    args = parser.parse_args()
    try:
        from gwpy.timeseries import TimeSeries
    except ImportError as exc:
        raise SystemExit("Instale o extra GW: pip install -e '.[gw]'") from exc

    cfg = load_toml(args.config)
    templates = {name: tuple(value) for name, value in cfg["templates"].items()}
    signal_cfg = cfg["signal"]
    gate_cfg = cfg["gate"]
    expected = str(gate_cfg["expected_template"])
    fs_target = int(signal_cfg["sample_rate"])
    results: dict[str, dict[str, dict[str, float]]] = {name: {} for name in templates}
    bands: dict[str, dict[str, float]] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, masses in templates.items():
        band = template_band(
            *masses,
            max_template_seconds=float(signal_cfg["max_template_seconds"]),
            floor_hz=float(signal_cfg["f_low_hz"]),
            analysis_f_high_hz=float(signal_cfg["analysis_f_high_hz"]),
            extension_factor=float(signal_cfg["extension_factor"]),
        )
        bands[name] = band.__dict__
        if band.bandwidth_inspiral < float(gate_cfg["minimum_template_bandwidth_hz"]):
            raise SystemExit(
                f"Template inerte: {name} tem apenas {band.bandwidth_inspiral:.1f} Hz "
                "de banda inspiral"
            )

    for detector in ("H1", "L1"):
        strain = TimeSeries.fetch_open_data(
            detector,
            GPS_MERGER - 16,
            GPS_MERGER + 16,
            sample_rate=fs_target,
            cache=True,
        )
        values = np.asarray(strain.value, dtype=float)
        fs = float(strain.sample_rate.value)
        frequencies = np.fft.rfftfreq(len(values), 1.0 / fs)
        psd = estimate_psd(values, fs, segment_seconds=4.0)
        t_rel = np.asarray(strain.times.value, dtype=float) - GPS_MERGER
        merger_index = int(np.argmin(np.abs(t_rel)))
        half_window = int(float(gate_cfg["peak_window_seconds"]) * fs)

        for name, masses in templates.items():
            band = bands[name]
            inspiral = taylorf2_template(
                *masses,
                frequencies,
                float(band["f_low"]),
                float(band["f_high_inspiral"]),
            )
            extended = taylorf2_tapered_extension(
                *masses,
                frequencies,
                float(band["f_low"]),
                float(band["f_high_extended"]),
                isco_frequency=float(band["f_isco"]),
            )
            detector_result: dict[str, float] = {}
            for series_name, template, high in (
                ("inspiral", inspiral, float(band["f_high_inspiral"])),
                ("extended", extended, float(band["f_high_extended"])),
            ):
                snr = matched_filter_snr(
                    values,
                    fs,
                    template,
                    psd.frequencies,
                    psd.values,
                    f_low=float(band["f_low"]),
                    f_high=high,
                )
                local = snr[max(0, merger_index - half_window):merger_index + half_window + 1]
                detector_result[series_name] = float(np.max(local))
                safe_name = name.replace(" ", "_").replace("+", "p")
                np.save(
                    args.output_dir / f"{detector}_{safe_name}_{series_name}_snr.npy",
                    snr.astype(np.float32),
                )
            results[name][detector] = detector_result

    rankings: dict[str, list[dict]] = {}
    for series_name in ("inspiral", "extended"):
        table = []
        for name in templates:
            h1 = results[name].get("H1", {}).get(series_name, 0.0)
            l1 = results[name].get("L1", {}).get(series_name, 0.0)
            table.append({
                "template": name,
                "snr_H1": h1,
                "snr_L1": l1,
                "network_snr": float(np.hypot(h1, l1)),
                **bands[name],
            })
        table.sort(key=lambda row: row["network_snr"], reverse=True)
        rankings[series_name] = table

    ranking_series = str(gate_cfg["ranking_series"])
    ranking = rankings[ranking_series]
    ratio = top2_ratio(ranking)
    rank_pass = (
        not bool(gate_cfg["require_expected_template_rank_1"])
        or ranking[0]["template"] == expected
    )
    snr_pass = ranking[0]["network_snr"] >= float(gate_cfg["network_snr_min"])
    margin_pass = ratio >= float(gate_cfg["minimum_top2_snr_ratio"])
    passed = rank_pass and snr_pass and margin_pass
    summary = {
        "rankings": rankings,
        "gate": {
            "ranking_series": ranking_series,
            "winner": ranking[0]["template"],
            "expected": expected,
            "network_snr": ranking[0]["network_snr"],
            "top2_snr_ratio": ratio,
            "rank_passed": rank_pass,
            "snr_passed": snr_pass,
            "margin_passed": margin_pass,
            "passed": passed,
        },
        "scope_warning": (
            "O ranking extended usa TaylorF2 extrapolado e afunilado acima de f_ISCO. "
            "Ele e diagnostico pedagogico, nao substitui IMRPhenom/PyCBC nem estima FAR."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_manifest(args.output_dir / "manifest.json", {"config": cfg, **summary})
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
