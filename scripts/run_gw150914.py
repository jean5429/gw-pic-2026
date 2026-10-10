from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from gwproject.alignment import best_lag_alignment
from gwproject.config import load_toml
from gwproject.preprocessing import preprocess
from gwproject.provenance import write_manifest

GPS_MERGER = 1126259462.4


def main() -> int:
    parser = argparse.ArgumentParser(description="Processa GW150914 com gates de whitening v8")
    parser.add_argument("--config", type=Path, default=Path("configs/processing.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gw150914"))
    args = parser.parse_args()
    try:
        from gwpy.timeseries import TimeSeries
    except ImportError as exc:
        raise SystemExit("Instale o extra GW: pip install -e '.[gw]'") from exc

    cfg = load_toml(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    processed: dict[str, np.ndarray] = {}
    for detector in ("H1", "L1"):
        strain = TimeSeries.fetch_open_data(
            detector,
            GPS_MERGER - 16,
            GPS_MERGER + 16,
            sample_rate=int(cfg["signal"]["sample_rate"]),
            cache=True,
        )
        values = np.asarray(strain.value, dtype=float)
        fs = float(strain.sample_rate.value)
        filtered, _, diagnostics = preprocess(
            values,
            fs,
            psd_segment_seconds=float(cfg["psd"]["segment_seconds"]),
            psd_overlap_fraction=float(cfg["psd"]["overlap_fraction"]),
            psd_average=str(cfg["psd"]["average"]),
            whitening_fmin=float(cfg["whitening"]["fmin_hz"]),
            whitening_fmax=float(cfg["whitening"]["fmax_hz"]),
            whitening_tukey_alpha=float(cfg["whitening"]["tukey_alpha"]),
            bandpass_fmin=35.0,
            bandpass_fmax=350.0,
            bandpass_order=int(cfg["bandpass"]["order"]),
            edge_crop_seconds=float(cfg["signal"]["edge_crop_seconds"]),
            white_std_min=float(cfg["gates"]["white_std_min"]),
            white_std_max=float(cfg["gates"]["white_std_max"]),
            max_zero_fraction=float(cfg["data_quality"]["max_zero_fraction"]),
            max_zero_run_seconds=float(cfg["data_quality"]["max_zero_run_seconds"]),
            return_diagnostics=True,
        )
        np.save(args.output_dir / f"{detector}_processed.npy", filtered.astype(np.float32))
        processed[detector] = filtered
        results[detector] = {
            "white_std_core": diagnostics.white_std_core,
            "zero_fraction": diagnostics.quality.zero_fraction,
            "longest_zero_run_seconds": diagnostics.quality.longest_zero_run_seconds,
        }

    center = int(16 * float(cfg["signal"]["sample_rate"]))
    half = int(0.25 * float(cfg["signal"]["sample_rate"]))
    aligned = best_lag_alignment(
        processed["H1"][center-half:center+half],
        processed["L1"][center-half:center+half],
        float(cfg["signal"]["sample_rate"]),
        max_lag_seconds=0.02,
        allow_sign_flip=True,
    )
    results["H1_L1_alignment"] = {
        "correlation": aligned.correlation,
        "lag_ms": 1000 * aligned.lag_seconds,
        "sign": aligned.sign,
        "diagnostic_only": True,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    write_manifest(args.output_dir / "manifest.json", {"config": cfg, **results})
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
