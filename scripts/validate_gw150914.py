from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from gwproject.config import load_toml
from gwproject.downloads import download_with_resume
from gwproject.morphology import physical_morphology_series, strict_interpolate_reference, validate_morphology
from gwproject.preprocessing import preprocess, q_transform_constant_q, q_transform_self_test
from gwproject.provenance import sha256_file, write_manifest
from gwproject.ridge import estimate_chirp_mass_from_ridge_points, ridge_from_energy

GPS_MERGER = 1126259462.4
FIG1_T0 = 1126259462.0
FIG1_WINDOW = (0.25, 0.45)
URL_BASES = [
    "https://gwosc.org/GW150914data/P150914",
    "https://www.gw-openscience.org/GW150914data/P150914",
    "https://raw.githubusercontent.com/gwastro/gw150914_investigation/master",
]
FILES = {"H1": "fig1-observed-H.txt", "L1": "fig1-observed-L.txt"}


def download_first(filename: str, destination: Path) -> Path:
    last_error: Exception | None = None
    for base in URL_BASES:
        try:
            return download_with_resume(f"{base.rstrip('/')}/{filename}", destination)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"falha ao baixar {filename}: {last_error}")


def aligned_time_axis(times: np.ndarray, lag_samples: int) -> np.ndarray:
    if lag_samples > 0:
        return times[lag_samples:]
    if lag_samples < 0:
        return times[:lag_samples]
    return times


def preprocessing_kwargs(cfg: dict) -> dict:
    return {
        "psd_segment_seconds": float(cfg["psd"]["segment_seconds"]),
        "psd_overlap_fraction": float(cfg["psd"]["overlap_fraction"]),
        "psd_average": str(cfg["psd"]["average"]),
        "whitening_fmin": float(cfg["whitening"]["fmin_hz"]),
        "whitening_fmax": float(cfg["whitening"]["fmax_hz"]),
        "whitening_tukey_alpha": float(cfg["whitening"]["tukey_alpha"]),
        "bandpass_order": int(cfg["bandpass"]["order"]),
        "edge_crop_seconds": float(cfg["signal"]["edge_crop_seconds"]),
        "white_std_min": float(cfg["gates"]["white_std_min"]),
        "white_std_max": float(cfg["gates"]["white_std_max"]),
        "max_zero_fraction": float(cfg["data_quality"]["max_zero_fraction"]),
        "max_zero_run_seconds": float(cfg["data_quality"]["max_zero_run_seconds"]),
        "return_diagnostics": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validacao quantitativa da Camada 2 v8 em GW150914")
    parser.add_argument("--config", type=Path, default=Path("configs/processing.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/camada2_gw150914"))
    args = parser.parse_args()
    try:
        from gwpy.timeseries import TimeSeries
    except ImportError as exc:
        raise SystemExit("Instale o extra GW: pip install -e '.[gw]'") from exc

    cfg = load_toml(args.config)
    gates = cfg["gates"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    losc_dir = args.output_dir / "losc"
    losc_dir.mkdir(exist_ok=True)

    qcfg = cfg["qtransform"]
    q_self_ratio = q_transform_self_test(
        fs=float(cfg["signal"]["sample_rate"]),
        duration_seconds=float(qcfg["self_test_seconds"]),
        fmin=float(qcfg["fmin_hz"]),
        fmax=float(qcfg["fmax_hz"]),
        n_bins=int(qcfg["n_bins"]),
        q=float(qcfg["display_q"]),
        seed=int(qcfg["self_test_seed"]),
    )
    q_self_passed = q_self_ratio <= float(gates["q_flatness_max_ratio"])

    raw = {
        detector: TimeSeries.fetch_open_data(
            detector,
            GPS_MERGER - 16,
            GPS_MERGER + 16,
            sample_rate=int(cfg["signal"]["sample_rate"]),
            cache=True,
        )
        for detector in ("H1", "L1")
    }

    morphology: dict[str, dict] = {}
    chirp: dict[str, dict] = {}
    reference_hashes: dict[str, str] = {}
    white_quality: dict[str, dict] = {}

    for detector in ("H1", "L1"):
        strain = raw[detector]
        fs = float(strain.sample_rate.value)
        values = np.asarray(strain.value, dtype=float)

        observed_path = download_first(FILES[detector], losc_dir / FILES[detector])
        reference_hashes[FILES[detector]] = sha256_file(observed_path)
        observed = np.loadtxt(observed_path)

        physical = physical_morphology_series(
            values,
            fs,
            fmin=float(cfg["morphology"]["fmin_hz"]),
            fmax=float(cfg["morphology"]["fmax_hz"]),
            order=int(cfg["bandpass"]["order"]),
            notches_hz=cfg["morphology"]["notches_hz"],
            notch_q=float(cfg["morphology"]["notch_q"]),
        ) * 1e21
        t_fig = np.asarray(strain.times.value, dtype=float) - FIG1_T0
        mask = (t_fig >= FIG1_WINDOW[0]) & (t_fig <= FIG1_WINDOW[1])
        t_local = t_fig[mask]
        ours = physical[mask]
        reference = strict_interpolate_reference(t_local, observed)
        alignment = validate_morphology(
            ours,
            reference,
            fs,
            max_lag_seconds=float(cfg["morphology"]["max_lag_seconds"]),
            allow_sign_flip=bool(cfg["morphology"]["allow_sign_flip"]),
        )
        t_aligned = aligned_time_axis(t_local, alignment.lag_samples)
        gain = float(
            np.dot(alignment.a_aligned, alignment.b_aligned)
            / (np.dot(alignment.b_aligned, alignment.b_aligned) + 1e-30)
        )
        residual = alignment.a_aligned - gain * alignment.b_aligned
        np.savez(
            args.output_dir / f"{detector}_alignment.npz",
            time=t_aligned,
            ours=alignment.a_aligned,
            reference=alignment.b_aligned,
            residual=residual,
        )
        morphology[detector] = {
            "correlation": alignment.correlation,
            "lag_ms": 1000.0 * alignment.lag_seconds,
            "sign": alignment.sign,
            "gain": gain,
            "normalized_residual_rms": float(np.std(residual) / (np.std(alignment.a_aligned) + 1e-30)),
            "passed": alignment.correlation >= float(gates["morphology_correlation_min"]),
        }

        wide, _, diagnostics = preprocess(
            values,
            fs,
            bandpass_fmin=30.0,
            bandpass_fmax=400.0,
            **preprocessing_kwargs(cfg),
        )
        white_quality[detector] = {
            "white_std_core": diagnostics.white_std_core,
            "zero_fraction": diagnostics.quality.zero_fraction,
            "longest_zero_run_seconds": diagnostics.quality.longest_zero_run_seconds,
            "passed": float(gates["white_std_min"]) <= diagnostics.white_std_core <= float(gates["white_std_max"]),
        }

        t_rel = np.asarray(strain.times.value, dtype=float) - GPS_MERGER
        ridge_cfg = cfg["ridge"]
        time_mask = (
            (t_rel >= float(ridge_cfg["window_start_seconds"]))
            & (t_rel <= float(ridge_cfg["window_stop_seconds"]))
        )
        qresult = q_transform_constant_q(
            wide,
            fs,
            fmin=float(ridge_cfg["fmin_hz"]),
            fmax=float(ridge_cfg["fmax_hz"]),
            n_bins=int(ridge_cfg["n_bins"]),
            q=float(qcfg["ridge_q"]),
        )
        ridge_t, ridge_f = ridge_from_energy(
            qresult.energy[:, time_mask],
            qresult.frequencies,
            t_rel[time_mask],
            min_quantile=float(ridge_cfg["min_quantile"]),
            max_points=int(ridge_cfg["max_points"]),
        )
        estimate = estimate_chirp_mass_from_ridge_points(
            ridge_t,
            ridge_f,
            n_bootstrap=int(gates["chirp_bootstrap_samples"]),
            seed=42 if detector == "H1" else 43,
        )
        reference_mass = float(gates["chirp_reference_mass_solar"])
        relative_error = abs(estimate.mass_solar - reference_mass) / reference_mass
        chirp[detector] = {
            "mass_solar": estimate.mass_solar,
            "ridge_bootstrap_dispersion_solar": [
                estimate.dispersion_low_solar,
                estimate.dispersion_high_solar,
            ],
            "relative_error": relative_error,
            "ridge_q": float(qcfg["ridge_q"]),
            "n_points": estimate.n_points,
            "n_bootstrap": estimate.n_bootstrap,
            "passed_individual": relative_error <= float(gates["chirp_relative_error_max"]),
        }
        np.savez(args.output_dir / f"{detector}_ridge.npz", time=ridge_t, frequency=ridge_f)

    masses = [chirp[detector]["mass_solar"] for detector in ("H1", "L1")]
    disagreement = abs(masses[0] - masses[1]) / np.mean(masses)
    morphology_passed = all(item["passed"] for item in morphology.values())
    white_passed = all(item["passed"] for item in white_quality.values())
    chirp_passed = (
        all(item["passed_individual"] for item in chirp.values())
        and disagreement <= float(gates["chirp_detector_disagreement_max"])
    )
    summary = {
        "gate_q_transform_self_test": {
            "flatness_ratio": q_self_ratio,
            "maximum": float(gates["q_flatness_max_ratio"]),
            "passed": q_self_passed,
        },
        "gate_whitening": {"detectors": white_quality, "passed": white_passed},
        "gate_morphology": {"detectors": morphology, "passed": morphology_passed},
        "gate_chirp_mass": {
            "detectors": chirp,
            "relative_detector_disagreement": disagreement,
            "passed": chirp_passed,
        },
        "reference_sha256": reference_hashes,
        "overall_passed": q_self_passed and white_passed and morphology_passed and chirp_passed,
        "scope_warning": (
            "Validacao pedagogica de condicionamento, morfologia e ridge. "
            "O intervalo do bootstrap mede dispersao do ridge, nao cobertura completa."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_manifest(args.output_dir / "manifest.json", {"config": cfg, **summary})
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["overall_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
