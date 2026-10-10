"""Freeze inputs, evaluate four paired conditions, and report event-level effects.

No training, download, calibration fit or test-set selection is performed.
Only the explicitly supplied new output directory is written.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gwproject.config import load_toml
from gwproject.provenance import package_versions, sha256_file
from gwproject.research.paired_intensity import (
    CONDITIONS, factorial_contrasts, fast_mahalanobis, intensity_variant,
    paired_bootstrap,
)


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _input_rows(paths, expected):
    from gwproject.gravity_spy import (
        attach_image_paths, index_spectrograms, official_splits,
        validate_harmonized_dataset,
    )
    approved = pd.read_csv(paths["approved"])
    if len(approved) != expected["channels"] or approved.event.nunique() != expected["events"]:
        raise ValueError("catalog coverage differs from 475 channels / 218 events; no rows dropped")
    if approved[["event", "detector", "qgram_path"]].isna().any().any():
        raise ValueError("catalog contains missing identifiers or paths")
    if approved.duplicated(["event", "detector"]).any():
        raise ValueError("duplicate event/detector")
    if not approved["approved"].astype(str).str.lower().eq("true").all():
        raise ValueError("input contains channels not approved by QC")
    validate_harmonized_dataset(paths["images_root"], paths["metadata"])
    metadata = pd.read_csv(paths["metadata"])
    splits = metadata.sample_type.astype(str).str.strip().str.lower()
    if not splits.isin(["train", "training", "validation", "validate", "valid", "test", "testing"]).all():
        raise ValueError(f"unknown official split names: {sorted(splits.unique())}")
    val = official_splits(metadata)["val"].copy()
    if len(val) != expected["validation_images"] or val.duplicated(["ifo", "gravityspy_id"]).any():
        raise ValueError("validation count or unique IFO/ID contract failed")
    val = attach_image_paths(val, index_spectrograms(paths["images_root"], "1.0"))
    rows = []
    for row in approved.sort_values(["event", "detector"]).itertuples():
        rows.append(dict(sample_id=f"catalog:{row.event}:{row.detector}",
                         domain="catalog", group=str(row.event), event=str(row.event),
                         detector=str(row.detector), label="", split="external",
                         path=str(Path(row.qgram_path).resolve())))
    for row in val.sort_values(["ifo", "gravityspy_id"]).itertuples():
        rows.append(dict(sample_id=f"gravity_spy:{row.ifo}:{row.gravityspy_id}",
                         domain="gravity_spy_val", group=f"{row.ifo}:{row.gravityspy_id}",
                         event="", detector=str(row.ifo), label=str(row.label), split="val",
                         path=str(Path(row.path).resolve())))
    frame = pd.DataFrame(rows)
    if frame.sample_id.duplicated().any():
        raise ValueError("duplicate sample identifiers")
    print(f"Hashing {len(frame)} source files; no source files will be changed.", flush=True)
    frame["source_sha256"] = [sha256_file(path) for path in frame.path]
    return frame


def _cache_images(frame, renderer, output, fingerprint, resume):
    from PIL import Image
    from torchvision import transforms
    from gwproject.qgram_inference import qgram_to_uint8
    cache = output / "baseline_images.npy"
    receipt = output / "baseline_images.json"
    shape = (len(frame), int(renderer["resize_height"]), int(renderer["resize_width"]))
    if resume and receipt.exists():
        meta = json.loads(receipt.read_text(encoding="utf-8"))
        if meta["fingerprint"] != fingerprint or sha256_file(cache) != meta["sha256"]:
            raise ValueError("baseline cache integrity failure; use a new run directory")
        pixels = np.load(cache, mmap_mode="r", allow_pickle=False)
        if pixels.shape != shape or pixels.dtype != np.uint8:
            raise ValueError("invalid baseline cache shape/dtype")
        return pixels
    resize = transforms.Resize(shape[1:])
    pixels = np.empty(shape, dtype=np.uint8)
    for i, row in enumerate(frame.itertuples()):
        if row.domain == "catalog":
            raw = np.load(row.path, allow_pickle=False)
            array = qgram_to_uint8(
                raw, lower_percentile=float(renderer["lower_percentile"]),
                upper_percentile=float(renderer["upper_percentile"]),
                source_duration_seconds=float(renderer["source_outseg_seconds"]),
                target_duration_seconds=float(renderer["target_duration_seconds"]),
                baseline_mode=str(renderer["baseline_mode"]),
                flip_frequency=bool(renderer["flip_frequency"]),
                render_mode=str(renderer.get("render_mode", "percentile")),
            )
            pixels[i] = np.asarray(resize(Image.fromarray(array)))
        else:
            with Image.open(row.path) as image:
                pixels[i] = np.asarray(resize(image.convert("L")))
        if sha256_file(row.path) != row.source_sha256:
            raise ValueError(f"source changed during preparation: {row.path}")
        if (i + 1) % 200 == 0:
            print(f"Baseline images: {i + 1}/{len(frame)}", flush=True)
    temporary = cache.with_suffix(".npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, pixels, allow_pickle=False)
    temporary.replace(cache)
    _json(receipt, dict(fingerprint=fingerprint, sha256=sha256_file(cache), shape=list(shape)))
    return pixels


def _evaluate(pixels, classes, checkpoints, references, cfg, args, output, fingerprint, device):
    import torch
    from gwproject.qgram_inference import load_gravity_spy_checkpoint
    from gwproject.research.uncertainty import empirical_cdf, energy_score, mahalanobis_score
    results = []
    experiment = cfg["experiment"]
    n = len(pixels)
    order = np.random.default_rng(experiment["seed"]).permutation(n * len(CONDITIONS))
    # The same randomized sample/condition schedule is used for each fixed model.
    pd.DataFrame({"sample_index": order // 4, "condition_index": order % 4}).to_csv(
        output / "inference_schedule.csv", index=False)
    for model_index, (checkpoint, reference) in enumerate(zip(checkpoints, references)):
        path = output / f"model_{model_index:02d}.npz"
        receipt = path.with_suffix(".json")
        if args.resume and receipt.exists():
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            if saved["fingerprint"] != fingerprint or saved["sha256"] != sha256_file(path):
                raise ValueError(f"invalid resume artifact: {path}")
            with np.load(path, allow_pickle=False) as bundle:
                arrays = {key: bundle[key].copy() for key in bundle.files}
            if arrays["logits"].shape != (n, 4, len(classes)):
                raise ValueError("resumed logits have incompatible shape")
            results.append(arrays)
            print(f"Reused verified model {model_index + 1}/{len(checkpoints)}", flush=True)
            continue
        model, model_classes, _ = load_gravity_spy_checkpoint(checkpoint, device)
        if model_classes != classes or not np.isclose(model.gw_temperature, reference["temperature"], rtol=0, atol=1e-12):
            raise ValueError("checkpoint taxonomy or temperature differs from frozen OOD reference")
        classifier = model.fc
        model.fc = torch.nn.Identity()
        if device.type == "cuda":
            model.to(memory_format=torch.channels_last)
        arrays = {"logits": np.empty((n, 4, len(classes)), dtype=np.float32)}
        for key in ("energy", "mahalanobis", "energy_percentile", "mahalanobis_percentile"):
            arrays[key] = np.empty((n, 4), dtype=np.float64)
        checked_scorer = False
        with torch.inference_mode():
            for start in range(0, len(order), args.batch_size):
                indices = order[start:start + args.batch_size]
                sample, condition = indices // 4, indices % 4
                batch = np.stack([intensity_variant(
                    pixels[s], CONDITIONS[c], gamma=experiment["gamma"],
                    lower=experiment["contrast_lower_percentile"],
                    upper=experiment["contrast_upper_percentile"],
                )[0] for s, c in zip(sample, condition)])
                # Same ToTensor -> Normalize arithmetic as the canonical pipeline.
                images = torch.from_numpy(batch.copy()).unsqueeze(1).float().div_(255.0)
                images.sub_(0.5).div_(0.5)
                images = images.to(device)
                if device.type == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)
                with torch.amp.autocast(device_type=device.type,
                                       enabled=device.type == "cuda" and cfg["runtime"]["mixed_precision"]):
                    features = model(images)
                    logits = classifier(features) / float(model.gw_temperature)
                embedding = features.float().cpu().numpy()
                values = logits.float().cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError("nonfinite logits; no samples may be dropped")
                distance = fast_mahalanobis(embedding, reference["reference"])
                if not checked_scorer:
                    canonical = mahalanobis_score(embedding[:2], reference["reference"])
                    np.testing.assert_allclose(distance[:2], canonical, rtol=1e-8, atol=1e-7)
                    checked_scorer = True
                energy = energy_score(values)
                arrays["logits"][sample, condition] = values
                arrays["energy"][sample, condition] = energy
                arrays["mahalanobis"][sample, condition] = distance
                arrays["energy_percentile"][sample, condition] = empirical_cdf(energy, reference["validation_energy"])
                arrays["mahalanobis_percentile"][sample, condition] = empirical_cdf(distance, reference["validation_mahalanobis"])
                if start % (args.batch_size * 100) == 0:
                    print(f"Model {model_index + 1}/{len(checkpoints)}: {start}/{len(order)} paired inputs", flush=True)
        _npz(path, **arrays)
        _json(receipt, dict(fingerprint=fingerprint, sha256=sha256_file(path),
                            checkpoint=str(checkpoint), checkpoint_sha256=reference["checkpoint_sha256"]))
        results.append(arrays)
        del model, classifier, images, features, logits, embedding, values
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def _summarize(frame, pixels, classes, results, references, cfg, renderer_config, output):
    from gwproject.research.uncertainty import softmax_probabilities
    experiment = cfg["experiment"]
    probabilities = np.mean([softmax_probabilities(r["logits"]) for r in results], axis=0)
    winner = probabilities.argmax(axis=-1)
    minimum = float(renderer_config["open_world"]["minimum_ood_model_fraction"])
    energy_rank = np.stack([r["energy_percentile"] for r in results])
    distance_rank = np.stack([r["mahalanobis_percentile"] for r in results])
    # Rank saturation at 1 can hide changes; report a within-model continuous score too.
    log_excess = np.median([
        np.log1p(r["mahalanobis"]) - np.log1p(np.quantile(ref["validation_mahalanobis"], 0.99))
        for r, ref in zip(results, references)
    ], axis=0)
    metrics = {
        "mahalanobis_reject_99": ((distance_rank >= 0.99).mean(axis=0) >= minimum).astype(float),
        "energy_reject_99": ((energy_rank >= 0.99).mean(axis=0) >= minimum).astype(float),
        "mahalanobis_log_excess_99": log_excess,
        "closed_label_changed": (winner != winner[:, :1]).astype(float),
        "max_probability": probabilities.max(axis=-1),
    }
    pieces = []
    for c, name in enumerate(CONDITIONS):
        piece = frame.drop(columns=["path", "source_sha256"]).copy()
        piece["condition"] = name
        piece["forced_closed_set_class"] = np.asarray(classes)[winner[:, c]]
        for key, value in metrics.items():
            piece[key] = value[:, c]
        truth = frame.label.map({label: i for i, label in enumerate(classes)})
        piece["correct_validation_only"] = np.where(truth.notna(), winner[:, c] == truth.to_numpy(), np.nan)
        piece["contrast_degenerate"] = [intensity_variant(
            image, name, gamma=experiment["gamma"],
            lower=experiment["contrast_lower_percentile"],
            upper=experiment["contrast_upper_percentile"],
        )[1] for image in pixels]
        pieces.append(piece)
    detail = pd.concat(pieces, ignore_index=True)
    detail.to_csv(output / "per_sample_condition.csv", index=False)
    event_detail = detail[detail.domain.eq("catalog")].groupby(["event", "condition"], sort=True)[list(metrics)].mean()
    if len(event_detail) != cfg["coverage"]["events"] * 4:
        raise ValueError("incomplete per-event paired results")
    event_detail.to_csv(output / "per_event_condition.csv")
    contrasts = []
    summaries = []
    for domain, part in detail.groupby("domain", sort=True):
        names = list(metrics) + (["correct_validation_only"] if domain == "gravity_spy_val" else [])
        # Detector measurements are first averaged inside each event.
        blocked = part.groupby(["group", "condition"])[names].mean()
        for metric in names:
            wide = blocked[metric].unstack("condition").reindex(columns=CONDITIONS)
            for name, value in wide.mean(axis=0).items():
                summaries.append(dict(domain=domain, metric=metric, condition=name,
                                      mean=float(value), blocks=len(wide)))
            for contrast, deltas in factorial_contrasts(wide.to_numpy()).items():
                contrasts.append(dict(domain=domain, metric=metric, contrast=contrast,
                                      **paired_bootstrap(deltas, seed=experiment["seed"],
                                                         replicates=experiment["bootstrap_replicates"])))
    pd.DataFrame(summaries).to_csv(output / "condition_summary.csv", index=False)
    pd.DataFrame(contrasts).to_csv(output / "paired_contrasts.csv", index=False)
    report = {
        "status": "completed_exploratory_not_confirmatory", "coverage": cfg["coverage"],
        "conditions": list(CONDITIONS), "condition_summaries": summaries,
        "interpretation": [
            "Intensity sensitivity of fixed CNNs only; not paired official-vs-local rendering of the same strain.",
            "Catalog labels are forced diagnostics, never truth; no catalog classification accuracy is defined.",
            "Gravity Spy validation also calibrated OOD: its rates are descriptive, not independent FPR estimates.",
            "Bootstrap conditions on this ensemble; catalog blocks are events, validation blocks are IFO/ID triggers.",
            "Validation triggers may share observing conditions; trigger-level intervals do not remove temporal dependence.",
            "Exploratory pointwise 95% intervals; no multiplicity correction or confirmatory significance claims.",
            "A decreased OOD rate is not physical validation. A null result does not rule out other domain shifts.",
        ],
    }
    _json(output / "summary.json", report)
    lines = ["# Paired intensity experiment — exploratory results", "",
             f"Coverage: {cfg['coverage']['events']} events, {cfg['coverage']['channels']} catalog channels, "
             f"{cfg['coverage']['validation_images']} validation images, {len(results)} fixed models.", "",
             "| Domain | Condition | Mahalanobis rejection 99% (block mean) |",
             "|---|---|---:|"]
    for item in summaries:
        if item["metric"] == "mahalanobis_reject_99":
            lines.append(f"| {item['domain']} | {item['condition']} | {item['mean']:.4f} |")
    lines.extend(["", "## Interpretation limits", ""] + [f"- {line}" for line in report["interpretation"]])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paired_intensity.toml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_toml(args.config)
    if args.batch_size is None:
        args.batch_size = int(cfg["runtime"]["batch_size"])
    if args.batch_size < 1 or cfg["runtime"]["cpu_threads"] < 1:
        raise ValueError("batch size and CPU threads must be positive")
    experiment = cfg["experiment"]
    intensity_variant(np.zeros((1, 1), dtype=np.uint8), "identity",
                      gamma=experiment["gamma"], lower=experiment["contrast_lower_percentile"],
                      upper=experiment["contrast_upper_percentile"])
    if int(experiment["bootstrap_replicates"]) < 100:
        raise ValueError("at least 100 bootstrap replicates required")
    from gwproject.commands.classify_catalog_qgrams_ensemble import _checkpoint_paths, _load_ood_bundle
    from gwproject.gravity_spy import resolve_device, set_deterministic_seed
    import torch
    torch.set_num_threads(int(cfg["runtime"]["cpu_threads"]))
    set_deterministic_seed(int(cfg["experiment"]["seed"]))
    device = resolve_device(args.device)
    paths = {key: Path(value).resolve() for key, value in cfg["inputs"].items()}
    output = args.output_dir.resolve()
    # Never allow a run to be placed in the source, data or canonical result tree.
    allowed = Path("outputs/research").resolve()
    if output.parent != allowed or not output.name.startswith("paired_intensity_"):
        raise ValueError("output must be a direct outputs/research/paired_intensity_* directory")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("nonempty output directory; use --resume with unchanged inputs or a new run")
    if args.resume and not (output / "contract.json").is_file():
        raise ValueError("--resume requires an existing contract.json")
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < cfg["runtime"]["minimum_free_disk_gib"] * 1024**3:
        raise RuntimeError("insufficient free disk space")
    checkpoints = _checkpoint_paths([paths["checkpoints"]])
    if len(checkpoints) != cfg["coverage"]["models"]:
        raise ValueError("expected exactly five checkpoints")
    references = _load_ood_bundle(paths["ood_reference"], checkpoints)
    ood_receipt_path = paths["ood_reference"].with_suffix(".json")
    ood_receipt = json.loads(ood_receipt_path.read_text(encoding="utf-8"))
    if (ood_receipt.get("output_sha256") != sha256_file(paths["ood_reference"])
            or ood_receipt.get("metadata", {}).get("sha256") != sha256_file(paths["metadata"])):
        raise ValueError("OOD reference receipt does not match the bundle/current metadata")
    if len({r["checkpoint_sha256"] for r in references}) != len(references):
        raise ValueError("duplicate checkpoint contents")
    with np.load(paths["ood_reference"], allow_pickle=False) as bundle:
        classes = bundle["classes"].astype(str).tolist()
    if len(classes) != 22 or len(set(classes)) != len(classes):
        raise ValueError("expected the fixed 22-class taxonomy")
    for reference in references:
        for key in ("validation_energy", "validation_mahalanobis"):
            scores = np.asarray(reference[key])
            if scores.shape != (cfg["coverage"]["validation_images"],) or not np.isfinite(scores).all():
                raise ValueError("invalid OOD calibration coverage or nonfinite scores")
    frame = _input_rows(paths, cfg["coverage"])
    if not frame.loc[frame.domain.eq("gravity_spy_val"), "label"].isin(classes).all():
        raise ValueError("validation labels outside checkpoint taxonomy")
    renderer_config = load_toml(paths["render_config"])
    renderer = renderer_config["harmonized_inference"]
    if (int(renderer["resize_height"]), int(renderer["resize_width"])) != (140, 170):
        raise ValueError("this protocol freezes the final image dimensions at 140 x 170")
    if not 0 < float(renderer_config["open_world"]["minimum_ood_model_fraction"]) <= 1:
        raise ValueError("invalid ensemble OOD fraction")
    if renderer_config["open_world"]["mahalanobis_quantiles"][-1] != 0.99 or renderer_config["open_world"]["energy_quantiles"][-1] != 0.99:
        raise ValueError("this protocol freezes the 99% OOD thresholds")
    source_root = Path(__file__).resolve().parents[1]
    contract = dict(
        protocol="paired_intensity_v1", config=cfg, device=str(device), batch_size=args.batch_size,
        python=sys.version, packages=package_versions(), conditions=list(CONDITIONS),
        accelerator=dict(cuda_runtime=torch.version.cuda,
                         cudnn=torch.backends.cudnn.version(),
                         device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"),
        sources={str(p.relative_to(source_root)): sha256_file(p) for p in sorted(source_root.rglob("*.py"))},
        inputs={key: sha256_file(paths[key]) for key in ("approved", "metadata", "ood_reference", "render_config")},
        ood_receipt_sha256=sha256_file(ood_receipt_path),
        checkpoints=[dict(path=str(p), sha256=r["checkpoint_sha256"]) for p, r in zip(checkpoints, references)],
        samples=frame.to_dict(orient="records"),
    )
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    contract_path = output / "contract.json"
    if args.resume:
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous != dict(fingerprint=fingerprint, contract=contract):
            raise ValueError("code/config/data/environment changed; create a new run, do not mix results")
    else:
        _json(contract_path, dict(fingerprint=fingerprint, contract=contract))
    frame.to_csv(output / "inputs.csv", index=False)
    _json(output / "STATUS.json", dict(status="running", fingerprint=fingerprint))
    pixels = _cache_images(frame, renderer_config["harmonized_inference"], output, fingerprint, args.resume)
    results = _evaluate(pixels, classes, checkpoints, references, cfg, args, output, fingerprint, device)
    _summarize(frame, pixels, classes, results, references, cfg, renderer_config, output)
    # Only artifacts of this run are hashed; previous research outputs are untouched.
    hashes = [f"{sha256_file(p)}  {p.name}" for p in sorted(output.iterdir())
              if p.is_file() and p.name not in {"ARTIFACT_HASHES.sha256", "STATUS.json"}
              and not p.name.endswith(".tmp")]
    manifest = output / "ARTIFACT_HASHES.sha256"
    manifest.write_text("\n".join(hashes) + "\n", encoding="utf-8")
    # Final marker is written last; hashing failure cannot leave a complete status.
    _json(output / "STATUS.json", dict(status="complete", fingerprint=fingerprint,
                                        artifact_manifest_sha256=sha256_file(manifest)))
    print(f"Completed. Read {output / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
