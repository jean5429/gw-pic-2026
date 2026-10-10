from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gwproject.config import load_toml
from gwproject.gravity_spy import resolve_device
from gwproject.provenance import write_manifest
from gwproject.qgram_inference import CatalogQgramDataset, load_gravity_spy_checkpoint
from gwproject.research.uncertainty import (
    MahalanobisReference,
    ensemble_statistics,
    mahalanobis_score,
)


def _checkpoint_paths(values: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        if value.is_dir():
            paths.extend(sorted(value.rglob("best_model.pt")))
        else:
            paths.append(value)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            unique.append(path)
            seen.add(key)
    if not unique:
        raise ValueError("nenhum checkpoint encontrado")
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inferencia ensemble, harmonizada e open-world sobre Q-grams do catalogo."
    )
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--approved", type=Path, default=Path("outputs/gwtc/approved_channels.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/research_program.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research/gravity_spy_ensemble"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--ood-reference", type=Path, default=None)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    checkpoints = _checkpoint_paths(args.checkpoints)
    cfg_all = load_toml(args.config)
    cfg = cfg_all["harmonized_inference"]
    device = resolve_device(args.device)
    loaded = [load_gravity_spy_checkpoint(path, device) for path in checkpoints]
    classes = loaded[0][1]
    for _, model_classes, _ in loaded[1:]:
        if model_classes != classes:
            raise ValueError("checkpoints com taxonomias diferentes")
    models = [item[0] for item in loaded]
    approved = pd.read_csv(args.approved)
    approved = approved[approved["qgram_path"].notna()].reset_index(drop=True)
    transform = transforms.Compose([
        transforms.Resize((int(cfg["resize_height"]), int(cfg["resize_width"]))),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    loader = DataLoader(
        CatalogQgramDataset(
            approved,
            transform,
            lower_percentile=float(cfg["lower_percentile"]),
            upper_percentile=float(cfg["upper_percentile"]),
            source_duration_seconds=float(cfg["source_outseg_seconds"]),
            target_duration_seconds=float(cfg["target_duration_seconds"]),
            baseline_mode=str(cfg["baseline_mode"]),
            flip_frequency=bool(cfg["flip_frequency"]),
            render_mode=str(cfg.get("render_mode", "percentile")),
        ),
        batch_size=int(args.batch_size or cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg["num_workers"]) > 0,
    )
    for model in models:
        model.eval()
        if device.type == "cuda":
            model.to(memory_format=torch.channels_last)

    ood = None
    if args.ood_reference:
        data = np.load(args.ood_reference, allow_pickle=False)
        ood = {
            "reference": MahalanobisReference(
                class_means=data["class_means"],
                precision=data["precision"],
                class_labels=data["class_labels"],
            ),
            "energy_q95": float(data["energy_q95"]),
            "energy_q99": float(data["energy_q99"]),
            "mahalanobis_q95": float(data["mahalanobis_q95"]),
            "mahalanobis_q99": float(data["mahalanobis_q99"]),
        }

    rows: list[dict] = []
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            logits_models: list[np.ndarray] = []
            first_embeddings = None
            for model_index, model in enumerate(models):
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    if model_index == 0 and ood is not None:
                        original_fc = model.fc
                        model.fc = torch.nn.Identity()
                        features = model(images)
                        logits = original_fc(features)
                        model.fc = original_fc
                        first_embeddings = features.float().cpu().numpy()
                    else:
                        logits = model(images)
                logits_models.append(logits.float().cpu().numpy())
            stack = np.stack(logits_models, axis=0)
            stats = ensemble_statistics(stack)
            probabilities = stats["mean_probability"]
            mahal = mahalanobis_score(first_embeddings, ood["reference"]) if ood is not None else np.full(len(indices), np.nan)
            for batch_index, source_index in enumerate(indices.numpy().tolist()):
                probability = probabilities[batch_index]
                order = np.argsort(probability)[::-1]
                source = approved.iloc[int(source_index)]
                energy_value = float(stats["energy_mean"][batch_index])
                mahal_value = float(mahal[batch_index])
                row = {
                    "event": source["event"],
                    "detector": source["detector"],
                    "qgram_path": source["qgram_path"],
                    "predicted_class": classes[int(order[0])],
                    "ensemble_confidence": float(probability[order[0]]),
                    "second_class": classes[int(order[1])],
                    "second_probability": float(probability[order[1]]),
                    "predictive_entropy": float(stats["predictive_entropy"][batch_index]),
                    "expected_entropy": float(stats["expected_entropy"][batch_index]),
                    "mutual_information": float(stats["mutual_information"][batch_index]),
                    "vote_fraction": float(stats["vote_fraction"][batch_index]),
                    "probability_margin": float(stats["probability_margin"][batch_index]),
                    "energy_mean": energy_value,
                    "energy_std": float(stats["energy_std"][batch_index]),
                    "mahalanobis_distance": mahal_value,
                    "ood_energy_95": bool(ood is not None and energy_value > ood["energy_q95"]),
                    "ood_energy_99": bool(ood is not None and energy_value > ood["energy_q99"]),
                    "ood_mahalanobis_95": bool(ood is not None and mahal_value > ood["mahalanobis_q95"]),
                    "ood_mahalanobis_99": bool(ood is not None and mahal_value > ood["mahalanobis_q99"]),
                }
                row["open_world_status"] = (
                    "reject_ood"
                    if row["ood_energy_99"] or row["ood_mahalanobis_99"]
                    else "review"
                    if row["ood_energy_95"] or row["ood_mahalanobis_95"] or row["vote_fraction"] < float(cfg["minimum_vote_fraction"])
                    else "in_domain_candidate"
                )
                rows.append(row)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["event", "detector"])
    frame.to_csv(output / "per_channel.csv", index=False)
    event_rows = []
    for event, group in frame.groupby("event"):
        event_rows.append({
            "event": event,
            "n_channels": int(len(group)),
            "detectors": ",".join(group["detector"].astype(str)),
            "classes": ",".join(group["predicted_class"].astype(str)),
            "detector_class_disagreement": bool(group["predicted_class"].nunique() > 1),
            "maximum_ood_severity": "reject_ood" if (group["open_world_status"] == "reject_ood").any() else "review" if (group["open_world_status"] == "review").any() else "in_domain_candidate",
            "mean_vote_fraction": float(group["vote_fraction"].mean()),
            "mean_mutual_information": float(group["mutual_information"].mean()),
        })
    pd.DataFrame(event_rows).to_csv(output / "per_event.csv", index=False)
    summary = {
        "checkpoints": [str(p) for p in checkpoints],
        "n_models": len(models),
        "channels": int(len(frame)),
        "events": int(frame["event"].nunique()),
        "class_counts": frame["predicted_class"].value_counts().to_dict(),
        "open_world_counts": frame["open_world_status"].value_counts().to_dict(),
        "detector_disagreement_events": int(pd.DataFrame(event_rows)["detector_class_disagreement"].sum()),
        "preprocessing": {key: cfg[key] for key in cfg},
        "interpretation_warning": (
            "in_domain_candidate nao e prova de classe fisica. OOD e ensemble medem compatibilidade "
            "estatistica com o dominio harmonizado, nao origem causal do transiente."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_manifest(output / "manifest.json", {"config": cfg_all, "summary": summary})
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
