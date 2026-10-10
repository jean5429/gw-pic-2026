from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gwproject.config import load_toml
from gwproject.gravity_spy import (
    GravitySpyImageDataset,
    attach_image_paths,
    index_spectrograms,
    official_splits,
    resolve_device,
    validate_harmonized_dataset,
)
from gwproject.provenance import sha256_file, write_manifest
from gwproject.qgram_inference import load_gravity_spy_checkpoint
from gwproject.research.uncertainty import (
    energy_score,
    fit_mahalanobis_reference,
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
        resolved = str(path.resolve())
        if resolved not in seen:
            if not path.is_file():
                raise FileNotFoundError(path)
            unique.append(path)
            seen.add(resolved)
    if not unique:
        raise ValueError("nenhum checkpoint encontrado")
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Constroi referencias OOD independentes por checkpoint: ajusta a "
            "geometria Mahalanobis no treino e calibra os escores na validacao."
        )
    )
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/gravity_spy.toml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/research/ood_reference.npz"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    checkpoints = _checkpoint_paths(args.checkpoints)
    cfg = load_toml(args.config)
    device = resolve_device(args.device)
    dataset_validation = validate_harmonized_dataset(
        args.images_root, args.metadata
    )

    loaded = [load_gravity_spy_checkpoint(path, device) for path in checkpoints]
    classes = loaded[0][1]
    for _, model_classes, _ in loaded[1:]:
        if model_classes != classes:
            raise ValueError("checkpoints com taxonomias diferentes")

    metadata = pd.read_csv(args.metadata)
    metadata = attach_image_paths(
        metadata,
        index_spectrograms(args.images_root, str(cfg["data"]["image_duration"])),
    )
    splits = official_splits(metadata)
    fit_split = splits["train"].copy()
    calibration_split = splits["val"].copy()
    class_to_idx = {name: index for index, name in enumerate(classes)}
    fit_split = fit_split[fit_split["label"].isin(classes)].copy()
    calibration_split = calibration_split[
        calibration_split["label"].isin(classes)
    ].copy()
    fit_split["class_idx"] = fit_split["label"].map(class_to_idx).astype(int)
    calibration_split["class_idx"] = (
        calibration_split["label"].map(class_to_idx).astype(int)
    )
    if fit_split.empty:
        raise RuntimeError("split de treino vazio apos alinhar a taxonomia")
    if calibration_split.empty:
        raise RuntimeError("split de validacao vazio apos alinhar a taxonomia")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((140, 170)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    fit_loader = DataLoader(
        GravitySpyImageDataset(fit_split, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    calibration_loader = DataLoader(
        GravitySpyImageDataset(calibration_split, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    class_means: list[np.ndarray] = []
    precisions: list[np.ndarray] = []
    validation_energy: list[np.ndarray] = []
    validation_mahalanobis: list[np.ndarray] = []
    profiles: list[dict] = []
    labels_reference: np.ndarray | None = None

    for (model, _, profile), checkpoint in zip(loaded, checkpoints):
        model.eval()
        original_fc = model.fc
        model.fc = torch.nn.Identity()
        fit_embeddings: list[np.ndarray] = []
        fit_labels: list[np.ndarray] = []
        calibration_embeddings: list[np.ndarray] = []
        calibration_logits: list[np.ndarray] = []
        with torch.no_grad():
            for images, labels in fit_loader:
                images = images.to(device, non_blocking=device.type == "cuda")
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    features = model(images)
                fit_embeddings.append(features.float().cpu().numpy())
                fit_labels.append(labels.numpy())
            for images, _ in calibration_loader:
                images = images.to(device, non_blocking=device.type == "cuda")
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    features = model(images)
                    logits = original_fc(features) / float(model.gw_temperature)
                calibration_embeddings.append(features.float().cpu().numpy())
                calibration_logits.append(logits.float().cpu().numpy())
        model.fc = original_fc

        reference = fit_mahalanobis_reference(
            np.concatenate(fit_embeddings),
            np.concatenate(fit_labels),
        )
        calibration_embedding = np.concatenate(calibration_embeddings)
        calibration_logit = np.concatenate(calibration_logits)
        energy = energy_score(calibration_logit)
        mahalanobis = mahalanobis_score(calibration_embedding, reference)

        if labels_reference is None:
            labels_reference = reference.class_labels
        elif not np.array_equal(labels_reference, reference.class_labels):
            raise RuntimeError("classes da referencia Mahalanobis diferem entre checkpoints")

        class_means.append(reference.class_means)
        precisions.append(reference.precision)
        validation_energy.append(energy)
        validation_mahalanobis.append(mahalanobis)
        profiles.append(profile)
        print(f"referencia concluida: {checkpoint}")

    checkpoint_hashes = [sha256_file(path) for path in checkpoints]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        format_version=np.asarray("4"),
        reference_fit_split=np.asarray("train"),
        calibration_split=np.asarray("val"),
        checkpoint_paths=np.asarray([str(path) for path in checkpoints], dtype=str),
        checkpoint_sha256=np.asarray(checkpoint_hashes, dtype=str),
        classes=np.asarray(classes, dtype=str),
        class_labels=np.asarray(labels_reference, dtype=np.int64),
        class_means=np.stack(class_means, axis=0),
        precision=np.stack(precisions, axis=0),
        validation_energy=np.stack(validation_energy, axis=0),
        validation_mahalanobis=np.stack(validation_mahalanobis, axis=0),
        temperatures=np.asarray(
            [float(model.gw_temperature) for model, _, _ in loaded],
            dtype=float,
        ),
    )

    summary = {
        "format_version": 4,
        "models": len(checkpoints),
        "reference_fit_split": "train",
        "reference_fit_samples_per_model": int(len(fit_split)),
        "calibration_split": "val",
        "calibration_samples_per_model": int(len(calibration_split)),
        "samples_per_model": int(len(calibration_split)),
        "embedding_dimension": int(class_means[0].shape[1]),
        "checkpoints": [
            {"path": str(path), "sha256": digest, "profile": profile}
            for path, digest, profile in zip(checkpoints, checkpoint_hashes, profiles)
        ],
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "metadata": {
            "path": str(args.metadata),
            "sha256": sha256_file(args.metadata),
        },
        "images_root": str(args.images_root),
        "dataset_validation": dataset_validation,
        "temperature_calibration_applied": True,
        "calibration": (
            "Cada checkpoint possui media e precisao ajustadas no split de treino, "
            "enquanto as distribuicoes empiricas de calibracao usam exclusivamente "
            "o split de validacao. Nenhum embedding ou limiar bruto e transferido "
            "entre modelos; o split de teste permanece controle final in-domain."
        ),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_manifest(args.output.with_suffix(".manifest.json"), {"ood_reference": summary})
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
