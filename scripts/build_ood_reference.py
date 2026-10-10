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
)
from gwproject.qgram_inference import load_gravity_spy_checkpoint
from gwproject.research.uncertainty import energy_score, fit_mahalanobis_reference


def main() -> int:
    parser = argparse.ArgumentParser(description="Constroi referencia OOD no split de validacao Gravity Spy.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("images_root", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/gravity_spy.toml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/research/ood_reference_seed42.npz"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    cfg = load_toml(args.config)
    device = resolve_device(args.device)
    model, classes, profile = load_gravity_spy_checkpoint(args.checkpoint, device)
    metadata = pd.read_csv(args.metadata)
    metadata = attach_image_paths(metadata, index_spectrograms(args.images_root, str(cfg["data"]["image_duration"])))
    split = official_splits(metadata)["val"].copy()
    class_to_idx = {name: index for index, name in enumerate(classes)}
    split = split[split["label"].isin(classes)].copy()
    split["class_idx"] = split["label"].map(class_to_idx).astype(int)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((140, 170)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    loader = DataLoader(
        GravitySpyImageDataset(split, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model.eval()
    original_fc = model.fc
    model.fc = torch.nn.Identity()
    embeddings: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                features = model(images)
                logits = original_fc(features)
            embeddings.append(features.float().cpu().numpy())
            logits_all.append(logits.float().cpu().numpy())
            labels_all.append(labels.numpy())
    model.fc = original_fc
    embedding = np.concatenate(embeddings)
    logits = np.concatenate(logits_all)
    labels = np.concatenate(labels_all)
    reference = fit_mahalanobis_reference(embedding, labels)
    energy = energy_score(logits)
    mahalanobis = np.asarray(
        __import__("gwproject.research.uncertainty", fromlist=["mahalanobis_score"]).mahalanobis_score(
            embedding, reference
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        class_means=reference.class_means,
        precision=reference.precision,
        class_labels=reference.class_labels,
        classes=np.asarray(classes, dtype=str),
        energy_q95=np.quantile(energy, 0.95),
        energy_q99=np.quantile(energy, 0.99),
        mahalanobis_q95=np.quantile(mahalanobis, 0.95),
        mahalanobis_q99=np.quantile(mahalanobis, 0.99),
        checkpoint=str(args.checkpoint),
    )
    summary = {
        "samples": int(len(labels)),
        "embedding_dimension": int(embedding.shape[1]),
        "energy_q95": float(np.quantile(energy, 0.95)),
        "energy_q99": float(np.quantile(energy, 0.99)),
        "mahalanobis_q95": float(np.quantile(mahalanobis, 0.95)),
        "mahalanobis_q99": float(np.quantile(mahalanobis, 0.99)),
        "checkpoint": str(args.checkpoint),
        "output": str(args.output),
        "profile": profile,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
