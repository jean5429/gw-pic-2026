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
    MahalanobisReference,
    empirical_cdf,
    energy_score,
    mahalanobis_score,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Valida a calibracao OOD no split oficial de teste, que nao participa "
            "do ajuste Mahalanobis nem da calibracao empirica."
        )
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gravity_spy.toml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/research/ood_test_control.json"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    validate_harmonized_dataset(args.images_root, args.metadata)
    cfg = load_toml(args.config)
    device = resolve_device(args.device)
    with np.load(args.reference) as bundle:
        if str(bundle["format_version"].item()) != "4":
            raise ValueError("o controle independente exige referencia OOD formato 4")
        if str(bundle["reference_fit_split"].item()) != "train":
            raise ValueError("a geometria Mahalanobis deve ser ajustada no treino")
        if str(bundle["calibration_split"].item()) != "val":
            raise ValueError("os percentis OOD devem ser calibrados na validacao")
        checkpoint_paths = [Path(value) for value in bundle["checkpoint_paths"].tolist()]
        expected_hashes = bundle["checkpoint_sha256"].astype(str).tolist()
        reference_classes = bundle["classes"].astype(str).tolist()
        class_labels = bundle["class_labels"].copy()
        class_means = bundle["class_means"].copy()
        precisions = bundle["precision"].copy()
        validation_energy = bundle["validation_energy"].copy()
        validation_mahalanobis = bundle["validation_mahalanobis"].copy()

    for path, expected_hash in zip(checkpoint_paths, expected_hashes):
        if sha256_file(path) != expected_hash:
            raise ValueError(f"hash do checkpoint nao corresponde a referencia: {path}")

    metadata = pd.read_csv(args.metadata)
    metadata = attach_image_paths(
        metadata,
        index_spectrograms(args.images_root, str(cfg["data"]["image_duration"])),
    )
    test = official_splits(metadata)["test"].copy()
    class_to_idx = {name: index for index, name in enumerate(reference_classes)}
    test = test[test["label"].isin(reference_classes)].copy()
    test["class_idx"] = test["label"].map(class_to_idx).astype(int)
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((140, 170)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    loader = DataLoader(
        GravitySpyImageDataset(test, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    models: list[dict] = []
    for index, checkpoint in enumerate(checkpoint_paths):
        model, classes, _ = load_gravity_spy_checkpoint(checkpoint, device)
        if classes != reference_classes:
            raise ValueError(f"taxonomia divergente no checkpoint {checkpoint}")
        original_fc = model.fc
        model.fc = torch.nn.Identity()
        embeddings: list[np.ndarray] = []
        logits: list[np.ndarray] = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device, non_blocking=device.type == "cuda")
                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=device.type == "cuda",
                ):
                    features = model(images)
                    output = original_fc(features) / float(model.gw_temperature)
                embeddings.append(features.float().cpu().numpy())
                logits.append(output.float().cpu().numpy())
        model.fc = original_fc

        embedding = np.concatenate(embeddings)
        logit = np.concatenate(logits)
        reference = MahalanobisReference(
            class_means=class_means[index],
            precision=precisions[index],
            class_labels=class_labels,
        )
        energy_percentile = empirical_cdf(
            energy_score(logit),
            validation_energy[index],
        )
        mahalanobis_percentile = empirical_cdf(
            mahalanobis_score(embedding, reference),
            validation_mahalanobis[index],
        )
        models.append(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": expected_hashes[index],
                "test_samples": int(len(test)),
                "energy_fraction_ge_0_95": float(np.mean(energy_percentile >= 0.95)),
                "energy_fraction_ge_0_99": float(np.mean(energy_percentile >= 0.99)),
                "mahalanobis_fraction_ge_0_95": float(
                    np.mean(mahalanobis_percentile >= 0.95)
                ),
                "mahalanobis_fraction_ge_0_99": float(
                    np.mean(mahalanobis_percentile >= 0.99)
                ),
                "mahalanobis_percentile_median": float(
                    np.median(mahalanobis_percentile)
                ),
            }
        )

    summary = {
        "format_version": 1,
        "reference": str(args.reference),
        "reference_sha256": sha256_file(args.reference),
        "reference_fit_split": "train",
        "calibration_split": "val",
        "control_split": "test",
        "control_samples": int(len(test)),
        "expected_tail_fractions": {"0.95": 0.05, "0.99": 0.01},
        "models": models,
        "interpretation": (
            "As fracoes sao controles descritivos in-domain. Proximidade aos valores "
            "esperados sustenta a calibracao empirica; nao valida poder OOD universal."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_manifest(args.output.with_suffix(".manifest.json"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
