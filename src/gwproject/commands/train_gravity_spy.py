from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gwproject.config import load_toml
from gwproject.gravity_spy import (
    assert_real_dataset,
    attach_image_paths,
    build_resnet18_grayscale,
    choose_profile,
    freeze_backbone,
    index_spectrograms,
    official_splits,
    resolve_device,
    set_deterministic_seed,
    validate_harmonized_dataset,
    GravitySpyImageDataset,
    RandomIntensityScale,
)
from gwproject.plotting import FIGURE_DPI, plot_confusion_matrix
from gwproject.provenance import accelerator_manifest, sha256_file, write_manifest
from gwproject.research.uncertainty import (
    expected_calibration_error,
    fit_temperature,
    multiclass_brier_score,
    negative_log_likelihood,
    softmax_probabilities,
    temperature_scale_logits,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Treina ResNet-18 no conjunto Gravity Spy")
    parser.add_argument("metadata", type=Path)
    parser.add_argument("images_root", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/gravity_spy.toml"))
    parser.add_argument("--profile", default=None, help="auto, cpu, cuda_balanced ou cuda_highres")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--smoke-per-class", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument(
        "--allow-unharmonized",
        action="store_true",
        help="somente para reproducao legada; desativa a barreira de QC canônica",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/gravity_spy_harmonized_v1d1"))
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from PIL import Image
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            classification_report,
            cohen_kappa_score,
            confusion_matrix,
            f1_score,
        )
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Instale scikit-learn/Pillow com pip install -e '.[ml]' e PyTorch separadamente."
        ) from exc

    cfg = load_toml(args.config)
    training_cfg = cfg["training"]
    evaluation_cfg = cfg["evaluation"]
    seed = int(args.seed if args.seed is not None else training_cfg["seeds"][0])
    epochs = int(args.epochs if args.epochs is not None else training_cfg["epochs"])
    device = resolve_device(args.device)
    requested_profile = args.profile or str(training_cfg["default_profile"])
    profile = choose_profile(cfg, device.type, requested=requested_profile)
    batch_size = int(args.batch_size if args.batch_size is not None else profile.batch_size)
    num_workers = int(args.num_workers if args.num_workers is not None else profile.num_workers)
    pretrained = bool(training_cfg["pretrained"]) and not args.no_pretrained
    run_baseline = bool(training_cfg["run_baseline"]) and not args.no_baseline
    deterministic = bool(training_cfg["deterministic"])

    set_deterministic_seed(seed, deterministic=deterministic)
    if device.type == "cuda":
        # TF32 e desativado no modo deterministico para evitar uma promessa
        # de reprodutibilidade incompatível com kernels aproximados.
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
        torch.backends.cudnn.allow_tf32 = not deterministic

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / f"seed_{seed}_{profile.name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_validation = None
    if not args.allow_unharmonized:
        dataset_validation = validate_harmonized_dataset(
            args.images_root, args.metadata
        )
    metadata = pd.read_csv(args.metadata)
    image_index = index_spectrograms(args.images_root, str(cfg["data"]["image_duration"]))
    metadata = attach_image_paths(metadata, image_index)
    assert_real_dataset(metadata["path"])
    split_frames = official_splits(metadata)
    for required in ("train", "val", "test"):
        if required not in split_frames or split_frames[required].empty:
            raise RuntimeError(f"split oficial ausente ou vazio: {required}")

    classes = sorted(metadata["label"].dropna().unique())
    class_to_idx = {name: index for index, name in enumerate(classes)}
    for name, frame in list(split_frames.items()):
        frame = frame.copy()
        frame["class_idx"] = frame["label"].map(class_to_idx).astype(int)
        if args.smoke_per_class > 0:
            frame = (
                frame.groupby("label", group_keys=False)
                .head(args.smoke_per_class)
                .reset_index(drop=True)
            )
        split_frames[name] = frame
    split_membership = pd.concat(
        [
            frame.assign(split=name)[
                ["ifo", "gravityspy_id", "label", "split", "path"]
            ]
            for name, frame in split_frames.items()
        ],
        ignore_index=True,
    )
    split_membership.to_csv(run_dir / "split_membership.csv", index=False)

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((140, 170)),
        transforms.ToTensor(),
        RandomIntensityScale(),
        transforms.Normalize([0.5], [0.5]),
    ])
    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((140, 170)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        GravitySpyImageDataset(split_frames["train"], train_transform),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        GravitySpyImageDataset(split_frames["val"], eval_transform),
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        GravitySpyImageDataset(split_frames["test"], eval_transform),
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model = build_resnet18_grayscale(
        len(classes),
        pretrained=pretrained,
        conv1_stride=profile.conv1_stride,
        use_maxpool=profile.use_maxpool,
    ).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    freeze_backbone(model, True)

    train_counts = np.bincount(
        split_frames["train"]["class_idx"].to_numpy(), minlength=len(classes)
    ).astype(float)
    class_weights = train_counts.sum() / np.maximum(train_counts, 1.0)
    class_weights /= np.mean(class_weights)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    lr_head = float(training_cfg["learning_rate_head"])
    lr_backbone = float(training_cfg["learning_rate_backbone"])
    weight_decay = float(training_cfg["weight_decay"])
    freeze_epochs = int(training_cfg["freeze_epochs"])
    patience = int(training_cfg["patience"])

    def optimizer_for(frozen: bool):
        if frozen:
            return torch.optim.AdamW(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=lr_head,
                weight_decay=weight_decay,
            )
        backbone = [parameter for name, parameter in model.named_parameters() if not name.startswith("fc.")]
        head = [parameter for name, parameter in model.named_parameters() if name.startswith("fc.")]
        return torch.optim.AdamW(
            [
                {"params": backbone, "lr": lr_backbone},
                {"params": head, "lr": lr_head},
            ],
            weight_decay=weight_decay,
        )

    optimizer = optimizer_for(True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    amp_enabled = bool(profile.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    def move_images(images):
        images = images.to(device, non_blocking=device.type == "cuda")
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        return images

    def evaluate(loader):
        model.eval()
        labels_all: list[int] = []
        predictions_all: list[int] = []
        logits_all: list[np.ndarray] = []
        losses: list[float] = []
        with torch.no_grad():
            for images, labels in loader:
                images = move_images(images)
                labels = labels.to(device, non_blocking=device.type == "cuda")
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(images)
                    loss = criterion(logits, labels)
                losses.append(float(loss.item()) * len(labels))
                labels_all.extend(labels.cpu().numpy().tolist())
                predictions_all.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                logits_all.append(logits.float().cpu().numpy())
        y_true = np.asarray(labels_all, dtype=int)
        y_pred = np.asarray(predictions_all, dtype=int)
        logits_array = np.concatenate(logits_all, axis=0)
        return {
            "loss": sum(losses) / max(len(loader.dataset), 1),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "kappa": float(cohen_kappa_score(y_true, y_pred)),
            "y_true": y_true,
            "y_pred": y_pred,
            "logits": logits_array,
        }

    history: list[dict] = []
    best_f1 = -np.inf
    patience_count = 0
    checkpoint = run_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        if epoch == freeze_epochs + 1:
            freeze_backbone(model, False)
            optimizer = optimizer_for(False)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=2
            )

        model.train()
        running_loss = 0.0
        train_true: list[int] = []
        train_pred: list[int] = []
        for images, labels in train_loader:
            images = move_images(images)
            labels = labels.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * len(labels)
            train_true.extend(labels.cpu().numpy().tolist())
            train_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

        validation = evaluate(val_loader)
        scheduler.step(validation["macro_f1"])
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(len(train_loader.dataset), 1),
            "train_macro_f1": float(
                f1_score(train_true, train_pred, average="macro", zero_division=0)
            ),
            "val_loss": validation["loss"],
            "val_macro_f1": validation["macro_f1"],
            "val_balanced_accuracy": validation["balanced_accuracy"],
            "val_kappa": validation["kappa"],
            "stage": "conv1_bn1_head" if epoch <= freeze_epochs else "fine_tune",
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if validation["macro_f1"] > best_f1 + 1e-6:
            best_f1 = validation["macro_f1"]
            patience_count = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": classes,
                    "class_to_idx": class_to_idx,
                    "seed": seed,
                    "pretrained": pretrained,
                    "profile": profile.__dict__,
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                },
                checkpoint,
            )
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["state_dict"])
    validation_final = evaluate(val_loader)
    test = evaluate(test_loader)

    temperature = fit_temperature(
        validation_final["logits"], validation_final["y_true"]
    )
    raw_probabilities = softmax_probabilities(test["logits"])
    calibrated_probabilities = softmax_probabilities(
        temperature_scale_logits(test["logits"], temperature)
    )
    calibrated_predictions = np.argmax(calibrated_probabilities, axis=1)
    calibration_metrics = {
        "temperature": temperature,
        "raw_ece": expected_calibration_error(raw_probabilities, test["y_true"]),
        "calibrated_ece": expected_calibration_error(
            calibrated_probabilities, test["y_true"]
        ),
        "raw_brier": multiclass_brier_score(raw_probabilities, test["y_true"]),
        "calibrated_brier": multiclass_brier_score(
            calibrated_probabilities, test["y_true"]
        ),
        "raw_nll": negative_log_likelihood(raw_probabilities, test["y_true"]),
        "calibrated_nll": negative_log_likelihood(
            calibrated_probabilities, test["y_true"]
        ),
    }
    np.savez_compressed(
        run_dir / "test_predictions.npz",
        ifo=split_frames["test"]["ifo"].astype(str).to_numpy(),
        gravityspy_id=split_frames["test"]["gravityspy_id"].astype(str).to_numpy(),
        label=split_frames["test"]["label"].astype(str).to_numpy(),
        y_true=test["y_true"],
        y_pred=test["y_pred"],
        calibrated_y_pred=calibrated_predictions,
        logits=test["logits"],
        probabilities=raw_probabilities,
        calibrated_probabilities=calibrated_probabilities,
        classes=np.asarray(classes, dtype=str),
        temperature=np.asarray(temperature),
    )

    # A temperatura pertence ao artefato do modelo, nao apenas ao relatorio.
    saved["temperature"] = float(temperature)
    saved["metadata_sha256"] = sha256_file(args.metadata)
    saved["preprocessing_summary_sha256"] = (
        sha256_file(args.images_root / "preprocessing_summary.json")
        if (args.images_root / "preprocessing_summary.json").is_file()
        else None
    )
    saved["qc_summary_sha256"] = (
        sha256_file(args.images_root / "qc_summary.json")
        if (args.images_root / "qc_summary.json").is_file()
        else None
    )
    saved["input_contract"] = {
        "image_height": 140,
        "image_width": 170,
        "channels": 1,
        "normalization_mean": [0.5],
        "normalization_std": [0.5],
        "panel_crop_mode": "official" if not args.allow_unharmonized else "legacy_override",
    }
    torch.save(saved, checkpoint)

    report = classification_report(
        test["y_true"],
        test["y_pred"],
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
        output_dict=True,
    )
    report_frame = pd.DataFrame(report).T
    report_frame.to_csv(run_dir / "classification_report.csv")
    per_class_f1 = {
        class_name: float(report[class_name]["f1-score"]) for class_name in classes
    }
    cm = confusion_matrix(
        test["y_true"], test["y_pred"], labels=list(range(len(classes)))
    )
    np.save(run_dir / "confusion_matrix.npy", cm)
    plot_confusion_matrix(cm, classes, run_dir / "confusion_matrix.png")

    confidence = np.max(calibrated_probabilities, axis=1)
    correctness = calibrated_predictions == test["y_true"]
    bin_edges = np.linspace(0.0, 1.0, 16)
    bin_centers: list[float] = []
    bin_accuracy: list[float] = []
    bin_confidence: list[float] = []
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidence > left) & (confidence <= right)
        if np.any(mask):
            bin_centers.append(float((left + right) / 2.0))
            bin_accuracy.append(float(correctness[mask].mean()))
            bin_confidence.append(float(confidence[mask].mean()))
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", label="calibracao ideal")
    axis.plot(bin_confidence, bin_accuracy, marker="o", label="modelo calibrado")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Confianca media")
    axis.set_ylabel("Acuracia empirica")
    axis.set_title("Diagrama de confiabilidade")
    axis.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "reliability_diagram.png", dpi=FIGURE_DPI)
    plt.close(fig)

    baseline_metrics = None
    if run_baseline:
        def downsample(frame: pd.DataFrame):
            features = np.empty((len(frame), 32 * 32), dtype=np.float32)
            labels = frame["class_idx"].to_numpy()
            for index, path in enumerate(frame["path"]):
                with Image.open(path) as image_file:
                    image_file = image_file.convert("L").resize((32, 32))
                    features[index] = np.asarray(image_file, dtype=np.float32).ravel() / 255.0
            return features, labels

        xb_train, yb_train = downsample(split_frames["train"])
        xb_test, yb_test = downsample(split_frames["test"])
        baseline = LogisticRegression(
            max_iter=2000,
            C=0.1,
            class_weight="balanced",
            random_state=seed,
        )
        baseline.fit(xb_train, yb_train)
        yb_pred = baseline.predict(xb_test)
        baseline_metrics = {
            "macro_f1": float(f1_score(yb_test, yb_pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(yb_test, yb_pred)),
            "kappa": float(cohen_kappa_score(yb_test, yb_pred)),
            "accuracy": float(accuracy_score(yb_test, yb_pred)),
        }

    test_metrics = {
        "macro_f1": test["macro_f1"],
        "balanced_accuracy": test["balanced_accuracy"],
        "kappa": test["kappa"],
        "accuracy": float(accuracy_score(test["y_true"], test["y_pred"])),
        "minimum_per_class_f1": float(min(per_class_f1.values())),
        "calibrated_ece": calibration_metrics["calibrated_ece"],
        "calibrated_brier": calibration_metrics["calibrated_brier"],
        "calibrated_nll": calibration_metrics["calibrated_nll"],
    }
    bootstrap_rng = np.random.default_rng(seed + 100_003)
    bootstrap_macro_f1: list[float] = []
    bootstrap_accuracy: list[float] = []
    n_test = len(test["y_true"])
    for _ in range(2000):
        indices = bootstrap_rng.integers(0, n_test, size=n_test)
        bootstrap_macro_f1.append(
            float(
                f1_score(
                    test["y_true"][indices],
                    test["y_pred"][indices],
                    average="macro",
                    zero_division=0,
                )
            )
        )
        bootstrap_accuracy.append(
            float(accuracy_score(test["y_true"][indices], test["y_pred"][indices]))
        )
    uncertainty_intervals = {
        "method": "nonparametric_bootstrap_test_samples",
        "replicates": 2000,
        "confidence_level": 0.95,
        "macro_f1": {
            "lower": float(np.quantile(bootstrap_macro_f1, 0.025)),
            "upper": float(np.quantile(bootstrap_macro_f1, 0.975)),
        },
        "accuracy": {
            "lower": float(np.quantile(bootstrap_accuracy, 0.025)),
            "upper": float(np.quantile(bootstrap_accuracy, 0.975)),
        },
    }
    gates = {
        "macro_f1": test_metrics["macro_f1"] >= float(evaluation_cfg["macro_f1_min"]),
        "balanced_accuracy": test_metrics["balanced_accuracy"] >= float(evaluation_cfg["balanced_accuracy_min"]),
        "kappa": test_metrics["kappa"] >= float(evaluation_cfg["kappa_min"]),
        "per_class_f1": test_metrics["minimum_per_class_f1"] >= float(evaluation_cfg["per_class_f1_min"]),
        "calibration": calibration_metrics["calibrated_ece"] <= float(evaluation_cfg["ece_max"]),
    }
    summary = {
        "seed": seed,
        "device": str(device),
        "accelerator": accelerator_manifest(),
        "profile": profile.__dict__,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "amp_enabled": amp_enabled,
        "classes": classes,
        "n_classes": len(classes),
        "split_sizes": {name: len(frame) for name, frame in split_frames.items()},
        "pretrained": pretrained,
        "synthetic_data_allowed": False,
        "harmonized_dataset_required": not args.allow_unharmonized,
        "dataset_validation": dataset_validation,
        "metadata": {
            "path": str(args.metadata),
            "sha256": sha256_file(args.metadata),
        },
        "images_root": str(args.images_root),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "contains_temperature_calibration": True,
            "contains_optimizer_scheduler_scaler": True,
        },
        "best_epoch": int(saved["epoch"]),
        "best_val_macro_f1": float(saved["val_macro_f1"]),
        "test_metrics": test_metrics,
        "per_class_f1": per_class_f1,
        "calibration": calibration_metrics,
        "uncertainty_intervals": uncertainty_intervals,
        "baseline_metrics": baseline_metrics,
        "gates": gates,
        "overall_passed": all(gates.values()),
    }
    if baseline_metrics is not None:
        summary["cnn_minus_baseline_macro_f1"] = (
            test_metrics["macro_f1"] - baseline_metrics["macro_f1"]
        )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_manifest(run_dir / "manifest.json", {"config": cfg, **summary})
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["overall_passed"] or args.smoke_per_class > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
