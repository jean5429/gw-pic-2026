from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gwproject.config import load_toml
from gwproject.gravity_spy import resolve_device
from gwproject.provenance import sha256_file, write_manifest
from gwproject.qgram_inference import (
    CatalogQgramDataset,
    load_gravity_spy_checkpoint,
    probability_entropy,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aplica um checkpoint Gravity Spy aos Q-grams aprovados")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--approved", type=Path, default=Path("outputs/gwtc4_218/approved_channels.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/research_program.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research/gwtc4_218/gravity_spy_single"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
        from torchvision import transforms
    except ImportError as exc:
        raise SystemExit("Instale PyTorch, Torchvision e Pillow para classificar Q-grams") from exc

    config = load_toml(args.config)
    cfg = config["harmonized_inference"]
    device = resolve_device(args.device)
    model, classes, profile = load_gravity_spy_checkpoint(args.checkpoint, device)
    approved = pd.read_csv(args.approved)
    approved = approved[approved["qgram_path"].notna()].reset_index(drop=True)

    transform = transforms.Compose(
        [
            transforms.Resize((int(cfg["resize_height"]), int(cfg["resize_width"]))),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    batch_size = args.batch_size or int(cfg["batch_size"])
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
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg["num_workers"]) > 0,
    )
    amp = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model = model.to(memory_format=torch.channels_last)
    predictions: list[dict] = []
    model.eval()
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                probabilities = torch.softmax(
                    model(images) / float(model.gw_temperature), dim=1
                ).cpu().numpy()
            for probability, source_index in zip(probabilities, indices.numpy().tolist()):
                order = np.argsort(probability)[::-1]
                row = approved.iloc[int(source_index)]
                predictions.append(
                    {
                        "event": row["event"],
                        "detector": row["detector"],
                        "qgram_path": row["qgram_path"],
                        "predicted_gravity_spy_class": classes[int(order[0])],
                        "confidence": float(probability[order[0]]),
                        "second_class": classes[int(order[1])] if len(order) > 1 else None,
                        "second_probability": float(probability[order[1]]) if len(order) > 1 else None,
                        "normalized_entropy": probability_entropy(probability),
                        "closed_set_only": True,
                    }
                )

    per_channel = pd.DataFrame(predictions).sort_values(["event", "detector"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_channel.to_csv(args.output_dir / "per_channel.csv", index=False)
    event_rows = []
    for event, group in per_channel.groupby("event"):
        event_rows.append(
            {
                "event": event,
                "n_channels": int(len(group)),
                "detectors": ",".join(sorted(group["detector"].astype(str))),
                "maximum_confidence": float(group["confidence"].max()),
                "mean_confidence": float(group["confidence"].mean()),
                "classes": ",".join(group["predicted_gravity_spy_class"].astype(str)),
            }
        )
    pd.DataFrame(event_rows).to_csv(args.output_dir / "per_event.csv", index=False)
    summary = {
        "channels_classified": int(len(per_channel)),
        "events_classified": int(per_channel["event"].nunique()),
        "classes": classes,
        "temperature": float(model.gw_temperature),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "qgram_source_outseg_seconds": float(cfg["source_outseg_seconds"]),
        "model_target_duration_seconds": float(cfg["target_duration_seconds"]),
        "preprocessing": dict(cfg),
        "interpretation_warning": (
            "Saida fechada para diagnostico e comparacao apenas. Use o ensemble com "
            "referencia OOD para qualquer classificacao principal. As classes Gravity Spy "
            "descrevem semelhanca morfologica, nao a origem causal do transiente."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_manifest(
        args.output_dir / "manifest.json",
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_profile": profile,
            "device": str(device),
            "config": config,
            "summary": summary,
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
