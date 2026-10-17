"""Grad-CAM de um checkpoint Gravity Spy sobre um painel ou um Q-gram do catalogo.

Q-grams passam pela mesma renderizacao e pelo mesmo Resize/Normalize do ensemble.
O mapa explica apenas a decisao fechada do checkpoint escolhido.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from gwproject.config import load_toml
from gwproject.gravity_spy import resolve_device
from gwproject.plotting import FIGURE_DPI
from gwproject.provenance import sha256_file, write_manifest
from gwproject.qgram_inference import (
    RENDER_MODES,
    catalog_network_input,
    load_gravity_spy_checkpoint,
    qgram_to_uint8,
    render_settings,
)
from gwproject.research.gradcam import grad_cam, upsample_heatmap
from gwproject.research.uncertainty import softmax_probabilities

DEFAULT_CHECKPOINT = Path("models/gravity_spy_harmonized_v1d1/seed_42_cuda_balanced/best_model.pt")
WARNING = (
    "Grad-CAM e diagnostico qualitativo da decisao fechada deste checkpoint. Nao valida "
    "a classe, nao substitui a referencia OOD do ensemble e nao identifica a origem "
    "fisica do transiente."
)


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _qgram_from_approved(approved_csv: Path, event: str, detector: str) -> Path:
    approved = pd.read_csv(approved_csv)
    match = approved[(approved["event"] == event) & (approved["detector"] == detector)]
    if len(match) != 1 or pd.isna(match["qgram_path"].iloc[0]):
        raise ValueError(f"canal {event}/{detector} ausente ou sem Q-gram em {approved_csv}")
    return Path(str(match["qgram_path"].iloc[0]))


def save_gradcam_figure(network_input, heatmap, path: Path, *, title: str) -> None:
    from matplotlib.figure import Figure

    upsampled = upsample_heatmap(heatmap, network_input.shape)
    figure = Figure(figsize=(15, 5))
    axes = figure.subplots(1, 3)
    axes[0].imshow(network_input, cmap="gray", vmin=0, vmax=255, aspect="auto")
    axes[0].set_title("1. Entrada da rede (140 × 170)")
    axes[1].imshow(upsampled, cmap="jet", vmin=0, vmax=1, aspect="auto")
    axes[1].set_title("2. Grad-CAM (vermelho = maior peso)")
    axes[2].imshow(network_input, cmap="gray", vmin=0, vmax=255, aspect="auto")
    axes[2].imshow(upsampled, cmap="jet", vmin=0, vmax=1, alpha=0.4, aspect="auto")
    axes[2].set_title("3. Sobreposição")
    for axis in axes:
        axis.set_xlabel("tempo")
        axis.set_ylabel("frequência (alta no topo)")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=FIGURE_DPI)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="painel harmonizado (PNG) do Gravity Spy")
    source.add_argument("--qgram", type=Path, help="Q-gram .npy no contrato (tempo, frequencia)")
    source.add_argument("--event", help="evento do catalogo; exige --detector")
    parser.add_argument("--detector", help="detector do canal, por exemplo H1")
    parser.add_argument("--approved", type=Path, default=Path("outputs/gwtc4_218/approved_channels.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/research_program.toml"))
    parser.add_argument("--render-mode", choices=RENDER_MODES, default=None)
    parser.add_argument("--target-class", default=None, help="classe a explicar; padrao: a prevista")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research/gradcam"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.event and not args.detector:
        parser.error("--event exige --detector")
    if args.detector and not args.event:
        parser.error("--detector so pode ser usado com --event")

    import torch
    from PIL import Image

    cfg = load_toml(args.config)["harmonized_inference"]
    height, width = int(cfg["resize_height"]), int(cfg["resize_width"])
    device = resolve_device(args.device)
    model, classes, profile = load_gravity_spy_checkpoint(args.checkpoint, device)

    settings = None
    if args.image is not None:
        input_path = args.image
        with Image.open(input_path) as handle:
            rendered = np.asarray(handle.convert("L"), dtype=np.uint8)
        stem = input_path.stem
    else:
        input_path = args.qgram or _qgram_from_approved(args.approved, args.event, args.detector)
        settings = render_settings(cfg, args.render_mode)
        rendered = qgram_to_uint8(np.load(input_path, allow_pickle=False), **settings)
        label = f"{args.event}_{args.detector}" if args.event else input_path.stem
        stem = f"{label}_{settings['render_mode']}"
    network_input = catalog_network_input(rendered, height=height, width=width)
    # Mesma aritmetica de ToTensor -> Normalize([0.5], [0.5]) dos comandos de inferencia.
    tensor = torch.from_numpy(network_input.astype(np.float32) / 255.0)
    tensor = ((tensor - 0.5) / 0.5)[None, None].to(device)

    target_index = None
    if args.target_class is not None:
        if args.target_class not in classes:
            raise ValueError(f"classe desconhecida: {args.target_class}")
        target_index = classes.index(args.target_class)
    result = grad_cam(model, tensor, class_index=target_index)
    probabilities = softmax_probabilities(result.logits[None, :] / float(model.gw_temperature))[0]
    order = np.argsort(probabilities)[::-1]
    explained = classes[result.class_index]
    predicted = classes[int(order[0])]

    figure_path = args.output_dir / f"gradcam_{_safe_name(stem)}__{_safe_name(explained)}.png"
    save_gradcam_figure(
        network_input,
        result.heatmap,
        figure_path,
        title=(
            f"{stem} | explicada: {explained} | prevista: {predicted} "
            f"(p = {probabilities[order[0]]:.3f}, checkpoint isolado)"
        ),
    )
    summary = {
        "figure": str(figure_path),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_profile": profile,
        "temperature": float(model.gw_temperature),
        "input": {
            "kind": "image" if args.image is not None else "qgram",
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "event": args.event,
            "detector": args.detector,
            "render_settings": settings,
            "network_input_shape": [height, width],
        },
        "target_layer": "layer4[-1]",
        "heatmap_shape": list(result.heatmap.shape),
        "explained_class": explained,
        "predicted_class": predicted,
        "top_probabilities": {
            classes[int(index)]: float(probabilities[index]) for index in order[:5]
        },
        "interpretation_warning": WARNING,
    }
    write_manifest(figure_path.with_suffix(".json"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
