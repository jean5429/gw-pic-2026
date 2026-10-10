from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from gwproject.research.panel_preprocessing import PanelCropConfig, prepare_gravity_spy_panel


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove eixos/colorbar dos PNGs Gravity Spy e harmoniza a banda fisica."
    )
    parser.add_argument("images_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--source-fmin", type=float, default=10.0)
    parser.add_argument("--source-fmax", type=float, default=2048.0)
    parser.add_argument("--target-fmin", type=float, default=20.0)
    parser.add_argument("--target-fmax", type=float, default=512.0)
    parser.add_argument("--height", type=int, default=140)
    parser.add_argument("--width", type=int, default=170)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.images_root.exists():
        raise FileNotFoundError(args.images_root)
    images = sorted(args.images_root.rglob("*spectrogram_1.0.png"))
    if not images:
        raise RuntimeError("nenhum PNG Gravity Spy de 1.0 s encontrado")
    cfg = PanelCropConfig(
        source_fmin_hz=args.source_fmin,
        source_fmax_hz=args.source_fmax,
        target_fmin_hz=args.target_fmin,
        target_fmax_hz=args.target_fmax,
        output_height=args.height,
        output_width=args.width,
    )
    succeeded = 0
    failed: list[dict] = []
    args.output_root.mkdir(parents=True, exist_ok=True)
    for source in tqdm(images, desc="harmonizando Gravity Spy"):
        relative = source.relative_to(args.images_root)
        destination = args.output_root / relative
        if destination.exists() and not args.overwrite:
            succeeded += 1
            continue
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            panel = prepare_gravity_spy_panel(source, config=cfg, normalize=args.normalize)
            panel.save(destination, format="PNG", optimize=True)
            succeeded += 1
        except Exception as exc:
            failed.append({"path": str(source), "error": f"{type(exc).__name__}: {exc}"})

    summary = {
        "images_found": len(images),
        "images_succeeded": succeeded,
        "images_failed": len(failed),
        "input_root": str(args.images_root),
        "output_root": str(args.output_root),
        "configuration": cfg.__dict__,
        "normalization_applied": bool(args.normalize),
        "failures": failed[:100],
        "scientific_warning": (
            "Os limites 10-2048 Hz para PNGs originais sao uma hipotese de renderizacao. "
            "Confirme-os contra metadados ou geracao oficial antes de conclusoes fisicas."
        ),
    }
    (args.output_root / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    # Permite ate 50 imagens corrompidas no dataset sem abortar o treinamento inteiro
    return 0 if len(failed) < 50 else 2


if __name__ == "__main__":
    raise SystemExit(main())
