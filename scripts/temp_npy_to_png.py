import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from gwproject.config import load_toml
from gwproject.qgram_inference import qgram_to_uint8


def npy_to_png(npy_path, png_path, cfg):
    # Mesmo pre-processamento do ensemble (classify_catalog_qgrams_ensemble.py):
    # o .npy do gwpy e (tempo, frequencia) e precisa ser transposto.
    data = qgram_to_uint8(
        np.load(npy_path),
        lower_percentile=float(cfg["lower_percentile"]),
        upper_percentile=float(cfg["upper_percentile"]),
        source_duration_seconds=float(cfg["source_outseg_seconds"]),
        target_duration_seconds=float(cfg["target_duration_seconds"]),
        baseline_mode=str(cfg["baseline_mode"]),
        flip_frequency=bool(cfg["flip_frequency"]),
        layout="gwpy",
        render_mode=str(cfg.get("render_mode", "percentile")),
    )
    img = Image.fromarray(data).convert("L")
    img = img.resize((int(cfg["resize_width"]), int(cfg["resize_height"])))
    img.save(png_path)
    print(f"Salvo: {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Converte Q-grams .npy do GWTC em PNG para o Grad-CAM")
    parser.add_argument("--event", default="GW151012-v4")
    parser.add_argument("--detectors", nargs="+", default=["H1", "L1"])
    parser.add_argument("--config", type=Path, default=Path("configs/research_program.toml"))
    args = parser.parse_args()

    cfg = load_toml(args.config)["harmonized_inference"]
    folder = Path("outputs/gwtc/qgrams") / args.event
    for detector in args.detectors:
        npy_to_png(folder / f"{detector}_qgram.npy", folder / f"{detector}_{args.event}.png", cfg)


if __name__ == "__main__":
    main()
