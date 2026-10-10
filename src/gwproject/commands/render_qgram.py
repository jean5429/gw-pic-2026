"""Exporta como PNG a entrada da CNN (140 x 170) gerada a partir de Q-grams do catalogo.

Usa a renderizacao de ``[harmonized_inference]`` e o mesmo Resize da inferencia, para
inspecao visual e para Grad-CAM. Nao altera Q-grams nem resultados existentes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gwproject.config import load_toml
from gwproject.provenance import sha256_file, write_manifest
from gwproject.qgram_inference import (
    RENDER_MODES,
    catalog_network_input,
    qgram_to_uint8,
    render_settings,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument("--detectors", nargs="+", default=None, help="padrao: todos os aprovados")
    parser.add_argument("--approved", type=Path, default=Path("outputs/gwtc4_218/approved_channels.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/research_program.toml"))
    parser.add_argument(
        "--render-modes",
        nargs="+",
        choices=RENDER_MODES,
        default=None,
        help="padrao: o modo da configuracao",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research/rendered_qgrams"))
    args = parser.parse_args()

    from PIL import Image

    cfg = load_toml(args.config)["harmonized_inference"]
    height, width = int(cfg["resize_height"]), int(cfg["resize_width"])
    modes = args.render_modes or [str(cfg.get("render_mode", "percentile"))]
    approved = pd.read_csv(args.approved)
    channels = approved[approved["event"] == args.event]
    if args.detectors:
        channels = channels[channels["detector"].isin(args.detectors)]
        missing = sorted(set(args.detectors) - set(channels["detector"]))
        if missing:
            raise ValueError(f"detectores sem canal aprovado em {args.event}: {missing}")
    if channels.empty:
        raise ValueError(f"nenhum canal aprovado para {args.event}")

    renders = []
    for row in channels.sort_values("detector").itertuples():
        qgram_path = Path(str(row.qgram_path))
        qgram = np.load(qgram_path, allow_pickle=False)
        for mode in modes:
            settings = render_settings(cfg, mode)
            pixels = catalog_network_input(qgram_to_uint8(qgram, **settings), height=height, width=width)
            output = args.output_dir / f"{args.event}_{row.detector}_{mode}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(output)
            renders.append({
                "event": args.event,
                "detector": str(row.detector),
                "render_settings": settings,
                "qgram_path": str(qgram_path),
                "qgram_sha256": sha256_file(qgram_path),
                "png": str(output),
                "png_sha256": sha256_file(output),
            })
            print(f"Salvo: {output}")
    summary = {"network_input_shape": [height, width], "renders": renders}
    write_manifest(args.output_dir / f"{args.event}_render_manifest.json", summary)
    print(json.dumps({"renders": len(renders)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
