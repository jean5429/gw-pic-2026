from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gwproject.config import load_toml
from gwproject.provenance import sha256_file, write_manifest
from gwproject.research.domain_shift import rendered_qgram_statistics


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calcula estatisticas de baixo nivel dos Q-grams exatamente apos a "
            "renderizacao usada pela CNN catalogal."
        )
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research_program.toml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/research/catalog_rendered_statistics.csv"),
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    cfg = load_toml(args.config)["harmonized_inference"]
    render_mode = str(cfg.get("render_mode", "percentile"))
    statistics = rendered_qgram_statistics(
        frame,
        lower_percentile=float(cfg["lower_percentile"]),
        upper_percentile=float(cfg["upper_percentile"]),
        source_duration_seconds=float(cfg["source_outseg_seconds"]),
        target_duration_seconds=float(cfg["target_duration_seconds"]),
        baseline_mode=str(cfg["baseline_mode"]),
        flip_frequency=bool(cfg["flip_frequency"]),
        render_mode=render_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    statistics.to_csv(args.output, index=False)
    summary = {
        "input": str(args.input_csv),
        "input_sha256": sha256_file(args.input_csv),
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "channels": int(len(statistics)),
        "events": int(statistics["event"].nunique()),
        "rendering": {
            "lower_percentile": float(cfg["lower_percentile"]),
            "upper_percentile": float(cfg["upper_percentile"]),
            "source_duration_seconds": float(cfg["source_outseg_seconds"]),
            "target_duration_seconds": float(cfg["target_duration_seconds"]),
            "baseline_mode": str(cfg["baseline_mode"]),
            "flip_frequency": bool(cfg["flip_frequency"]),
            "render_mode": render_mode,
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_manifest(args.output.with_suffix(".manifest.json"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
