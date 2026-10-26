"""Piloto de pixels do Plano 1 v1.1: qual receita de Q-scan reproduz o painel oficial?

Para glitches do split de TREINO (o conjunto de avaliacao fica intocado), gera a imagem
de entrada da CNN por cada variante de ``[pilot].variants`` (taxa de amostragem, janela,
receita do Q-transform) e mede a semelhanca com o painel oficial recortado em 20-512 Hz
do MESMO glitch: erro absoluto medio e correlacao de pixels. Nao usa a rede neural.

A variante e escolhida pela regra fixada em ``configs/plano1_v1_1.toml`` e gravada em
``escolha.json``, que o lote seguinte usa para montar o conjunto de fechamento.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from gwproject.commands.build_closure_set import download, locate_url, public_segments, read_gwosc
from gwproject.config import load_toml
from gwproject.research.closure import annotate_files, select_files

COST_ORDER = ("sample_rate", "seconds")  # mais barato primeiro: 4 kHz, depois janela menor


def render_variant(strain, gps: float, variant: dict, processing: dict, inference: dict) -> np.ndarray:
    """Imagem 140x170 (cinza Gravity Spy, 20-512 Hz) de um glitch numa variante."""
    from PIL import Image

    from gwproject.gwosc_pipeline import gwpy_qtransform_consistent
    from gwproject.qgram_inference import qgram_to_uint8

    half = float(variant["seconds"]) / 2
    qcfg = processing["qtransform"]
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.hdf5"
        strain.crop(gps - half, gps + half).write(raw, format="hdf5", overwrite=True)
        result = gwpy_qtransform_consistent(
            raw, gps, Path(tmp) / "q.npy",
            frange=(float(qcfg["fmin_hz"]), float(qcfg["fmax_hz"])),
            config=processing, recipe=str(variant["recipe"]),
        )
        if result.get("status") != "ok":
            raise ValueError(result.get("reason") or result.get("status"))
        array = np.load(Path(tmp) / "q.npy")
    image = qgram_to_uint8(
        array,
        source_duration_seconds=float(inference["source_outseg_seconds"]),
        target_duration_seconds=float(inference["target_duration_seconds"]),
        flip_frequency=bool(inference["flip_frequency"]),
        render_mode="gravity_spy",
    )
    width, height = int(inference["resize_width"]), int(inference["resize_height"])
    return np.asarray(Image.fromarray(image).resize((width, height)), dtype=float)


def official_panel(root: Path, row, inference: dict) -> np.ndarray:
    from PIL import Image

    path = root / "TrainingSet" / row.label / f"{row.ifo}_{row.gravityspy_id}_spectrogram_1.0.png"
    width, height = int(inference["resize_width"]), int(inference["resize_height"])
    return np.asarray(Image.open(path).convert("L").resize((width, height)), dtype=float)


def choose_variant(scores: pd.DataFrame, variants: list[dict], min_gain: float) -> dict:
    """Aplica a regra preregistrada: maior media de correlacao mediana por classe; ganhos
    menores que ``min_gain`` nao justificam uma variante mais cara."""
    by_name = {variant["name"]: variant for variant in variants}
    means = scores.groupby("variant")["median_corr"].mean()
    best = float(means.max())
    tied = [name for name, value in means.items() if best - value < min_gain]
    cheapest = min(tied, key=lambda name: tuple(by_name[name][key] for key in COST_ORDER))
    return {**by_name[cheapest], "mean_median_corr": float(means[cheapest]),
            "best_mean_median_corr": best, "means": {k: float(v) for k, v in means.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description="Piloto de pixels do Plano 1 v1.1.")
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/gravity_spy_harmonized/clean_metadata.csv")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/plano1_v1_1.toml"))
    parser.add_argument("--processing-config", type=Path, default=Path("configs/processing.toml"))
    parser.add_argument("--research-config", type=Path, default=Path("configs/research_program.toml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/research/plano1_v1_1/piloto"))
    parser.add_argument("--dry-run", action="store_true", help="so grava o plano; sem download")
    args = parser.parse_args()

    plan_cfg = load_toml(args.config)
    pilot, selection = plan_cfg["pilot"], plan_cfg["selection"]
    processing = load_toml(args.processing_config)
    inference = load_toml(args.research_config)["harmonized_inference"]
    variants = list(pilot["variants"])
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata)
    widest = max(float(variant["seconds"]) for variant in variants) / 2
    segments = public_segments(
        sorted(metadata["ifo"].unique()), float(selection["run_start_gps"]),
        float(selection["run_end_gps"]), output / "public_segments.json",
    )
    annotated = annotate_files(
        metadata[metadata["label"].isin(pilot["classes"])],
        run_start_gps=float(selection["run_start_gps"]), run_end_gps=float(selection["run_end_gps"]),
        file_seconds=int(selection["file_seconds"]), before_seconds=widest, after_seconds=widest,
        public_segments=segments,
    )
    plan = select_files(
        annotated, evaluation_splits=["train"], max_files=int(pilot["max_files"]),
        per_class_target=int(pilot["max_glitches_per_class"]),
    )
    plan.to_csv(output / "plan_files.csv", index=False)
    keys = set(zip(plan["ifo"], plan["file_start"]))
    glitches = annotated[
        annotated["usable"] & (annotated["sample_type"] == "train")
        & np.array([key in keys for key in zip(annotated["ifo"], annotated["file_start"])])
    ]
    glitches = glitches.groupby("label", group_keys=False).head(int(pilot["max_glitches_per_class"]))
    rates = sorted({int(variant["sample_rate"]) for variant in variants})
    print(json.dumps({
        "files": len(plan), "glitches": len(glitches),
        "by_class": glitches["label"].value_counts().to_dict(),
        "estimated_download_gb": round(len(plan) * sum(r / 4096 for r in rates) * 0.13, 1),
    }, indent=2))
    if args.dry_run:
        return 0

    rows: list[dict] = []
    panels = Path(pilot["panels_root"])
    for file_row in plan.itertuples(index=False):
        in_file = glitches[(glitches["ifo"] == file_row.ifo) & (glitches["file_start"] == file_row.file_start)]
        for rate in rates:
            url = locate_url(file_row.ifo, int(file_row.file_start), int(selection["file_seconds"]), rate)
            local = download(url, output / "_download" / Path(url).name)
            strain = read_gwosc(local)
            for glitch in in_file.itertuples(index=False):
                reference = official_panel(panels, glitch, inference)
                for variant in (v for v in variants if int(v["sample_rate"]) == rate):
                    row = {"variant": variant["name"], "label": glitch.label,
                           "gravityspy_id": glitch.gravityspy_id, "ifo": glitch.ifo}
                    try:
                        image = render_variant(strain, float(glitch.event_time), variant, processing, inference)
                        row["mae"] = float(np.abs(image - reference).mean())
                        row["corr"] = float(np.corrcoef(image.ravel(), reference.ravel())[0, 1])
                        row["median_shift"] = float(np.median(image) - np.median(reference))
                    except Exception as exc:  # registra e segue: falha conta contra a variante
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    rows.append(row)
            local.unlink(missing_ok=True)
            print(f"{file_row.ifo} {file_row.file_start} @ {rate} Hz: {len(in_file)} glitches")

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "pilot_pixels.csv", index=False)
    scores = (
        frame.dropna(subset=["corr"]).groupby(["variant", "label"])
        .agg(median_corr=("corr", "median"), median_mae=("mae", "median"), n=("corr", "size"))
        .reset_index()
    )
    scores.to_csv(output / "pilot_scores.csv", index=False)
    choice = choose_variant(scores, variants, float(pilot["min_gain_to_prefer_costlier"]))
    choice["errors"] = int(frame.get("error", pd.Series(dtype=object)).notna().sum())
    (output / "escolha.json").write_text(json.dumps(choice, indent=2), encoding="utf-8")
    print(scores.pivot(index="label", columns="variant", values="median_corr").round(3).to_string())
    print(json.dumps(choice, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
