"""Monta o conjunto de fechamento do Plano 1 a partir do strain publico de O1.

Para cada arquivo GWOSC de 4096 s escolhido em ``configs/plano1.toml``:

1. baixa o arquivo (com retomada) para uma pasta temporaria;
2. recorta 32 s em torno de cada glitch Gravity Spy rotulado que cabe no arquivo,
   gravando no mesmo formato dos eventos do catalogo (``raw/<id>/<ifo>.hdf5``);
3. gera o Q-gram com ``gwpy_qtransform_consistent`` e a configuracao de
   ``configs/processing.toml``, a mesma chamada de ``run_catalog``;
4. registra o resultado de cada glitch e apaga o arquivo baixado.

A execucao e retomavel: cada arquivo concluido grava ``done/<ifo>_<inicio>.json``.
``--dry-run`` so calcula e grava o plano de arquivos, sem acessar a rede.
O CSV final tem as colunas ``event``/``detector``/``qgram_path`` esperadas por
``classify-ensemble`` (``event`` = ``gravityspy_id``).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from gwproject.config import load_toml
from gwproject.provenance import sha256_file, write_manifest
from gwproject.research.closure import annotate_files, coverage_by_class, select_files

# Tamanho medido de um arquivo O1 de 4 kHz (H-H1_LOSC_4_V1-*-4096.hdf5) em 21/10/2026.
O1_4KHZ_FILE_BYTES = 130_268_561


class NoPublicData(Exception):
    """O GWOSC nao publica strain nesse intervalo (detector fora do modo cientifico)."""


def locate_url(ifo: str, start: int, file_seconds: int, sample_rate: int) -> str:
    from gwosc.locate import get_urls

    try:
        urls = get_urls(ifo, start, start + file_seconds, sample_rate=sample_rate, format="hdf5")
    except ValueError as exc:
        if "Cannot find a GWOSC dataset" in str(exc):
            raise NoPublicData(str(exc)) from exc
        raise
    matches = [url for url in urls if re.search(rf"-{start}-{file_seconds}\.hdf5$", url)]
    if len(matches) != 1:
        raise ValueError(f"URL GWOSC ambigua ou ausente para {ifo} {start}: {urls}")
    return matches[0]


def public_segments(
    ifos: list[str], start: float, end: float, cache: Path
) -> dict[str, list[tuple[float, float]]]:
    """Trechos ``{ifo}_DATA`` do GWOSC (onde o strain publico nao e NaN), com cache local."""
    if cache.exists():
        return {ifo: [tuple(item) for item in value] for ifo, value in json.loads(cache.read_text()).items()}
    from gwosc.timeline import get_segments

    segments = {ifo: [tuple(item) for item in get_segments(f"{ifo}_DATA", start, end)] for ifo in ifos}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(segments), encoding="utf-8")
    return segments


def download(url: str, destination: Path) -> Path:
    from gwproject.downloads import download_with_resume

    return download_with_resume(url, destination)


def read_gwosc(path: Path):
    from gwpy.timeseries import TimeSeries

    return TimeSeries.read(path, format="hdf5.gwosc")


def process_file(
    file_row: pd.Series,
    glitches: pd.DataFrame,
    *,
    output_root: Path,
    download_dir: Path,
    selection: dict,
    processing: dict,
    keep_files: bool,
) -> list[dict]:
    from gwproject.gwosc_pipeline import gwpy_qtransform_consistent

    ifo, start = str(file_row["ifo"]), int(file_row["file_start"])
    file_seconds = int(selection["file_seconds"])
    url = locate_url(ifo, start, file_seconds, int(selection["sample_rate"]))
    local = download(url, download_dir / Path(url).name)
    source_sha256 = sha256_file(local)
    strain = read_gwosc(local)
    before = float(selection["before_seconds"])
    after = float(selection["after_seconds"])
    qcfg = processing["qtransform"]
    rows: list[dict] = []
    for glitch in glitches.itertuples(index=False):
        gps = float(glitch.event_time)
        raw_path = output_root / "raw" / str(glitch.gravityspy_id) / f"{ifo}.hdf5"
        q_path = output_root / "qgrams" / str(glitch.gravityspy_id) / f"{ifo}_qgram.npy"
        row = {
            "event": glitch.gravityspy_id,
            "detector": ifo,
            "label": glitch.label,
            "sample_type": glitch.sample_type,
            "event_time": gps,
            "file_start": start,
            "source_url": url,
            "source_sha256": source_sha256,
            "raw_path": str(raw_path),
            "qgram_path": None,
        }
        try:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            strain.crop(gps - before, gps + after).write(raw_path, format="hdf5", overwrite=True)
            result = gwpy_qtransform_consistent(
                raw_path,
                gps,
                q_path,
                frange=(float(qcfg["fmin_hz"]), float(qcfg["fmax_hz"])),
                config=processing,
                recipe=str(selection.get("qtransform_recipe", "catalog")),
            )
            row["status"] = result.get("status")
            row["reason"] = result.get("reason")
            if result.get("status") == "ok":
                row["qgram_path"] = str(q_path)
        except Exception as exc:  # um glitch ruim nao derruba o arquivo inteiro
            row["status"] = "error"
            row["reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    if not keep_files:
        local.unlink(missing_ok=True)
    return rows


def mark_render_invalid(channels: pd.DataFrame, inference_cfg: dict) -> pd.DataFrame:
    """Marca Q-grams que algum renderer recusa, para que todas as condicoes usem o mesmo conjunto.

    Glitches muito fortes (sobretudo Extremely_Loud) podem gerar valores negativos na
    interpolacao do Q-transform; o renderer ``percentile`` recusa a mediana por
    frequencia nao positiva. Esses canais saem de todas as condicoes, com o motivo.
    """
    from gwproject.qgram_inference import RENDER_MODES, qgram_to_uint8

    channels = channels.copy()
    for index, row in channels[channels["status"] == "ok"].iterrows():
        array = np.load(row["qgram_path"])
        for mode in RENDER_MODES:
            try:
                qgram_to_uint8(
                    array,
                    lower_percentile=float(inference_cfg["lower_percentile"]),
                    upper_percentile=float(inference_cfg["upper_percentile"]),
                    source_duration_seconds=float(inference_cfg["source_outseg_seconds"]),
                    target_duration_seconds=float(inference_cfg["target_duration_seconds"]),
                    baseline_mode=str(inference_cfg["baseline_mode"]),
                    flip_frequency=bool(inference_cfg["flip_frequency"]),
                    render_mode=mode,
                )
            except ValueError as exc:
                channels.loc[index, ["status", "reason", "qgram_path"]] = [
                    "render_invalid", f"{mode}: {exc}", None,
                ]
                break
    return channels


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plano 1: reconstroi glitches Gravity Spy de O1 pela cadeia do catalogo."
    )
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/gravity_spy_harmonized/clean_metadata.csv")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/plano1.toml"))
    parser.add_argument("--processing-config", type=Path, default=Path("configs/processing.toml"))
    parser.add_argument(
        "--research-config", type=Path, default=Path("configs/research_program.toml")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/research/plano1/fechamento")
    )
    parser.add_argument("--download-dir", type=Path, default=None)
    parser.add_argument("--max-files", type=int, default=None, help="sobrepoe [selection].max_files")
    parser.add_argument("--sample-rate", type=int, default=None, help="sobrepoe [selection].sample_rate")
    parser.add_argument(
        "--window-seconds", type=float, default=None,
        help="janela total centrada no glitch; sobrepoe before/after_seconds",
    )
    parser.add_argument(
        "--recipe", choices=["catalog", "gravity_spy"], default=None,
        help="sobrepoe [selection].qtransform_recipe",
    )
    parser.add_argument("--dry-run", action="store_true", help="so grava o plano; sem rede")
    parser.add_argument("--keep-files", action="store_true", help="nao apaga os arquivos baixados")
    args = parser.parse_args()

    plan_cfg = load_toml(args.config)
    selection = dict(plan_cfg["selection"])
    if args.max_files is not None:
        selection["max_files"] = int(args.max_files)
    if args.sample_rate is not None:
        selection["sample_rate"] = int(args.sample_rate)
    if args.window_seconds is not None:
        selection["before_seconds"] = selection["after_seconds"] = float(args.window_seconds) / 2
    if args.recipe is not None:
        selection["qtransform_recipe"] = args.recipe
    processing = load_toml(args.processing_config)
    output_root = args.output_root
    download_dir = args.download_dir or output_root / "_download"
    output_root.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata)
    segments = public_segments(
        sorted(metadata["ifo"].unique()),
        float(selection["run_start_gps"]),
        float(selection["run_end_gps"]),
        output_root / "public_segments.json",
    )
    annotated = annotate_files(
        metadata,
        run_start_gps=float(selection["run_start_gps"]),
        run_end_gps=float(selection["run_end_gps"]),
        file_seconds=int(selection["file_seconds"]),
        before_seconds=float(selection["before_seconds"]),
        after_seconds=float(selection["after_seconds"]),
        public_segments=segments,
    )
    splits = [str(value) for value in selection["evaluation_splits"]]
    plan = select_files(
        annotated,
        evaluation_splits=splits,
        max_files=int(selection["max_files"]),
        per_class_target=int(selection["per_class_target"]),
    )
    coverage = coverage_by_class(annotated, plan, splits)
    plan.to_csv(output_root / "plan_files.csv", index=False)
    coverage.to_csv(output_root / "plan_coverage_by_class.csv")
    plan_summary = {
        "files": int(len(plan)),
        "estimated_download_gb": round(
            len(plan) * O1_4KHZ_FILE_BYTES * int(selection["sample_rate"]) / 4096 / 1e9, 1
        ),
        "sample_rate": int(selection["sample_rate"]),
        "window_seconds": float(selection["before_seconds"]) + float(selection["after_seconds"]),
        "qtransform_recipe": str(selection.get("qtransform_recipe", "catalog")),
        "evaluation_glitches": int(coverage["evaluation"].sum()),
        "train_glitches_in_same_files": int(coverage["train"].sum()),
        "o1_glitches_outside_file_window": int((~annotated["window_inside_file"]).sum()),
        "o1_glitches_without_public_data": int((~annotated["window_public"]).sum()),
    }
    print(json.dumps(plan_summary, indent=2))
    print(coverage.to_string())
    if args.dry_run:
        (output_root / "plan_summary.json").write_text(
            json.dumps(plan_summary, indent=2), encoding="utf-8"
        )
        return 0

    done_dir = output_root / "done"
    done_dir.mkdir(exist_ok=True)
    inside = annotated[annotated["usable"]]
    failures: list[dict] = []
    for position, file_row in enumerate(plan.itertuples(index=False), start=1):
        marker = done_dir / f"{file_row.ifo}_{file_row.file_start}.json"
        if marker.exists():
            continue
        glitches = inside[
            (inside["ifo"] == file_row.ifo) & (inside["file_start"] == file_row.file_start)
        ]
        print(f"[{position}/{len(plan)}] {file_row.ifo} {file_row.file_start}: {len(glitches)} glitches")
        try:
            rows = process_file(
                pd.Series(file_row._asdict()),
                glitches,
                output_root=output_root,
                download_dir=download_dir,
                selection=selection,
                processing=processing,
                keep_files=args.keep_files,
            )
        except NoPublicData as exc:  # permanente: registra e nao tenta de novo
            rows = [
                {
                    "event": glitch.gravityspy_id, "detector": file_row.ifo, "label": glitch.label,
                    "sample_type": glitch.sample_type, "event_time": float(glitch.event_time),
                    "file_start": int(file_row.file_start), "qgram_path": None,
                    "status": "no_public_data", "reason": str(exc),
                }
                for glitch in glitches.itertuples(index=False)
            ]
            print("  sem dado publico no GWOSC; registrado")
        except Exception as exc:  # rede ou arquivo: fica sem marcador e e refeito na retomada
            failures.append({"ifo": file_row.ifo, "file_start": int(file_row.file_start), "error": str(exc)})
            print(f"  FALHOU: {exc}")
            continue
        marker.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    rows = []
    for marker in sorted(done_dir.glob("*.json")):
        rows.extend(json.loads(marker.read_text(encoding="utf-8")))
    channels = pd.DataFrame(rows)
    research_cfg = load_toml(args.research_config)
    channels = mark_render_invalid(channels, research_cfg["harmonized_inference"])
    channels.to_csv(output_root / "closure_channels.csv", index=False)
    ok = channels[channels["status"] == "ok"]
    summary = {
        **plan_summary,
        "files_done": len(list(done_dir.glob("*.json"))),
        "files_failed": failures,
        "glitches_processed": int(len(channels)),
        "qgrams_ok": int(len(ok)),
        "status_counts": channels["status"].value_counts().to_dict(),
        "qgrams_ok_by_split": ok["sample_type"].value_counts().to_dict(),
        "complete": not failures and len(list(done_dir.glob("*.json"))) == len(plan),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_manifest(
        output_root / "manifest.json",
        {"plan": plan_cfg, "processing": processing, "summary": summary},
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
