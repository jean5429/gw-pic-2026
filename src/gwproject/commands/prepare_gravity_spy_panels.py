from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from gwproject.config import load_toml
from gwproject.gravity_spy import (
    attach_image_paths,
    index_spectrograms,
    official_splits,
)
from gwproject.provenance import sha256_file, write_manifest
from gwproject.research.panel_preprocessing import PanelCropConfig, prepare_gravity_spy_panel


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extrai o painel dos PNGs Gravity Spy e produz entradas 140x170."
    )
    parser.add_argument("images_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="CSV oficial trainingset_v1d1_metadata.csv",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/research_program.toml"))
    parser.add_argument("--crop-mode", choices=["official", "auto"], default=None)
    parser.add_argument(
        "--official-bbox",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        default=None,
    )
    parser.add_argument("--apply-frequency-crop", action="store_true", default=None)
    parser.add_argument("--source-fmin", type=float, default=None)
    parser.add_argument("--source-fmax", type=float, default=None)
    parser.add_argument("--target-fmin", type=float, default=None)
    parser.add_argument("--target-fmax", type=float, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--normalize", action="store_true", default=None)
    parser.add_argument("--blank-std-min", type=float, default=1.0)
    parser.add_argument("--blank-range-min", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_all = load_toml(args.config)
    panel_cfg = config_all.get("gravity_spy_panel", {})

    crop_mode = args.crop_mode or str(panel_cfg.get("crop_mode", "official"))
    official_bbox = args.official_bbox or (
        int(panel_cfg.get("crop_left", 105)),
        int(panel_cfg.get("crop_top", 66)),
        int(panel_cfg.get("crop_right", 671)),
        int(panel_cfg.get("crop_bottom", 532)),
    )
    apply_frequency_crop = (
        bool(panel_cfg.get("apply_frequency_crop", False))
        if args.apply_frequency_crop is None
        else bool(args.apply_frequency_crop)
    )
    source_fmin = float(panel_cfg.get("source_fmin_hz", 10.0) if args.source_fmin is None else args.source_fmin)
    source_fmax = float(panel_cfg.get("source_fmax_hz", 2048.0) if args.source_fmax is None else args.source_fmax)
    target_fmin = float(panel_cfg.get("target_fmin_hz", 20.0) if args.target_fmin is None else args.target_fmin)
    target_fmax = float(panel_cfg.get("target_fmax_hz", 512.0) if args.target_fmax is None else args.target_fmax)
    height = int(panel_cfg.get("output_height", 140) if args.height is None else args.height)
    width = int(panel_cfg.get("output_width", 170) if args.width is None else args.width)
    normalize = (
        bool(panel_cfg.get("normalize_after_crop", False))
        if args.normalize is None
        else bool(args.normalize)
    )

    if not args.images_root.exists():
        raise FileNotFoundError(args.images_root)
    if not args.metadata.is_file():
        raise FileNotFoundError(args.metadata)
    metadata = pd.read_csv(args.metadata)
    image_index = index_spectrograms(args.images_root, "1.0")
    indexed = attach_image_paths(metadata, image_index, require_all=True)
    if len(indexed) != len(metadata):
        raise RuntimeError("nem todas as linhas da metadata receberam imagem")

    left, top, right, bottom = official_bbox
    cfg = PanelCropConfig(
        crop_mode=crop_mode,
        crop_left=left,
        crop_top=top,
        crop_right=right,
        crop_bottom=bottom,
        apply_frequency_crop=apply_frequency_crop,
        source_fmin_hz=source_fmin,
        source_fmax_hz=source_fmax,
        target_fmin_hz=target_fmin,
        target_fmax_hz=target_fmax,
        output_height=height,
        output_width=width,
    )

    succeeded = 0
    failed: list[dict] = []
    qc_rows: list[dict] = []
    args.output_root.mkdir(parents=True, exist_ok=True)
    for row in tqdm(
        indexed.to_dict(orient="records"),
        desc="preparando Gravity Spy",
        total=len(indexed),
    ):
        source = Path(row["path"])
        relative = source.relative_to(args.images_root)
        destination = args.output_root / relative
        try:
            if destination.exists() and not args.overwrite:
                from PIL import Image

                with Image.open(destination) as image:
                    panel_array = np.asarray(image.convert("L"), dtype=np.uint8)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                panel = prepare_gravity_spy_panel(source, config=cfg, normalize=normalize)
                panel.save(destination, format="PNG", optimize=True)
                panel_array = np.asarray(panel.convert("L"), dtype=np.uint8)
            if tuple(panel_array.shape) != (height, width):
                raise ValueError(
                    f"shape inesperado {panel_array.shape}; esperado {(height, width)}"
                )
            pixel_hash = hashlib.sha256(panel_array.tobytes()).hexdigest()
            pixel_std = float(np.std(panel_array))
            pixel_range = float(np.ptp(panel_array))
            qc_rows.append({
                "ifo": str(row["ifo"]).upper(),
                "gravityspy_id": str(row["gravityspy_id"]),
                "label": str(row["label"]),
                "sample_type": str(row["sample_type"]),
                "split": (
                    "val"
                    if str(row["sample_type"]).lower().startswith("valid")
                    else "test"
                    if str(row["sample_type"]).lower().startswith("test")
                    else "train"
                ),
                "source_path": str(source),
                "prepared_path": str(destination),
                "pixel_sha256": pixel_hash,
                "pixel_std": pixel_std,
                "pixel_range": pixel_range,
                "blank": bool(
                    pixel_std < args.blank_std_min
                    or pixel_range < args.blank_range_min
                ),
            })
            succeeded += 1
        except Exception as exc:
            failed.append({"path": str(source), "error": f"{type(exc).__name__}: {exc}"})

    qc_frame = pd.DataFrame(qc_rows)
    if qc_frame.empty:
        raise RuntimeError("nenhuma imagem passou pelo preprocessamento")
    hash_splits = qc_frame.groupby("pixel_sha256")["split"].nunique()
    cross_split_hashes = set(hash_splits[hash_splits > 1].index)
    hash_counts = qc_frame["pixel_sha256"].value_counts()
    qc_frame["exact_duplicate_count"] = (
        qc_frame["pixel_sha256"].map(hash_counts).astype(int)
    )
    qc_frame["exact_duplicate_cross_split"] = qc_frame["pixel_sha256"].isin(
        cross_split_hashes
    )
    qc_frame["excluded"] = (
        qc_frame["blank"] | qc_frame["exact_duplicate_cross_split"]
    )
    qc_frame["exclusion_reason"] = np.select(
        [
            qc_frame["blank"] & qc_frame["exact_duplicate_cross_split"],
            qc_frame["blank"],
            qc_frame["exact_duplicate_cross_split"],
        ],
        [
            "blank_and_exact_cross_split_duplicate",
            "blank",
            "exact_cross_split_duplicate",
        ],
        default="",
    )
    qc_frame.to_csv(args.output_root / "qc_samples.csv", index=False)

    clean_keys = set(
        zip(
            qc_frame.loc[~qc_frame["excluded"], "ifo"],
            qc_frame.loc[~qc_frame["excluded"], "gravityspy_id"],
        )
    )
    clean_metadata = metadata.copy()
    clean_metadata["ifo"] = clean_metadata["ifo"].astype(str).str.upper()
    clean_mask = [
        (ifo, gravityspy_id) in clean_keys
        for ifo, gravityspy_id in zip(
            clean_metadata["ifo"], clean_metadata["gravityspy_id"].astype(str)
        )
    ]
    clean_metadata = clean_metadata.loc[clean_mask].reset_index(drop=True)
    clean_metadata_path = args.output_root / "clean_metadata.csv"
    clean_metadata.to_csv(clean_metadata_path, index=False)

    clean_splits = official_splits(clean_metadata)
    classes = sorted(metadata["label"].dropna().astype(str).unique())
    missing_class_splits: list[dict] = []
    for split_name in ("train", "val", "test"):
        present = (
            set(clean_splits.get(split_name, pd.DataFrame()).get("label", []))
            if split_name in clean_splits
            else set()
        )
        for class_name in classes:
            if class_name not in present:
                missing_class_splits.append(
                    {"split": split_name, "label": class_name}
                )

    warnings: list[str] = []
    if crop_mode == "auto":
        warnings.append(
            "A deteccao automatica e uma variante experimental e deve ser validada contra o recorte oficial."
        )
    if apply_frequency_crop:
        warnings.append(
            "O recorte em frequencia pressupoe limites fisicos do renderer e requer validacao independente."
        )

    clean_metadata_hash = sha256_file(clean_metadata_path)
    qc_summary = {
        "format_version": 1,
        "metadata_rows": int(len(metadata)),
        "prepared_samples": int(len(qc_frame)),
        "clean_samples": int(len(clean_metadata)),
        "excluded_samples": int(qc_frame["excluded"].sum()),
        "blank_samples": int(qc_frame["blank"].sum()),
        "exact_cross_split_duplicate_samples": int(
            qc_frame["exact_duplicate_cross_split"].sum()
        ),
        "exact_cross_split_duplicate_groups": int(len(cross_split_hashes)),
        "missing_class_splits": missing_class_splits,
        "unprocessed_failures": int(len(failed)),
        "metadata_sha256": sha256_file(args.metadata),
        "clean_metadata_path": str(clean_metadata_path),
        "clean_metadata_sha256": clean_metadata_hash,
        "composite_key": ["ifo", "gravityspy_id"],
        "blank_thresholds": {
            "pixel_std_min": float(args.blank_std_min),
            "pixel_range_min": float(args.blank_range_min),
        },
        "qc_passed": bool(not failed and not missing_class_splits),
    }
    (args.output_root / "qc_summary.json").write_text(
        json.dumps(qc_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = {
        "format_version": 2,
        "images_found": len(indexed),
        "images_succeeded": succeeded,
        "images_failed": len(failed),
        "input_root": str(args.images_root),
        "output_root": str(args.output_root),
        "metadata": str(args.metadata),
        "metadata_sha256": sha256_file(args.metadata),
        "configuration_file": str(args.config),
        "configuration": cfg.__dict__,
        "normalization_applied": normalize,
        "qc_summary": qc_summary,
        "failures": failed[:100],
        "warnings": warnings,
    }
    (args.output_root / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_manifest(
        args.output_root / "manifest.json",
        {"preprocessing": summary, "qc": qc_summary},
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if qc_summary["qc_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
