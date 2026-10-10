"""Audit HDF5 identity keys and fixed pixel encodings; no fitting or inference."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gwproject.provenance import sha256_file


def canonical_split(value):
    mapping = {"train": "train", "training": "train", "validation": "val",
               "valid": "val", "val": "val", "test": "test", "testing": "test"}
    key = str(value).strip().lower()
    if key not in mapping:
        raise ValueError(f"unknown split: {value}")
    return mapping[key]


def index_hdf5(handle):
    """Visit every hard-linked object; do not dereference external/soft links."""
    import h5py
    rows, issues = [], []
    counts = Counter()
    seen = set()

    def walk(group):
        for name in group:
            path = group.name.rstrip("/") + "/" + name
            link = group.get(name, getlink=True)
            if not isinstance(link, h5py.HardLink):
                issues.append(dict(path=path, reason="non_hard_link_not_followed"))
                continue
            item = group[name]
            address = int(h5py.h5o.get_info(item.id).addr)
            if address in seen:
                issues.append(dict(path=path, reason="hard_link_alias_not_followed"))
                continue
            seen.add(address)
            if isinstance(item, h5py.Group):
                counts["groups"] += 1
                walk(item)
                continue
            counts["datasets"] += 1
            parts = path.strip("/").split("/")
            if len(parts) != 4:
                issues.append(dict(path=path, reason="unexpected_dataset_path"))
                continue
            label, split, identifier, duration = parts
            try:
                split = canonical_split(split)
            except ValueError:
                issues.append(dict(path=path, reason="unknown_split"))
                continue
            counts[f"duration:{duration}"] += 1
            if duration != "1.0.png":
                continue
            rows.append(dict(hdf5_path=path, label=label, split=split,
                             gravityspy_id=identifier, shape=str(item.shape), dtype=str(item.dtype),
                             readable_image=bool(item.shape == (1, 140, 170)
                                                 and item.dtype == np.dtype("float32")
                                                 and not item.is_virtual and not item.external)))
    seen.add(int(h5py.h5o.get_info(handle["/"].id).addr))
    walk(handle)
    return pd.DataFrame(rows, columns=["hdf5_path", "label", "split", "gravityspy_id", "shape", "dtype", "readable_image"]), dict(counts), issues


def pair_metadata(metadata, hdf_index, official_metadata):
    """An ID must be unique in full official metadata, not just after cleaning.

    HDF5 paths do not supply IFO. Recover it only through a globally unique ID,
    and require agreement of IFO, label and split. Never join by row position.
    """
    official = {str(key): part for key, part in official_metadata.groupby("gravityspy_id", dropna=False)}
    indexed = {str(key): part for key, part in hdf_index.groupby("gravityspy_id", dropna=False)}
    duplicate_clean = metadata.gravityspy_id.astype(str).duplicated(keep=False)
    records = []
    for position, (_, row) in enumerate(metadata.iterrows()):
        identifier = str(row.gravityspy_id)
        split = canonical_split(row.sample_type)
        record = dict(gravityspy_id=identifier, ifo=str(row.ifo).upper(), label=str(row.label),
                      split=split, hdf5_path="", status="matched")
        raw = official.get(identifier)
        candidates = indexed.get(identifier)
        if duplicate_clean.iloc[position]:
            record["status"] = "ambiguous_clean_id"
        elif raw is None or len(raw) != 1:
            record["status"] = "missing_or_ambiguous_official_id"
        elif (str(raw.iloc[0].ifo).upper(), str(raw.iloc[0].label), canonical_split(raw.iloc[0].sample_type)) != (record["ifo"], record["label"], split):
            record["status"] = "official_metadata_disagreement"
        elif candidates is None or len(candidates) != 1:
            record["status"] = "missing_or_ambiguous_hdf5_id"
        else:
            candidate = candidates.iloc[0]
            record["hdf5_path"] = candidate.hdf5_path
            if (candidate.label, candidate.split) != (record["label"], split):
                record["status"] = "hdf5_label_or_split_disagreement"
            elif not candidate.readable_image:
                record["status"] = "unsupported_array_schema"
        records.append(record)
    return pd.DataFrame(records)


def pixel_metrics(official, local):
    """Compare three fixed encodings, with no per-image rescaling or fitting."""
    array = np.asarray(official)
    pixels = np.asarray(local)
    if array.shape != (1, 140, 170) or pixels.shape != (140, 170) or pixels.dtype != np.uint8:
        raise ValueError("unexpected pixel shape or local dtype")
    x = array[0].astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite official pixels")
    result = dict(official_min=float(x.min()), official_max=float(x.max()),
                  official_mean=float(x.mean()), official_std=float(x.std()))
    for name, candidate in {"u8": pixels.astype(float), "unit": pixels.astype(float) / 255,
                            "signed": pixels.astype(float) / 127.5 - 1}.items():
        difference = np.abs(x - candidate)
        result[f"mae_{name}"] = float(difference.mean())
        result[f"max_abs_{name}"] = float(difference.max())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.parent != Path("outputs/research").resolve() or not output.name.startswith("official_pairs_"):
        raise ValueError("use a new outputs/research/official_pairs_* directory")
    output.mkdir(parents=True, exist_ok=False)
    source = Path("data/gravity_spy_official/trainingsetv1d1.h5")
    metadata_path = Path("data/gravity_spy_harmonized/clean_metadata.csv")
    raw_path = Path("data/gravity_spy/trainingset_v1d1_metadata.csv")
    panels = Path("data/gravity_spy_harmonized")
    from gwproject.gravity_spy import index_spectrograms, validate_harmonized_dataset
    import h5py
    from PIL import Image
    validate_harmonized_dataset(panels, metadata_path)
    if source.stat().st_size != 3051325064:
        raise ValueError("unexpected official HDF5 size")
    print("Verificando integridade do HDF5 por leitura sequencial; sem GPU.", flush=True)
    md5, digest = hashlib.md5(usedforsecurity=False), hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            md5.update(chunk)
            digest.update(chunk)
    if md5.hexdigest() != "a6665b4ebee7b4ea57164ee743c2bd11":
        raise ValueError("official HDF5 MD5 mismatch")
    if hashlib.md5(raw_path.read_bytes(), usedforsecurity=False).hexdigest() != "c16d135f5ad3f029a3ef32114e280e79":
        raise ValueError("raw metadata differs from the official inventory")
    metadata = pd.read_csv(metadata_path)
    raw_metadata = pd.read_csv(raw_path)
    for frame in (metadata, raw_metadata):
        if frame[["gravityspy_id", "ifo", "label", "sample_type"]].isna().any().any():
            raise ValueError("missing metadata identifiers")
    image_index = index_spectrograms(panels, "1.0")
    pixel_rows = []
    with h5py.File(source, "r") as handle:
        print("Inventariando a estrutura completa, sem limite de 2000 objetos.", flush=True)
        indexed, counts, issues = index_hdf5(handle)
        indexed.to_csv(output / "hdf5_index_1s.csv", index=False)
        paired = pair_metadata(metadata, indexed, raw_metadata)
        paired.to_csv(output / "pairing_all_clean.csv", index=False)
        validation = paired[paired.split.eq("val")]
        for i, row in enumerate(validation.itertuples()):
            item = dict(gravityspy_id=row.gravityspy_id, ifo=row.ifo, label=row.label,
                        hdf5_path=row.hdf5_path, status=row.status)
            if row.status == "matched":
                try:
                    path = Path(image_index[(row.ifo, row.gravityspy_id)])
                    with Image.open(path) as image:
                        pixels = np.asarray(image.convert("L"))
                    item.update(pixel_metrics(handle[row.hdf5_path][...], pixels))
                    item["png_sha256"] = sha256_file(path)
                except (ValueError, KeyError, OSError) as error:
                    item.update(status="pixel_read_or_validation_error", error=str(error))
            pixel_rows.append(item)
            if (i + 1) % 200 == 0:
                print(f"Pares de validacao: {i + 1}/{len(validation)}", flush=True)
    pixels_frame = pd.DataFrame(pixel_rows)
    pixels_frame.to_csv(output / "validation_pixel_comparison.csv", index=False)
    successes = pixels_frame[pixels_frame.status.eq("matched")] if len(pixels_frame) else pixels_frame
    metrics = [key for key in successes.columns if key.startswith(("mae_", "max_abs_", "official_"))]
    description = {key: {"min": float(successes[key].min()), "median": float(successes[key].median()),
                         "max": float(successes[key].max())} for key in metrics if len(successes)}
    matched = int(paired.status.eq("matched").sum())
    summary = dict(status="complete", structural_counts=counts, structural_issues=issues,
                   clean_rows=len(paired), matched_clean_rows=matched,
                   pairing_status_counts={str(k): int(v) for k, v in paired.status.value_counts().items()},
                   validation_rows=len(validation), validation_pixel_pairs=len(successes),
                   coverage_gate=bool(len(paired) == 7807 and matched == 7807 and len(validation) == 1177
                                      and len(successes) == 1177 and not issues),
                   pixel_metrics=description, test_pixel_arrays_read=0, train_pixel_arrays_read=0,
                   provenance=dict(hdf5_sha256=digest.hexdigest(), metadata_sha256=sha256_file(metadata_path),
                                   raw_metadata_sha256=sha256_file(raw_path), code_sha256=sha256_file(__file__)),
                   interpretation=["A completed audit can have a failed coverage gate; inspect counts and reasons.",
                       "Identity pairing and pixel agreement do not establish physical renderer equivalence.",
                       "u8, unit and signed are fixed diagnostic encodings, not fitted normalizations.",
                       "No per-image best encoding is chosen; no interpolation, clipping or orientation correction is applied.",
                       "No CNN inference, training, downloads or catalog reclassification were performed."])
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Concluido: {output / 'summary.json'}; coverage_gate={summary['coverage_gate']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
