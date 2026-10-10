"""Inventory official file descriptors or local HDF5 structure, without downloads.

Dataset names/shapes are not evidence of raw strain or of a valid trigger join.
No HDF5 dataset values are loaded. Soft/external links are not followed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def hdf5_schema(path, *, maximum_objects=2000):
    import h5py
    resolved = Path(path).resolve()
    if {part.lower() for part in resolved.parts} & {"atenuacao_ridge", "ridge_atenuacao", "ridge_atenuacao_v2"}:
        raise ValueError("directory excluded from this project review")
    objects = []
    seen = set()
    truncated = False
    with h5py.File(resolved, "r") as handle:
        def walk(group, depth=0):
            nonlocal truncated
            if depth > 20:
                truncated = True
                return
            for name in group:
                if len(objects) >= maximum_objects:
                    truncated = True
                    return
                full_name = group.name.rstrip("/") + "/" + name
                link = group.get(name, getlink=True)
                if not isinstance(link, h5py.HardLink):
                    objects.append(dict(name=full_name, kind=type(link).__name__, followed=False))
                    continue
                item = group[name]
                address = int(h5py.h5o.get_info(item.id).addr)
                if address in seen:
                    objects.append(dict(name=full_name, kind="hard_link_alias", followed=False))
                    continue
                seen.add(address)
                if isinstance(item, h5py.Dataset):
                    objects.append(dict(name=full_name, kind="dataset", shape=item.shape,
                                        dtype=str(item.dtype), chunks=item.chunks,
                                        compression=item.compression, virtual=bool(item.is_virtual),
                                        attribute_names=list(item.attrs.keys())))
                else:
                    objects.append(dict(name=full_name, kind="group", attribute_names=list(item.attrs.keys())))
                    walk(item, depth + 1)
        walk(handle)
    return dict(path=str(resolved), bytes=resolved.stat().st_size,
                truncated=truncated, objects=objects, dataset_values_read=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-id", type=int,
                        help="Query this Zenodo record (default 1486046 when no local files are given).")
    parser.add_argument("--local-hdf5", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new output filename; an existing inventory is never overwritten")
    record = args.record_id
    if record is None and not args.local_hdf5:
        record = 1486046
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  record_id=record, downloads_performed=False, local_schemas=[])
    if record is not None:
        from gwproject.downloads import zenodo_record_files
        descriptors = zenodo_record_files(record)
        result["remote_files"] = [dict(name=name, **descriptor,
            hdf5_extension_candidate=Path(name).suffix.lower() in {".h5", ".hdf5", ".hdf"})
            for name, descriptor in sorted(descriptors.items())]
    for path in args.local_hdf5:
        result["local_schemas"].append(hdf5_schema(path))
    result["next_gate"] = [
        "Confirm whether arrays are strain, numerical spectrograms or already rendered/normalized images.",
        "Identify explicit IFO + trigger ID or GPS keys, duration, frequency grid, units and preprocessing.",
        "Do not assume row order matches metadata; validate uniqueness and report unmatched entries.",
        "Only compare renderers on the same paired input after these identity/representation checks.",
        "No renderer equivalence or availability of raw strain has been established by this inventory.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against another process creating the file.
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Inventory written to {args.output}. No arrays downloaded or evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
