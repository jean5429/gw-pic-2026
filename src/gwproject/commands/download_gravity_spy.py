from __future__ import annotations

import argparse
from pathlib import Path

from gwproject.config import load_toml
from gwproject.downloads import download_with_resume, safe_extract_tar, verify_zenodo_file, zenodo_record_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Baixa Gravity Spy com checksum oficial do Zenodo")
    parser.add_argument("--config", type=Path, default=Path("configs/gravity_spy.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/gravity_spy"))
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    cfg = load_toml(args.config)
    data_cfg = cfg["data"]
    record_id = int(data_cfg["record_id"])
    metadata = str(data_cfg["metadata_filename"])
    archive = str(data_cfg["archive_filename"])

    descriptors = zenodo_record_files(record_id)
    for filename in (metadata, archive):
        if filename not in descriptors:
            raise SystemExit(f"Arquivo ausente no registro Zenodo: {filename}")
        target = args.output_dir / filename
        if not target.exists():
            download_with_resume(descriptors[filename]["url"], target)
        verify_zenodo_file(target, descriptors[filename])
        print(f"Verificado: {target}")

    if args.extract:
        safe_extract_tar(args.output_dir / archive, args.output_dir / "trainingset")
        print(f"Extraido em {args.output_dir / 'trainingset'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
