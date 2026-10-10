from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tarfile
from typing import Any

import requests
from tqdm import tqdm


def download_with_resume(url: str, destination: str | Path, timeout: int = 60) -> Path:
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    downloaded = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
        response.raise_for_status()
        if response.status_code == 200 and downloaded:
            downloaded = 0
        mode = "ab" if downloaded else "wb"
        total = int(response.headers.get("content-length", 0)) + downloaded
        with partial.open(mode) as handle, tqdm(total=total, initial=downloaded, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
                    bar.update(len(chunk))
    os.replace(partial, dest)
    return dest


def checksum(path: str | Path, algorithm: str = "md5") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def zenodo_record_files(record_id: int) -> dict[str, dict[str, Any]]:
    response = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=60)
    response.raise_for_status()
    files: dict[str, dict[str, Any]] = {}
    for item in response.json().get("files", []):
        files[item["key"]] = {
            "url": item.get("links", {}).get("content") or item.get("links", {}).get("self"),
            "checksum": item.get("checksum"),
            "size": item.get("size"),
        }
    return files


def verify_zenodo_file(path: str | Path, descriptor: dict[str, Any]) -> None:
    expected_size = descriptor.get("size")
    if expected_size is not None and Path(path).stat().st_size != int(expected_size):
        raise ValueError(f"tamanho invalido: {Path(path).stat().st_size} != {expected_size}")
    expected = descriptor.get("checksum")
    if expected:
        algorithm, expected_hash = expected.split(":", 1)
        actual = checksum(path, algorithm)
        if actual.lower() != expected_hash.lower():
            raise ValueError(f"checksum invalido: {actual} != {expected_hash}")


def safe_extract_tar(archive: str | Path, destination: str | Path) -> None:
    dest = Path(destination).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if dest not in target.parents and target != dest:
                raise ValueError(f"entrada insegura no tar: {member.name}")
        try:
            tar.extractall(dest, filter="data")
        except TypeError:
            tar.extractall(dest)
