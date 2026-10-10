from __future__ import annotations

import hashlib
import re
from importlib import metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_PACKAGES = (
    "numpy", "scipy", "pandas", "matplotlib", "h5py", "requests",
    "gwpy", "gwosc", "torch", "torchvision", "scikit-learn", "pillow",
)


def anonymize_paths(value: Any, home: str | Path | None = None) -> Any:
    """Troca a pasta do usuario por ``~`` em strings (recursivo em dicts/listas).

    Evita publicar o nome de usuario do sistema em manifestos e logs versionados.
    """
    home_text = str(home if home is not None else Path.home()).rstrip("\\/")
    if not home_text:
        return value
    variants = {home_text, home_text.replace("\\", "/")}
    pattern = re.compile("|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True)), re.IGNORECASE)

    def walk(item: Any) -> Any:
        if isinstance(item, str):
            return pattern.sub("~", item)
        if isinstance(item, dict):
            return {key: walk(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [walk(val) for val in item]
        return item

    return walk(value)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: str | Path = ".") -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def package_versions(packages: Iterable[str] = DEFAULT_PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def accelerator_manifest() -> dict[str, Any]:
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        result: dict[str, Any] = {
            "torch_cuda_available": cuda,
            "torch_cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
        if cuda:
            index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            result.update({
                "device_index": index,
                "device_name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": [properties.major, properties.minor],
            })
        return result
    except Exception as exc:
        return {"torch_cuda_available": False, "probe_error": str(exc)}


def runtime_manifest(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "git_commit": git_commit(),
        "packages": package_versions(),
        "accelerator": accelerator_manifest(),
        "environment": {
            "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
            "CONDA_PREFIX": os.environ.get("CONDA_PREFIX"),
        },
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(path: str | Path, extra: dict[str, Any] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(anonymize_paths(runtime_manifest(extra)), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
