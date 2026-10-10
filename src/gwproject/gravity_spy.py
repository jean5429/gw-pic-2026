from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BackboneProfile:
    name: str
    conv1_stride: int
    use_maxpool: bool
    batch_size: int
    num_workers: int
    amp: bool


def set_deterministic_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = not deterministic
    except ImportError:
        pass


def resolve_device(requested: str = "auto"):
    import torch
    requested = requested.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device deve ser auto, cpu ou cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitada, mas torch.cuda.is_available() e False")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def choose_profile(config: dict, device_type: str, requested: str = "auto") -> BackboneProfile:
    profiles = config["profiles"]
    if requested == "auto":
        name = "cuda_balanced" if device_type == "cuda" else "cpu"
    else:
        name = requested
    if name not in profiles:
        raise KeyError(f"perfil inexistente: {name}")
    profile = profiles[name]
    return BackboneProfile(
        name=name,
        conv1_stride=int(profile["conv1_stride"]),
        use_maxpool=bool(profile["use_maxpool"]),
        batch_size=int(profile["batch_size"]),
        num_workers=int(profile["num_workers"]),
        amp=bool(profile["amp"]),
    )


def md5_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: str | Path, expected_md5: str) -> None:
    actual = md5_file(path)
    if actual.lower() != expected_md5.lower():
        raise ValueError(f"MD5 invalido para {path}: {actual} != {expected_md5}")


def split_of(sample_type: object) -> str:
    value = str(sample_type).strip().lower()
    if value.startswith("valid"):
        return "val"
    if value.startswith("test"):
        return "test"
    return "train"


def official_splits(metadata_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    required = {"gravityspy_id", "label", "sample_type"}
    missing = required - set(metadata_frame.columns)
    if missing:
        raise ValueError(f"colunas ausentes: {sorted(missing)}")
    df = metadata_frame.copy()
    df["split"] = df["sample_type"].map(split_of)
    duplicated = df.groupby("gravityspy_id")["split"].nunique()
    if (duplicated > 1).any():
        bad = duplicated[duplicated > 1].index.tolist()[:5]
        raise ValueError(f"IDs presentes em mais de um split: {bad}")
    return {name: part.reset_index(drop=True) for name, part in df.groupby("split")}


def adapt_conv1_to_grayscale(weight, out_kernel: int | None = None):
    """Converte pesos RGB para um canal preservando o ganho da entrada RGB replicada."""
    import torch
    import torch.nn.functional as functional

    if weight.ndim != 4 or weight.shape[1] != 3:
        raise ValueError("weight deve ter shape (out_channels, 3, k, k)")
    gray = weight.sum(dim=1, keepdim=True)
    if out_kernel is not None and out_kernel != gray.shape[-1]:
        if out_kernel < 1:
            raise ValueError("out_kernel deve ser positivo")
        original_sum = gray.sum(dim=(-2, -1), keepdim=True)
        gray = functional.interpolate(gray, size=(out_kernel, out_kernel), mode="bilinear", align_corners=False)
        new_sum = gray.sum(dim=(-2, -1), keepdim=True)
        safe = torch.where(new_sum.abs() > 1e-12, new_sum, torch.ones_like(new_sum))
        gray = gray * (original_sum / safe)
    return gray


def build_resnet18_grayscale(
    num_classes: int,
    pretrained: bool = True,
    *,
    conv1_stride: int = 2,
    use_maxpool: bool = False,
):
    import torch
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    if conv1_stride not in {1, 2}:
        raise ValueError("conv1_stride deve ser 1 ou 2")
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    old = model.conv1
    new = nn.Conv2d(1, old.out_channels, kernel_size=7, stride=conv1_stride, padding=3, bias=False)
    with torch.no_grad():
        if pretrained:
            new.weight.copy_(adapt_conv1_to_grayscale(old.weight))
        else:
            nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
    model.conv1 = new
    if not use_maxpool:
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_backbone(model, freeze: bool = True) -> None:
    """Na etapa congelada, conv1/bn1 e cabeca continuam treinaveis."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (
            not freeze
            or name.startswith("conv1.")
            or name.startswith("bn1.")
            or name.startswith("fc.")
        )


def assert_real_dataset(paths: Iterable[str | os.PathLike[str]]) -> None:
    suspicious = [str(path) for path in paths if "synthetic" in Path(path).name.lower()]
    if suspicious:
        raise ValueError(f"dados sinteticos proibidos no fluxo cientifico: {suspicious[:3]}")


def index_spectrograms(root: str | Path, duration: str = "1.0") -> dict[str, str]:
    import re
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(root_path)
    pattern = re.compile(r"([A-Za-z0-9]{10})_spectrogram_" + re.escape(duration) + r"\.png$")
    index: dict[str, str] = {}
    for path in root_path.rglob(f"*spectrogram_{duration}.png"):
        match = pattern.search(path.name)
        if match:
            index[match.group(1)] = str(path)
    if not index:
        raise RuntimeError(f"nenhum espectrograma de {duration} s encontrado em {root_path}")
    return index


def attach_image_paths(metadata_frame: pd.DataFrame, image_index: dict[str, str]) -> pd.DataFrame:
    if "gravityspy_id" not in metadata_frame.columns:
        raise ValueError("metadata sem gravityspy_id")
    df = metadata_frame.copy()
    df["path"] = df["gravityspy_id"].map(image_index)
    return df.dropna(subset=["path"]).reset_index(drop=True)


class RandomIntensityScale:
    """Transformacao picklable para DataLoader com spawn no Windows."""

    def __init__(self, low: float = 0.9, high: float = 1.1):
        self.low = float(low)
        self.high = float(high)

    def __call__(self, tensor):
        import torch

        return torch.clamp(tensor * random.uniform(self.low, self.high), 0.0, 1.0)


class GravitySpyImageDataset:
    """Dataset definido no nivel do modulo para funcionar com workers no Windows."""

    def __init__(self, frame: pd.DataFrame, transform):
        self.records = frame[["path", "class_idx"]].to_dict(orient="records")
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image

        row = self.records[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("L"))
        return tensor, int(row["class_idx"])
