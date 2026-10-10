from __future__ import annotations

from pathlib import Path
from typing import Any
import tomllib


def load_toml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("rb") as handle:
        return tomllib.load(handle)
