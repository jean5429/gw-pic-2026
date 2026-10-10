from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from gwproject.config import load_toml


def main() -> int:
    parser = argparse.ArgumentParser(description="Executa as sementes definidas em gravity_spy.toml")
    parser.add_argument("metadata", type=Path)
    parser.add_argument("images_root", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/gravity_spy.toml"))
    parser.add_argument("--profile", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gravity_spy"))
    args = parser.parse_args()
    cfg = load_toml(args.config)
    failures = []
    for seed in cfg["training"]["seeds"]:
        command = [
            sys.executable,
            "-m",
            "scripts.train_gravity_spy",
            str(args.metadata),
            str(args.images_root),
            "--config",
            str(args.config),
            "--profile",
            args.profile,
            "--seed",
            str(seed),
            "--output-dir",
            str(args.output_dir),
        ]
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            failures.append({"seed": seed, "returncode": result.returncode})
    if failures:
        print(f"Falhas: {failures}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
