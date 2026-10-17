from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gwproject.research.phase import randomize_fourier_phase


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gera contrafactuais de fase que preservam o modulo espectral."
    )
    parser.add_argument("input", type=Path, help="NPY 1D ou CSV com coluna path")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research/phase_counterfactuals"))
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.input.suffix.lower() == ".csv":
        frame = pd.read_csv(args.input)
        if "path" not in frame.columns:
            raise ValueError("CSV deve conter coluna path")
        paths = [Path(value) for value in frame["path"]]
    else:
        paths = [args.input]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for source_index, source in enumerate(paths):
        signal = np.load(source, allow_pickle=False)
        if signal.ndim != 1:
            raise ValueError(f"{source}: esperado vetor 1D")
        original_magnitude = np.abs(np.fft.rfft(signal))
        for fraction in args.fractions:
            counterfactual = randomize_fourier_phase(
                signal,
                fraction=fraction,
                seed=args.seed + source_index,
            )
            destination = args.output_dir / f"{source.stem}_phase_{fraction:.2f}.npy"
            np.save(destination, counterfactual)
            mismatch = float(np.max(np.abs(np.abs(np.fft.rfft(counterfactual)) - original_magnitude)))
            manifest.append({
                "source": str(source),
                "output": str(destination),
                "phase_fraction": float(fraction),
                "maximum_magnitude_spectrum_difference": mismatch,
            })
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"counterfactuals": len(manifest), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
