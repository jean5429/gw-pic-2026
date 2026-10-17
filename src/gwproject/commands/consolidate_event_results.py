from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gwproject.provenance import sha256_file, write_manifest


def consolidate_event_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Une resultados por evento sob contrato estrito um-para-um."""

    if not frames:
        raise ValueError("nenhuma tabela fornecida")
    event_sets: dict[str, set[str]] = {}
    prepared: dict[str, pd.DataFrame] = {}
    for prefix, frame in frames.items():
        if "event" not in frame.columns:
            raise ValueError(f"tabela {prefix} sem coluna event")
        if frame["event"].astype(str).duplicated().any():
            raise ValueError(f"tabela {prefix} possui eventos duplicados")
        copy = frame.copy()
        copy["event"] = copy["event"].astype(str)
        event_sets[prefix] = set(copy["event"])
        copy = copy.rename(
            columns={
                column: f"{prefix}_{column}"
                for column in copy.columns
                if column != "event"
            }
        )
        prepared[prefix] = copy

    reference_name = next(iter(event_sets))
    reference_events = event_sets[reference_name]
    for name, events in event_sets.items():
        if events != reference_events:
            missing = sorted(reference_events - events)[:5]
            extra = sorted(events - reference_events)[:5]
            raise ValueError(
                f"escopo de eventos divergente em {name}: "
                f"ausentes={missing}, extras={extra}"
            )

    result: pd.DataFrame | None = None
    for frame in prepared.values():
        result = (
            frame
            if result is None
            else result.merge(frame, on="event", how="inner", validate="one_to_one")
        )
    if result is None:
        raise RuntimeError("falha ao consolidar tabelas")
    return result.sort_values("event").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Consolida QC, CNN/OOD, ridge e matched filter para todos os eventos."
    )
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--ridge", type=Path, required=True)
    parser.add_argument("--matched-filter", type=Path, required=True)
    parser.add_argument("--expected-events", type=int, default=218)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/research/gwtc4_218/all_218_events.csv"),
    )
    args = parser.parse_args()

    inputs = {
        "qc": args.quality,
        "cnn": args.ensemble,
        "ridge": args.ridge,
        "mf": args.matched_filter,
    }
    frames = {name: pd.read_csv(path) for name, path in inputs.items()}
    result = consolidate_event_frames(frames)
    if len(result) != args.expected_events:
        raise ValueError(
            f"esperados {args.expected_events} eventos, obtidos {len(result)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    summary = {
        "events": int(len(result)),
        "columns": int(len(result.columns)),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
        "contract": (
            "As quatro tabelas devem conter exatamente o mesmo conjunto de eventos, "
            "sem duplicatas, e sao unidas com validacao one-to-one."
        ),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_manifest(args.output.with_suffix(".manifest.json"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
