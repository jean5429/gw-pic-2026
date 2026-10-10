from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.fillna("").value_counts().items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume os resultados do ridge do catalogo")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/gwtc/ridge_catalog"),
    )
    parser.add_argument("--top-reasons", type=int, default=20)
    args = parser.parse_args()

    per_channel_path = args.input_dir / "per_channel.csv"
    per_event_path = args.input_dir / "per_event.csv"
    if not per_channel_path.exists() or not per_event_path.exists():
        raise FileNotFoundError("per_channel.csv ou per_event.csv ausente")

    channels = pd.read_csv(per_channel_path)
    events = pd.read_csv(per_event_path)
    ok = channels[channels["status"] == "ok"].copy()

    relative_error = pd.to_numeric(ok.get("relative_error"), errors="coerce")
    finite_error = relative_error[np.isfinite(relative_error)]
    fit_method_counts = _counts(ok["fit_method"]) if "fit_method" in ok else {}

    if "reason" not in channels.columns:
        channels["reason"] = ""
    reasons = channels.loc[channels["status"] != "ok", ["status", "reason"]].copy()
    reasons["reason"] = reasons["reason"].fillna("")
    reason_counts = (
        reasons.groupby(["status", "reason"], dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(max(1, args.top_reasons))
    )

    summary = {
        "channels_total": int(len(channels)),
        "channel_status_counts": _counts(channels["status"]),
        "status_by_detector": {
            str(detector): _counts(group["status"])
            for detector, group in channels.groupby("detector")
        },
        "successful_fit_method_counts": fit_method_counts,
        "successful_relative_error": {
            "count": int(len(finite_error)),
            "median": float(finite_error.median()) if len(finite_error) else None,
            "mean": float(finite_error.mean()) if len(finite_error) else None,
            "p90": float(finite_error.quantile(0.90)) if len(finite_error) else None,
            "passed_30_percent": int((finite_error <= 0.30).sum()),
        },
        "events_total": int(len(events)),
        "event_status_counts": _counts(events["status"]),
        "top_failure_reasons": [
            {"status": str(status), "reason": str(reason), "count": int(count)}
            for (status, reason), count in reason_counts.items()
        ],
    }

    output = args.input_dir / "diagnostic_summary.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
