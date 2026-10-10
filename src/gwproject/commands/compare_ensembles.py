"""Compara duas execucoes de classify-ensemble sobre os mesmos canais do catalogo.

A comparacao so e aceita com os mesmos canais, checkpoints, referencia OOD e limiares,
de modo que apenas o pre-processamento (ou o ambiente) difira. Os detectores sao
agregados dentro de cada evento antes do bootstrap pareado por evento. A analise e
exploratoria: nao ajusta limiares nem escolhe renderer pelo resultado.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gwproject.provenance import sha256_file, write_manifest
from gwproject.research.paired_intensity import paired_bootstrap

KEYS = ["event", "detector"]
STATUS_METRICS = {
    "reject_ood": "reject_ood",
    "review": "review",
    "in_domain_candidate": "in_domain_candidate",
}
FLAG_METRICS = (
    "ood_mahalanobis_99",
    "ood_energy_99",
    "ood_mahalanobis_95",
    "ood_energy_95",
)
CONTINUOUS_METRICS = (
    "mahalanobis_log_excess_99",
    "mahalanobis_percentile_median",
    "energy_percentile_median",
    "closed_set_ensemble_confidence",
    "vote_fraction",
    "mutual_information",
)
INTERPRETATION = [
    "Comparacao exploratoria e pareada pelos mesmos canais; nao ha teste confirmatorio nem correcao por multiplicidade.",
    "Rotulos fechados forcados sao diagnosticos, nunca verdade fisica dos eventos.",
    "Uma queda da rejeicao OOD nao valida fisicamente o renderer; uma rejeicao mantida nao identifica sozinha a causa do shift.",
    "Intervalos de 95% por bootstrap de eventos, condicionados a este ensemble e a este catalogo.",
]


def _load_run(directory: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(directory / "per_channel.csv")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if frame.duplicated(KEYS).any():
        raise ValueError(f"canais duplicados em {directory}")
    return frame, summary


def _checkpoint_hashes(summary: dict) -> list[str]:
    return sorted(str(item["sha256"]) for item in summary["checkpoints"])


def validate_pairing(baseline: tuple[pd.DataFrame, dict], variant: tuple[pd.DataFrame, dict]) -> dict:
    """Recusa pares que diferem em canais, modelos, referencia OOD ou limiares."""
    base_frame, base_summary = baseline
    var_frame, var_summary = variant
    base_keys = set(map(tuple, base_frame[KEYS].astype(str).to_numpy()))
    var_keys = set(map(tuple, var_frame[KEYS].astype(str).to_numpy()))
    if base_keys != var_keys:
        raise ValueError(
            f"canais diferentes: {len(base_keys - var_keys)} so na base, "
            f"{len(var_keys - base_keys)} so na variante"
        )
    merged = base_frame[KEYS + ["qgram_path"]].merge(
        var_frame[KEYS + ["qgram_path"]], on=KEYS, suffixes=("_base", "_variant")
    )
    if not merged["qgram_path_base"].astype(str).eq(merged["qgram_path_variant"].astype(str)).all():
        raise ValueError("o mesmo canal aponta para Q-grams diferentes")
    if _checkpoint_hashes(base_summary) != _checkpoint_hashes(var_summary):
        raise ValueError("checkpoints diferentes; a comparacao exige os mesmos modelos")
    base_ood = dict(base_summary["ood_calibration"])
    var_ood = dict(var_summary["ood_calibration"])
    for key in ("reference_sha256", "minimum_ood_model_fraction", "energy_quantiles",
                "mahalanobis_quantiles", "maximum_mutual_information"):
        if base_ood.get(key) != var_ood.get(key):
            raise ValueError(f"calibracao OOD diferente em {key}")
    base_pre = dict(base_summary.get("preprocessing", {}))
    var_pre = dict(var_summary.get("preprocessing", {}))
    differences = {
        key: {"baseline": base_pre.get(key), "variant": var_pre.get(key)}
        for key in sorted(set(base_pre) | set(var_pre))
        if base_pre.get(key) != var_pre.get(key)
    }
    return {
        "channels": len(base_keys),
        "events": int(base_frame["event"].nunique()),
        "preprocessing_differences": differences,
    }


def channel_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Metricas numericas por canal; ausentes na execucao ficam fora da comparacao."""
    result = frame[KEYS].astype(str).copy()
    status = frame["open_world_status"].astype(str)
    for name, value in STATUS_METRICS.items():
        result[name] = status.eq(value).astype(float)
    for name in FLAG_METRICS:
        result[name] = frame[name].astype(str).str.lower().eq("true").astype(float)
    for name in CONTINUOUS_METRICS:
        if name in frame.columns:
            result[name] = pd.to_numeric(frame[name], errors="raise").astype(float)
    result["forced_closed_set_class"] = frame["forced_closed_set_class"].astype(str)
    result["open_world_status"] = status
    return result


def compare(baseline: pd.DataFrame, variant: pd.DataFrame, *, seed: int, replicates: int) -> dict:
    base = channel_metrics(baseline)
    var = channel_metrics(variant)
    metrics = [
        name
        for name in [*STATUS_METRICS, *FLAG_METRICS, *CONTINUOUS_METRICS]
        if name in base.columns and name in var.columns
    ]
    paired = base.merge(var, on=KEYS, suffixes=("_baseline", "_variant"), validate="one_to_one")
    for name in metrics:
        paired[f"{name}_delta"] = paired[f"{name}_variant"] - paired[f"{name}_baseline"]
    paired["forced_class_changed"] = (
        paired["forced_closed_set_class_baseline"] != paired["forced_closed_set_class_variant"]
    ).astype(float)

    columns = [f"{name}_{side}" for name in metrics for side in ("baseline", "variant", "delta")]
    per_event = paired.groupby("event", sort=True)[columns + ["forced_class_changed"]].mean()
    per_event.insert(0, "n_channels", paired.groupby("event", sort=True).size())

    contrasts = []
    for name in metrics:
        contrasts.append({
            "metric": name,
            "baseline_event_mean": float(per_event[f"{name}_baseline"].mean()),
            "variant_event_mean": float(per_event[f"{name}_variant"].mean()),
            "baseline_channel_sum": float(paired[f"{name}_baseline"].sum()),
            "variant_channel_sum": float(paired[f"{name}_variant"].sum()),
            **{
                f"delta_{key}": value
                for key, value in paired_bootstrap(
                    per_event[f"{name}_delta"].to_numpy(), seed=seed, replicates=replicates
                ).items()
            },
        })
    changed = paired_bootstrap(per_event["forced_class_changed"].to_numpy(), seed=seed, replicates=replicates)
    contrasts.append({
        "metric": "forced_class_changed",
        "baseline_event_mean": 0.0,
        "variant_event_mean": changed["estimate"],
        "baseline_channel_sum": 0.0,
        "variant_channel_sum": float(paired["forced_class_changed"].sum()),
        **{f"delta_{key}": value for key, value in changed.items()},
    })
    status_transitions = pd.crosstab(
        paired["open_world_status_baseline"], paired["open_world_status_variant"]
    )
    class_transitions = (
        paired.groupby(["forced_closed_set_class_baseline", "forced_closed_set_class_variant"])
        .size()
        .rename("channels")
        .reset_index()
        .sort_values("channels", ascending=False)
    )
    return {
        "paired": paired,
        "per_event": per_event.reset_index(),
        "contrasts": pd.DataFrame(contrasts),
        "status_transitions": status_transitions,
        "class_transitions": class_transitions,
    }


def _report(summary: dict, contrasts: pd.DataFrame) -> str:
    rows = {row["metric"]: row for row in contrasts.to_dict(orient="records")}
    lines = [
        "# Comparacao pareada de execucoes do ensemble",
        "",
        f"Base: `{summary['baseline']['directory']}`  ",
        f"Variante: `{summary['variant']['directory']}`  ",
        f"Cobertura: {summary['validation']['events']} eventos, "
        f"{summary['validation']['channels']} canais, mesmos checkpoints e referencia OOD.",
        "",
        "Diferencas de pre-processamento: "
        + (
            ", ".join(
                f"`{key}`: {value['baseline']} -> {value['variant']}"
                for key, value in summary["validation"]["preprocessing_differences"].items()
            )
            or "nenhuma"
        ),
        "",
        "| Metrica | Base (media por evento) | Variante | Delta [IC 95%] | Canais base -> variante |",
        "|---|---:|---:|---|---|",
    ]
    for metric, row in rows.items():
        lines.append(
            f"| {metric} | {row['baseline_event_mean']:.4f} | {row['variant_event_mean']:.4f} | "
            f"{row['delta_estimate']:+.4f} [{row['delta_ci_low']:+.4f}, {row['delta_ci_high']:+.4f}] | "
            f"{row['baseline_channel_sum']:.0f} -> {row['variant_channel_sum']:.0f} |"
        )
    lines += ["", "Para metricas continuas, a coluna de canais soma valores e serve so como conferencia.", ""]
    lines += ["## Limites de interpretacao", ""] + [f"- {line}" for line in INTERPRETATION]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="pasta de classify-ensemble usada como base")
    parser.add_argument("variant", type=Path, help="pasta de classify-ensemble comparada")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()

    output = args.output_dir
    if output.resolve() in {args.baseline.resolve(), args.variant.resolve()}:
        raise ValueError("a saida nao pode ser uma das pastas comparadas")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"pasta de saida nao vazia: {output}")

    baseline = _load_run(args.baseline)
    variant = _load_run(args.variant)
    validation = validate_pairing(baseline, variant)
    result = compare(baseline[0], variant[0], seed=args.seed, replicates=args.bootstrap_replicates)

    output.mkdir(parents=True, exist_ok=True)
    result["paired"].to_csv(output / "per_channel_comparison.csv", index=False)
    result["per_event"].to_csv(output / "per_event_comparison.csv", index=False)
    result["contrasts"].to_csv(output / "contrasts.csv", index=False)
    result["status_transitions"].to_csv(output / "status_transitions.csv")
    result["class_transitions"].to_csv(output / "forced_class_transitions.csv", index=False)

    def describe(directory: Path, summary: dict) -> dict:
        return {
            "directory": str(directory),
            "per_channel_sha256": sha256_file(directory / "per_channel.csv"),
            "summary_sha256": sha256_file(directory / "summary.json"),
            "render_mode": summary.get("preprocessing", {}).get("render_mode", "percentile"),
            "open_world_counts": summary.get("open_world_counts"),
            "forced_closed_set_class_counts": summary.get("forced_closed_set_class_counts"),
        }

    summary = {
        "status": "completed_exploratory_not_confirmatory",
        "baseline": describe(args.baseline, baseline[1]),
        "variant": describe(args.variant, variant[1]),
        "validation": validation,
        "bootstrap": {"seed": args.seed, "replicates": args.bootstrap_replicates, "block": "event"},
        "contrasts": result["contrasts"].to_dict(orient="records"),
        "interpretation": INTERPRETATION,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (output / "REPORT.md").write_text(_report(summary, result["contrasts"]), encoding="utf-8")
    write_manifest(output / "manifest.json", {"comparison": summary})
    print((output / "REPORT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
