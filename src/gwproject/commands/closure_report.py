"""Avalia uma execucao de classify-ensemble sobre o conjunto de fechamento do Plano 1.

Junta ``per_channel.csv`` (saida de ``classify-ensemble``) aos rotulos de
``closure_channels.csv`` e calcula, para os splits de avaliacao:

- acuracia, acuracia balanceada e macro-F1 da classe fechada do ensemble;
- fracoes ``reject_ood`` / ``review`` / ``in_domain_candidate``;
- intervalos de 95% por bootstrap de arquivos GWOSC;
- os criterios C1 e C2 de ``configs/plano1.toml``.

O split de treino aparece apenas como analise secundaria.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gwproject.config import load_toml
from gwproject.provenance import sha256_file, write_manifest
from gwproject.research.closure import (
    classification_metrics,
    evaluate_criteria,
    grouped_bootstrap,
)

STATUSES = ("reject_ood", "review", "in_domain_candidate")


def _join(closure: pd.DataFrame, ensemble: pd.DataFrame) -> pd.DataFrame:
    labels = closure[closure["status"] == "ok"][
        ["event", "detector", "label", "sample_type", "file_start"]
    ]
    merged = ensemble.merge(labels, on=["event", "detector"], how="inner", validate="one_to_one")
    if len(merged) != len(ensemble):
        raise ValueError("per_channel.csv contem canais fora do conjunto de fechamento")
    return merged


def _block(frame: pd.DataFrame, min_class_samples: int) -> dict:
    metrics = classification_metrics(
        frame["label"].to_numpy(),
        frame["forced_closed_set_class"].to_numpy(),
        min_class_samples=min_class_samples,
    )
    status = frame["open_world_status"].value_counts(normalize=True)
    metrics["open_world_fractions"] = {name: float(status.get(name, 0.0)) for name in STATUSES}
    metrics["ood_mahalanobis_99_fraction"] = float(frame["ood_mahalanobis_99"].mean())
    metrics["ood_energy_99_fraction"] = float(frame["ood_energy_99"].mean())
    return metrics


def _per_class(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in frame.groupby("label"):
        predicted_as = frame[frame["forced_closed_set_class"] == label]
        recall = float(np.mean(group["forced_closed_set_class"] == label))
        precision = float(np.mean(predicted_as["label"] == label)) if len(predicted_as) else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(precision) and precision + recall > 0
            else 0.0
        )
        top_confusion = (
            group.loc[group["forced_closed_set_class"] != label, "forced_closed_set_class"]
            .value_counts()
            .head(1)
        )
        rows.append({
            "label": label,
            "n": int(len(group)),
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "reject_ood_fraction": float(np.mean(group["open_world_status"] == "reject_ood")),
            "most_common_error": top_confusion.index[0] if len(top_confusion) else "",
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    def cell(value) -> str:
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return lines


def _write_markdown(path: Path, label: str, report: dict, per_class: pd.DataFrame) -> None:
    primary = report["evaluation"]
    criteria = report["criteria"]
    lines = [
        f"# Teste de fechamento - {label}",
        "",
        f"Renderer: `{report['render_mode']}` | modelos: {report['n_models']} | "
        f"glitches de avaliacao: {primary['n']} em {report['evaluation_files']} arquivos",
        "",
        "| Criterio | Valor | Limiar | Passou |",
        "|---|---:|---:|---|",
    ]
    for name, item in criteria.items():
        if name == "all_passed":
            continue
        lines.append(
            f"| {name} | {item['value']:.3f} | {item['rule']} {item['threshold']:.2f} | "
            f"{'sim' if item['passed'] else 'NAO'} |"
        )
    ci = report["bootstrap_95"]
    lines += [
        "",
        f"**Veredito:** {'PASSOU' if criteria['all_passed'] else 'NAO PASSOU'}",
        "",
        f"Macro-F1 IC 95%: [{ci['macro_f1'][0]:.3f}; {ci['macro_f1'][1]:.3f}] | "
        f"reject_ood IC 95%: [{ci['reject_ood'][0]:.3f}; {ci['reject_ood'][1]:.3f}]",
        "",
        "Fracoes open-world: "
        + ", ".join(f"{k} {v:.3f}" for k, v in primary["open_world_fractions"].items()),
        "",
        "## Por classe (splits de avaliacao)",
        "",
        *_markdown_table(per_class),
        "",
        "Classes com menos de "
        f"{report['min_class_samples']} amostras nao entram no macro-F1.",
    ]
    if report.get("secondary_train"):
        train = report["secondary_train"]
        lines += [
            "",
            "## Secundario: triggers do split de treino",
            "",
            f"n={train['n']}, acuracia {train['accuracy']:.3f}, macro-F1 {train['macro_f1']:.3f}, "
            f"reject_ood {train['open_world_fractions']['reject_ood']:.3f}. "
            "A rede ja viu esses triggers como PNG; nao decide criterio.",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Relatorio do teste de fechamento (Plano 1).")
    parser.add_argument("ensemble_dir", type=Path, help="saida de classify-ensemble")
    parser.add_argument(
        "--closure",
        type=Path,
        default=Path("outputs/research/plano1/fechamento/closure_channels.csv"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/plano1.toml"))
    parser.add_argument("--label", default=None, help="nome curto da condicao (modelo x renderer)")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    plan_cfg = load_toml(args.config)
    criteria_cfg = plan_cfg["criteria"]
    boot_cfg = plan_cfg["bootstrap"]
    splits = [str(value) for value in plan_cfg["selection"]["evaluation_splits"]]
    min_class_samples = int(criteria_cfg["min_class_samples"])

    ensemble_summary = json.loads((args.ensemble_dir / "summary.json").read_text(encoding="utf-8"))
    frame = _join(pd.read_csv(args.closure), pd.read_csv(args.ensemble_dir / "per_channel.csv"))
    evaluation = frame[frame["sample_type"].isin(splits)]
    if evaluation.empty:
        raise ValueError("nenhum glitch de avaliacao no per_channel.csv")

    primary = _block(evaluation, min_class_samples)
    decisive = primary["decisive_classes"]

    def macro_f1(sample: pd.DataFrame) -> float:
        from sklearn.metrics import f1_score

        return float(f1_score(
            sample["label"], sample["forced_closed_set_class"],
            labels=decisive, average="macro", zero_division=0,
        ))

    def reject(sample: pd.DataFrame) -> float:
        return float(np.mean(sample["open_world_status"] == "reject_ood"))

    boot = {
        name: grouped_bootstrap(
            evaluation,
            statistic,
            group_column=str(boot_cfg["group_column"]),
            replicates=int(boot_cfg["replicates"]),
            seed=int(boot_cfg["seed"]),
        )
        for name, statistic in (("macro_f1", macro_f1), ("reject_ood", reject))
    }
    criteria = evaluate_criteria(primary, primary["open_world_fractions"]["reject_ood"], criteria_cfg)
    train = frame[frame["sample_type"] == "train"]

    label = args.label or args.ensemble_dir.name
    output = args.output_dir or args.ensemble_dir / "fechamento"
    output.mkdir(parents=True, exist_ok=True)
    per_class = _per_class(evaluation)
    per_class.to_csv(output / "per_class.csv", index=False)

    classes = sorted(set(evaluation["label"]) | set(evaluation["forced_closed_set_class"]))
    index = {name: position for position, name in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for true, predicted in zip(evaluation["label"], evaluation["forced_closed_set_class"]):
        matrix[index[true], index[predicted]] += 1
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(output / "confusion_matrix.csv")
    from gwproject.plotting import plot_confusion_matrix

    plot_confusion_matrix(
        matrix, classes, output / "confusion_matrix.png",
        title=f"Fechamento - {label}",
    )

    report = {
        "label": label,
        "render_mode": ensemble_summary.get("render_mode"),
        "n_models": ensemble_summary.get("n_models"),
        "checkpoints": ensemble_summary.get("checkpoints"),
        "evaluation_splits": splits,
        "evaluation_files": int(evaluation["file_start"].nunique()),
        "min_class_samples": min_class_samples,
        "evaluation": primary,
        "bootstrap_95": boot,
        "criteria": criteria,
        "secondary_train": _block(train, min_class_samples) if len(train) else None,
        "closure_sha256": sha256_file(args.closure),
        "plan_config": plan_cfg,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    _write_markdown(output / "REPORT.md", label, report, per_class)
    write_manifest(output / "manifest.json", {"report": report})
    print((output / "REPORT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
