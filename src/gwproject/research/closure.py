"""Teste de fechamento do renderer (Plano 1, "mesma camera").

Glitches Gravity Spy rotulados sao reconstruidos a partir do strain publico do GWOSC
e passam pela MESMA cadeia dos eventos do catalogo (janela de 32 s, Q-transform GWPy
em 20-512 Hz, renderer escolhido). Se a CNN reconhecer esses glitches como reconhece
os PNGs oficiais, a cadeia esta validada; se nao, a falha nos eventos pode vir da
representacao e nao do conteudo fisico.

Este modulo guarda apenas funcoes puras (selecao de arquivos e metricas), testaveis
sem rede nem GPU.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def file_start(gps: pd.Series | np.ndarray, file_seconds: int) -> np.ndarray:
    """Inicio do arquivo GWOSC alinhado que contem cada GPS."""
    values = np.asarray(gps, dtype=float)
    return (np.floor(values / file_seconds) * file_seconds).astype(np.int64)


def annotate_files(
    metadata: pd.DataFrame,
    *,
    run_start_gps: float,
    run_end_gps: float,
    file_seconds: int,
    before_seconds: float,
    after_seconds: float,
) -> pd.DataFrame:
    """Restringe ao run e marca arquivo e se a janela inteira cabe nele."""
    frame = metadata[
        (metadata["event_time"] >= run_start_gps) & (metadata["event_time"] < run_end_gps)
    ].copy()
    frame["file_start"] = file_start(frame["event_time"], file_seconds)
    frame["window_inside_file"] = (
        (frame["event_time"] - before_seconds >= frame["file_start"])
        & (frame["event_time"] + after_seconds <= frame["file_start"] + file_seconds)
    )
    return frame


def select_files(
    annotated: pd.DataFrame,
    *,
    evaluation_splits: list[str],
    max_files: int,
    per_class_target: int,
) -> pd.DataFrame:
    """Escolha gulosa de arquivos (detector, inicio) para cobrir todas as classes.

    A cobertura de cada classe, limitada a ``per_class_target`` amostras de avaliacao,
    vale ``log(1 + n)``. A cada passo entra o arquivo que mais aumenta a soma: a primeira
    amostra de uma classe rara (Tomte, No_Glitch aparecem um por arquivo) vale mais que
    a vigesima de uma classe comum. Empates vao para o arquivo com mais amostras de
    avaliacao. Para quando o orcamento acaba ou nenhum arquivo acrescenta cobertura.
    So conta glitches cuja janela cabe no arquivo.
    """
    usable = annotated[
        annotated["window_inside_file"] & annotated["sample_type"].isin(evaluation_splits)
    ]
    if usable.empty:
        raise ValueError("nenhum glitch de avaliacao cabe nos arquivos do run")
    counts = (
        usable.groupby(["ifo", "file_start", "label"]).size().unstack("label", fill_value=0)
    )
    classes = list(counts.columns)
    matrix = counts.to_numpy(dtype=np.int64)
    target = int(per_class_target)
    covered = np.zeros(len(classes), dtype=np.int64)
    available = np.ones(len(counts), dtype=bool)
    chosen: list[dict[str, Any]] = []
    for order in range(int(max_files)):
        after = np.minimum(covered + matrix, target)
        gains = (np.log1p(after) - np.log1p(covered)).sum(axis=1)
        gains = np.where(available, gains, -1.0)
        best_gain = float(gains.max())
        if best_gain <= 1e-12:
            break
        candidates = np.flatnonzero(np.isclose(gains, best_gain))
        totals = matrix[candidates].sum(axis=1)
        index = int(candidates[np.argmax(totals)])
        new_samples = int((after[index] - covered).sum())
        covered = after[index]
        available[index] = False
        ifo, start = counts.index[index]
        chosen.append({
            "order": order,
            "ifo": ifo,
            "file_start": int(start),
            "coverage_gain": round(best_gain, 4),
            "new_samples": new_samples,
            "evaluation_glitches": int(matrix[index].sum()),
        })
    plan = pd.DataFrame(chosen)
    all_in_files = annotated[annotated["window_inside_file"]]
    per_file = all_in_files.groupby(["ifo", "file_start"]).size().rename("all_glitches")
    return plan.merge(per_file, left_on=["ifo", "file_start"], right_index=True, how="left")


def coverage_by_class(
    annotated: pd.DataFrame, plan: pd.DataFrame, evaluation_splits: list[str]
) -> pd.DataFrame:
    """Quantos glitches por classe e split entram com os arquivos escolhidos."""
    keys = set(zip(plan["ifo"], plan["file_start"]))
    inside = annotated[annotated["window_inside_file"]]
    mask = [key in keys for key in zip(inside["ifo"], inside["file_start"])]
    selected = inside[mask]
    table = selected.groupby(["label", "sample_type"]).size().unstack(fill_value=0)
    for split in ("train", *evaluation_splits):
        if split not in table.columns:
            table[split] = 0
    table["evaluation"] = table[list(evaluation_splits)].sum(axis=1)
    return table.sort_values("evaluation", ascending=False)


def classification_metrics(
    labels: np.ndarray, predictions: np.ndarray, *, min_class_samples: int
) -> dict[str, Any]:
    """Acuracia, acuracia balanceada e macro-F1 restrito a classes com amostras suficientes."""
    from sklearn.metrics import balanced_accuracy_score, f1_score

    labels = np.asarray(labels, dtype=object)
    predictions = np.asarray(predictions, dtype=object)
    if len(labels) == 0:
        raise ValueError("conjunto de avaliacao vazio")
    classes, counts = np.unique(labels, return_counts=True)
    decisive = [str(name) for name, count in zip(classes, counts) if count >= min_class_samples]
    return {
        "n": int(len(labels)),
        "accuracy": float(np.mean(labels == predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": (
            float(f1_score(labels, predictions, labels=decisive, average="macro", zero_division=0))
            if decisive
            else float("nan")
        ),
        "decisive_classes": decisive,
    }


def grouped_bootstrap(
    frame: pd.DataFrame,
    statistic,
    *,
    group_column: str,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    """Intervalo de 95% reamostrando grupos inteiros (arquivos) com reposicao."""
    groups = [group for _, group in frame.groupby(group_column)]
    if len(groups) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(replicates)):
        picks = rng.integers(0, len(groups), size=len(groups))
        sample = pd.concat([groups[index] for index in picks], ignore_index=True)
        value = statistic(sample)
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def evaluate_criteria(
    metrics: dict[str, Any], reject_fraction: float, criteria: dict[str, Any]
) -> dict[str, Any]:
    """Compara as estimativas pontuais com os limiares preregistrados."""
    checks = {
        "C1_macro_f1": (metrics["macro_f1"], float(criteria["macro_f1_min"]), ">="),
        "C1_accuracy": (metrics["accuracy"], float(criteria["accuracy_min"]), ">="),
        "C2_reject_ood": (reject_fraction, float(criteria["reject_ood_max"]), "<="),
    }
    result: dict[str, Any] = {}
    for name, (value, threshold, direction) in checks.items():
        passed = bool(
            np.isfinite(value) and (value >= threshold if direction == ">=" else value <= threshold)
        )
        result[name] = {"value": float(value), "threshold": threshold, "rule": direction, "passed": passed}
    result["all_passed"] = all(item["passed"] for item in result.values())
    return result
