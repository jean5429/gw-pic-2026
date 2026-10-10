"""Padroes de exportacao de figuras do projeto."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

# Resolucao das figuras salvas (Grad-CAM, matriz de confusao, diagnosticos).
# Nao afeta a entrada da rede, definida em pixels (140 x 170).
FIGURE_DPI = 300


def plot_confusion_matrix(
    cm: ArrayLike,
    classes: Sequence[str],
    path: str | Path,
    *,
    title: str = "Gravity Spy — matriz de confusao normalizada",
) -> None:
    """Salva a matriz de confusao normalizada por linha (fracao da classe verdadeira).

    Linhas sem amostras ficam em zero em vez de produzir divisao por zero.
    """
    from matplotlib.figure import Figure

    counts = np.asarray(cm, dtype=float)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("matriz de confusao deve ser quadrada")
    if counts.shape[0] != len(classes):
        raise ValueError("numero de classes incompativel com a matriz de confusao")
    normalized = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)

    figure = Figure(figsize=(11, 9))
    axis = figure.subplots()
    image = axis.imshow(normalized, vmin=0, vmax=1)
    axis.set_xticks(range(len(classes)), labels=list(classes), rotation=90, fontsize=7)
    axis.set_yticks(range(len(classes)), labels=list(classes), fontsize=7)
    axis.set_xlabel("Classe prevista")
    axis.set_ylabel("Classe verdadeira")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="Fracao")
    figure.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=FIGURE_DPI)
