"""Padroes de exportacao de figuras do projeto."""

# Resolucao de todas as figuras salvas (Grad-CAM, matriz de confusao, diagnosticos).
# Nao afeta a entrada da rede, que e definida em pixels (140 x 170).
FIGURE_DPI = 300


def plot_confusion_matrix(cm, classes, path) -> None:
    """Salva a matriz de confusao normalizada por linha (fracao da classe verdadeira)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.asarray(cm, dtype=float)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(cm_norm, vmin=0, vmax=1)
    axis.set_xticks(range(len(classes)), labels=classes, rotation=90, fontsize=7)
    axis.set_yticks(range(len(classes)), labels=classes, fontsize=7)
    axis.set_xlabel("Classe prevista")
    axis.set_ylabel("Classe verdadeira")
    axis.set_title("Gravity Spy — matriz de confusao normalizada")
    fig.colorbar(image, ax=axis, label="Fracao")
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
