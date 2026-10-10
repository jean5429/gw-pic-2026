"""Grad-CAM para a ResNet-18 em escala de cinza do projeto.

Grad-CAM (Selvaraju et al., 2017) pondera os mapas de ativacao de uma camada pela
media espacial do gradiente do logit escolhido. O mapa e um diagnostico qualitativo
da decisao fechada de um checkpoint. Ele nao valida a classe, nao substitui a
referencia OOD e nao identifica a origem fisica do transiente.

Incorporado em 10/10/2026 a partir de jean5429/gw-pic-2026 (commit 1447593), com
gradiente obtido por ``torch.autograd.grad`` em vez de hooks de backward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GradCamResult:
    heatmap: FloatArray
    """Mapa em ``[0, 1]`` na resolucao da camada alvo; zero quando nada contribui."""
    class_index: int
    logits: FloatArray
    """Logits brutos do checkpoint, antes da temperatura de calibracao."""


def grad_cam(
    model: Any,
    input_tensor: Any,
    *,
    target_layer: Any | None = None,
    class_index: int | None = None,
) -> GradCamResult:
    """Calcula o Grad-CAM de uma unica imagem ``(1, canais, altura, largura)``.

    Sem ``target_layer``, usa a saida do ultimo bloco residual (``model.layer4[-1]``).
    Sem ``class_index``, explica a classe de maior logit.
    """
    import torch

    if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
        raise ValueError("input_tensor deve ter shape (1, canais, altura, largura)")
    layer = target_layer if target_layer is not None else model.layer4[-1]
    captured: dict[str, Any] = {}

    def keep_activation(_module, _inputs, output):
        captured["activation"] = output

    handle = layer.register_forward_hook(keep_activation)
    try:
        with torch.enable_grad():
            # O grafo existe mesmo se os parametros estiverem congelados.
            image = input_tensor.detach().clone().requires_grad_(True)
            logits = model(image)
            if logits.ndim != 2 or logits.shape[0] != 1:
                raise ValueError("o modelo deve devolver logits (1, classes)")
            index = int(logits[0].argmax()) if class_index is None else int(class_index)
            if not 0 <= index < logits.shape[1]:
                raise ValueError(f"classe fora da taxonomia: {index}")
            activation = captured["activation"]
            (gradient,) = torch.autograd.grad(logits[0, index], activation)
    finally:
        handle.remove()

    weights = gradient[0].mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * activation[0]).sum(dim=0))
    cam_array = cam.detach().double().cpu().numpy()
    peak = float(cam_array.max())
    heatmap = cam_array / peak if peak > 0 else np.zeros_like(cam_array)
    return GradCamResult(
        heatmap=heatmap,
        class_index=index,
        logits=logits[0].detach().double().cpu().numpy(),
    )


def upsample_heatmap(heatmap: FloatArray, shape: tuple[int, int]) -> FloatArray:
    """Interpola o mapa para ``(altura, largura)`` da imagem, mantendo ``[0, 1]``."""
    from PIL import Image

    values = np.asarray(heatmap, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("heatmap deve ser 2D")
    height, width = (int(shape[0]), int(shape[1]))
    resized = Image.fromarray(values).resize((width, height), Image.Resampling.BILINEAR)
    return np.clip(np.asarray(resized, dtype=np.float64), 0.0, 1.0)
