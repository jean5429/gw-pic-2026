from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

UInt8Array = NDArray[np.uint8]

RENDER_MODES = ("percentile", "gravity_spy")
# Topo da barra de cor dos PNGs Gravity Spy: energia normalizada de 0 a 25,5 em viridis.
GRAVITY_SPY_ENERGY_MAX = 25.5


def gravity_spy_gray(energy: ArrayLike) -> UInt8Array:
    """Converte energia normalizada em cinza pela cadeia dos PNGs Gravity Spy.

    A energia e recortada em ``[0, GRAVITY_SPY_ENERGY_MAX]``, mapeada em viridis pelo
    LUT de bytes do matplotlib e convertida para cinza por ``PIL.Image.convert("L")``,
    a mesma conversao que ``prepare_gravity_spy_panel`` aplica aos PNGs de treino.
    """
    from matplotlib import colormaps
    from PIL import Image

    values = np.asarray(energy, dtype=float)
    if values.ndim != 2:
        raise ValueError("energia deve ser uma matriz 2D")
    if not np.all(np.isfinite(values)):
        raise ValueError("energia contem NaN ou infinito")
    scaled = np.clip(values / GRAVITY_SPY_ENERGY_MAX, 0.0, 1.0)
    rgb = np.ascontiguousarray(colormaps["viridis"](scaled, bytes=True)[..., :3])
    return np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.uint8)


def qgram_to_uint8(
    qgram: ArrayLike,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.5,
    source_duration_seconds: float | None = None,
    target_duration_seconds: float | None = None,
    baseline_mode: str = "none",
    flip_frequency: bool = False,
    render_mode: str = "percentile",
) -> UInt8Array:
    """Converte um Q-gram bruto em uma imagem uint8 na orientacao Gravity Spy.

    Contrato de entrada (verificado empiricamente em 26/07/2026 contra o artefato real
    ``outputs/gwtc4_218/qgrams/GW150914-v4/H1_qgram.json``,
    ``shape == [1000, 500]``, e contra
    ``gwosc_pipeline.gwpy_qtransform_consistent``, que grava
    ``np.asarray(TimeSeries.q_transform(...))`` sem transpor): ``qgram`` tem shape
    ``(n_tempo, n_frequencia)`` — eixo 0 = tempo (ordem cronologica), eixo 1 = frequencia
    (ordem ascendente, convencao ``gwpy.spectrogram.Spectrogram``).

    Contrato de saida: ``(n_frequencia, n_tempo_recortado)`` — eixo 0 = frequencia, eixo 1 =
    tempo — para casar com as imagens de treino do Gravity Spy (ver
    ``research/panel_preprocessing.py``: ``output_height`` mapeia frequencia,
    ``output_width`` mapeia tempo; ``frequency_row_bounds`` documenta "frequencia alta no
    topo"). ``ridge_from_energy`` (``ridge.py``) usa o mesmo contrato de saida.

    Ate 26/07/2026 esta funcao recortava e invertia o eixo errado (eixo 1, frequencia, como
    se fosse duracao) e calculava ``per_frequency_median`` reduzindo sobre o eixo errado.
    Ver a memoria de projeto "bug_eixo_qgram" e a nota de correcao em ``PREREGISTRO.md``
    para o historico e a evidencia (execucao contra o artefato real de GW150914).

    ``render_mode`` escolhe a escala de intensidade aplicada depois do recorte de tempo.
    ``"percentile"`` e o renderer canonico (baseline, ``log1p`` e percentis).
    ``"gravity_spy"`` e exploratorio: trata o Q-gram como energia normalizada pela
    mediana (``q_transform(norm="median")``) e aplica ``gravity_spy_gray``; baseline e
    percentis sao ignorados. Ele so harmoniza a intensidade. A faixa de frequencia dos
    paineis de treino continua a do PNG oficial, nao 20-512 Hz. Incorporado em
    10/10/2026 a partir de jean5429/gw-pic-2026 (commit 1447593).
    """
    array = np.asarray(qgram, dtype=float)
    if array.ndim != 2:
        raise ValueError("qgram deve ser uma matriz 2D")
    if not np.all(np.isfinite(array)):
        raise ValueError("qgram contem NaN ou infinito")
    render_mode = str(render_mode).lower()
    if render_mode not in RENDER_MODES:
        raise ValueError("render_mode deve ser percentile ou gravity_spy")

    # 1) Recorte de duracao central: eixo 0 e' TEMPO no contrato de entrada.
    if source_duration_seconds is not None or target_duration_seconds is not None:
        if source_duration_seconds is None or target_duration_seconds is None:
            raise ValueError("source e target duration devem ser fornecidos juntos")
        if not 0 < target_duration_seconds <= source_duration_seconds:
            raise ValueError("duracoes de Q-gram invalidas")
        n_time = array.shape[0]
        target_time = max(1, int(round(n_time * target_duration_seconds / source_duration_seconds)))
        start = max(0, (n_time - target_time) // 2)
        array = array[start : start + target_time, :]

    if render_mode == "gravity_spy":
        # Mesmo layout do modo canonico: linhas = frequencia, colunas = tempo.
        energy = array.T
        if flip_frequency:
            energy = energy[::-1, :]
        return gravity_spy_gray(energy)

    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("percentis invalidos")

    # 2) Correcao de baseline. "per_frequency_median" precisa reduzir sobre o eixo do
    # TEMPO (0) para produzir um escalar por bin de FREQUENCIA (eixo 1) e propagar por
    # broadcasting sobre o tempo.
    baseline_mode = str(baseline_mode).lower()
    if baseline_mode == "none":
        excess = np.maximum(array, 0.0)
    elif baseline_mode == "subtract_one":
        excess = np.maximum(array - 1.0, 0.0)
    elif baseline_mode == "global_median":
        baseline = float(np.median(array))
        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError("baseline global invalido")
        excess = np.maximum(array / baseline - 1.0, 0.0)
    elif baseline_mode == "per_frequency_median":
        baseline = np.median(array, axis=0, keepdims=True)
        if not np.all(np.isfinite(baseline)) or np.any(baseline <= 0):
            raise ValueError("baseline por frequencia invalido")
        excess = np.maximum(array / baseline - 1.0, 0.0)
    else:
        raise ValueError(
            "baseline_mode deve ser none, subtract_one, global_median ou per_frequency_median"
        )
    transformed = np.log1p(excess)

    # 3) Alinha ao layout de imagem/CNN: linhas = frequencia, colunas = tempo.
    transformed = transformed.T

    # 4) flip_frequency opera sobre o eixo 0 pos-transposicao, que agora e' frequencia.
    if flip_frequency:
        transformed = transformed[::-1, :]

    low, high = np.percentile(transformed, [lower_percentile, upper_percentile])
    if not np.isfinite(high) or high <= low:
        raise ValueError("qgram sem contraste")
    normalized = np.clip((transformed - low) / (high - low), 0.0, 1.0)
    return np.round(255.0 * normalized).astype(np.uint8)


def render_settings(cfg: dict[str, Any], render_mode: str | None = None) -> dict[str, Any]:
    """Argumentos de ``qgram_to_uint8`` lidos de ``[harmonized_inference]``.

    ``render_mode`` explicito sobrepoe a configuracao; sem a chave, vale o canonico.
    """
    return {
        "lower_percentile": float(cfg["lower_percentile"]),
        "upper_percentile": float(cfg["upper_percentile"]),
        "source_duration_seconds": float(cfg["source_outseg_seconds"]),
        "target_duration_seconds": float(cfg["target_duration_seconds"]),
        "baseline_mode": str(cfg["baseline_mode"]),
        "flip_frequency": bool(cfg["flip_frequency"]),
        "render_mode": str(render_mode or cfg.get("render_mode", "percentile")),
    }


def catalog_network_input(image: ArrayLike, *, height: int, width: int) -> UInt8Array:
    """Pixels uint8 que a CNN recebe, antes de ToTensor/Normalize.

    Aplica o mesmo ``transforms.Resize`` dos comandos de inferencia sobre a imagem PIL.
    """
    from PIL import Image
    from torchvision import transforms

    pixels = np.asarray(image)
    if pixels.ndim != 2 or pixels.dtype != np.uint8:
        raise ValueError("esperada imagem 2D uint8")
    resized = transforms.Resize((int(height), int(width)))(Image.fromarray(pixels))
    return np.asarray(resized, dtype=np.uint8)


def load_gravity_spy_checkpoint(path: str | Path, device: Any):
    import torch

    from .gravity_spy import build_resnet18_grayscale

    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    classes = list(checkpoint["classes"])
    profile = checkpoint.get("profile") or {}
    model = build_resnet18_grayscale(
        len(classes),
        pretrained=False,
        conv1_stride=int(profile.get("conv1_stride", 2)),
        use_maxpool=bool(profile.get("use_maxpool", False)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    temperature = float(checkpoint.get("temperature", 1.0))
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperatura de calibracao invalida no checkpoint {path}")
    model.gw_temperature = temperature
    model.to(device)
    model.eval()
    return model, classes, profile


def probability_entropy(probabilities: ArrayLike) -> float:
    p = np.asarray(probabilities, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) == 0:
        return float("nan")
    entropy = -float(np.sum(p * np.log(p)))
    return entropy / float(np.log(max(2, len(probabilities))))


class CatalogQgramDataset:
    """Dataset picklable de Q-grams para inferencia com workers no Windows.

    Carrega ``qgram_path`` como ``np.load`` sem transpor (contrato ``(n_tempo,
    n_frequencia)``) e delega toda a normalizacao de eixo, recorte e orientacao de imagem a
    ``qgram_to_uint8`` — ver o docstring dessa funcao para o contrato completo.
    """

    def __init__(
        self,
        frame,
        transform,
        *,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.5,
        source_duration_seconds: float | None = None,
        target_duration_seconds: float | None = None,
        baseline_mode: str = "none",
        flip_frequency: bool = False,
        render_mode: str = "percentile",
    ):
        self.records = frame[["event", "detector", "qgram_path"]].to_dict(orient="records")
        self.transform = transform
        self.lower_percentile = float(lower_percentile)
        self.upper_percentile = float(upper_percentile)
        self.source_duration_seconds = source_duration_seconds
        self.target_duration_seconds = target_duration_seconds
        self.baseline_mode = str(baseline_mode)
        self.flip_frequency = bool(flip_frequency)
        self.render_mode = str(render_mode)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image

        row = self.records[index]
        array = np.load(row["qgram_path"])
        image_array = qgram_to_uint8(
            array,
            lower_percentile=self.lower_percentile,
            upper_percentile=self.upper_percentile,
            source_duration_seconds=self.source_duration_seconds,
            target_duration_seconds=self.target_duration_seconds,
            baseline_mode=self.baseline_mode,
            flip_frequency=self.flip_frequency,
            render_mode=self.render_mode,
        )
        image = Image.fromarray(image_array)
        return self.transform(image), index
