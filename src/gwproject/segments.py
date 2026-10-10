from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FiniteEventSegment:
    """Maior segmento finito continuo que contem o instante do evento."""

    start_index: int
    stop_index: int
    event_index_original: int
    event_index_segment: int
    sample_rate: float
    trimmed: bool
    valid_before_event_seconds: float
    valid_after_event_seconds: float
    valid_total_seconds: float
    nonfinite_fraction_original: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_index": self.start_index,
            "stop_index": self.stop_index,
            "event_index_original": self.event_index_original,
            "event_index_segment": self.event_index_segment,
            "sample_rate": self.sample_rate,
            "trimmed": self.trimmed,
            "valid_before_event_seconds": self.valid_before_event_seconds,
            "valid_after_event_seconds": self.valid_after_event_seconds,
            "valid_total_seconds": self.valid_total_seconds,
            "nonfinite_fraction_original": self.nonfinite_fraction_original,
        }


def _finite_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def select_finite_event_segment(
    data: ArrayLike,
    event_index: int,
    sample_rate: float,
    *,
    allow_edge_trim: bool = True,
    reject_internal_nonfinite: bool = True,
    min_valid_before_event_seconds: float = 8.0,
    min_valid_after_event_seconds: float = 8.0,
    min_contiguous_valid_seconds: float = 20.0,
) -> tuple[FloatArray, FiniteEventSegment]:
    """Seleciona de forma conservadora um trecho finito contendo o evento.

    Valores nao finitos podem ser removidos apenas quando ocupam as bordas da
    serie. Ilhas finitas separadas por gaps internos sao rejeitadas quando
    ``reject_internal_nonfinite`` esta ativo. Nenhum valor e interpolado.
    """

    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 1 or len(x) < 16:
        raise ValueError("serie invalida ou curta demais")
    if sample_rate <= 0:
        raise ValueError("sample_rate deve ser positivo")
    if not 0 <= event_index < len(x):
        raise ValueError("event_index fora da serie")

    finite = np.isfinite(x)
    nonfinite_fraction = float(1.0 - np.mean(finite))
    if np.all(finite):
        info = FiniteEventSegment(
            start_index=0,
            stop_index=len(x),
            event_index_original=event_index,
            event_index_segment=event_index,
            sample_rate=float(sample_rate),
            trimmed=False,
            valid_before_event_seconds=event_index / float(sample_rate),
            valid_after_event_seconds=(len(x) - event_index - 1) / float(sample_rate),
            valid_total_seconds=len(x) / float(sample_rate),
            nonfinite_fraction_original=0.0,
        )
        return x.copy(), info

    if not allow_edge_trim:
        raise ValueError("data contem NaN ou infinito")
    if not finite[event_index]:
        raise ValueError("instante do evento pertence a trecho NaN ou infinito")

    runs = _finite_runs(finite)
    containing = [run for run in runs if run[0] <= event_index < run[1]]
    if len(containing) != 1:
        raise ValueError("nao foi possivel isolar trecho finito contendo o evento")
    start, stop = containing[0]

    if reject_internal_nonfinite:
        finite_before = bool(np.any(finite[:start]))
        finite_after = bool(np.any(finite[stop:]))
        if finite_before or finite_after:
            raise ValueError("gap nao finito interno; somente recorte de borda e permitido")

    event_segment = event_index - start
    before = event_segment / float(sample_rate)
    after = (stop - event_index - 1) / float(sample_rate)
    total = (stop - start) / float(sample_rate)
    if before < min_valid_before_event_seconds:
        raise ValueError(
            f"dados finitos insuficientes antes do evento: {before:.3f} s < "
            f"{min_valid_before_event_seconds:.3f} s"
        )
    if after < min_valid_after_event_seconds:
        raise ValueError(
            f"dados finitos insuficientes depois do evento: {after:.3f} s < "
            f"{min_valid_after_event_seconds:.3f} s"
        )
    if total < min_contiguous_valid_seconds:
        raise ValueError(
            f"trecho finito continuo curto: {total:.3f} s < "
            f"{min_contiguous_valid_seconds:.3f} s"
        )

    selected = x[start:stop].copy()
    if not np.all(np.isfinite(selected)):
        raise RuntimeError("erro interno: trecho selecionado ainda contem nao finitos")
    info = FiniteEventSegment(
        start_index=start,
        stop_index=stop,
        event_index_original=event_index,
        event_index_segment=event_segment,
        sample_rate=float(sample_rate),
        trimmed=True,
        valid_before_event_seconds=before,
        valid_after_event_seconds=after,
        valid_total_seconds=total,
        nonfinite_fraction_original=nonfinite_fraction,
    )
    return selected, info
