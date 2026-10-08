"""Normalised search space generated from effect Param declarations (D-08)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from voxpipe.engine.entry import ChainEntry
from voxpipe.engine.params import Param
from voxpipe.engine.registry import EFFECTS


@dataclass(frozen=True)
class _Dimension:
    entry_index: int
    param: Param


def _to_unit(param: Param, value: float) -> float:
    if param.scale == "log":
        lo, hi = math.log(param.minimum), math.log(param.maximum)
        return (math.log(value) - lo) / (hi - lo)
    return (value - param.minimum) / (param.maximum - param.minimum)


def _from_unit(param: Param, unit: float) -> float:
    if param.scale == "log":
        lo, hi = math.log(param.minimum), math.log(param.maximum)
        value = math.exp(lo + unit * (hi - lo))
    else:
        value = param.minimum + unit * (param.maximum - param.minimum)
    return min(param.maximum, max(param.minimum, round(value, 6)))


class ParamSpace:
    def __init__(self, template: Sequence[ChainEntry]) -> None:
        self.template = tuple(template)
        self._dimensions = tuple(
            _Dimension(index, param)
            for index, entry in enumerate(self.template)
            for param in EFFECTS[entry.effect].params
        )

    @property
    def size(self) -> int:
        return len(self._dimensions)

    def encode(self, entries: Sequence[ChainEntry]) -> np.ndarray:
        return np.array(
            [_to_unit(d.param, entries[d.entry_index].params[d.param.name]) for d in self._dimensions]
        )

    def decode(self, x: np.ndarray) -> list[ChainEntry]:
        values: list[dict[str, float]] = [{} for _ in self.template]
        for dimension, unit in zip(self._dimensions, np.clip(x, 0.0, 1.0), strict=True):
            values[dimension.entry_index][dimension.param.name] = _from_unit(
                dimension.param, float(unit)
            )
        return [ChainEntry(e.effect, v) for e, v in zip(self.template, values, strict=True)]
