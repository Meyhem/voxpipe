"""Base class for streaming effects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar

import numpy as np

from voxpipe.audio import SAMPLE_RATE
from voxpipe.engine.params import Param


class Effect(ABC):
    """A stateful effect: process() accepts blocks of any length, and its output
    must not depend on how the signal is split into blocks."""

    name: ClassVar[str]
    params: ClassVar[tuple[Param, ...]]

    def __init__(
        self, values: Mapping[str, float] | None = None, sample_rate: int = SAMPLE_RATE
    ) -> None:
        values = dict(values or {})
        unknown = sorted(set(values) - {p.name for p in self.params})
        if unknown:
            raise ValueError(f"{self.name}: unknown parameter(s): {', '.join(unknown)}")
        self.sample_rate = sample_rate
        self.values = {p.name: float(values.get(p.name, p.default)) for p in self.params}
        self.reset()

    @classmethod
    def defaults(cls) -> dict[str, float]:
        return {p.name: p.default for p in cls.params}

    @property
    def latency_samples(self) -> int:
        return 0

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal state and derive coefficients from self.values."""

    @abstractmethod
    def process(self, block: np.ndarray) -> np.ndarray:
        """Process one block; returns float32 of the same length."""


def mix(dry: np.ndarray, wet: np.ndarray, amount: float) -> np.ndarray:
    return (dry + amount * (wet - dry)).astype(np.float32, copy=False)
