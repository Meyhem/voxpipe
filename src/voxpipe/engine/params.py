from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Param:
    """One tunable effect parameter. Drives the profile schema and the tuning space."""

    name: str
    minimum: float
    maximum: float
    default: float
    scale: Literal["linear", "log"] = "linear"

    def __post_init__(self) -> None:
        if not self.minimum < self.maximum:
            raise ValueError(f"{self.name}: minimum must be below maximum")
        if not self.minimum <= self.default <= self.maximum:
            raise ValueError(f"{self.name}: default {self.default} outside range")
        if self.scale == "log" and self.minimum <= 0:
            raise ValueError(f"{self.name}: log scale needs a positive minimum")
