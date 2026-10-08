from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ChainEntry:
    """One effect in a profile's chain: its registry name and parameter values."""

    effect: str
    params: Mapping[str, float]
