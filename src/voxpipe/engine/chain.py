from collections.abc import Sequence

import numpy as np

from voxpipe.audio import OFFLINE_CHUNK, SAMPLE_RATE
from voxpipe.engine.entry import ChainEntry
from voxpipe.engine.registry import create_effect


class Chain:
    """Runs effects in profile order. Shared by run, render and tune (D-05)."""

    def __init__(self, entries: Sequence[ChainEntry], sample_rate: int = SAMPLE_RATE) -> None:
        self.entries = tuple(entries)
        self.effects = [create_effect(e.effect, e.params, sample_rate) for e in self.entries]

    @property
    def latency_samples(self) -> int:
        return sum(effect.latency_samples for effect in self.effects)

    def reset(self) -> None:
        for effect in self.effects:
            effect.reset()

    def process_block(self, block: np.ndarray) -> np.ndarray:
        x = np.asarray(block, dtype=np.float32)
        for effect in self.effects:
            x = effect.process(x)
        return x

    def process_signal(self, signal: np.ndarray, chunk: int = OFFLINE_CHUNK) -> np.ndarray:
        """Process a whole signal from a fresh state. Identical to feeding live blocks."""
        self.reset()
        out = np.empty(len(signal), dtype=np.float32)
        for start in range(0, len(signal), chunk):
            out[start : start + chunk] = self.process_block(signal[start : start + chunk])
        return out
