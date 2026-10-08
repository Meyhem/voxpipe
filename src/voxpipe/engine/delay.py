"""Streaming delay-line helpers for modulated-delay effects."""

import numpy as np


class History:
    """Keeps the last `size` input samples so each block can read into the past."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.buffer = np.zeros(size, dtype=np.float64)

    def extend(self, block: np.ndarray) -> np.ndarray:
        full = np.concatenate([self.buffer, np.asarray(block, dtype=np.float64)])
        self.buffer = full[-self.size :].copy()
        return full


def read_delayed(full: np.ndarray, history_size: int, delays: np.ndarray) -> np.ndarray:
    """For new sample k (at index history_size + k of `full`), read `delays[k]` samples
    into the past with linear interpolation. Every delay must lie in [1, history_size]."""
    positions = history_size + np.arange(delays.size) - delays
    base = np.floor(positions).astype(np.int64)
    frac = positions - base
    return full[base] * (1.0 - frac) + full[base + 1] * frac
