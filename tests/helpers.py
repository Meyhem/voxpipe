"""Shared test utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import cycle

import numpy as np

from voxpipe.audio import SAMPLE_RATE


def sine(freq_hz: float, seconds: float = 1.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def voice_like(seconds: float = 2.0, seed: int = 0) -> np.ndarray:
    """Deterministic harmonic signal with a gliding pitch plus a little noise."""
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    f0 = 120.0 + 60.0 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
    voice = sum(0.3 / k * np.sin(k * phase) for k in range(1, 9))
    noise = np.random.default_rng(seed).normal(0.0, 0.02, t.size)
    return (voice + noise).astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def dominant_hz(x: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    return float(np.argmax(spectrum) * SAMPLE_RATE / x.size)


def run_in_chunks(effect, signal: np.ndarray, sizes: Sequence[int]) -> np.ndarray:
    effect.reset()
    out, start = [], 0
    for size in cycle(sizes):
        if start >= signal.size:
            break
        out.append(effect.process(signal[start : start + size]))
        start += size
    return np.concatenate(out)


def assert_block_invariant(factory: Callable[[], object], signal: np.ndarray) -> None:
    """Output must not depend on how the signal is split into blocks (D-05)."""
    whole = run_in_chunks(factory(), signal, [signal.size])
    for sizes in ([480], [4800], [1, 7, 480, 333, 4800]):
        chunked = run_in_chunks(factory(), signal, sizes)
        assert chunked.dtype == np.float32
        np.testing.assert_allclose(chunked, whole, atol=1e-5, err_msg=f"chunk sizes {sizes}")
