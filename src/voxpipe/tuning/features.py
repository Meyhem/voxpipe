"""Frame-level features for the tuning objective. All share a 10 ms hop."""

from __future__ import annotations

from functools import cache

import numpy as np

from voxpipe.audio import SAMPLE_RATE

HOP = 480
MEL_FFT = 1024
N_MELS = 40
MEL_FMIN, MEL_FMAX = 60.0, 12_000.0
PITCH_FRAME = 2048
PITCH_MIN_HZ, PITCH_MAX_HZ = 60.0, 500.0
VOICING_THRESHOLD = 0.5
SILENCE_RMS = 1e-3


def frame_count(n_samples: int) -> int:
    return max(1, -(-n_samples // HOP))


def frames(signal: np.ndarray, size: int) -> np.ndarray:
    n = frame_count(len(signal))
    padded = np.zeros((n - 1) * HOP + size, dtype=np.float64)
    m = min(len(signal), padded.size)
    padded[:m] = signal[:m]
    return np.lib.stride_tricks.sliding_window_view(padded, size)[::HOP][:n]


@cache
def _mel_filterbank() -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    edges = mel_to_hz(np.linspace(hz_to_mel(MEL_FMIN), hz_to_mel(MEL_FMAX), N_MELS + 2))
    bins = np.fft.rfftfreq(MEL_FFT, 1.0 / SAMPLE_RATE)
    bank = np.zeros((N_MELS, bins.size))
    for i in range(N_MELS):
        lo, centre, hi = edges[i], edges[i + 1], edges[i + 2]
        rising = (bins - lo) / (centre - lo)
        falling = (hi - bins) / (hi - centre)
        bank[i] = np.maximum(0.0, np.minimum(rising, falling))
    return bank


@cache
def _hann(size: int) -> np.ndarray:
    return np.hanning(size)


def log_mel(signal: np.ndarray) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(frames(signal, MEL_FFT) * _hann(MEL_FFT), axis=1)) ** 2
    return 10.0 * np.log10(spectrum @ _mel_filterbank().T + 1e-10)


def pitch(signal: np.ndarray) -> np.ndarray:
    """Autocorrelation pitch per frame; 0.0 where unvoiced or silent."""
    f = frames(signal, PITCH_FRAME)
    f = f - f.mean(axis=1, keepdims=True)
    loudness = np.sqrt(np.mean(f**2, axis=1))
    power = np.abs(np.fft.rfft(f, n=2 * PITCH_FRAME, axis=1)) ** 2
    autocorr = np.fft.irfft(power, axis=1)[:, :PITCH_FRAME]
    lag_min = int(SAMPLE_RATE / PITCH_MAX_HZ)
    lag_max = int(SAMPLE_RATE / PITCH_MIN_HZ)
    lags = np.arange(lag_min, lag_max + 1)
    # Unbiased: undo the (N - lag) / N decay of zero-padded autocorrelation, which
    # otherwise hides low (e.g. machine-like ~65-100 Hz) voices below the threshold.
    normalised = (
        autocorr[:, lag_min : lag_max + 1]
        / np.maximum(autocorr[:, :1], 1e-12)
        * (PITCH_FRAME / (PITCH_FRAME - lags))
    )
    best = np.argmax(normalised, axis=1)
    strength = normalised[np.arange(best.size), best]
    voiced = (strength > VOICING_THRESHOLD) & (loudness > SILENCE_RMS)
    return np.where(voiced, SAMPLE_RATE / (best + lag_min), 0.0)


def ltas(mel: np.ndarray) -> np.ndarray:
    return mel.mean(axis=0)
