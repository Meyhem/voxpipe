import numpy as np
from scipy.signal import lfilter

from voxpipe.engine.effect import Effect, mix
from voxpipe.engine.params import Param


class _StreamingFilter:
    """lfilter with carried state, so blocks of any length chain seamlessly."""

    def __init__(self, b: np.ndarray, a: np.ndarray) -> None:
        self.b, self.a = b, a
        self.zi = np.zeros(max(len(a), len(b)) - 1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = lfilter(self.b, self.a, x, zi=self.zi)
        return y


def feedback_comb(delay: int, feedback: float) -> _StreamingFilter:
    """y[n] = x[n] + feedback * y[n - delay]"""
    a = np.zeros(delay + 1)
    a[0], a[delay] = 1.0, -feedback
    return _StreamingFilter(np.array([1.0]), a)


def allpass(delay: int, gain: float) -> _StreamingFilter:
    """Schroeder allpass: y[n] = -g x[n] + x[n - d] + g y[n - d]"""
    b = np.zeros(delay + 1)
    b[0], b[delay] = -gain, 1.0
    a = np.zeros(delay + 1)
    a[0], a[delay] = 1.0, -gain
    return _StreamingFilter(b, a)


def ms_to_samples(ms: float, fs: int) -> int:
    return max(1, round(ms * fs / 1000))


class Comb(Effect):
    name = "comb"
    params = (
        Param("delay_ms", 0.5, 20.0, 5.0, "log"),
        Param("feedback", -0.95, 0.95, 0.5),
        Param("mix", 0.0, 1.0, 0.0),
    )

    def reset(self) -> None:
        fb = self.values["feedback"]
        self._filter = feedback_comb(ms_to_samples(self.values["delay_ms"], self.sample_rate), fb)
        self._scale = 1.0 - abs(fb)

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32)
        wet = self._filter(block.astype(np.float64)) * self._scale
        return mix(block, wet, self.values["mix"])


class Reverb(Effect):
    name = "reverb"
    params = (
        Param("room_size", 0.0, 1.0, 0.5),
        Param("mix", 0.0, 1.0, 0.0),
    )
    COMB_MS = (29.7, 37.1, 41.1, 43.7)
    ALLPASS_MS = (5.0, 1.7)
    ALLPASS_GAIN = 0.7

    def reset(self) -> None:
        fs = self.sample_rate
        feedback = 0.7 + 0.28 * self.values["room_size"]
        self._combs = [feedback_comb(ms_to_samples(ms, fs), feedback) for ms in self.COMB_MS]
        self._allpasses = [allpass(ms_to_samples(ms, fs), self.ALLPASS_GAIN) for ms in self.ALLPASS_MS]
        self._scale = (1.0 - feedback) / len(self.COMB_MS) * 4.0

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32)
        x = block.astype(np.float64)
        wet = sum(comb(x) for comb in self._combs) * self._scale
        for stage in self._allpasses:
            wet = stage(wet)
        return mix(block, wet, self.values["mix"])
