import numpy as np
from scipy.signal import butter, sosfilt

from voxpipe.engine.effect import Effect
from voxpipe.engine.params import Param


class _SosEffect(Effect):
    """Shared streaming second-order-sections filter; subclasses build self._sos."""

    def _build_sos(self) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        self._sos = self._build_sos()
        self._zi = np.zeros((self._sos.shape[0], 2))

    def process(self, block: np.ndarray) -> np.ndarray:
        out, self._zi = sosfilt(self._sos, np.asarray(block, dtype=np.float64), zi=self._zi)
        return out.astype(np.float32)


class Filter(_SosEffect):
    name = "filter"
    params = (
        Param("highpass_hz", 20.0, 2000.0, 20.0, "log"),
        Param("lowpass_hz", 1000.0, 20000.0, 20000.0, "log"),
    )

    def _build_sos(self) -> np.ndarray:
        fs = self.sample_rate
        lowpass = min(self.values["lowpass_hz"], 0.45 * fs)
        high = butter(2, self.values["highpass_hz"], "highpass", fs=fs, output="sos")
        low = butter(2, lowpass, "lowpass", fs=fs, output="sos")
        return np.vstack([high, low])


def peaking_sos(freq_hz: float, gain_db: float, q: float, fs: int) -> np.ndarray:
    """RBJ audio-EQ-cookbook peaking filter as one second-order section."""
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq_hz / fs
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    b = np.array([1 + alpha * a, -2 * cos_w0, 1 - alpha * a])
    den = np.array([1 + alpha / a, -2 * cos_w0, 1 - alpha / a])
    return np.concatenate([b / den[0], den / den[0]])


class Eq(_SosEffect):
    name = "eq"
    params = (
        Param("low_hz", 80.0, 800.0, 250.0, "log"),
        Param("low_gain_db", -18.0, 18.0, 0.0),
        Param("mid_hz", 500.0, 4000.0, 1500.0, "log"),
        Param("mid_gain_db", -18.0, 18.0, 0.0),
        Param("high_hz", 2000.0, 12000.0, 5000.0, "log"),
        Param("high_gain_db", -18.0, 18.0, 0.0),
        Param("q", 0.3, 5.0, 1.0, "log"),
    )

    def _build_sos(self) -> np.ndarray:
        v, fs = self.values, self.sample_rate
        return np.vstack(
            [
                peaking_sos(v[f"{band}_hz"], v[f"{band}_gain_db"], v["q"], fs)
                for band in ("low", "mid", "high")
            ]
        )
