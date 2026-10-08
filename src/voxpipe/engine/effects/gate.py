import numpy as np
from scipy.signal import lfilter

from voxpipe.engine.effect import Effect
from voxpipe.engine.params import Param

RATIO = 3.0  # downward expansion below the threshold
MAX_ATTENUATION_DB = 60.0


class Gate(Effect):
    """Noise gate (downward expander) for mic hiss in pauses, so later drive and
    distortion stages can't turn it into static. A one-pole power detector makes the
    gain smooth without a per-sample loop."""

    name = "gate"
    params = (
        Param("threshold_db", -80.0, -20.0, -80.0),
        Param("release_ms", 5.0, 200.0, 50.0, "log"),
    )

    def reset(self) -> None:
        pole = np.exp(-1000.0 / (self.values["release_ms"] * self.sample_rate))
        self._b, self._a = [1.0 - pole], [1.0, -pole]
        self._zi = np.zeros(1)

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32)
        x = block.astype(np.float64)
        power, self._zi = lfilter(self._b, self._a, x * x, zi=self._zi)
        level_db = 10.0 * np.log10(power + 1e-12)
        gain_db = np.clip(
            (level_db - self.values["threshold_db"]) * (RATIO - 1.0), -MAX_ATTENUATION_DB, 0.0
        )
        return (x * 10.0 ** (gain_db / 20.0)).astype(np.float32)
