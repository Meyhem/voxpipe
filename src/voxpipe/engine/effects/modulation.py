import numpy as np

from voxpipe.engine.effect import Effect, mix
from voxpipe.engine.params import Param

TWO_PI = 2 * np.pi


class RingMod(Effect):
    name = "ring_mod"
    params = (
        Param("freq_hz", 20.0, 2000.0, 100.0, "log"),
        Param("mix", 0.0, 1.0, 0.0),
    )

    def reset(self) -> None:
        self._phase = 0.0
        self._increment = TWO_PI * self.values["freq_hz"] / self.sample_rate

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32)
        phases = self._phase + self._increment * np.arange(block.size)
        self._phase = (self._phase + self._increment * block.size) % TWO_PI
        wet = block * np.sin(phases)
        return mix(block, wet, self.values["mix"])
