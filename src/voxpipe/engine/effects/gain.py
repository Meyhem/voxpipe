import numpy as np

from voxpipe.engine.effect import Effect
from voxpipe.engine.params import Param


class Gain(Effect):
    name = "gain"
    params = (Param("gain_db", -24.0, 24.0, 0.0),)

    def reset(self) -> None:
        self._factor = np.float32(10 ** (self.values["gain_db"] / 20))

    def process(self, block: np.ndarray) -> np.ndarray:
        return (np.asarray(block, dtype=np.float32) * self._factor).astype(np.float32)
