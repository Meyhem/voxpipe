import numpy as np

from voxpipe.engine.effect import Effect, mix
from voxpipe.engine.params import Param


class Distortion(Effect):
    """Normalised tanh saturation followed by bit-crush quantisation. Stateless."""

    name = "distortion"
    params = (
        Param("drive_db", 0.0, 40.0, 0.0),
        Param("bits", 4.0, 16.0, 16.0),
        Param("mix", 0.0, 1.0, 0.0),
    )

    def reset(self) -> None:
        self._drive = 10 ** (self.values["drive_db"] / 20)
        self._norm = 1.0 / np.tanh(self._drive)
        self._steps = 2 ** (self.values["bits"] - 1)

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32)
        wet = np.tanh(self._drive * block.astype(np.float64)) * self._norm
        wet = np.round(wet * self._steps) / self._steps
        return mix(block, wet, self.values["mix"])
