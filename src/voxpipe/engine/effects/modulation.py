import numpy as np

from voxpipe.engine.delay import History, read_delayed
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


class Chorus(Effect):
    """Single modulated-delay voice mixed with the dry signal (doubling)."""

    name = "chorus"
    params = (
        Param("delay_ms", 5.0, 40.0, 15.0),
        Param("depth_ms", 0.0, 10.0, 2.0),
        Param("rate_hz", 0.1, 5.0, 0.8, "log"),
        Param("mix", 0.0, 1.0, 0.0),
    )

    def reset(self) -> None:
        ms = self.sample_rate / 1000
        self._delay = self.values["delay_ms"] * ms
        self._depth = self.values["depth_ms"] * ms
        self._history = History(int(np.ceil(self._delay + self._depth)) + 2)
        self._phase = 0.0
        self._increment = TWO_PI * self.values["rate_hz"] / self.sample_rate

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32)
        phases = self._phase + self._increment * np.arange(block.size)
        self._phase = (self._phase + self._increment * block.size) % TWO_PI
        delays = np.clip(self._delay + self._depth * np.sin(phases), 1.0, self._history.size)
        full = self._history.extend(block)
        wet = read_delayed(full, self._history.size, delays)
        return mix(block, wet, self.values["mix"])
