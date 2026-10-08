import numpy as np

from voxpipe.engine.delay import History, read_delayed
from voxpipe.engine.effect import Effect, mix
from voxpipe.engine.params import Param

WINDOW_MS = 30.0


class PitchShift(Effect):
    """Two-tap rotating delay-line pitch shifter. Each tap's delay sweeps across a
    30 ms window at the rate that gives the requested pitch ratio. The taps are half a
    window apart with sin^2/cos^2 gains, so each tap is silent at the moment it jumps."""

    name = "pitch_shift"
    params = (
        Param("semitones", -12.0, 12.0, 0.0),
        Param("mix", 0.0, 1.0, 0.0),
    )

    def reset(self) -> None:
        self._window = WINDOW_MS * self.sample_rate / 1000
        self._history = History(int(self._window) + 3)
        ratio = 2 ** (self.values["semitones"] / 12)
        self._increment = (1.0 - ratio) / self._window
        self._phase = 0.0

    @property
    def latency_samples(self) -> int:
        return int(self._window / 2) if self.values["mix"] > 0 else 0

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32)
        phase_a = (self._phase + self._increment * np.arange(block.size)) % 1.0
        phase_b = (phase_a + 0.5) % 1.0
        self._phase = (self._phase + self._increment * block.size) % 1.0
        full = self._history.extend(block)
        size = self._history.size
        tap_a = read_delayed(full, size, 1.0 + phase_a * self._window)
        tap_b = read_delayed(full, size, 1.0 + phase_b * self._window)
        wet = np.sin(np.pi * phase_a) ** 2 * tap_a + np.sin(np.pi * phase_b) ** 2 * tap_b
        return mix(block, wet, self.values["mix"])
