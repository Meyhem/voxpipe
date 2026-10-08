import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from voxpipe.engine.effect import Effect, mix
from voxpipe.engine.params import Param

BANDS = 16
LOW_HZ, HIGH_HZ = 100.0, 8000.0
NOISE_SEED = 0  # fixed, so renders and tuning stay deterministic (N-06)
_EPS = 1e-4


class Vocoder(Effect):
    """Channel vocoder: the voice's band envelopes shape a sawtooth+noise carrier at a
    fixed pitch. This gives the frame-by-frame "machine" spectrum that EQ and
    distortion cannot. Carrier band levels are normalised by their own envelopes, so the
    output follows the voice's level whatever the carrier."""

    name = "vocoder"
    params = (
        Param("carrier_hz", 40.0, 400.0, 100.0, "log"),
        Param("noise", 0.0, 1.0, 0.1),
        Param("smooth_ms", 2.0, 40.0, 10.0, "log"),
        Param("mix", 0.0, 1.0, 0.0),
    )

    def reset(self) -> None:
        edges = np.geomspace(LOW_HZ, HIGH_HZ, BANDS + 1)
        self._sos = [
            butter(2, [lo, hi], btype="bandpass", fs=self.sample_rate, output="sos")
            for lo, hi in zip(edges[:-1], edges[1:], strict=True)
        ]
        sections = self._sos[0].shape[0]
        # Per band: filter state for [voice, carrier], filtered together in one call.
        self._zi = np.zeros((BANDS, sections, 2, 2))
        pole = np.exp(-1000.0 / (self.values["smooth_ms"] * self.sample_rate))
        self._env_b, self._env_a = [1.0 - pole], [1.0, -pole]
        self._zi_env = np.zeros((2, BANDS, 1))
        self._samples = 0
        self._increment = self.values["carrier_hz"] / self.sample_rate
        self._rng = np.random.default_rng(NOISE_SEED)

    def _bands(self, signals: np.ndarray) -> np.ndarray:
        """(2, n) voice+carrier -> (2, BANDS, n) band signals."""
        out = np.empty((2, BANDS, signals.shape[1]))
        for i, sos in enumerate(self._sos):
            out[:, i], self._zi[i] = sosfilt(sos, signals, axis=-1, zi=self._zi[i])
        return out

    def _carrier(self, n: int) -> np.ndarray:
        # Phase from an integer sample counter: a sawtooth jumps at the wrap, so an
        # accumulated float phase would make the output depend on block size.
        phases = ((self._samples + np.arange(n)) * self._increment) % 1.0
        self._samples += n
        noise = self._rng.random(n) * 2.0 - 1.0  # uniform doubles: chunking-invariant
        amount = self.values["noise"]
        return (1.0 - amount) * (2.0 * phases - 1.0) + amount * noise

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32)
        if self.values["mix"] == 0.0:
            return block.copy()
        bands = self._bands(np.stack([block.astype(np.float64), self._carrier(block.size)]))
        env, self._zi_env = lfilter(
            self._env_b, self._env_a, np.abs(bands), axis=-1, zi=self._zi_env
        )
        wet = np.sum(bands[1] * env[0] / (env[1] + _EPS), axis=0)
        return mix(block, wet.astype(np.float32), self.values["mix"])
