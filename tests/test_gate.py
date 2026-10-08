import numpy as np

from helpers import assert_block_invariant, rms, sine, voice_like
from voxpipe.engine.effects.gate import Gate

SETTLE = 4_800


def db(x: np.ndarray) -> float:
    return 20 * np.log10(rms(x))


def test_default_is_transparent_for_speech():
    x = voice_like(0.5)
    assert abs(db(Gate().process(x)[SETTLE:]) - db(x[SETTLE:])) < 0.1


def test_quiet_noise_below_threshold_is_attenuated():
    noise = np.random.default_rng(0).normal(0.0, 0.003, 48_000).astype(np.float32)  # ~-50 dB
    y = Gate({"threshold_db": -40.0}).process(noise)
    assert db(y[SETTLE:]) < db(noise[SETTLE:]) - 20.0


def test_loud_signal_above_threshold_passes():
    x = sine(300.0, amplitude=0.3)  # about -13 dB
    y = Gate({"threshold_db": -40.0}).process(x)
    assert abs(db(y[SETTLE:]) - db(x[SETTLE:])) < 0.5


def test_gate_is_block_invariant():
    signal = np.concatenate([voice_like(0.3), np.zeros(9_600, dtype=np.float32), voice_like(0.2)])
    assert_block_invariant(lambda: Gate({"threshold_db": -35.0, "release_ms": 30.0}), signal)
