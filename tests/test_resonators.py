import numpy as np

from helpers import assert_block_invariant, rms, voice_like
from voxpipe.audio import SAMPLE_RATE
from voxpipe.engine.effects.resonators import Comb, Reverb


def impulse(seconds: float) -> np.ndarray:
    x = np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)
    x[0] = 1.0
    return x


def test_comb_impulse_response_echoes_at_delay():
    y = Comb({"delay_ms": 5.0, "feedback": 0.5, "mix": 1.0}).process(impulse(0.1))
    d = 240  # 5 ms at 48 kHz
    np.testing.assert_allclose([y[0], y[d], y[2 * d]], [0.5, 0.25, 0.125], atol=1e-6)
    assert abs(y[d // 2]) < 1e-9


def test_reverb_tail_decays():
    y = Reverb({"room_size": 0.5, "mix": 1.0}).process(impulse(1.0))
    early = rms(y[4_800:9_600])
    late = rms(y[38_400:43_200])
    assert early > late > 0.0


def test_mix_zero_is_identity():
    x = voice_like(0.2)
    np.testing.assert_array_equal(Comb({"mix": 0.0}).process(x), x)
    np.testing.assert_array_equal(Reverb({"mix": 0.0}).process(x), x)


def test_resonators_are_block_invariant():
    signal = voice_like(0.5)
    assert_block_invariant(lambda: Comb({"delay_ms": 3.3, "feedback": -0.7, "mix": 0.5}), signal)
    assert_block_invariant(lambda: Reverb({"room_size": 0.8, "mix": 0.4}), signal)
