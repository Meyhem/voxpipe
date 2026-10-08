import numpy as np

from helpers import assert_block_invariant, voice_like
from voxpipe.audio import SAMPLE_RATE
from voxpipe.engine.effects.distortion import Distortion
from voxpipe.engine.effects.modulation import RingMod


def test_ring_mod_turns_dc_into_carrier():
    x = np.full(SAMPLE_RATE // 10, 0.5, dtype=np.float32)
    y = RingMod({"freq_hz": 100.0, "mix": 1.0}).process(x)
    quarter_period = SAMPLE_RATE // 400  # 100 Hz carrier, quarter period = 120 samples
    assert abs(y[quarter_period] - 0.5) < 1e-4
    assert abs(y[2 * quarter_period]) < 1e-4


def test_ring_mod_mix_zero_is_identity():
    x = voice_like(0.2)
    np.testing.assert_array_equal(RingMod({"mix": 0.0}).process(x), x)


def test_bitcrush_quantises_to_levels():
    x = voice_like(0.2)
    y = Distortion({"bits": 4.0, "mix": 1.0}).process(x)
    levels = y * 8  # 4 bits -> steps of 1/8
    np.testing.assert_allclose(levels, np.round(levels), atol=1e-5)


def test_drive_saturates():
    x = np.array([0.1, 0.9], dtype=np.float32)
    y = Distortion({"drive_db": 30.0, "mix": 1.0}).process(x)
    assert y[1] < 1.0 + 1e-6
    assert y[0] / x[0] > y[1] / x[1]  # small signals amplified more than large ones


def test_effects_are_block_invariant():
    signal = voice_like(0.5)
    assert_block_invariant(lambda: RingMod({"freq_hz": 73.0, "mix": 0.7}), signal)
    assert_block_invariant(
        lambda: Distortion({"drive_db": 18.0, "bits": 8.0, "mix": 0.6}), signal
    )
