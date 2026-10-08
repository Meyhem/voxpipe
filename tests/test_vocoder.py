import numpy as np

from helpers import assert_block_invariant, rms, voice_like
from voxpipe.audio import SAMPLE_RATE
from voxpipe.engine.effects.vocoder import Vocoder
from voxpipe.tuning.features import pitch

SETTLE = 4_800


def test_mix_zero_is_identity():
    x = voice_like(0.5)
    np.testing.assert_array_equal(Vocoder().process(x), x)


def test_output_takes_the_carrier_pitch():
    x = voice_like(1.0)  # glides between 60 and 180 Hz
    y = Vocoder({"carrier_hz": 100.0, "noise": 0.0, "mix": 1.0}).process(x)
    f0 = pitch(y[SETTLE:])
    voiced = f0[f0 > 0]
    assert voiced.size > 0.5 * f0.size
    assert abs(np.median(voiced) - 100.0) < 5.0


def test_level_follows_the_voice():
    x = voice_like(1.0)
    y = Vocoder({"mix": 1.0}).process(x)
    assert 0.25 < rms(y[SETTLE:]) / rms(x[SETTLE:]) < 4.0


def test_silence_in_silence_out():
    x = np.concatenate([voice_like(0.5), np.zeros(SAMPLE_RATE // 2, dtype=np.float32)])
    y = Vocoder({"noise": 0.5, "mix": 1.0}).process(x)
    assert rms(y[-SAMPLE_RATE // 4 :]) < 0.01 * rms(y[SETTLE : SAMPLE_RATE // 2])


def test_vocoder_is_block_invariant():
    assert_block_invariant(
        lambda: Vocoder({"carrier_hz": 90.0, "noise": 0.3, "smooth_ms": 8.0, "mix": 0.8}),
        voice_like(0.5),
    )
