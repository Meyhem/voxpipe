import numpy as np

from helpers import assert_block_invariant, dominant_hz, sine, voice_like
from voxpipe.audio import SAMPLE_RATE
from voxpipe.engine.delay import History, read_delayed
from voxpipe.engine.effects.modulation import Chorus
from voxpipe.engine.effects.pitch import PitchShift


def test_read_delayed_integer_delay_is_exact():
    history = History(10)
    first = history.extend(np.arange(1, 6, dtype=np.float32))
    np.testing.assert_array_equal(read_delayed(first, 10, np.full(5, 2.0)), [0, 0, 1, 2, 3])
    second = history.extend(np.arange(6, 9, dtype=np.float32))
    np.testing.assert_array_equal(read_delayed(second, 10, np.full(3, 2.0)), [4, 5, 6])


def test_read_delayed_interpolates():
    full = History(4).extend(np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32))
    out = read_delayed(full, 4, np.array([1.5, 1.5, 1.5, 1.5]))
    np.testing.assert_allclose(out[2:], [0.5, 1.5])


def test_chorus_without_depth_is_pure_delay():
    x = np.zeros(SAMPLE_RATE // 10, dtype=np.float32)
    x[0] = 1.0
    y = Chorus({"delay_ms": 10.0, "depth_ms": 0.0, "mix": 1.0}).process(x)
    assert np.argmax(y) == 480
    assert abs(y[480] - 1.0) < 1e-6


# 400 Hz is a whole number of cycles per half-window (15 ms), so the two taps stay in phase;
# other inputs split energy into +-33 Hz crossfade sidebands (known artefact, K-02).
def test_pitch_shift_octave_up_doubles_frequency():
    y = PitchShift({"semitones": 12.0, "mix": 1.0}).process(sine(400.0))
    assert abs(dominant_hz(y[4_800:]) - 800.0) < 15.0


def test_pitch_shift_down_lowers_frequency():
    y = PitchShift({"semitones": -12.0, "mix": 1.0}).process(sine(800.0))
    assert abs(dominant_hz(y[4_800:]) - 400.0) < 15.0


def test_pitch_shift_latency_only_when_active():
    assert PitchShift({"mix": 0.0}).latency_samples == 0
    assert PitchShift({"semitones": 3.0, "mix": 0.5}).latency_samples == 720


def test_delay_effects_are_block_invariant():
    signal = voice_like(0.5)
    assert_block_invariant(
        lambda: Chorus({"delay_ms": 12.0, "depth_ms": 4.0, "rate_hz": 1.3, "mix": 0.5}), signal
    )
    assert_block_invariant(lambda: PitchShift({"semitones": -5.0, "mix": 0.8}), signal)
    assert_block_invariant(lambda: PitchShift({"semitones": 7.0, "mix": 1.0}), signal)
