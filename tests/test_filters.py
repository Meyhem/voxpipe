from helpers import assert_block_invariant, rms, sine, voice_like
from voxpipe.engine.effects.filters import Eq, Filter

SETTLE = 4_800  # skip the first 100 ms of filter transient


def test_highpass_removes_low_tone():
    x = sine(100.0)
    y = Filter({"highpass_hz": 1000.0}).process(x)
    assert rms(y[SETTLE:]) < 0.05 * rms(x[SETTLE:])


def test_lowpass_removes_high_tone():
    x = sine(8000.0)
    y = Filter({"lowpass_hz": 2000.0}).process(x)
    assert rms(y[SETTLE:]) < 0.1 * rms(x[SETTLE:])


def test_filter_passes_midband():
    x = sine(1000.0)
    y = Filter({"highpass_hz": 100.0, "lowpass_hz": 10000.0}).process(x)
    assert 0.9 < rms(y[SETTLE:]) / rms(x[SETTLE:]) < 1.1


def test_eq_mid_boost_12db_at_centre():
    x = sine(1500.0)
    y = Eq({"mid_hz": 1500.0, "mid_gain_db": 12.0}).process(x)
    assert 3.7 < rms(y[SETTLE:]) / rms(x[SETTLE:]) < 4.3


def test_eq_flat_by_default():
    x = voice_like(0.5)
    y = Eq().process(x)
    assert rms(y - x) < 1e-5


def test_filters_are_block_invariant():
    signal = voice_like(0.5)
    assert_block_invariant(lambda: Filter({"highpass_hz": 300.0, "lowpass_hz": 3000.0}), signal)
    assert_block_invariant(
        lambda: Eq({"low_gain_db": -6.0, "mid_gain_db": 9.0, "high_gain_db": 3.0, "q": 2.0}),
        signal,
    )
