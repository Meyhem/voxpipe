import numpy as np
import pytest

from helpers import voice_like
from voxpipe.engine.chain import Chain
from voxpipe.engine.entry import ChainEntry
from voxpipe.engine.registry import EFFECTS, TUNING_CHAIN, create_effect, default_entries


def test_registry_contains_every_effect():
    assert set(EFFECTS) == {
        "gain", "pitch_shift", "filter", "eq", "ring_mod",
        "comb", "distortion", "chorus", "reverb",
    }
    assert all(name in EFFECTS for name in TUNING_CHAIN)


def test_create_effect_unknown_name():
    with pytest.raises(ValueError, match="unknown effect"):
        create_effect("flanger", {})


def test_default_chain_is_near_identity():
    x = voice_like(0.5)
    y = Chain(default_entries()).process_signal(x)
    settle = 4_800
    # the default 20 Hz high-pass phase-shifts the lowest harmonics by a few degrees
    assert np.max(np.abs(y[settle:] - x[settle:])) < 0.15


def test_chain_applies_effects_in_order():
    entries = [ChainEntry("gain", {"gain_db": 6.0206}), ChainEntry("distortion", {"bits": 4.0, "mix": 1.0})]
    x = voice_like(0.2) * 0.4
    y = Chain(entries).process_signal(x)
    levels = y * 8
    np.testing.assert_allclose(levels, np.round(levels), atol=1e-5)  # crush ran last


def test_live_blocks_match_offline_signal():
    entries = [
        ChainEntry("pitch_shift", {"semitones": -3.0, "mix": 0.7}),
        ChainEntry("reverb", {"room_size": 0.4, "mix": 0.3}),
    ]
    x = voice_like(1.0)
    offline = Chain(entries).process_signal(x)
    live_chain = Chain(entries)
    live = np.concatenate([live_chain.process_block(x[i : i + 480]) for i in range(0, x.size, 480)])
    np.testing.assert_allclose(live, offline, atol=1e-5)


def test_latency_sums_effects():
    assert Chain(default_entries()).latency_samples == 0
    assert Chain([ChainEntry("pitch_shift", {"semitones": 2.0, "mix": 1.0})]).latency_samples == 720
