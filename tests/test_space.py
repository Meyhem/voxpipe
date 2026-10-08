import numpy as np

from voxpipe.engine.entry import ChainEntry
from voxpipe.engine.registry import EFFECTS, default_entries
from voxpipe.tuning.space import ParamSpace


def test_size_counts_every_parameter():
    template = default_entries()
    expected = sum(len(EFFECTS[e.effect].params) for e in template)
    assert ParamSpace(template).size == expected == 33


def test_round_trip_defaults():
    space = ParamSpace(default_entries())
    decoded = space.decode(space.encode(default_entries()))
    for got, want in zip(decoded, default_entries(), strict=True):
        assert got.effect == want.effect
        for name, value in want.params.items():
            assert abs(got.params[name] - value) < 1e-4


def test_log_scale_midpoint_is_geometric_mean():
    space = ParamSpace([ChainEntry("ring_mod", {"freq_hz": 100.0, "mix": 0.0})])
    entry = space.decode(np.array([0.5, 0.5]))[0]
    assert abs(entry.params["freq_hz"] - np.sqrt(20.0 * 2000.0)) < 1e-3
    assert entry.params["mix"] == 0.5


def test_decode_clips_out_of_bounds():
    space = ParamSpace([ChainEntry("gain", {"gain_db": 0.0})])
    assert space.decode(np.array([1.7]))[0].params["gain_db"] == 24.0
    assert space.decode(np.array([-3.0]))[0].params["gain_db"] == -24.0


def test_decoded_values_stay_in_schema_range():
    space = ParamSpace(default_entries())
    rng = np.random.default_rng(0)
    for _ in range(50):
        for entry in space.decode(rng.random(space.size)):
            for param in EFFECTS[entry.effect].params:
                assert param.minimum <= entry.params[param.name] <= param.maximum
