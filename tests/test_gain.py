import numpy as np
import pytest

from helpers import assert_block_invariant, sine, voice_like
from voxpipe.engine.effects.gain import Gain
from voxpipe.engine.params import Param


def test_param_rejects_default_outside_range():
    with pytest.raises(ValueError):
        Param("x", 0.0, 1.0, 2.0)


def test_param_rejects_log_scale_with_non_positive_minimum():
    with pytest.raises(ValueError):
        Param("x", 0.0, 1.0, 0.5, "log")


def test_unknown_parameter_rejected():
    with pytest.raises(ValueError, match="unknown parameter"):
        Gain({"volume": 1.0})


def test_defaults_fill_missing_values():
    assert Gain().values == {"gain_db": 0.0}
    assert Gain.defaults() == {"gain_db": 0.0}


def test_gain_6db_doubles_amplitude():
    x = sine(440.0)
    y = Gain({"gain_db": 6.0206}).process(x)
    assert y.dtype == np.float32
    np.testing.assert_allclose(y, 2 * x, rtol=1e-4, atol=1e-6)


def test_gain_is_block_invariant():
    assert_block_invariant(lambda: Gain({"gain_db": -7.5}), voice_like(0.5))
