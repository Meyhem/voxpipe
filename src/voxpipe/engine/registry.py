from collections.abc import Mapping

from voxpipe.audio import SAMPLE_RATE
from voxpipe.engine.effect import Effect
from voxpipe.engine.effects.distortion import Distortion
from voxpipe.engine.effects.filters import Eq, Filter
from voxpipe.engine.effects.gain import Gain
from voxpipe.engine.effects.gate import Gate
from voxpipe.engine.effects.modulation import Chorus, RingMod
from voxpipe.engine.effects.pitch import PitchShift
from voxpipe.engine.effects.resonators import Comb, Reverb
from voxpipe.engine.effects.vocoder import Vocoder
from voxpipe.engine.entry import ChainEntry

EFFECTS: dict[str, type[Effect]] = {
    cls.name: cls
    for cls in (Gate, Gain, PitchShift, Filter, Eq, Vocoder, RingMod, Comb, Distortion, Chorus, Reverb)
}

# The chain order `tune` always emits (tech spec section 5). Gain appears twice: input and output.
# The gate comes first so mic hiss is removed before any stage can amplify it.
TUNING_CHAIN: tuple[str, ...] = (
    "gate", "gain", "pitch_shift", "filter", "eq", "vocoder", "ring_mod",
    "comb", "distortion", "chorus", "reverb", "gain",
)


def create_effect(
    name: str, params: Mapping[str, float], sample_rate: int = SAMPLE_RATE
) -> Effect:
    try:
        cls = EFFECTS[name]
    except KeyError:
        raise ValueError(f"unknown effect: {name}") from None
    return cls(params, sample_rate)


def default_entries() -> list[ChainEntry]:
    return [ChainEntry(name, EFFECTS[name].defaults()) for name in TUNING_CHAIN]
