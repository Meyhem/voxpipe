from collections.abc import Mapping

from voxpipe.audio import SAMPLE_RATE
from voxpipe.engine.effect import Effect
from voxpipe.engine.effects.distortion import Distortion
from voxpipe.engine.effects.filters import Eq, Filter
from voxpipe.engine.effects.gain import Gain
from voxpipe.engine.effects.modulation import Chorus, RingMod
from voxpipe.engine.effects.pitch import PitchShift
from voxpipe.engine.effects.resonators import Comb, Reverb
from voxpipe.engine.entry import ChainEntry

EFFECTS: dict[str, type[Effect]] = {
    cls.name: cls
    for cls in (Gain, PitchShift, Filter, Eq, RingMod, Comb, Distortion, Chorus, Reverb)
}

# The chain order `tune` always emits (tech spec section 5). Gain appears twice: input and output.
TUNING_CHAIN: tuple[str, ...] = (
    "gain", "pitch_shift", "filter", "eq", "ring_mod",
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
