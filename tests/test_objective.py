import shutil

import numpy as np
import pytest

from helpers import voice_like
from voxpipe.engine.entry import ChainEntry
from voxpipe.engine.registry import default_entries
from voxpipe.media import write_wav
from voxpipe.tuning.objective import (
    chain_distance,
    floor_lift,
    pair_distance,
    pitch_distance,
    prepare_pair,
    prepare_pair_from_arrays,
    similarity,
)
from voxpipe.tuning.samples import SamplePair


def test_identical_audio_scores_near_100():
    x = voice_like(1.0)
    data = prepare_pair_from_arrays("a", x, x)
    assert pair_distance(x, data) < 1e-6
    assert similarity(chain_distance(default_entries(), [data])) > 97.0


def test_matching_gain_beats_identity():
    reference = voice_like(1.0)
    target = reference * np.float32(0.25)  # target is 12 dB quieter
    data = prepare_pair_from_arrays("a", reference, target)
    identity = chain_distance([ChainEntry("gain", {"gain_db": 0.0})], [data])
    matched = chain_distance([ChainEntry("gain", {"gain_db": -12.04})], [data])
    assert matched < 0.2 * identity


def test_pitch_mismatch_is_penalised():
    reference = voice_like(1.0)
    data = prepare_pair_from_arrays("a", reference, reference)
    shifted = chain_distance([ChainEntry("pitch_shift", {"semitones": 5.0, "mix": 1.0})], [data])
    assert shifted > chain_distance(default_entries(), [data]) + 0.1


def test_alignment_absorbs_timing_differences():
    # Both recordings start with silence, but the target's lead-in is 200 ms longer.
    voice = voice_like(1.0)
    reference = np.concatenate([np.zeros(4_800, dtype=np.float32), voice])
    target = np.concatenate([np.zeros(14_400, dtype=np.float32), voice])
    data = prepare_pair_from_arrays("a", reference, target)
    assert similarity(pair_distance(reference, data)) > 85.0


def test_similarity_monotonic():
    assert similarity(0.0) == 100.0
    assert similarity(0.5) > similarity(1.0) > 0.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_prepare_pair_warns_on_length_mismatch(tmp_path):
    write_wav(tmp_path / "a-target.wav", voice_like(1.0))
    write_wav(tmp_path / "a-reference.wav", voice_like(2.0))
    pair = SamplePair("a", tmp_path / "a-target.wav", tmp_path / "a-reference.wav")
    data, warnings = prepare_pair(pair)
    assert data.id == "a" and data.reference.size == 96_000
    assert len(warnings) == 1 and "50%" in warnings[0]


def test_pitch_distance_penalises_dropped_voicing():
    # The tuner must not win by destroying pitch: an unvoiced frame where the target is
    # voiced costs as much as a full octave error.
    target = np.array([100.0, 100.0, 100.0, 0.0])
    assert pitch_distance(target, target) == 0.0
    assert pitch_distance(np.zeros(4), target) == pytest.approx(1.0)
    assert pitch_distance(np.array([400.0, 400.0, 400.0, 0.0]), target) == pytest.approx(1.0)
    assert pitch_distance(np.array([100.0, 100.0, 100.0, 100.0]), target) == pytest.approx(0.5)


def test_objective_version_bumped():
    from voxpipe.tuning.objective import OBJECTIVE_VERSION

    assert OBJECTIVE_VERSION == 3


def _with_pauses(voice: np.ndarray, floor: float, seed: int = 0) -> np.ndarray:
    pause = np.zeros(14_400, dtype=np.float32)
    signal = np.concatenate([pause, voice, pause])
    return signal + np.random.default_rng(seed).normal(0.0, floor, signal.size).astype(np.float32)


def test_target_background_is_not_rewarded():
    # A target with a loud background bed must not teach the tuner to fill pauses with
    # noise: the clean reference beats the reference with the same bed added.
    voice = voice_like(1.0)
    reference = _with_pauses(voice, 3e-4)  # clean mic, about -70 dB floor
    target = _with_pauses(voice, 0.015, seed=1)  # background at about -36 dB
    data = prepare_pair_from_arrays("a", reference, target)
    noisy = reference + np.random.default_rng(2).normal(0.0, 0.015, reference.size).astype(np.float32)
    assert pair_distance(reference, data) < pair_distance(noisy, data)


def test_floor_lift_ignores_uniform_gain_but_catches_compression():
    reference = _with_pauses(voice_like(1.0), 3e-4)
    data = prepare_pair_from_arrays("a", reference, reference)
    assert floor_lift(reference * np.float32(4.0), data) < 0.05
    compressed = np.tanh(reference * np.float32(300.0)).astype(np.float32)  # lifts pauses
    assert floor_lift(compressed, data) > 1.0
