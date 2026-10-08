import numpy as np
import pytest

from helpers import voice_like
from voxpipe.engine.chain import Chain
from voxpipe.engine.entry import ChainEntry
from voxpipe.tuning.objective import chain_distance, prepare_pair_from_arrays, similarity
from voxpipe.tuning.search import TuneSettings, tune

TEMPLATE = [
    ChainEntry("gain", {"gain_db": 0.0}),
    ChainEntry("ring_mod", {"freq_hz": 100.0, "mix": 0.0}),
]


@pytest.fixture(scope="module")
def pairs():
    reference = voice_like(1.0)
    hidden = [ChainEntry("gain", {"gain_db": -9.0}), ChainEntry("ring_mod", {"freq_hz": 300.0, "mix": 0.6})]
    target = Chain(hidden).process_signal(reference)
    return [prepare_pair_from_arrays("synthetic", reference, target)]


def test_search_improves_on_baseline(pairs):
    baseline = similarity(chain_distance(TEMPLATE, pairs))
    result = tune(pairs, TuneSettings(evaluations=240, seed=3, workers=1), template=TEMPLATE)
    assert result.score > baseline + 10.0
    assert not result.partial
    assert 240 <= result.evaluations < 240 + 24 + 1


def test_deterministic_regardless_of_workers(pairs):
    settings = dict(evaluations=96, seed=7)
    a = tune(pairs, TuneSettings(**settings, workers=1), template=TEMPLATE)
    b = tune(pairs, TuneSettings(**settings, workers=2), template=TEMPLATE)
    assert a.entries == b.entries
    assert a.score == b.score


def test_progress_reports_and_interrupt_returns_partial(pairs):
    seen = []

    def progress(done, score):
        seen.append((done, score))
        if len(seen) == 2:
            raise KeyboardInterrupt

    result = tune(pairs, TuneSettings(evaluations=1000, seed=1, workers=1),
                  progress=progress, template=TEMPLATE)
    assert result.partial
    assert len(seen) == 2
    assert seen[1][0] > seen[0][0]
    assert result.score >= seen[0][1]


def test_never_worse_than_default_chain(pairs):
    result = tune(pairs, TuneSettings(evaluations=24, seed=1, workers=1), template=TEMPLATE)
    assert result.score >= similarity(chain_distance(TEMPLATE, pairs)) - 1e-9


def test_entries_are_complete(pairs):
    result = tune(pairs, TuneSettings(evaluations=24, seed=1, workers=1), template=TEMPLATE)
    assert [e.effect for e in result.entries] == ["gain", "ring_mod"]
    assert set(result.entries[1].params) == {"freq_hz", "mix"}
    assert np.isfinite(result.duration_s)
