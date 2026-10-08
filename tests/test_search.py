import numpy as np
import pytest

from helpers import voice_like
from voxpipe.engine.chain import Chain
from voxpipe.engine.entry import ChainEntry
from voxpipe.tuning import search
from voxpipe.tuning.features import frame_db
from voxpipe.tuning.objective import chain_distance, prepare_pair_from_arrays, similarity
from voxpipe.tuning.search import TuneSettings, tune

TEMPLATE = [
    ChainEntry("gain", {"gain_db": 0.0}),
    ChainEntry("ring_mod", {"freq_hz": 100.0, "mix": 0.0}),
]


HIDDEN = [ChainEntry("gain", {"gain_db": -9.0}), ChainEntry("ring_mod", {"freq_hz": 300.0, "mix": 0.6})]


@pytest.fixture(scope="module")
def pairs():
    reference = voice_like(1.0)
    target = Chain(HIDDEN).process_signal(reference)
    return [prepare_pair_from_arrays("synthetic", reference, target)]


def test_search_improves_on_baseline(pairs):
    # The overall level is free (objective v4), so only the ring mod is left to find. The
    # synthetic voice aligns loosely, which caps even the hidden chain's own score; the
    # search must at least reach it.
    baseline = similarity(chain_distance(TEMPLATE, pairs))
    result = tune(pairs, TuneSettings(evaluations=240, seed=3, workers=1), template=TEMPLATE)
    assert result.score > baseline + 3.0
    assert result.score >= similarity(chain_distance(HIDDEN, pairs))
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


def test_restarts_from_best_when_stalled(pairs, monkeypatch):
    # Nothing beats the starting point, so the search must restart around the best point
    # found so far, with a fresh seed each time (D-07).
    starts = []
    real = search._new_strategy

    def spy(x0, sigma, seed, budget):
        starts.append((np.array(x0), sigma, seed))
        return real(x0, sigma, seed, budget)

    monkeypatch.setattr(search, "_new_strategy", spy)
    x0 = search.ParamSpace(TEMPLATE).encode(TEMPLATE)
    monkeypatch.setattr(search, "_evaluate", lambda x: 1.0 + float(np.sum((np.asarray(x) - x0) ** 2)))
    budget = 24 * (2 * search.STALL_GENERATIONS + 1) + 1
    tune(pairs, TuneSettings(evaluations=budget, seed=5, workers=1), template=TEMPLATE)
    assert [s[2] for s in starts] == [5, 6, 7]
    assert starts[0][1] == search.SIGMA0 and starts[1][1] == search.RESTART_SIGMA
    assert all(np.array_equal(s[0], x0) for s in starts)


def test_output_level_matches_reference(pairs):
    # The score ignores overall level, so the final gain is set to keep the user's own
    # speech level and never inherits whatever loudness the search drifted to.
    template = TEMPLATE + [ChainEntry("gain", {"gain_db": 0.0})]
    result = tune(pairs, TuneSettings(evaluations=48, seed=2, workers=1), template=template)
    data = pairs[0]
    processed = Chain(result.entries).process_signal(data.reference)
    speech = data.reference_speech if data.reference_speech.any() else np.ones_like(data.reference_speech)
    level = np.mean(frame_db(processed)[: speech.size][speech])
    assert level == pytest.approx(np.mean(data.reference_db[speech]), abs=0.5)
