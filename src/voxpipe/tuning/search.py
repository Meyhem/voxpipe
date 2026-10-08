"""Seeded CMA-ES over the normalised parameter space (D-07)."""

from __future__ import annotations

import multiprocessing
import os
import signal
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from voxpipe.engine.chain import Chain
from voxpipe.engine.entry import ChainEntry
from voxpipe.engine.registry import EFFECTS, default_entries
from voxpipe.tuning.features import frame_db
from voxpipe.tuning.objective import PairData, chain_distance, compensate_latency, similarity
from voxpipe.tuning.space import ParamSpace

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Could not import matplotlib")
    import cma

POPULATION = 24  # fixed, never derived from CPU count, so results are machine-independent
SIGMA0 = 0.25
STALL_GENERATIONS = 15  # generations without a new best before restarting around it
RESTART_SIGMA = 0.15  # restarts refine around the best point, so they search more narrowly

_state: dict = {}


@dataclass(frozen=True)
class TuneSettings:
    evaluations: int = 4000
    seed: int = 1
    workers: int = 0  # 0 = all CPUs; 1 = evaluate in-process


@dataclass(frozen=True)
class TuneResult:
    entries: list[ChainEntry]
    score: float
    evaluations: int
    partial: bool
    duration_s: float


def _set_state(pairs: Sequence[PairData], template: Sequence[ChainEntry]) -> None:
    _state["pairs"] = list(pairs)
    _state["space"] = ParamSpace(template)


def _init_worker(pairs: Sequence[PairData], template: Sequence[ChainEntry]) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent handles Ctrl+C
    _set_state(pairs, template)


def _evaluate(x: np.ndarray) -> float:
    return chain_distance(_state["space"].decode(x), _state["pairs"])


def _new_strategy(x0: np.ndarray, sigma: float, seed: int, budget: int) -> cma.CMAEvolutionStrategy:
    return cma.CMAEvolutionStrategy(
        x0,
        sigma,
        {"seed": seed, "bounds": [0.0, 1.0], "popsize": POPULATION, "maxfevals": budget, "verbose": -9},
    )


def _level_output(entries: list[ChainEntry], pairs: Sequence[PairData]) -> list[ChainEntry]:
    """Set a trailing gain so processed speech is as loud as the user's own mic.

    The objective ignores overall level, so the search leaves output loudness wherever it
    drifted; a final gain is a pure level change and can be fixed without changing the score."""
    if not entries or entries[-1].effect != "gain":
        return entries
    chain = Chain(entries)
    deltas = []
    for data in pairs:
        processed = compensate_latency(chain.process_signal(data.reference), chain.latency_samples)
        speech = data.reference_speech
        if not speech.any():  # no pauses to tell speech from background: use every frame
            speech = np.ones_like(speech)
        deltas.append(np.mean(data.reference_db[speech]) - np.mean(frame_db(processed)[: speech.size][speech]))
    (param,) = EFFECTS["gain"].params
    gain_db = entries[-1].params["gain_db"] + float(np.mean(deltas))
    gain_db = min(param.maximum, max(param.minimum, round(gain_db, 6)))
    return [*entries[:-1], ChainEntry("gain", {"gain_db": gain_db})]


def tune(
    pairs: Sequence[PairData],
    settings: TuneSettings | None = None,
    progress: Callable[[int, float], None] | None = None,
    template: Sequence[ChainEntry] | None = None,
) -> TuneResult:
    started = time.monotonic()
    settings = settings or TuneSettings()
    template = list(template or default_entries())
    _set_state(pairs, template)
    space = _state["space"]
    x0 = space.encode(template)
    best_x, best_f = x0, _evaluate(x0)
    evaluations, partial = 1, False
    restarts, stalled = 0, 0

    strategy = _new_strategy(x0, SIGMA0, settings.seed, settings.evaluations)
    workers = settings.workers or os.cpu_count() or 1
    pool = None
    if workers > 1:
        pool = multiprocessing.get_context("forkserver").Pool(
            workers, initializer=_init_worker, initargs=(list(pairs), template)
        )
    try:
        while evaluations < settings.evaluations:
            # CMA-ES follows its population mean, so a lucky outlier can stay unbeaten
            # for a long time; when progress stalls, restart around the best point (D-07).
            if strategy.stop() or stalled >= STALL_GENERATIONS:
                restarts, stalled = restarts + 1, 0
                strategy = _new_strategy(
                    best_x, RESTART_SIGMA, settings.seed + restarts, settings.evaluations - evaluations
                )
            candidates = strategy.ask()
            scores = pool.map(_evaluate, candidates) if pool else [_evaluate(c) for c in candidates]
            strategy.tell(candidates, scores)
            evaluations += len(candidates)
            i = int(np.argmin(scores))
            if scores[i] < best_f:
                best_x, best_f = np.array(candidates[i]), scores[i]
                stalled = 0
            else:
                stalled += 1
            if progress is not None:
                progress(evaluations, similarity(best_f))
    except KeyboardInterrupt:
        partial = True
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    return TuneResult(
        entries=_level_output(space.decode(best_x), pairs),
        score=similarity(best_f),
        evaluations=evaluations,
        partial=partial,
        duration_s=time.monotonic() - started,
    )
