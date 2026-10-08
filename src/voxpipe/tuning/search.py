"""Seeded CMA-ES over the normalised parameter space (D-07)."""

from __future__ import annotations

import multiprocessing
import os
import signal
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cma
import numpy as np

from voxpipe.engine.entry import ChainEntry
from voxpipe.engine.registry import default_entries
from voxpipe.tuning.objective import PairData, chain_distance, similarity
from voxpipe.tuning.space import ParamSpace

POPULATION = 24  # fixed, never derived from CPU count, so results are machine-independent
SIGMA0 = 0.25

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

    strategy = cma.CMAEvolutionStrategy(
        x0,
        SIGMA0,
        {
            "seed": settings.seed,
            "bounds": [0.0, 1.0],
            "popsize": POPULATION,
            "maxfevals": settings.evaluations,
            "verbose": -9,
        },
    )
    workers = settings.workers or os.cpu_count() or 1
    pool = None
    if workers > 1:
        pool = multiprocessing.get_context("forkserver").Pool(
            workers, initializer=_init_worker, initargs=(list(pairs), template)
        )
    try:
        while not strategy.stop() and evaluations < settings.evaluations:
            candidates = strategy.ask()
            scores = pool.map(_evaluate, candidates) if pool else [_evaluate(c) for c in candidates]
            strategy.tell(candidates, scores)
            evaluations += len(candidates)
            i = int(np.argmin(scores))
            if scores[i] < best_f:
                best_x, best_f = np.array(candidates[i]), scores[i]
            if progress is not None:
                progress(evaluations, similarity(best_f))
    except KeyboardInterrupt:
        partial = True
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    return TuneResult(
        entries=space.decode(best_x),
        score=similarity(best_f),
        evaluations=evaluations,
        partial=partial,
        duration_s=time.monotonic() - started,
    )
