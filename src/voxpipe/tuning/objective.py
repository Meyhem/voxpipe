"""Tuning objective (D-06): distance between processed reference and target speech."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from voxpipe.engine.chain import Chain
from voxpipe.engine.entry import ChainEntry
from voxpipe.media import decode
from voxpipe.tuning.align import cosine_cost, dtw_path
from voxpipe.tuning.features import log_mel, ltas, pitch
from voxpipe.tuning.samples import SamplePair

OBJECTIVE_VERSION = 2  # bump whenever scoring changes; stored in profile provenance
LENGTH_MISMATCH_TOLERANCE = 0.30  # R-12 (A-03)
W_MEL, W_PITCH, W_LTAS = 1.0, 0.5, 0.5
DB_UNIT = 10.0  # spectral distances are measured in units of 10 dB


@dataclass(frozen=True)
class PairData:
    id: str
    reference: np.ndarray
    path_ref: np.ndarray
    path_tgt: np.ndarray
    target_mel: np.ndarray
    target_f0: np.ndarray
    target_ltas: np.ndarray


def prepare_pair_from_arrays(pair_id: str, reference: np.ndarray, target: np.ndarray) -> PairData:
    """Align once on the unprocessed reference; effects do not change timing."""
    ref_mel, tgt_mel = log_mel(reference), log_mel(target)
    path_ref, path_tgt = dtw_path(cosine_cost(ref_mel, tgt_mel))
    return PairData(
        id=pair_id,
        reference=np.asarray(reference, dtype=np.float32),
        path_ref=path_ref,
        path_tgt=path_tgt,
        target_mel=tgt_mel,
        target_f0=pitch(target),
        target_ltas=ltas(tgt_mel[path_tgt]),  # over aligned frames: unequal silence must not bias it
    )


def prepare_pair(pair: SamplePair) -> tuple[PairData, list[str]]:
    reference, target = decode(pair.reference), decode(pair.target)
    warnings = []
    mismatch = abs(reference.size - target.size) / max(reference.size, target.size)
    if mismatch > LENGTH_MISMATCH_TOLERANCE:
        warnings.append(
            f"pair '{pair.id}': target and reference lengths differ by {mismatch:.0%}; "
            "do they contain the same words?"
        )
    return prepare_pair_from_arrays(pair.id, reference, target), warnings


def pair_distance(processed: np.ndarray, data: PairData) -> float:
    mel = log_mel(processed)
    d_mel = np.mean(np.abs(mel[data.path_ref] - data.target_mel[data.path_tgt])) / DB_UNIT

    d_pitch = pitch_distance(pitch(processed)[data.path_ref], data.target_f0[data.path_tgt])

    d_ltas = np.mean(np.abs(ltas(mel[data.path_ref]) - data.target_ltas)) / DB_UNIT
    return float(W_MEL * d_mel + W_PITCH * d_pitch + W_LTAS * d_ltas)


def pitch_distance(f0_proc: np.ndarray, f0_tgt: np.ndarray) -> float:
    """Aligned pitch contours (0 = unvoiced) -> distance in octaves.

    Where the target is voiced, each frame costs its octave error (capped at 1); an
    unvoiced processed frame costs the full 1, so destroying pitch never pays. Where the
    target is unvoiced, a voiced processed frame costs 0.5."""
    voiced_proc, voiced_tgt = f0_proc > 0, f0_tgt > 0
    on_voiced = 0.0
    if voiced_tgt.any():
        safe_proc = np.where(voiced_proc, f0_proc, 1.0)
        octaves = np.minimum(np.abs(np.log2(safe_proc / np.where(voiced_tgt, f0_tgt, 1.0))), 1.0)
        on_voiced = float(np.mean(np.where(voiced_proc, octaves, 1.0)[voiced_tgt]))
    on_unvoiced = float(np.mean(voiced_proc[~voiced_tgt])) if (~voiced_tgt).any() else 0.0
    return on_voiced + 0.5 * on_unvoiced


def _compensate_latency(processed: np.ndarray, latency: int) -> np.ndarray:
    if latency <= 0:
        return processed
    return np.concatenate([processed[latency:], np.zeros(latency, dtype=np.float32)])


def chain_distance(entries: Sequence[ChainEntry], pairs: Sequence[PairData]) -> float:
    chain = Chain(entries)
    total = 0.0
    for data in pairs:
        processed = _compensate_latency(chain.process_signal(data.reference), chain.latency_samples)
        total += pair_distance(processed, data)
    return total / len(pairs)


def similarity(distance: float) -> float:
    return float(100.0 * np.exp(-distance))
