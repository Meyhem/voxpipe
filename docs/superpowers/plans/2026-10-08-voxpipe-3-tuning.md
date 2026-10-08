# voxpipe Plan 3: Tuning (`voxpipe tune`)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `voxpipe tune`. It searches the effect chain's settings so the user's processed reference recordings sound as close as possible to the target samples, then saves the result as a profile.

**Architecture:**
- **Sample pairs.** Pairs are found by file name. Each pair is decoded once.
- **Alignment.** Each target is aligned to its *unprocessed* reference once, with DTW on log-mel features. Effects don't change timing, so that alignment is reused for every candidate (D-06).
- **Scoring.** A candidate setting is scored by processing each reference through a fresh `Chain` and comparing to the target along the stored alignment. The comparison weighs three terms: frame-wise log-mel, pitch contour, and long-term average spectrum.
- **Search.** A seeded CMA-ES (`cma` package) searches the normalised parameter space generated from the effects' `Param` declarations. Evaluations are spread across CPU cores with a process pool, and results are collected in order so a run is deterministic (D-07, N-06).
- **Interrupting.** Ctrl+C saves the best result so far, marked partial.

**Tech Stack:** Python 3.12, numpy, `cma` (pycma), multiprocessing (forkserver), pytest.

**Specs:** `design/tech-spec-voxpipe.md` (D-06, D-07, N-05, N-06, section 9), `design/domain.md` (R-05, R-06, R-09, R-12, R-16, R-24, R-25, R-26).

**Prerequisite:** Plan 1 is complete. This plan uses:
- `Chain`, `ChainEntry`, `EFFECTS`, `default_entries`, `Param`;
- `Profile`, `Provenance`, `save_profile`, `load_profile`, `now_iso`;
- `decode`, `write_wav`, `VoxpipeError`, `cli.COMMANDS`.

## Global Constraints

- Sample pairs: `<id>-target.<ext>` and `<id>-reference.<ext>`, in any ffmpeg-readable format (R-09). An incomplete pair is skipped with a warning. Tuning fails only if no complete pair remains (R-16).
- If target and reference lengths differ by more than 30 %, warn (R-12, A-03). This doesn't fail the run.
- `tune` always emits the `TUNING_CHAIN` order with every parameter set (D-04).
- Deterministic: the same samples, `--seed`, `--evaluations` and code give the same profile, whatever the core count. Population size is a fixed constant, not derived from CPU count (N-06).
- The default budget is 4000 evaluations, which targets ≤ 10 min for 10 pairs of 10 s (N-05, A-02).
- The profile is always saved, even with a low score. The score is reported. Ctrl+C saves the best so far with `provenance.partial = true` and exits 130.
- The output path is `<--profiles-dir>/<--name>.json`, and an existing file is overwritten (R-24, R-25). The samples folder defaults to the current directory (R-26).
- `provenance.objective_version` = `OBJECTIVE_VERSION` (1). Bump it whenever the scoring changes.

---

## File Structure

```
src/voxpipe/
  tuning/
    __init__.py
    samples.py        SamplePair, SampleError, find_pairs()
    features.py       frame_count(), frames(), log_mel(), pitch(), ltas()
    align.py          cosine_cost(), dtw_path()
    objective.py      PairData, prepare_pair_from_arrays(), prepare_pair(), pair_distance(),
                      chain_distance(), similarity(), OBJECTIVE_VERSION
    space.py          ParamSpace
    search.py         TuneSettings, TuneResult, tune()
  commands/
    tune.py           `voxpipe tune`
tests/
  test_samples.py
  test_features.py
  test_align.py
  test_objective.py
  test_space.py
  test_search.py
  test_tune_cmd.py
```

---

### Task 1: Sample pair discovery

**Files:**
- Create: `src/voxpipe/tuning/__init__.py` (empty), `src/voxpipe/tuning/samples.py`
- Test: `tests/test_samples.py`

**Interfaces:**
- Consumes: `VoxpipeError` (plan 1).
- Produces:
  - `SampleError(VoxpipeError)`.
  - `SamplePair(id: str, target: Path, reference: Path)`, frozen.
  - `find_pairs(folder: Path) -> tuple[list[SamplePair], list[str]]`. Returns (complete pairs sorted by id, warnings). It ignores hidden files and files that don't match the pattern. It raises `SampleError` when the folder is missing, when an id has two files for the same role, or when there are no complete pairs.

- [ ] **Step 1: Write the failing tests**

`tests/test_samples.py`:
```python
import pytest

from voxpipe.tuning.samples import SampleError, SamplePair, find_pairs


def touch(folder, *names):
    for name in names:
        (folder / name).write_bytes(b"")


def test_pairs_matched_by_id_any_extension(tmp_path):
    touch(tmp_path, "02-target.ogg", "02-reference.wav", "01-target.wav", "01-reference.mp3")
    pairs, warnings = find_pairs(tmp_path)
    assert pairs == [
        SamplePair("01", tmp_path / "01-target.wav", tmp_path / "01-reference.mp3"),
        SamplePair("02", tmp_path / "02-target.ogg", tmp_path / "02-reference.wav"),
    ]
    assert warnings == []


def test_incomplete_pairs_skipped_with_warning(tmp_path):
    touch(tmp_path, "a-target.wav", "a-reference.wav", "b-target.wav", "c-reference.wav")
    pairs, warnings = find_pairs(tmp_path)
    assert [p.id for p in pairs] == ["a"]
    assert any("'b'" in w and "reference" in w for w in warnings)
    assert any("'c'" in w and "target" in w for w in warnings)


def test_ignores_hidden_and_unrelated_files(tmp_path):
    touch(tmp_path, "x-target.wav", "x-reference.wav", ".x-target.wav", "notes.txt", "profile.json")
    pairs, warnings = find_pairs(tmp_path)
    assert [p.id for p in pairs] == ["x"] and warnings == []


def test_id_may_contain_dashes(tmp_path):
    touch(tmp_path, "tech-priest-1-target.wav", "tech-priest-1-reference.wav")
    assert find_pairs(tmp_path)[0][0].id == "tech-priest-1"


def test_duplicate_role_is_an_error(tmp_path):
    touch(tmp_path, "a-target.wav", "a-target.mp3", "a-reference.wav")
    with pytest.raises(SampleError, match="more than one target"):
        find_pairs(tmp_path)


def test_no_complete_pairs(tmp_path):
    touch(tmp_path, "a-target.wav")
    with pytest.raises(SampleError, match="no complete sample pairs"):
        find_pairs(tmp_path)


def test_missing_folder(tmp_path):
    with pytest.raises(SampleError, match="not found"):
        find_pairs(tmp_path / "nope")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_samples.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voxpipe.tuning'`

- [ ] **Step 3: Write the implementation**

`src/voxpipe/tuning/samples.py`:
```python
"""Sample pairs: <id>-target.* plus <id>-reference.* (R-09, R-16)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from voxpipe.errors import VoxpipeError

_PATTERN = re.compile(r"^(?P<id>.+)-(?P<role>target|reference)$")


class SampleError(VoxpipeError):
    pass


@dataclass(frozen=True)
class SamplePair:
    id: str
    target: Path
    reference: Path


def find_pairs(folder: Path) -> tuple[list[SamplePair], list[str]]:
    if not folder.is_dir():
        raise SampleError(f"{folder}: samples folder not found")
    found: dict[str, dict[str, Path]] = {}
    for path in sorted(folder.iterdir()):
        if path.name.startswith(".") or not path.is_file():
            continue
        match = _PATTERN.match(path.stem)
        if match is None:
            continue
        roles = found.setdefault(match["id"], {})
        if match["role"] in roles:
            raise SampleError(
                f"pair '{match['id']}' has more than one {match['role']} file: "
                f"{roles[match['role']].name}, {path.name}"
            )
        roles[match["role"]] = path

    pairs, warnings = [], []
    for pair_id in sorted(found):
        roles = found[pair_id]
        if "target" in roles and "reference" in roles:
            pairs.append(SamplePair(pair_id, roles["target"], roles["reference"]))
        else:
            missing = "reference" if "target" in roles else "target"
            warnings.append(f"skipping pair '{pair_id}': no {missing} recording")
    if not pairs:
        raise SampleError(f"{folder}: no complete sample pairs (<id>-target + <id>-reference)")
    return pairs, warnings
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_samples.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/voxpipe/tuning tests/test_samples.py
git commit -m "feat(tune): sample pair discovery"
```

---

### Task 2: Audio features

**Files:**
- Create: `src/voxpipe/tuning/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `SAMPLE_RATE` (plan 1).
- Produces: all features use hop `HOP = 480` (10 ms), and every feature array has `frame_count(len(signal))` rows.
  - `frame_count(n_samples: int) -> int`, which is `max(1, ceil(n / HOP))`.
  - `frames(signal, size: int) -> np.ndarray` of shape `(frame_count, size)`, zero-padded.
  - `log_mel(signal) -> np.ndarray` of shape `(frames, N_MELS=40)`, in dB.
  - `pitch(signal) -> np.ndarray` of shape `(frames,)`, in Hz, with 0.0 for unvoiced or silent frames. Range 60–500 Hz.
  - `ltas(mel: np.ndarray) -> np.ndarray` of shape `(N_MELS,)`, the mean over frames.

- [ ] **Step 1: Write the failing tests**

`tests/test_features.py`:
```python
import numpy as np
from helpers import sine, voice_like

from voxpipe.tuning.features import HOP, N_MELS, frame_count, frames, log_mel, ltas, pitch


def test_frame_count_and_padding():
    assert frame_count(0) == 1
    assert frame_count(480) == 1
    assert frame_count(481) == 2
    f = frames(np.ones(500, dtype=np.float32), 1024)
    assert f.shape == (2, 1024)
    assert f[0, :500].sum() == 500 and f[0, 500:].sum() == 0
    assert f[1, 0] == 1.0 and f[1, 20] == 0.0  # second frame starts at sample 480


def test_log_mel_shape_and_band_energy():
    low, high = log_mel(sine(200.0)), log_mel(sine(5000.0))
    assert low.shape == (frame_count(48_000), N_MELS)
    assert np.argmax(low.mean(axis=0)) < np.argmax(high.mean(axis=0))


def test_log_mel_tracks_level():
    x = voice_like(0.5)
    diff = log_mel(x) - log_mel(x * 0.5)
    assert abs(np.median(diff) - 6.02) < 0.1


def test_pitch_of_harmonic_tone():
    f0 = pitch(sum(sine(200.0 * k, amplitude=0.3 / k) for k in range(1, 6)))
    voiced = f0[f0 > 0]
    assert voiced.size > 0.8 * f0.size
    assert abs(np.median(voiced) - 200.0) < 4.0


def test_pitch_silence_is_unvoiced():
    assert np.all(pitch(np.zeros(24_000, dtype=np.float32)) == 0.0)


def test_ltas_is_mean_over_frames():
    mel = log_mel(voice_like(0.5))
    np.testing.assert_allclose(ltas(mel), mel.mean(axis=0))


def test_hop_is_10ms():
    assert HOP == 480
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/voxpipe/tuning/features.py`:
```python
"""Frame-level features for the tuning objective. All share a 10 ms hop."""

from __future__ import annotations

from functools import cache

import numpy as np

from voxpipe.audio import SAMPLE_RATE

HOP = 480
MEL_FFT = 1024
N_MELS = 40
MEL_FMIN, MEL_FMAX = 60.0, 12_000.0
PITCH_FRAME = 2048
PITCH_MIN_HZ, PITCH_MAX_HZ = 60.0, 500.0
VOICING_THRESHOLD = 0.5
SILENCE_RMS = 1e-3


def frame_count(n_samples: int) -> int:
    return max(1, -(-n_samples // HOP))


def frames(signal: np.ndarray, size: int) -> np.ndarray:
    n = frame_count(len(signal))
    padded = np.zeros((n - 1) * HOP + size, dtype=np.float64)
    m = min(len(signal), padded.size)
    padded[:m] = signal[:m]
    return np.lib.stride_tricks.sliding_window_view(padded, size)[::HOP][:n]


@cache
def _mel_filterbank() -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    edges = mel_to_hz(np.linspace(hz_to_mel(MEL_FMIN), hz_to_mel(MEL_FMAX), N_MELS + 2))
    bins = np.fft.rfftfreq(MEL_FFT, 1.0 / SAMPLE_RATE)
    bank = np.zeros((N_MELS, bins.size))
    for i in range(N_MELS):
        lo, centre, hi = edges[i], edges[i + 1], edges[i + 2]
        rising = (bins - lo) / (centre - lo)
        falling = (hi - bins) / (hi - centre)
        bank[i] = np.maximum(0.0, np.minimum(rising, falling))
    return bank


@cache
def _hann(size: int) -> np.ndarray:
    return np.hanning(size)


def log_mel(signal: np.ndarray) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(frames(signal, MEL_FFT) * _hann(MEL_FFT), axis=1)) ** 2
    return 10.0 * np.log10(spectrum @ _mel_filterbank().T + 1e-10)


def pitch(signal: np.ndarray) -> np.ndarray:
    """Autocorrelation pitch per frame; 0.0 where unvoiced or silent."""
    f = frames(signal, PITCH_FRAME)
    f = f - f.mean(axis=1, keepdims=True)
    loudness = np.sqrt(np.mean(f**2, axis=1))
    power = np.abs(np.fft.rfft(f, n=2 * PITCH_FRAME, axis=1)) ** 2
    autocorr = np.fft.irfft(power, axis=1)[:, :PITCH_FRAME]
    lag_min = int(SAMPLE_RATE / PITCH_MAX_HZ)
    lag_max = int(SAMPLE_RATE / PITCH_MIN_HZ)
    normalised = autocorr[:, lag_min : lag_max + 1] / np.maximum(autocorr[:, :1], 1e-12)
    best = np.argmax(normalised, axis=1)
    strength = normalised[np.arange(best.size), best]
    voiced = (strength > VOICING_THRESHOLD) & (loudness > SILENCE_RMS)
    return np.where(voiced, SAMPLE_RATE / (best + lag_min), 0.0)


def ltas(mel: np.ndarray) -> np.ndarray:
    return mel.mean(axis=0)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_features.py -v`
Expected: 7 passed. If `test_pitch_of_harmonic_tone` reports about 100 Hz (an octave error), check that the biased autocorrelation is used (no division by `N - lag`): the bias is what favours the shortest true period.

- [ ] **Step 5: Commit**

```bash
git add src/voxpipe/tuning/features.py tests/test_features.py
git commit -m "feat(tune): log-mel, pitch and long-term spectrum features"
```

---

### Task 3: DTW alignment

**Files:**
- Create: `src/voxpipe/tuning/align.py`
- Test: `tests/test_align.py`

**Interfaces:**
- Produces:
  - `cosine_cost(a: np.ndarray, b: np.ndarray) -> np.ndarray` of shape `(len(a), len(b))`, with values in [0, 2]. Rows are mean-centred before the cosine.
  - `dtw_path(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]`. Monotonic index arrays `(ia, ib)` from `(0, 0)` to `(n-1, m-1)` that minimise the accumulated cost, with steps (1,1), (1,0) and (0,1).

- [ ] **Step 1: Write the failing tests**

`tests/test_align.py`:
```python
import numpy as np

from voxpipe.tuning.align import cosine_cost, dtw_path


def brute_force_dtw_cost(cost):
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
    return acc[n, m]


def path_cost(cost, ia, ib):
    return cost[ia, ib].sum()


def test_identical_sequences_align_diagonally():
    feats = np.random.default_rng(0).normal(size=(30, 8))
    ia, ib = dtw_path(cosine_cost(feats, feats))
    np.testing.assert_array_equal(ia, np.arange(30))
    np.testing.assert_array_equal(ib, np.arange(30))


def test_stretched_sequence_maps_back():
    a = np.random.default_rng(1).normal(size=(20, 8))
    b = np.repeat(a, 2, axis=0)  # b is a slowed down 2x
    ia, ib = dtw_path(cosine_cost(a, b))
    np.testing.assert_array_equal(ia, ib // 2)


def test_path_is_monotonic_and_complete():
    rng = np.random.default_rng(2)
    ia, ib = dtw_path(cosine_cost(rng.normal(size=(25, 6)), rng.normal(size=(40, 6))))
    assert (ia[0], ib[0]) == (0, 0) and (ia[-1], ib[-1]) == (24, 39)
    steps = np.stack([np.diff(ia), np.diff(ib)], axis=1)
    assert {tuple(s) for s in steps} <= {(1, 1), (1, 0), (0, 1)}


def test_matches_brute_force_optimum():
    rng = np.random.default_rng(3)
    cost = rng.random((15, 22))
    ia, ib = dtw_path(cost)
    np.testing.assert_allclose(path_cost(cost, ia, ib), brute_force_dtw_cost(cost))


def test_cosine_cost_range():
    rng = np.random.default_rng(4)
    c = cosine_cost(rng.normal(size=(5, 4)), rng.normal(size=(7, 4)))
    assert c.shape == (5, 7)
    assert c.min() >= -1e-9 and c.max() <= 2 + 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_align.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/voxpipe/tuning/align.py`:
```python
"""Dynamic time warping between reference and target feature sequences."""

from __future__ import annotations

import numpy as np


def cosine_cost(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    def normalise(x: np.ndarray) -> np.ndarray:
        centred = x - x.mean(axis=1, keepdims=True)
        return centred / (np.linalg.norm(centred, axis=1, keepdims=True) + 1e-9)

    return 1.0 - normalise(a) @ normalise(b).T


def dtw_path(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Classic DTW. Each row is computed vectorised: with v[k] the best arrival into
    (i, k) from row i-1 and S the row's cumulative cost, horizontal moves give
    acc[i, j] = S[j] + min_{k<=j}(v[k] - S[k])."""
    n, m = cost.shape
    acc = np.empty((n, m))
    acc[0] = np.cumsum(cost[0])
    for i in range(1, n):
        prev = acc[i - 1]
        diagonal = np.concatenate(([np.inf], prev[:-1]))
        arrival = cost[i] + np.minimum(prev, diagonal)
        running = np.cumsum(cost[i])
        acc[i] = running + np.minimum.accumulate(arrival - running)

    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step = int(np.argmin((acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])))
            i, j = (i - 1, j - 1) if step == 0 else (i - 1, j) if step == 1 else (i, j - 1)
        path.append((i, j))
    path.reverse()
    indices = np.array(path)
    return indices[:, 0], indices[:, 1]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_align.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/voxpipe/tuning/align.py tests/test_align.py
git commit -m "feat(tune): vectorised DTW alignment"
```

---

### Task 4: Objective

**Files:**
- Create: `src/voxpipe/tuning/objective.py`
- Test: `tests/test_objective.py`

**Interfaces:**
- Consumes:
  - `log_mel`, `pitch`, `ltas` (Task 2); `cosine_cost`, `dtw_path` (Task 3); `SamplePair` (Task 1);
  - `decode`, `Chain`, `ChainEntry` (plan 1).
- Produces:
  - `OBJECTIVE_VERSION = 1`, `LENGTH_MISMATCH_TOLERANCE = 0.30`.
  - `PairData(id, reference, path_ref, path_tgt, target_mel, target_f0, target_ltas)`, a frozen dataclass of numpy arrays.
  - `prepare_pair_from_arrays(pair_id: str, reference: np.ndarray, target: np.ndarray) -> PairData`.
  - `prepare_pair(pair: SamplePair) -> tuple[PairData, list[str]]`. Decodes both files and returns the R-12 warning when lengths differ by more than 30 %.
  - `pair_distance(processed: np.ndarray, data: PairData) -> float` (≥ 0; 0 means identical).
  - `chain_distance(entries: Sequence[ChainEntry], pairs: Sequence[PairData]) -> float`. Mean over pairs, with the chain's latency compensated.
  - `similarity(distance: float) -> float`, which is `100 * exp(-distance)`, in (0, 100].

- [ ] **Step 1: Write the failing tests**

`tests/test_objective.py`:
```python
import shutil

import numpy as np
import pytest
from helpers import voice_like

from voxpipe.engine.entry import ChainEntry
from voxpipe.engine.registry import default_entries
from voxpipe.media import write_wav
from voxpipe.tuning.objective import (
    chain_distance,
    pair_distance,
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
    reference = voice_like(1.0)
    target = np.concatenate([np.zeros(9_600, dtype=np.float32), reference])  # 200 ms late
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_objective.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/voxpipe/tuning/objective.py`:
```python
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

OBJECTIVE_VERSION = 1  # bump whenever scoring changes; stored in profile provenance
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
        target_ltas=ltas(tgt_mel),
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

    f0_proc = pitch(processed)[data.path_ref]
    f0_tgt = data.target_f0[data.path_tgt]
    voiced_proc, voiced_tgt = f0_proc > 0, f0_tgt > 0
    both = voiced_proc & voiced_tgt
    octaves = (
        np.mean(np.abs(np.log2(f0_proc[both] / f0_tgt[both]))) if both.any() else 0.0
    )
    d_pitch = octaves + np.mean(voiced_proc != voiced_tgt)

    d_ltas = np.mean(np.abs(ltas(mel) - data.target_ltas)) / DB_UNIT
    return float(W_MEL * d_mel + W_PITCH * d_pitch + W_LTAS * d_ltas)


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
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_objective.py -v`
Expected: 6 passed. If `test_alignment_absorbs_timing_differences` scores below 85, the leading silence frames are probably scored against speech. Check that `cosine_cost` mean-centres rows, so silence matches silence.

- [ ] **Step 5: Commit**

```bash
git add src/voxpipe/tuning/objective.py tests/test_objective.py
git commit -m "feat(tune): DTW-aligned spectral and pitch objective"
```

---

### Task 5: Parameter space

**Files:**
- Create: `src/voxpipe/tuning/space.py`
- Test: `tests/test_space.py`

**Interfaces:**
- Consumes: `EFFECTS`, `ChainEntry`, `Param`, `default_entries` (plan 1).
- Produces: `ParamSpace(template: Sequence[ChainEntry])` with:
  - `.template`;
  - `.size: int`, one dimension per parameter of every entry, in chain then declaration order;
  - `.encode(entries) -> np.ndarray` in [0, 1];
  - `.decode(x) -> list[ChainEntry]`. Clips to [0, 1], maps linear or log per `Param.scale`, rounds to 6 decimals, and clamps to the parameter range.

- [ ] **Step 1: Write the failing tests**

`tests/test_space.py`:
```python
import numpy as np

from voxpipe.engine.entry import ChainEntry
from voxpipe.engine.registry import EFFECTS, default_entries
from voxpipe.tuning.space import ParamSpace


def test_size_counts_every_parameter():
    template = default_entries()
    expected = sum(len(EFFECTS[e.effect].params) for e in template)
    assert ParamSpace(template).size == expected == 27


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_space.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/voxpipe/tuning/space.py`:
```python
"""Normalised search space generated from effect Param declarations (D-08)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from voxpipe.engine.entry import ChainEntry
from voxpipe.engine.params import Param
from voxpipe.engine.registry import EFFECTS


@dataclass(frozen=True)
class _Dimension:
    entry_index: int
    param: Param


def _to_unit(param: Param, value: float) -> float:
    if param.scale == "log":
        lo, hi = math.log(param.minimum), math.log(param.maximum)
        return (math.log(value) - lo) / (hi - lo)
    return (value - param.minimum) / (param.maximum - param.minimum)


def _from_unit(param: Param, unit: float) -> float:
    if param.scale == "log":
        lo, hi = math.log(param.minimum), math.log(param.maximum)
        value = math.exp(lo + unit * (hi - lo))
    else:
        value = param.minimum + unit * (param.maximum - param.minimum)
    return min(param.maximum, max(param.minimum, round(value, 6)))


class ParamSpace:
    def __init__(self, template: Sequence[ChainEntry]) -> None:
        self.template = tuple(template)
        self._dimensions = tuple(
            _Dimension(index, param)
            for index, entry in enumerate(self.template)
            for param in EFFECTS[entry.effect].params
        )

    @property
    def size(self) -> int:
        return len(self._dimensions)

    def encode(self, entries: Sequence[ChainEntry]) -> np.ndarray:
        return np.array(
            [_to_unit(d.param, entries[d.entry_index].params[d.param.name]) for d in self._dimensions]
        )

    def decode(self, x: np.ndarray) -> list[ChainEntry]:
        values: list[dict[str, float]] = [{} for _ in self.template]
        for dimension, unit in zip(self._dimensions, np.clip(x, 0.0, 1.0), strict=True):
            values[dimension.entry_index][dimension.param.name] = _from_unit(
                dimension.param, float(unit)
            )
        return [ChainEntry(e.effect, v) for e, v in zip(self.template, values, strict=True)]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_space.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/voxpipe/tuning/space.py tests/test_space.py
git commit -m "feat(tune): normalised parameter space from effect declarations"
```

---

### Task 6: Deterministic parallel CMA-ES search

**Files:**
- Modify: `pyproject.toml` (add the `cma` dependency)
- Create: `src/voxpipe/tuning/search.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `ParamSpace` (Task 5), `PairData`, `chain_distance`, `similarity` (Task 4), `default_entries`, `ChainEntry` (plan 1).
- Produces:
  - `POPULATION = 24`, `SIGMA0 = 0.25`.
  - `TuneSettings(evaluations: int = 4000, seed: int = 1, workers: int = 0)`, frozen. `workers=0` means all CPUs; `workers=1` evaluates in-process.
  - `TuneResult(entries: list[ChainEntry], score: float, evaluations: int, partial: bool, duration_s: float)`, frozen.
  - `tune(pairs: Sequence[PairData], settings: TuneSettings | None = None, progress: Callable[[int, float], None] | None = None, template: Sequence[ChainEntry] | None = None) -> TuneResult`.
    - The default chain is evaluated first as the baseline. The result is never worse than that.
    - `progress(evaluations_done, best_score)` is called after every generation.
    - `KeyboardInterrupt` (from the terminal or from `progress`) stops the search and returns the best so far with `partial=True`.

- [ ] **Step 1: Add the dependency**

Run: `uv add "cma>=3.3"`
Expected: `pyproject.toml` lists `cma>=3.3` and `uv.lock` updates.

- [ ] **Step 2: Write the failing tests**

`tests/test_search.py`:
```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voxpipe.tuning.search'`

- [ ] **Step 4: Write the implementation**

`src/voxpipe/tuning/search.py`:
```python
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
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_search.py -v`
Expected: 5 passed. If `test_deterministic_regardless_of_workers` fails, check that nothing else draws from numpy's global RNG between `ask()` and `tell()`, and that `pool.map` (ordered) is used, not `imap_unordered`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/voxpipe/tuning/search.py tests/test_search.py
git commit -m "feat(tune): deterministic parallel CMA-ES search"
```

---

### Task 7: `voxpipe tune` command

**Files:**
- Create: `src/voxpipe/commands/tune.py`
- Modify: `src/voxpipe/cli.py` (`COMMANDS`)
- Test: `tests/test_tune_cmd.py`

**Interfaces:**
- Consumes: Tasks 1–6, plus `Profile`, `Provenance`, `save_profile`, `load_profile`, `now_iso`, `VoxpipeError`, `__version__` (plan 1).
- Produces: `voxpipe tune --name NAME [--samples DIR] [--profiles-dir DIR] [--evaluations N] [--seed N] [--workers N]`.
  - Writes `<profiles-dir>/<name>.json`, overwriting any existing file.
  - Prints warnings, progress (at most once per second) and `saved PATH (score S[, partial])` to stderr.
  - Exit 0 when complete, 130 when interrupted (the profile is still saved), 1 on error.

- [ ] **Step 1: Write the failing tests**

`tests/test_tune_cmd.py`:
```python
import shutil

import pytest
from helpers import voice_like

from voxpipe.cli import main
from voxpipe.engine.chain import Chain
from voxpipe.engine.entry import ChainEntry
from voxpipe.engine.registry import TUNING_CHAIN
from voxpipe.media import write_wav
from voxpipe.profile import load_profile
from voxpipe.tuning import search

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture
def samples(tmp_path):
    folder = tmp_path / "samples"
    folder.mkdir()
    reference = voice_like(1.0)
    target = Chain([ChainEntry("gain", {"gain_db": -6.0})]).process_signal(reference)
    write_wav(folder / "01-reference.wav", reference)
    write_wav(folder / "01-target.wav", target)
    write_wav(folder / "02-target.wav", target)  # incomplete pair
    return folder


def run_tune(samples, profiles, *extra):
    return main([
        "tune", "--name", "robot", "--samples", str(samples), "--profiles-dir", str(profiles),
        "--evaluations", "48", "--workers", "1", *extra,
    ])


def test_tune_saves_valid_profile(samples, tmp_path, capsys):
    profiles = tmp_path / "profiles"
    assert run_tune(samples, profiles) == 0
    profile = load_profile(profiles / "robot.json")
    assert profile.name == "robot"
    assert [e.effect for e in profile.chain] == list(TUNING_CHAIN)
    prov = profile.provenance
    assert prov.pairs == ("01",) and prov.seed == 1 and prov.objective_version == 1
    assert prov.partial is False and 0 < prov.score <= 100
    err = capsys.readouterr().err
    assert "skipping pair '02'" in err
    assert "saved" in err and "robot.json" in err


def test_tune_overwrites_existing_profile(samples, tmp_path):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "robot.json").write_text("old")
    assert run_tune(samples, profiles) == 0
    assert load_profile(profiles / "robot.json").name == "robot"


def test_tune_is_deterministic(samples, tmp_path):
    run_tune(samples, tmp_path / "a")
    run_tune(samples, tmp_path / "b")
    a = load_profile(tmp_path / "a" / "robot.json")
    b = load_profile(tmp_path / "b" / "robot.json")
    assert a.chain == b.chain and a.provenance.score == b.provenance.score


def test_interrupt_saves_partial_and_exits_130(samples, tmp_path, monkeypatch):
    real_tune = search.tune

    def interrupting_tune(pairs, settings, progress=None, template=None):
        def stop_after_first(done, score):
            raise KeyboardInterrupt

        return real_tune(pairs, settings, progress=stop_after_first, template=template)

    monkeypatch.setattr("voxpipe.commands.tune.tune", interrupting_tune)
    assert run_tune(samples, tmp_path / "p") == 130
    assert load_profile(tmp_path / "p" / "robot.json").provenance.partial is True


def test_no_pairs_is_an_error(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run_tune(empty, tmp_path) == 1
    assert "no complete sample pairs" in capsys.readouterr().err


def test_name_must_be_a_plain_file_name(samples, tmp_path, capsys):
    code = main(["tune", "--name", "../evil", "--samples", str(samples),
                 "--profiles-dir", str(tmp_path)])
    assert code == 1
    assert "--name" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tune_cmd.py -v`
Expected: FAIL with `invalid choice: 'tune'` (SystemExit 2)

- [ ] **Step 3: Write the implementation**

`src/voxpipe/commands/tune.py`:
```python
"""`voxpipe tune`: fit effect settings to sample pairs and save a profile (R-06)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from voxpipe import __version__
from voxpipe.errors import VoxpipeError
from voxpipe.profile import Profile, Provenance, now_iso, save_profile
from voxpipe.tuning.objective import OBJECTIVE_VERSION, prepare_pair
from voxpipe.tuning.samples import find_pairs
from voxpipe.tuning.search import TuneSettings, tune

EXIT_INTERRUPTED = 130


def register(subparsers) -> None:
    parser = subparsers.add_parser("tune", help="fit effect settings to sample pairs")
    parser.add_argument("--name", required=True, help="profile name (saved as <name>.json)")
    parser.add_argument("--samples", type=Path, default=Path("."),
                        help="folder with <id>-target / <id>-reference files (default: .)")
    parser.add_argument("--profiles-dir", type=Path, default=Path("."),
                        help="where the profile is written (default: current directory)")
    parser.add_argument("--evaluations", type=int, default=TuneSettings.evaluations,
                        help="search budget (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=TuneSettings.seed)
    parser.add_argument("--workers", type=int, default=0, help="processes (default: all CPUs)")
    parser.set_defaults(handler=run)


def _say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class _Progress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.last = 0.0

    def __call__(self, done: int, score: float) -> None:
        now = time.monotonic()
        if now - self.last >= 1.0 or done >= self.total:
            self.last = now
            _say(f"  {min(done, self.total)}/{self.total} evaluations, best score {score:.1f}")


def run(args: argparse.Namespace) -> int:
    if Path(args.name).name != args.name or args.name in ("", ".", ".."):
        raise VoxpipeError("--name must be a plain file name without directories")
    pairs, warnings = find_pairs(args.samples)
    prepared = []
    for pair in pairs:
        data, pair_warnings = prepare_pair(pair)
        warnings.extend(pair_warnings)
        prepared.append(data)
    for warning in warnings:
        _say(f"warning: {warning}")

    _say(f"tuning '{args.name}' on {len(prepared)} pair(s), {args.evaluations} evaluations "
         "(Ctrl+C saves the best so far)")
    settings = TuneSettings(evaluations=args.evaluations, seed=args.seed, workers=args.workers)
    result = tune(prepared, settings, progress=_Progress(args.evaluations))

    profile = Profile(
        name=args.name,
        created=now_iso(),
        provenance=Provenance(
            tool_version=__version__,
            seed=args.seed,
            pairs=tuple(p.id for p in prepared),
            score=round(result.score, 2),
            partial=result.partial,
            duration_s=round(result.duration_s, 1),
            objective_version=OBJECTIVE_VERSION,
        ),
        chain=tuple(result.entries),
    )
    path = args.profiles_dir / f"{args.name}.json"
    save_profile(profile, path)
    _say(f"saved {path} (score {result.score:.1f}{', partial' if result.partial else ''})")
    return EXIT_INTERRUPTED if result.partial else 0
```

In `src/voxpipe/cli.py`:
```python
from voxpipe.commands import devices, render, run, tune

COMMANDS: tuple = (devices, run, tune, render)
```

- [ ] **Step 4: Run the full suite and lint**

Run: `uv run pytest -v && uv run ruff check`
Expected: all pass; `All checks passed!`

- [ ] **Step 5: Check the time budget (N-05) by hand**

Build a realistic sample folder: 10 pairs of about 10 s each, for example your own recordings plus Mechanicus clips. Then run:
```bash
time uv run voxpipe tune --name mech --samples ./samples --profiles-dir ./profiles
uv run voxpipe render ./samples/01-reference.wav --profile mech --profiles-dir ./profiles -o /tmp/mech-01.wav
```
Expected: it finishes in ≤ 10 min (N-05), and the score is reported. Listen to `/tmp/mech-01.wav` next to `01-target`. Note the wall time and score in `design/tech-spec-voxpipe.md` under A-02. If it takes longer than 10 min, raise it with the user before tuning defaults. The `--evaluations` default and `POPULATION` affect determinism and results.

- [ ] **Step 6: Commit**

```bash
git add src/voxpipe/commands/tune.py src/voxpipe/cli.py tests/test_tune_cmd.py
git commit -m "feat: voxpipe tune command"
```

---

## Self-review notes

- **Spec coverage:**
  - R-05, R-09, R-16 (Task 1).
  - R-12 (Task 4).
  - D-06 (Tasks 2–4).
  - D-07, N-06 (Task 6).
  - R-06, R-24, R-25, R-26, partial saving and low-score saving (Task 7).
  - N-05 is checked by hand (Task 7, Step 5).
  - The profile schema guarantees decoded values are valid (Task 5 range test, plus a `load_profile` round trip in Task 7).
- **Open:** domain Q-10b (auto-previews from `tune`) is not implemented; `render` covers it.
