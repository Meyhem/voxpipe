# voxpipe Plan 1: Effect Engine, Profiles and `render`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the streaming effect engine, the strict versioned profile format, and the `voxpipe render` command. Together these turn any audio or video file into an effected WAV, with no PipeWire involved.

**Architecture:**
- **Effects.** Each effect is a stateful object that processes float32 mono 48 kHz blocks of any length. It declares its parameters, and those declarations generate both the profile schema and (in plan 3) the tuning search space.
- **Chain.** A chain runs effects in profile order.
- **Block-size invariance.** Every effect must give the same output however the signal is split into blocks. A shared test helper proves this. It's what lets `render` and `tune` use 100 ms chunks for speed while sounding identical to `run`'s 10 ms blocks (C-07, D-05).
- **Profiles** are JSON, validated in two stages:
  - the envelope (format, version, provenance, chain item shape);
  - each chain item's params, against that effect's generated schema.

  Every error names the failing field.

**Tech Stack:** Python 3.12, uv, numpy, scipy (`signal`, `io.wavfile`), jsonschema, pytest, ruff, ffmpeg (child process).

**Specs:** `design/tech-spec-voxpipe.md` (C-xx, N-xx, D-xx), `design/domain.md` (R-xx). `CLAUDE.md` summarises both.

## Global Constraints

- Internal audio: float32, mono, 48 000 Hz. Live block size: 480 samples (10 ms). Offline chunk: 4 800 samples (D-09).
- Python ≥ 3.12. Run everything with `uv run …`. No packaging beyond `[project.scripts]` (C-03).
- Profile `format` is exactly `"voxpipe-profile"`. `version` is the integer `1`.
- Profile validation is strict: an unknown, missing or out-of-range value is an error naming the field. Profiles from newer versions are refused. Defaults are never silently applied (R-15, C-06).
- Profile writes are atomic: write a temporary file in the same directory, then `os.replace`.
- Child processes are called with argument lists, never `shell=True`.
- All user errors raise a subclass of `voxpipe.errors.VoxpipeError`. The CLI prints `voxpipe: error: <message>` to stderr and exits 1.
- Every path the user can name is a CLI option, and its default is the current directory (R-25, R-26).

---

## File Structure

```
pyproject.toml                    project, deps, ruff, pytest config
.gitignore
src/voxpipe/
  __init__.py                     __version__
  audio.py                        SAMPLE_RATE, BLOCK_SIZE, OFFLINE_CHUNK
  errors.py                       VoxpipeError
  cli.py                          argparse root, main(), entry()
  media.py                        decode() via ffmpeg, write_wav()
  profile.py                      Profile/Provenance dataclasses, parse/load/save, path resolution
  profile_schema.py               envelope_schema(), params_schema()
  engine/
    __init__.py
    params.py                     Param
    effect.py                     Effect base class, mix()
    delay.py                      History, read_delayed() for modulated delay lines
    entry.py                      ChainEntry
    registry.py                   EFFECTS, create_effect(), TUNING_CHAIN, default_entries()
    chain.py                      Chain
    effects/
      __init__.py
      gain.py                     Gain
      filters.py                  Filter (HP+LP), Eq (3 peaking bands)
      modulation.py               RingMod, Chorus
      distortion.py               Distortion (drive + bit-crush)
      resonators.py               Comb, Reverb
      pitch.py                    PitchShift
  commands/
    __init__.py
    render.py                     `voxpipe render`
scripts/
  update_golden.py                regenerates tests/golden/mechanicus.npy
tests/
  helpers.py                      sine(), voice_like(), rms(), dominant_hz(), assert_block_invariant()
  data/mechanicus.json            fixture profile using every effect
  golden/mechanicus.npy           golden render of voice_like() through the fixture
  test_cli.py
  test_gain.py
  test_filters.py
  test_modulation_distortion.py
  test_resonators.py
  test_pitch_chorus.py
  test_chain.py
  test_profile.py
  test_media.py
  test_render.py
  test_perf.py
```

---

### Task 1: Project scaffold and CLI skeleton

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/voxpipe/__init__.py`, `src/voxpipe/audio.py`, `src/voxpipe/errors.py`, `src/voxpipe/cli.py`, `src/voxpipe/commands/__init__.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces:
  - `voxpipe.__version__: str` (`"0.1.0"`)
  - `voxpipe.audio.SAMPLE_RATE = 48_000`, `BLOCK_SIZE = 480`, `OFFLINE_CHUNK = 4_800`
  - `voxpipe.errors.VoxpipeError(Exception)`
  - `voxpipe.cli.COMMANDS: tuple[ModuleType, ...]`. Each module has `register(subparsers) -> None`, which calls `parser.set_defaults(handler=run)` with `run(args: argparse.Namespace) -> int`.
  - `voxpipe.cli.main(argv: Sequence[str] | None = None) -> int`
  - `voxpipe.cli.entry() -> None`

- [ ] **Step 1: Write the project files**

`pyproject.toml`:
```toml
[project]
name = "voxpipe"
version = "0.1.0"
description = "Real-time voice effects into a PipeWire virtual microphone"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2.0",
    "scipy>=1.13",
    "jsonschema>=4.22",
]

[project.scripts]
voxpipe = "voxpipe.cli:entry"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/voxpipe"]

[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.5"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
ignore = ["E501"]  # line length is left to `ruff format`

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore`:
```
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
```

`src/voxpipe/__init__.py`:
```python
"""voxpipe: real-time voice effects into a PipeWire virtual microphone."""

__version__ = "0.1.0"
```

`src/voxpipe/audio.py`:
```python
"""Internal audio format (D-09): float32, mono, 48 kHz."""

SAMPLE_RATE = 48_000
BLOCK_SIZE = 480  # 10 ms, the live block size
OFFLINE_CHUNK = 4_800  # 100 ms, used by render and tune; output is block-size invariant
```

`src/voxpipe/errors.py`:
```python
class VoxpipeError(Exception):
    """A user-facing error: printed as `voxpipe: error: <message>`, exit code 1."""
```

`src/voxpipe/commands/__init__.py`: empty file.

- [ ] **Step 2: Write the failing test**

`tests/test_cli.py`:
```python
import argparse

import pytest

from voxpipe import __version__, cli
from voxpipe.errors import VoxpipeError


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"voxpipe {__version__}"


def test_voxpipe_error_is_printed_and_returns_1(monkeypatch, capsys):
    def failing(args: argparse.Namespace) -> int:
        raise VoxpipeError("boom")

    class FakeCommand:
        @staticmethod
        def register(subparsers) -> None:
            subparsers.add_parser("fail").set_defaults(handler=failing)

    monkeypatch.setattr(cli, "COMMANDS", (FakeCommand,))
    assert cli.main(["fail"]) == 1
    assert capsys.readouterr().err == "voxpipe: error: boom\n"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv sync && uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ImportError: cannot import name 'cli'`

- [ ] **Step 4: Write the implementation**

`src/voxpipe/cli.py`:
```python
"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from voxpipe import __version__
from voxpipe.errors import VoxpipeError

COMMANDS: tuple = ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxpipe",
        description="Real-time voice effects into a PipeWire virtual microphone.",
    )
    parser.add_argument("--version", action="version", version=f"voxpipe {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        command.register(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except VoxpipeError as exc:
        print(f"voxpipe: error: {exc}", file=sys.stderr)
        return 1


def entry() -> None:
    sys.exit(main())
```

- [ ] **Step 5: Run the tests and lint**

Run: `uv run pytest tests/test_cli.py -v && uv run ruff check`
Expected: 2 passed, `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .gitignore src tests
git commit -m "feat: scaffold voxpipe package and CLI skeleton"
```

---

### Task 2: Effect base, parameters, test helpers and `gain`

**Files:**
- Create: `src/voxpipe/engine/__init__.py` (empty), `src/voxpipe/engine/params.py`, `src/voxpipe/engine/effect.py`, `src/voxpipe/engine/effects/__init__.py` (empty), `src/voxpipe/engine/effects/gain.py`
- Test: `tests/helpers.py`, `tests/test_gain.py`

**Interfaces:**
- Produces:
  - `Param(name: str, minimum: float, maximum: float, default: float, scale: Literal["linear", "log"] = "linear")`, a frozen dataclass.
  - `Effect(values: Mapping[str, float] | None = None, sample_rate: int = SAMPLE_RATE)`. Abstract base with:
    - class attributes `name: ClassVar[str]` and `params: ClassVar[tuple[Param, ...]]`;
    - `values: dict[str, float]`;
    - `reset() -> None`;
    - `process(block: np.ndarray) -> np.ndarray` (float32, same length);
    - property `latency_samples -> int` (default 0);
    - classmethod `defaults() -> dict[str, float]`.
    - Unknown parameter names raise `ValueError`. Missing names take defaults. This lets the tuner and tests build effects directly; profiles are range-checked before they reach here.
  - `mix(dry: np.ndarray, wet: np.ndarray, amount: float) -> np.ndarray` (float32).
  - `Gain`: name `"gain"`, params `gain_db ∈ [-24, 24]`, default 0.
  - Test helpers: `sine(freq_hz, seconds=1.0, amplitude=0.5)`, `voice_like(seconds=2.0, seed=0)`, `rms(x)`, `dominant_hz(x)`, `run_in_chunks(effect, signal, sizes)`, `assert_block_invariant(factory, signal)`.

- [ ] **Step 1: Write the test helpers**

`tests/helpers.py`:
```python
"""Shared test utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import cycle

import numpy as np

from voxpipe.audio import SAMPLE_RATE


def sine(freq_hz: float, seconds: float = 1.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def voice_like(seconds: float = 2.0, seed: int = 0) -> np.ndarray:
    """Deterministic harmonic signal with a gliding pitch plus a little noise."""
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    f0 = 120.0 + 60.0 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
    voice = sum(0.3 / k * np.sin(k * phase) for k in range(1, 9))
    noise = np.random.default_rng(seed).normal(0.0, 0.02, t.size)
    return (voice + noise).astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def dominant_hz(x: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    return float(np.argmax(spectrum) * SAMPLE_RATE / x.size)


def run_in_chunks(effect, signal: np.ndarray, sizes: Sequence[int]) -> np.ndarray:
    effect.reset()
    out, start = [], 0
    for size in cycle(sizes):
        if start >= signal.size:
            break
        out.append(effect.process(signal[start : start + size]))
        start += size
    return np.concatenate(out)


def assert_block_invariant(factory: Callable[[], object], signal: np.ndarray) -> None:
    """Output must not depend on how the signal is split into blocks (D-05)."""
    whole = run_in_chunks(factory(), signal, [signal.size])
    for sizes in ([480], [4800], [1, 7, 480, 333, 4800]):
        chunked = run_in_chunks(factory(), signal, sizes)
        assert chunked.dtype == np.float32
        np.testing.assert_allclose(chunked, whole, atol=1e-5, err_msg=f"chunk sizes {sizes}")
```

- [ ] **Step 2: Write the failing tests**

`tests/test_gain.py`:
```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_gain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voxpipe.engine'`

- [ ] **Step 4: Write the implementation**

`src/voxpipe/engine/params.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Param:
    """One tunable effect parameter. Drives the profile schema and the tuning space."""

    name: str
    minimum: float
    maximum: float
    default: float
    scale: Literal["linear", "log"] = "linear"

    def __post_init__(self) -> None:
        if not self.minimum < self.maximum:
            raise ValueError(f"{self.name}: minimum must be below maximum")
        if not self.minimum <= self.default <= self.maximum:
            raise ValueError(f"{self.name}: default {self.default} outside range")
        if self.scale == "log" and self.minimum <= 0:
            raise ValueError(f"{self.name}: log scale needs a positive minimum")
```

`src/voxpipe/engine/effect.py`:
```python
"""Base class for streaming effects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar

import numpy as np

from voxpipe.audio import SAMPLE_RATE
from voxpipe.engine.params import Param


class Effect(ABC):
    """A stateful effect: process() accepts blocks of any length, and its output
    must not depend on how the signal is split into blocks."""

    name: ClassVar[str]
    params: ClassVar[tuple[Param, ...]]

    def __init__(
        self, values: Mapping[str, float] | None = None, sample_rate: int = SAMPLE_RATE
    ) -> None:
        values = dict(values or {})
        unknown = sorted(set(values) - {p.name for p in self.params})
        if unknown:
            raise ValueError(f"{self.name}: unknown parameter(s): {', '.join(unknown)}")
        self.sample_rate = sample_rate
        self.values = {p.name: float(values.get(p.name, p.default)) for p in self.params}
        self.reset()

    @classmethod
    def defaults(cls) -> dict[str, float]:
        return {p.name: p.default for p in cls.params}

    @property
    def latency_samples(self) -> int:
        return 0

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal state and derive coefficients from self.values."""

    @abstractmethod
    def process(self, block: np.ndarray) -> np.ndarray:
        """Process one block; returns float32 of the same length."""


def mix(dry: np.ndarray, wet: np.ndarray, amount: float) -> np.ndarray:
    return (dry + amount * (wet - dry)).astype(np.float32, copy=False)
```

`src/voxpipe/engine/effects/gain.py`:
```python
import numpy as np

from voxpipe.engine.effect import Effect
from voxpipe.engine.params import Param


class Gain(Effect):
    name = "gain"
    params = (Param("gain_db", -24.0, 24.0, 0.0),)

    def reset(self) -> None:
        self._factor = np.float32(10 ** (self.values["gain_db"] / 20))

    def process(self, block: np.ndarray) -> np.ndarray:
        return (np.asarray(block, dtype=np.float32) * self._factor).astype(np.float32)
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_gain.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add src/voxpipe/engine tests/helpers.py tests/test_gain.py
git commit -m "feat(engine): effect base class, params and gain"
```

---

### Task 3: `filter` and `eq` effects

**Files:**
- Create: `src/voxpipe/engine/effects/filters.py`
- Test: `tests/test_filters.py`

**Interfaces:**
- Consumes: `Effect`, `Param` (Task 2).
- Produces:
  - `Filter`: name `"filter"`. Params `highpass_hz ∈ [20, 2000]` (default 20, log) and `lowpass_hz ∈ [1000, 20000]` (default 20000, log). 2nd-order Butterworth for each.
  - `Eq`: name `"eq"`. Params:
    - `low_hz ∈ [80, 800]` (default 250, log), `low_gain_db ∈ [-18, 18]` (0);
    - `mid_hz ∈ [500, 4000]` (1500, log), `mid_gain_db ∈ [-18, 18]` (0);
    - `high_hz ∈ [2000, 12000]` (5000, log), `high_gain_db ∈ [-18, 18]` (0);
    - `q ∈ [0.3, 5]` (1, log).

    Three RBJ peaking biquads.

- [ ] **Step 1: Write the failing tests**

`tests/test_filters.py`:
```python
from helpers import assert_block_invariant, rms, sine, voice_like

from voxpipe.engine.effects.filters import Eq, Filter

SETTLE = 4_800  # skip the first 100 ms of filter transient


def test_highpass_removes_low_tone():
    x = sine(100.0)
    y = Filter({"highpass_hz": 1000.0}).process(x)
    assert rms(y[SETTLE:]) < 0.05 * rms(x[SETTLE:])


def test_lowpass_removes_high_tone():
    x = sine(8000.0)
    y = Filter({"lowpass_hz": 2000.0}).process(x)
    assert rms(y[SETTLE:]) < 0.1 * rms(x[SETTLE:])


def test_filter_passes_midband():
    x = sine(1000.0)
    y = Filter({"highpass_hz": 100.0, "lowpass_hz": 10000.0}).process(x)
    assert 0.9 < rms(y[SETTLE:]) / rms(x[SETTLE:]) < 1.1


def test_eq_mid_boost_12db_at_centre():
    x = sine(1500.0)
    y = Eq({"mid_hz": 1500.0, "mid_gain_db": 12.0}).process(x)
    assert 3.7 < rms(y[SETTLE:]) / rms(x[SETTLE:]) < 4.3


def test_eq_flat_by_default():
    x = voice_like(0.5)
    y = Eq().process(x)
    assert rms(y - x) < 1e-5


def test_filters_are_block_invariant():
    signal = voice_like(0.5)
    assert_block_invariant(lambda: Filter({"highpass_hz": 300.0, "lowpass_hz": 3000.0}), signal)
    assert_block_invariant(
        lambda: Eq({"low_gain_db": -6.0, "mid_gain_db": 9.0, "high_gain_db": 3.0, "q": 2.0}),
        signal,
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_filters.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voxpipe.engine.effects.filters'`

- [ ] **Step 3: Write the implementation**

`src/voxpipe/engine/effects/filters.py`:
```python
import numpy as np
from scipy.signal import butter, sosfilt

from voxpipe.engine.effect import Effect
from voxpipe.engine.params import Param


class _SosEffect(Effect):
    """Shared streaming second-order-sections filter; subclasses build self._sos."""

    def _build_sos(self) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        self._sos = self._build_sos()
        self._zi = np.zeros((self._sos.shape[0], 2))

    def process(self, block: np.ndarray) -> np.ndarray:
        out, self._zi = sosfilt(self._sos, np.asarray(block, dtype=np.float64), zi=self._zi)
        return out.astype(np.float32)


class Filter(_SosEffect):
    name = "filter"
    params = (
        Param("highpass_hz", 20.0, 2000.0, 20.0, "log"),
        Param("lowpass_hz", 1000.0, 20000.0, 20000.0, "log"),
    )

    def _build_sos(self) -> np.ndarray:
        fs = self.sample_rate
        lowpass = min(self.values["lowpass_hz"], 0.45 * fs)
        high = butter(2, self.values["highpass_hz"], "highpass", fs=fs, output="sos")
        low = butter(2, lowpass, "lowpass", fs=fs, output="sos")
        return np.vstack([high, low])


def peaking_sos(freq_hz: float, gain_db: float, q: float, fs: int) -> np.ndarray:
    """RBJ audio-EQ-cookbook peaking filter as one second-order section."""
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq_hz / fs
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    b = np.array([1 + alpha * a, -2 * cos_w0, 1 - alpha * a])
    den = np.array([1 + alpha / a, -2 * cos_w0, 1 - alpha / a])
    return np.concatenate([b / den[0], den / den[0]])


class Eq(_SosEffect):
    name = "eq"
    params = (
        Param("low_hz", 80.0, 800.0, 250.0, "log"),
        Param("low_gain_db", -18.0, 18.0, 0.0),
        Param("mid_hz", 500.0, 4000.0, 1500.0, "log"),
        Param("mid_gain_db", -18.0, 18.0, 0.0),
        Param("high_hz", 2000.0, 12000.0, 5000.0, "log"),
        Param("high_gain_db", -18.0, 18.0, 0.0),
        Param("q", 0.3, 5.0, 1.0, "log"),
    )

    def _build_sos(self) -> np.ndarray:
        v, fs = self.values, self.sample_rate
        return np.vstack(
            [
                peaking_sos(v[f"{band}_hz"], v[f"{band}_gain_db"], v["q"], fs)
                for band in ("low", "mid", "high")
            ]
        )
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_filters.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/voxpipe/engine/effects/filters.py tests/test_filters.py
git commit -m "feat(engine): filter and eq effects"
```

---

### Task 4: `ring_mod` and `distortion` effects

**Files:**
- Create: `src/voxpipe/engine/effects/modulation.py` (RingMod only; Chorus is added in Task 6), `src/voxpipe/engine/effects/distortion.py`
- Test: `tests/test_modulation_distortion.py`

**Interfaces:**
- Consumes: `Effect`, `Param`, `mix` (Task 2).
- Produces:
  - `RingMod`: name `"ring_mod"`. Params `freq_hz ∈ [20, 2000]` (default 100, log), `mix ∈ [0, 1]` (default 0).
  - `Distortion`: name `"distortion"`. Params `drive_db ∈ [0, 40]` (0), `bits ∈ [4, 16]` (16), `mix ∈ [0, 1]` (0).

- [ ] **Step 1: Write the failing tests**

`tests/test_modulation_distortion.py`:
```python
import numpy as np
from helpers import assert_block_invariant, voice_like

from voxpipe.audio import SAMPLE_RATE
from voxpipe.engine.effects.distortion import Distortion
from voxpipe.engine.effects.modulation import RingMod


def test_ring_mod_turns_dc_into_carrier():
    x = np.full(SAMPLE_RATE // 10, 0.5, dtype=np.float32)
    y = RingMod({"freq_hz": 100.0, "mix": 1.0}).process(x)
    quarter_period = SAMPLE_RATE // 400  # 100 Hz carrier, quarter period = 120 samples
    assert abs(y[quarter_period] - 0.5) < 1e-4
    assert abs(y[2 * quarter_period]) < 1e-4


def test_ring_mod_mix_zero_is_identity():
    x = voice_like(0.2)
    np.testing.assert_array_equal(RingMod({"mix": 0.0}).process(x), x)


def test_bitcrush_quantises_to_levels():
    x = voice_like(0.2)
    y = Distortion({"bits": 4.0, "mix": 1.0}).process(x)
    levels = y * 8  # 4 bits -> steps of 1/8
    np.testing.assert_allclose(levels, np.round(levels), atol=1e-5)


def test_drive_saturates():
    x = np.array([0.1, 0.9], dtype=np.float32)
    y = Distortion({"drive_db": 30.0, "mix": 1.0}).process(x)
    assert y[1] < 1.0 + 1e-6
    assert y[0] / x[0] > y[1] / x[1]  # small signals amplified more than large ones


def test_effects_are_block_invariant():
    signal = voice_like(0.5)
    assert_block_invariant(lambda: RingMod({"freq_hz": 73.0, "mix": 0.7}), signal)
    assert_block_invariant(
        lambda: Distortion({"drive_db": 18.0, "bits": 8.0, "mix": 0.6}), signal
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_modulation_distortion.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/voxpipe/engine/effects/modulation.py`:
```python
import numpy as np

from voxpipe.engine.effect import Effect, mix
from voxpipe.engine.params import Param

TWO_PI = 2 * np.pi


class RingMod(Effect):
    name = "ring_mod"
    params = (
        Param("freq_hz", 20.0, 2000.0, 100.0, "log"),
        Param("mix", 0.0, 1.0, 0.0),
    )

    def reset(self) -> None:
        self._phase = 0.0
        self._increment = TWO_PI * self.values["freq_hz"] / self.sample_rate

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32)
        phases = self._phase + self._increment * np.arange(block.size)
        self._phase = (self._phase + self._increment * block.size) % TWO_PI
        wet = block * np.sin(phases)
        return mix(block, wet, self.values["mix"])
```

`src/voxpipe/engine/effects/distortion.py`:
```python
import numpy as np

from voxpipe.engine.effect import Effect, mix
from voxpipe.engine.params import Param


class Distortion(Effect):
    """Normalised tanh saturation followed by bit-crush quantisation. Stateless."""

    name = "distortion"
    params = (
        Param("drive_db", 0.0, 40.0, 0.0),
        Param("bits", 4.0, 16.0, 16.0),
        Param("mix", 0.0, 1.0, 0.0),
    )

    def reset(self) -> None:
        self._drive = 10 ** (self.values["drive_db"] / 20)
        self._norm = 1.0 / np.tanh(self._drive)
        self._steps = 2 ** (self.values["bits"] - 1)

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32)
        wet = np.tanh(self._drive * block.astype(np.float64)) * self._norm
        wet = np.round(wet * self._steps) / self._steps
        return mix(block, wet, self.values["mix"])
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_modulation_distortion.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/voxpipe/engine/effects/modulation.py src/voxpipe/engine/effects/distortion.py tests/test_modulation_distortion.py
git commit -m "feat(engine): ring modulator and distortion effects"
```

---

### Task 5: `comb` and `reverb` effects

**Files:**
- Create: `src/voxpipe/engine/effects/resonators.py`
- Test: `tests/test_resonators.py`

**Interfaces:**
- Consumes: `Effect`, `Param`, `mix` (Task 2).
- Produces:
  - `Comb`: name `"comb"`. Params `delay_ms ∈ [0.5, 20]` (default 5, log), `feedback ∈ [-0.95, 0.95]` (0.5), `mix ∈ [0, 1]` (0). The wet signal is scaled by `1 - |feedback|`.
  - `Reverb`: name `"reverb"`. Params `room_size ∈ [0, 1]` (0.5), `mix ∈ [0, 1]` (0). A Schroeder design: 4 parallel feedback combs (29.7, 37.1, 41.1, 43.7 ms) and 2 series allpasses (5.0, 1.7 ms, g = 0.7).

- [ ] **Step 1: Write the failing tests**

`tests/test_resonators.py`:
```python
import numpy as np
from helpers import assert_block_invariant, rms, voice_like

from voxpipe.audio import SAMPLE_RATE
from voxpipe.engine.effects.resonators import Comb, Reverb


def impulse(seconds: float) -> np.ndarray:
    x = np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)
    x[0] = 1.0
    return x


def test_comb_impulse_response_echoes_at_delay():
    y = Comb({"delay_ms": 5.0, "feedback": 0.5, "mix": 1.0}).process(impulse(0.1))
    d = 240  # 5 ms at 48 kHz
    np.testing.assert_allclose([y[0], y[d], y[2 * d]], [0.5, 0.25, 0.125], atol=1e-6)
    assert abs(y[d // 2]) < 1e-9


def test_reverb_tail_decays():
    y = Reverb({"room_size": 0.5, "mix": 1.0}).process(impulse(1.0))
    early = rms(y[4_800:9_600])
    late = rms(y[38_400:43_200])
    assert early > late > 0.0


def test_mix_zero_is_identity():
    x = voice_like(0.2)
    np.testing.assert_array_equal(Comb({"mix": 0.0}).process(x), x)
    np.testing.assert_array_equal(Reverb({"mix": 0.0}).process(x), x)


def test_resonators_are_block_invariant():
    signal = voice_like(0.5)
    assert_block_invariant(lambda: Comb({"delay_ms": 3.3, "feedback": -0.7, "mix": 0.5}), signal)
    assert_block_invariant(lambda: Reverb({"room_size": 0.8, "mix": 0.4}), signal)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_resonators.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/voxpipe/engine/effects/resonators.py`:
```python
import numpy as np
from scipy.signal import lfilter

from voxpipe.engine.effect import Effect, mix
from voxpipe.engine.params import Param


class _StreamingFilter:
    """lfilter with carried state, so blocks of any length chain seamlessly."""

    def __init__(self, b: np.ndarray, a: np.ndarray) -> None:
        self.b, self.a = b, a
        self.zi = np.zeros(max(len(a), len(b)) - 1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = lfilter(self.b, self.a, x, zi=self.zi)
        return y


def feedback_comb(delay: int, feedback: float) -> _StreamingFilter:
    """y[n] = x[n] + feedback * y[n - delay]"""
    a = np.zeros(delay + 1)
    a[0], a[delay] = 1.0, -feedback
    return _StreamingFilter(np.array([1.0]), a)


def allpass(delay: int, gain: float) -> _StreamingFilter:
    """Schroeder allpass: y[n] = -g x[n] + x[n - d] + g y[n - d]"""
    b = np.zeros(delay + 1)
    b[0], b[delay] = -gain, 1.0
    a = np.zeros(delay + 1)
    a[0], a[delay] = 1.0, -gain
    return _StreamingFilter(b, a)


def ms_to_samples(ms: float, fs: int) -> int:
    return max(1, round(ms * fs / 1000))


class Comb(Effect):
    name = "comb"
    params = (
        Param("delay_ms", 0.5, 20.0, 5.0, "log"),
        Param("feedback", -0.95, 0.95, 0.5),
        Param("mix", 0.0, 1.0, 0.0),
    )

    def reset(self) -> None:
        fb = self.values["feedback"]
        self._filter = feedback_comb(ms_to_samples(self.values["delay_ms"], self.sample_rate), fb)
        self._scale = 1.0 - abs(fb)

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32)
        wet = self._filter(block.astype(np.float64)) * self._scale
        return mix(block, wet, self.values["mix"])


class Reverb(Effect):
    name = "reverb"
    params = (
        Param("room_size", 0.0, 1.0, 0.5),
        Param("mix", 0.0, 1.0, 0.0),
    )
    COMB_MS = (29.7, 37.1, 41.1, 43.7)
    ALLPASS_MS = (5.0, 1.7)
    ALLPASS_GAIN = 0.7

    def reset(self) -> None:
        fs = self.sample_rate
        feedback = 0.7 + 0.28 * self.values["room_size"]
        self._combs = [feedback_comb(ms_to_samples(ms, fs), feedback) for ms in self.COMB_MS]
        self._allpasses = [allpass(ms_to_samples(ms, fs), self.ALLPASS_GAIN) for ms in self.ALLPASS_MS]
        self._scale = (1.0 - feedback) / len(self.COMB_MS) * 4.0

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32)
        x = block.astype(np.float64)
        wet = sum(comb(x) for comb in self._combs) * self._scale
        for stage in self._allpasses:
            wet = stage(wet)
        return mix(block, wet, self.values["mix"])
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_resonators.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/voxpipe/engine/effects/resonators.py tests/test_resonators.py
git commit -m "feat(engine): comb and reverb effects"
```

---

### Task 6: Modulated delay lines: `chorus` and `pitch_shift`

**Files:**
- Create: `src/voxpipe/engine/delay.py`, `src/voxpipe/engine/effects/pitch.py`
- Modify: `src/voxpipe/engine/effects/modulation.py` (append `Chorus`)
- Test: `tests/test_pitch_chorus.py`

**Interfaces:**
- Consumes: `Effect`, `Param`, `mix` (Task 2).
- Produces:
  - `History(size: int)` with `.extend(block) -> np.ndarray`. It returns `[history..., block...]` as float64 and keeps the last `size` samples.
  - `read_delayed(full: np.ndarray, history_size: int, delays: np.ndarray) -> np.ndarray`. Linear-interpolated read at fractional delays (in samples, each in `[1, history_size]`) behind each new sample.
  - `Chorus`: name `"chorus"`. Params `delay_ms ∈ [5, 40]` (15), `depth_ms ∈ [0, 10]` (2), `rate_hz ∈ [0.1, 5]` (0.8, log), `mix ∈ [0, 1]` (0).
  - `PitchShift`: name `"pitch_shift"`. Params `semitones ∈ [-12, 12]` (0), `mix ∈ [0, 1]` (0). A two-tap rotating-delay shifter with a fixed 30 ms window. `latency_samples` is 720 when `mix > 0`, otherwise 0 (risk K-02).

  **Note:** the tech spec lists a "formant shift" effect. A real-time formant shifter doesn't fit the latency budget. The three-band EQ (Task 3) covers tone colour, so formant shifting is deferred. Record this in the commit message.

- [ ] **Step 1: Write the failing tests**

`tests/test_pitch_chorus.py`:
```python
import numpy as np
from helpers import assert_block_invariant, dominant_hz, sine, voice_like

from voxpipe.audio import SAMPLE_RATE
from voxpipe.engine.delay import History, read_delayed
from voxpipe.engine.effects.modulation import Chorus
from voxpipe.engine.effects.pitch import PitchShift


def test_read_delayed_integer_delay_is_exact():
    history = History(10)
    first = history.extend(np.arange(1, 6, dtype=np.float32))
    np.testing.assert_array_equal(read_delayed(first, 10, np.full(5, 2.0)), [0, 0, 1, 2, 3])
    second = history.extend(np.arange(6, 9, dtype=np.float32))
    np.testing.assert_array_equal(read_delayed(second, 10, np.full(3, 2.0)), [4, 5, 6])


def test_read_delayed_interpolates():
    full = History(4).extend(np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32))
    out = read_delayed(full, 4, np.array([1.5, 1.5, 1.5, 1.5]))
    np.testing.assert_allclose(out[2:], [0.5, 1.5])


def test_chorus_without_depth_is_pure_delay():
    x = np.zeros(SAMPLE_RATE // 10, dtype=np.float32)
    x[0] = 1.0
    y = Chorus({"delay_ms": 10.0, "depth_ms": 0.0, "mix": 1.0}).process(x)
    assert np.argmax(y) == 480
    assert abs(y[480] - 1.0) < 1e-6


def test_pitch_shift_octave_up_doubles_frequency():
    y = PitchShift({"semitones": 12.0, "mix": 1.0}).process(sine(300.0))
    assert abs(dominant_hz(y[4_800:]) - 600.0) < 15.0


def test_pitch_shift_down_lowers_frequency():
    y = PitchShift({"semitones": -12.0, "mix": 1.0}).process(sine(600.0))
    assert abs(dominant_hz(y[4_800:]) - 300.0) < 15.0


def test_pitch_shift_latency_only_when_active():
    assert PitchShift({"mix": 0.0}).latency_samples == 0
    assert PitchShift({"semitones": 3.0, "mix": 0.5}).latency_samples == 720


def test_delay_effects_are_block_invariant():
    signal = voice_like(0.5)
    assert_block_invariant(
        lambda: Chorus({"delay_ms": 12.0, "depth_ms": 4.0, "rate_hz": 1.3, "mix": 0.5}), signal
    )
    assert_block_invariant(lambda: PitchShift({"semitones": -5.0, "mix": 0.8}), signal)
    assert_block_invariant(lambda: PitchShift({"semitones": 7.0, "mix": 1.0}), signal)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pitch_chorus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voxpipe.engine.delay'`

- [ ] **Step 3: Write the implementation**

`src/voxpipe/engine/delay.py`:
```python
"""Streaming delay-line helpers for modulated-delay effects."""

import numpy as np


class History:
    """Keeps the last `size` input samples so each block can read into the past."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.buffer = np.zeros(size, dtype=np.float64)

    def extend(self, block: np.ndarray) -> np.ndarray:
        full = np.concatenate([self.buffer, np.asarray(block, dtype=np.float64)])
        self.buffer = full[-self.size :].copy()
        return full


def read_delayed(full: np.ndarray, history_size: int, delays: np.ndarray) -> np.ndarray:
    """For new sample k (at index history_size + k of `full`), read `delays[k]` samples
    into the past with linear interpolation. Every delay must lie in [1, history_size]."""
    positions = history_size + np.arange(delays.size) - delays
    base = np.floor(positions).astype(np.int64)
    frac = positions - base
    return full[base] * (1.0 - frac) + full[base + 1] * frac
```

In `src/voxpipe/engine/effects/modulation.py`, add `from voxpipe.engine.delay import History, read_delayed` to the top import block, then append:
```python
class Chorus(Effect):
    """Single modulated-delay voice mixed with the dry signal (doubling)."""

    name = "chorus"
    params = (
        Param("delay_ms", 5.0, 40.0, 15.0),
        Param("depth_ms", 0.0, 10.0, 2.0),
        Param("rate_hz", 0.1, 5.0, 0.8, "log"),
        Param("mix", 0.0, 1.0, 0.0),
    )

    def reset(self) -> None:
        ms = self.sample_rate / 1000
        self._delay = self.values["delay_ms"] * ms
        self._depth = self.values["depth_ms"] * ms
        self._history = History(int(np.ceil(self._delay + self._depth)) + 2)
        self._phase = 0.0
        self._increment = TWO_PI * self.values["rate_hz"] / self.sample_rate

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32)
        phases = self._phase + self._increment * np.arange(block.size)
        self._phase = (self._phase + self._increment * block.size) % TWO_PI
        delays = np.clip(self._delay + self._depth * np.sin(phases), 1.0, self._history.size)
        full = self._history.extend(block)
        wet = read_delayed(full, self._history.size, delays)
        return mix(block, wet, self.values["mix"])
```

`src/voxpipe/engine/effects/pitch.py`:
```python
import numpy as np

from voxpipe.engine.delay import History, read_delayed
from voxpipe.engine.effect import Effect, mix
from voxpipe.engine.params import Param

WINDOW_MS = 30.0


class PitchShift(Effect):
    """Two-tap rotating delay-line pitch shifter. Each tap's delay sweeps across a
    30 ms window at the rate that gives the requested pitch ratio. The taps are half a
    window apart with sin^2/cos^2 gains, so each tap is silent at the moment it jumps."""

    name = "pitch_shift"
    params = (
        Param("semitones", -12.0, 12.0, 0.0),
        Param("mix", 0.0, 1.0, 0.0),
    )

    def reset(self) -> None:
        self._window = WINDOW_MS * self.sample_rate / 1000
        self._history = History(int(self._window) + 3)
        ratio = 2 ** (self.values["semitones"] / 12)
        self._increment = (1.0 - ratio) / self._window
        self._phase = 0.0

    @property
    def latency_samples(self) -> int:
        return int(self._window / 2) if self.values["mix"] > 0 else 0

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32)
        phase_a = (self._phase + self._increment * np.arange(block.size)) % 1.0
        phase_b = (phase_a + 0.5) % 1.0
        self._phase = (self._phase + self._increment * block.size) % 1.0
        full = self._history.extend(block)
        size = self._history.size
        tap_a = read_delayed(full, size, 1.0 + phase_a * self._window)
        tap_b = read_delayed(full, size, 1.0 + phase_b * self._window)
        wet = np.sin(np.pi * phase_a) ** 2 * tap_a + np.sin(np.pi * phase_b) ** 2 * tap_b
        return mix(block, wet, self.values["mix"])
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_pitch_chorus.py -v && uv run ruff check`
Expected: 7 passed, `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/voxpipe/engine tests/test_pitch_chorus.py
git commit -m "feat(engine): chorus and pitch-shift via modulated delay lines

Formant shift (tech spec section 4) is deferred: no real-time formant shifter
fits the latency budget; the 3-band EQ covers tone colour."
```

---

### Task 7: Registry and `Chain`

**Files:**
- Create: `src/voxpipe/engine/entry.py`, `src/voxpipe/engine/registry.py`, `src/voxpipe/engine/chain.py`
- Test: `tests/test_chain.py`

**Interfaces:**
- Consumes: all effect classes (Tasks 2–6).
- Produces:
  - `ChainEntry(effect: str, params: Mapping[str, float])`, a frozen dataclass.
  - `EFFECTS: dict[str, type[Effect]]`, keyed by effect name.
  - `create_effect(name: str, params: Mapping[str, float], sample_rate: int = SAMPLE_RATE) -> Effect`. Raises `ValueError` for an unknown name.
  - `TUNING_CHAIN: tuple[str, ...]` = `("gain", "pitch_shift", "filter", "eq", "ring_mod", "comb", "distortion", "chorus", "reverb", "gain")`.
  - `default_entries() -> list[ChainEntry]` (the TUNING_CHAIN with default params).
  - `Chain(entries: Sequence[ChainEntry], sample_rate: int = SAMPLE_RATE)` with:
    - `.entries: tuple[ChainEntry, ...]`
    - `.reset()`
    - `.process_block(block) -> np.ndarray`
    - `.process_signal(signal, chunk: int = OFFLINE_CHUNK) -> np.ndarray` (resets first)
    - `.latency_samples -> int`

- [ ] **Step 1: Write the failing tests**

`tests/test_chain.py`:
```python
import numpy as np
import pytest
from helpers import voice_like

from voxpipe.engine.chain import Chain
from voxpipe.engine.entry import ChainEntry
from voxpipe.engine.registry import EFFECTS, TUNING_CHAIN, create_effect, default_entries


def test_registry_contains_every_effect():
    assert set(EFFECTS) == {
        "gain", "pitch_shift", "filter", "eq", "ring_mod",
        "comb", "distortion", "chorus", "reverb",
    }
    assert all(name in EFFECTS for name in TUNING_CHAIN)


def test_create_effect_unknown_name():
    with pytest.raises(ValueError, match="unknown effect"):
        create_effect("flanger", {})


def test_default_chain_is_near_identity():
    x = voice_like(0.5)
    y = Chain(default_entries()).process_signal(x)
    settle = 4_800
    assert np.max(np.abs(y[settle:] - x[settle:])) < 0.05


def test_chain_applies_effects_in_order():
    entries = [ChainEntry("gain", {"gain_db": 6.0206}), ChainEntry("distortion", {"bits": 4.0, "mix": 1.0})]
    x = voice_like(0.2) * 0.4
    y = Chain(entries).process_signal(x)
    levels = y * 8
    np.testing.assert_allclose(levels, np.round(levels), atol=1e-5)  # crush ran last


def test_live_blocks_match_offline_signal():
    entries = [
        ChainEntry("pitch_shift", {"semitones": -3.0, "mix": 0.7}),
        ChainEntry("reverb", {"room_size": 0.4, "mix": 0.3}),
    ]
    x = voice_like(1.0)
    offline = Chain(entries).process_signal(x)
    live_chain = Chain(entries)
    live = np.concatenate([live_chain.process_block(x[i : i + 480]) for i in range(0, x.size, 480)])
    np.testing.assert_allclose(live, offline, atol=1e-5)


def test_latency_sums_effects():
    assert Chain(default_entries()).latency_samples == 0
    assert Chain([ChainEntry("pitch_shift", {"semitones": 2.0, "mix": 1.0})]).latency_samples == 720
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chain.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/voxpipe/engine/entry.py`:
```python
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ChainEntry:
    """One effect in a profile's chain: its registry name and parameter values."""

    effect: str
    params: Mapping[str, float]
```

`src/voxpipe/engine/registry.py`:
```python
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
```

`src/voxpipe/engine/chain.py`:
```python
from collections.abc import Sequence

import numpy as np

from voxpipe.audio import OFFLINE_CHUNK, SAMPLE_RATE
from voxpipe.engine.entry import ChainEntry
from voxpipe.engine.registry import create_effect


class Chain:
    """Runs effects in profile order. Shared by run, render and tune (D-05)."""

    def __init__(self, entries: Sequence[ChainEntry], sample_rate: int = SAMPLE_RATE) -> None:
        self.entries = tuple(entries)
        self.effects = [create_effect(e.effect, e.params, sample_rate) for e in self.entries]

    @property
    def latency_samples(self) -> int:
        return sum(effect.latency_samples for effect in self.effects)

    def reset(self) -> None:
        for effect in self.effects:
            effect.reset()

    def process_block(self, block: np.ndarray) -> np.ndarray:
        x = np.asarray(block, dtype=np.float32)
        for effect in self.effects:
            x = effect.process(x)
        return x

    def process_signal(self, signal: np.ndarray, chunk: int = OFFLINE_CHUNK) -> np.ndarray:
        """Process a whole signal from a fresh state. Identical to feeding live blocks."""
        self.reset()
        out = np.empty(len(signal), dtype=np.float32)
        for start in range(0, len(signal), chunk):
            out[start : start + chunk] = self.process_block(signal[start : start + chunk])
        return out
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_chain.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/voxpipe/engine tests/test_chain.py
git commit -m "feat(engine): effect registry and chain"
```

---

### Task 8: Profile schema and strict parsing

**Files:**
- Create: `src/voxpipe/profile_schema.py`, `src/voxpipe/profile.py`, `tests/data/mechanicus.json`
- Test: `tests/test_profile.py`

**Interfaces:**
- Consumes: `EFFECTS` (Task 7), `ChainEntry` (Task 7), `Param` (Task 2), `VoxpipeError` (Task 1).
- Produces:
  - `profile_schema.FORMAT = "voxpipe-profile"`, `CURRENT_VERSION = 1`, `envelope_schema() -> dict`, `params_schema(effect_cls) -> dict`.
  - `profile.ProfileError(VoxpipeError)`.
  - `profile.Provenance(tool_version: str, seed: int | None, pairs: tuple[str, ...], score: float | None, partial: bool, duration_s: float | None, objective_version: int | None)`, frozen, with classmethod `Provenance.manual() -> Provenance`.
  - `profile.Profile(name: str, created: str, provenance: Provenance, chain: tuple[ChainEntry, ...])`, frozen.
  - `profile.parse_profile(data: object) -> Profile` (raises `ProfileError`).
  - `profile.to_dict(profile: Profile) -> dict`.
  - `profile.now_iso() -> str` (UTC, seconds precision).

- [ ] **Step 1: Write the fixture profile**

`tests/data/mechanicus.json`:
```json
{
  "format": "voxpipe-profile",
  "version": 1,
  "name": "mechanicus-fixture",
  "created": "2026-10-08T00:00:00+00:00",
  "provenance": {
    "tool_version": "0.1.0",
    "seed": null,
    "pairs": [],
    "score": null,
    "partial": false,
    "duration_s": null,
    "objective_version": null
  },
  "chain": [
    {"effect": "gain", "params": {"gain_db": 3.0}},
    {"effect": "pitch_shift", "params": {"semitones": -4.0, "mix": 0.8}},
    {"effect": "filter", "params": {"highpass_hz": 150.0, "lowpass_hz": 6000.0}},
    {"effect": "eq", "params": {"low_hz": 250.0, "low_gain_db": -3.0, "mid_hz": 1500.0, "mid_gain_db": 6.0, "high_hz": 5000.0, "high_gain_db": -6.0, "q": 1.0}},
    {"effect": "ring_mod", "params": {"freq_hz": 60.0, "mix": 0.4}},
    {"effect": "comb", "params": {"delay_ms": 4.0, "feedback": 0.6, "mix": 0.5}},
    {"effect": "distortion", "params": {"drive_db": 12.0, "bits": 10.0, "mix": 0.3}},
    {"effect": "chorus", "params": {"delay_ms": 15.0, "depth_ms": 2.0, "rate_hz": 0.8, "mix": 0.3}},
    {"effect": "reverb", "params": {"room_size": 0.3, "mix": 0.15}},
    {"effect": "gain", "params": {"gain_db": -3.0}}
  ]
}
```

- [ ] **Step 2: Write the failing tests**

`tests/test_profile.py`:
```python
import copy
import json
from pathlib import Path

import pytest

from voxpipe.engine.entry import ChainEntry
from voxpipe.profile import ProfileError, Provenance, parse_profile, to_dict

FIXTURE = Path(__file__).parent / "data" / "mechanicus.json"


@pytest.fixture
def data() -> dict:
    return json.loads(FIXTURE.read_text())


def error_for(data: object) -> str:
    with pytest.raises(ProfileError) as exc:
        parse_profile(data)
    return str(exc.value)


def test_valid_fixture_parses(data):
    profile = parse_profile(data)
    assert profile.name == "mechanicus-fixture"
    assert profile.chain[0] == ChainEntry("gain", {"gain_db": 3.0})
    assert profile.provenance == Provenance(
        tool_version="0.1.0", seed=None, pairs=(), score=None,
        partial=False, duration_s=None, objective_version=None,
    )


def test_round_trip(data):
    assert to_dict(parse_profile(data)) == data


def test_not_an_object():
    assert "expected a JSON object" in error_for([1, 2])


def test_wrong_format(data):
    data["format"] = "something-else"
    assert "not a voxpipe profile" in error_for(data)


def test_newer_version_refused(data):
    data["version"] = 2
    assert "newer" in error_for(data)


def test_bad_version(data):
    data["version"] = "1"
    assert "profile.version" in error_for(data)


def test_missing_param_names_field(data):
    del data["chain"][0]["params"]["gain_db"]
    message = error_for(data)
    assert "profile.chain[0].params" in message
    assert "gain_db" in message


def test_out_of_range_names_field(data):
    data["chain"][4]["params"]["freq_hz"] = 5000.0
    message = error_for(data)
    assert "profile.chain[4].params.freq_hz" in message
    assert "maximum" in message


def test_unknown_param_rejected(data):
    data["chain"][0]["params"]["volume"] = 1.0
    assert "volume" in error_for(data)


def test_unknown_effect_rejected(data):
    data["chain"][0]["effect"] = "flanger"
    assert "profile.chain[0].effect" in error_for(data)


def test_unknown_top_level_key_rejected(data):
    data["comment"] = "hi"
    assert "comment" in error_for(data)


def test_boolean_is_not_a_number(data):
    data["chain"][0]["params"]["gain_db"] = True
    assert "profile.chain[0].params.gain_db" in error_for(data)


def test_missing_provenance_field(data):
    broken = copy.deepcopy(data)
    del broken["provenance"]["seed"]
    assert "profile.provenance" in error_for(broken)


def test_empty_chain_rejected(data):
    data["chain"] = []
    assert "profile.chain" in error_for(data)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_profile.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voxpipe.profile'`

- [ ] **Step 4: Write the implementation**

`src/voxpipe/profile_schema.py`:
```python
"""JSON schemas for profiles. Effect param schemas are generated from Param
declarations, so the engine and the schema cannot drift apart (D-08)."""

from voxpipe.engine.effect import Effect
from voxpipe.engine.registry import EFFECTS

FORMAT = "voxpipe-profile"
CURRENT_VERSION = 1

_NULLABLE_NUMBER = {"type": ["number", "null"]}


def envelope_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["format", "version", "name", "created", "provenance", "chain"],
        "properties": {
            "format": {"const": FORMAT},
            "version": {"const": CURRENT_VERSION},
            "name": {"type": "string", "minLength": 1},
            "created": {"type": "string", "minLength": 1},
            "provenance": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "tool_version", "seed", "pairs", "score",
                    "partial", "duration_s", "objective_version",
                ],
                "properties": {
                    "tool_version": {"type": "string"},
                    "seed": {"type": ["integer", "null"]},
                    "pairs": {"type": "array", "items": {"type": "string"}},
                    "score": _NULLABLE_NUMBER,
                    "partial": {"type": "boolean"},
                    "duration_s": {"type": ["number", "null"], "minimum": 0},
                    "objective_version": {"type": ["integer", "null"]},
                },
            },
            "chain": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["effect", "params"],
                    "properties": {
                        "effect": {"enum": sorted(EFFECTS)},
                        "params": {"type": "object"},
                    },
                },
            },
        },
    }


def params_schema(effect_cls: type[Effect]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [p.name for p in effect_cls.params],
        "properties": {
            p.name: {"type": "number", "minimum": p.minimum, "maximum": p.maximum}
            for p in effect_cls.params
        },
    }
```

`src/voxpipe/profile.py`:
```python
"""Profiles: strict, versioned JSON effect configurations (D-04, D-08, R-15)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match

from voxpipe import __version__
from voxpipe.engine.entry import ChainEntry
from voxpipe.engine.registry import EFFECTS
from voxpipe.errors import VoxpipeError
from voxpipe.profile_schema import CURRENT_VERSION, FORMAT, envelope_schema, params_schema

# Maps a version to the function that upgrades a profile dict from it to version + 1.
MIGRATIONS: dict[int, Callable[[dict], dict]] = {}


class ProfileError(VoxpipeError):
    pass


@dataclass(frozen=True)
class Provenance:
    tool_version: str
    seed: int | None
    pairs: tuple[str, ...]
    score: float | None
    partial: bool
    duration_s: float | None
    objective_version: int | None

    @classmethod
    def manual(cls) -> Provenance:
        return cls(__version__, None, (), None, False, None, None)


@dataclass(frozen=True)
class Profile:
    name: str
    created: str
    provenance: Provenance
    chain: tuple[ChainEntry, ...]


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _format_path(parts: Iterable[str | int]) -> str:
    out = "profile"
    for part in parts:
        out += f"[{part}]" if isinstance(part, int) else f".{part}"
    return out


@cache
def _envelope_validator() -> Draft202012Validator:
    return Draft202012Validator(envelope_schema())


@cache
def _params_validator(effect: str) -> Draft202012Validator:
    return Draft202012Validator(params_schema(EFFECTS[effect]))


def _check(validator: Draft202012Validator, data: object, prefix: list[str | int]) -> None:
    error = best_match(validator.iter_errors(data))
    if error is not None:
        raise ProfileError(f"{_format_path([*prefix, *error.absolute_path])}: {error.message}")


def _migrate(data: dict) -> dict:
    while data["version"] < CURRENT_VERSION:
        data = MIGRATIONS[data["version"]](data)
    return data


def parse_profile(data: object) -> Profile:
    if not isinstance(data, dict):
        raise ProfileError("profile: expected a JSON object")
    if data.get("format") != FORMAT:
        raise ProfileError("profile: not a voxpipe profile (missing or wrong 'format')")
    version = data.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ProfileError("profile.version: must be a positive integer")
    if version > CURRENT_VERSION:
        raise ProfileError(
            f"profile.version: version {version} is newer than this voxpipe supports "
            f"({CURRENT_VERSION}); upgrade voxpipe"
        )
    data = _migrate(data)
    _check(_envelope_validator(), data, [])
    for index, item in enumerate(data["chain"]):
        _check(_params_validator(item["effect"]), item["params"], ["chain", index, "params"])

    prov = data["provenance"]
    return Profile(
        name=data["name"],
        created=data["created"],
        provenance=Provenance(
            tool_version=prov["tool_version"],
            seed=prov["seed"],
            pairs=tuple(prov["pairs"]),
            score=prov["score"],
            partial=prov["partial"],
            duration_s=prov["duration_s"],
            objective_version=prov["objective_version"],
        ),
        chain=tuple(
            ChainEntry(item["effect"], {k: float(v) for k, v in item["params"].items()})
            for item in data["chain"]
        ),
    )


def to_dict(profile: Profile) -> dict:
    prov = profile.provenance
    return {
        "format": FORMAT,
        "version": CURRENT_VERSION,
        "name": profile.name,
        "created": profile.created,
        "provenance": {
            "tool_version": prov.tool_version,
            "seed": prov.seed,
            "pairs": list(prov.pairs),
            "score": prov.score,
            "partial": prov.partial,
            "duration_s": prov.duration_s,
            "objective_version": prov.objective_version,
        },
        "chain": [{"effect": e.effect, "params": dict(e.params)} for e in profile.chain],
    }
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_profile.py -v`
Expected: 14 passed. If `test_unknown_effect_rejected` fails because the params check runs first, check that `_check(_envelope_validator(), …)` runs before the per-item loop. The envelope `enum` catches the bad effect name.

- [ ] **Step 6: Commit**

```bash
git add src/voxpipe/profile.py src/voxpipe/profile_schema.py tests/data tests/test_profile.py
git commit -m "feat(profile): strict versioned profile schema and parsing"
```

---

### Task 9: Profile files: load, atomic save, path resolution

**Files:**
- Modify: `src/voxpipe/profile.py` (append functions)
- Test: `tests/test_profile.py` (append tests)

**Interfaces:**
- Consumes: `parse_profile`, `to_dict`, `ProfileError` (Task 8).
- Produces:
  - `load_profile(path: Path) -> Profile`. Errors are prefixed with the path.
  - `save_profile(profile: Profile, path: Path) -> None`. Atomic; creates parent dirs.
  - `resolve_profile_path(arg: str, profiles_dir: Path) -> Path`. An argument ending in `.json`, or containing a path separator, is a path. Otherwise it's `<profiles_dir>/<arg>.json`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_profile.py`:
```python
from voxpipe import profile as profile_module
from voxpipe.profile import load_profile, resolve_profile_path, save_profile


def test_load_fixture():
    assert load_profile(FIXTURE).name == "mechanicus-fixture"


def test_load_missing_file(tmp_path):
    with pytest.raises(ProfileError, match="profile not found"):
        load_profile(tmp_path / "nope.json")


def test_load_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(ProfileError, match="invalid JSON"):
        load_profile(path)


def test_load_error_is_prefixed_with_path(tmp_path, data):
    data["version"] = 9
    path = tmp_path / "future.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ProfileError, match=str(path)):
        load_profile(path)


def test_save_then_load_round_trip(tmp_path):
    original = load_profile(FIXTURE)
    path = tmp_path / "sub" / "copy.json"
    save_profile(original, path)
    assert load_profile(path) == original
    assert list(path.parent.iterdir()) == [path]  # no temp file left behind


def test_failed_save_keeps_old_file(tmp_path, monkeypatch):
    path = tmp_path / "keep.json"
    path.write_text("old")

    def explode(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(profile_module.json, "dump", explode)
    with pytest.raises(RuntimeError):
        save_profile(load_profile(FIXTURE), path)
    assert path.read_text() == "old"
    assert list(tmp_path.iterdir()) == [path]


def test_resolve_profile_path(tmp_path):
    assert resolve_profile_path("robot", tmp_path) == tmp_path / "robot.json"
    assert resolve_profile_path("robot.json", tmp_path) == Path("robot.json")
    assert resolve_profile_path("dir/robot", tmp_path) == Path("dir/robot")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_profile.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_profile'`

- [ ] **Step 3: Write the implementation**

Add `import json`, `import os`, `import tempfile` and `from pathlib import Path` to the imports of `src/voxpipe/profile.py`, then append:
```python
def load_profile(path: Path) -> Profile:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ProfileError(f"{path}: profile not found") from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProfileError(f"{path}: invalid JSON: {exc}") from None
    try:
        return parse_profile(data)
    except ProfileError as exc:
        raise ProfileError(f"{path}: {exc}") from None


def save_profile(profile: Profile, path: Path) -> None:
    """Write atomically: a crash or Ctrl+C never leaves a truncated profile."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(to_dict(profile), fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def resolve_profile_path(arg: str, profiles_dir: Path) -> Path:
    candidate = Path(arg)
    if candidate.suffix == ".json" or len(candidate.parts) > 1:
        return candidate
    return profiles_dir / f"{arg}.json"
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_profile.py -v`
Expected: 21 passed

- [ ] **Step 5: Commit**

```bash
git add src/voxpipe/profile.py tests/test_profile.py
git commit -m "feat(profile): load, atomic save and path resolution"
```

---

### Task 10: Media decoding and WAV writing

**Files:**
- Create: `src/voxpipe/media.py`
- Test: `tests/test_media.py`

**Interfaces:**
- Consumes: `SAMPLE_RATE` (Task 1), `VoxpipeError` (Task 1).
- Produces:
  - `MediaError(VoxpipeError)`.
  - `decode(path: Path) -> np.ndarray`. Any ffmpeg-readable audio or video becomes float32 mono 48 kHz. Raises `MediaError` when the file is missing, ffmpeg is missing, the file can't be decoded, or it has no audio.
  - `write_wav(path: Path, samples: np.ndarray) -> None`. 16-bit PCM, 48 kHz, mono, clipped to ±1.

- [ ] **Step 1: Write the failing tests**

`tests/test_media.py`:
```python
import shutil
import subprocess

import numpy as np
import pytest
from helpers import dominant_hz, sine

from voxpipe.audio import SAMPLE_RATE
from voxpipe.media import MediaError, decode, write_wav

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], check=True)


def test_wav_round_trip(tmp_path):
    path = tmp_path / "tone.wav"
    x = sine(440.0, seconds=0.5)
    write_wav(path, x)
    y = decode(path)
    assert y.dtype == np.float32
    assert y.size == x.size
    np.testing.assert_allclose(y, x, atol=1e-3)


def test_write_wav_clips(tmp_path):
    path = tmp_path / "loud.wav"
    write_wav(path, np.array([2.0, -2.0, 0.5], dtype=np.float32))
    np.testing.assert_allclose(decode(path), [1.0, -1.0, 0.5], atol=1e-3)


def test_decodes_stereo_44k_mp3_to_mono_48k(tmp_path):
    path = tmp_path / "tone.mp3"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=300:duration=1:sample_rate=44100", "-ac", "2", str(path))
    y = decode(path)
    assert abs(y.size - SAMPLE_RATE) < 2_000  # mp3 padding
    assert abs(dominant_hz(y) - 300.0) < 5.0


def test_decodes_audio_from_video(tmp_path):
    path = tmp_path / "clip.mkv"
    ffmpeg(
        "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=500:duration=1",
        "-c:v", "mpeg4", "-c:a", "flac", "-shortest", str(path),
    )
    assert abs(dominant_hz(decode(path)) - 500.0) < 5.0


def test_video_without_audio_is_an_error(tmp_path):
    path = tmp_path / "silent.mkv"
    ffmpeg("-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=10", "-c:v", "mpeg4", str(path))
    with pytest.raises(MediaError, match="silent.mkv"):
        decode(path)


def test_missing_file(tmp_path):
    with pytest.raises(MediaError, match="file not found"):
        decode(tmp_path / "nope.wav")


def test_garbage_file(tmp_path):
    path = tmp_path / "garbage.wav"
    path.write_bytes(b"not audio at all")
    with pytest.raises(MediaError, match="cannot decode"):
        decode(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_media.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voxpipe.media'`

- [ ] **Step 3: Write the implementation**

`src/voxpipe/media.py`:
```python
"""File decoding through ffmpeg and WAV output."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from voxpipe.audio import SAMPLE_RATE
from voxpipe.errors import VoxpipeError


class MediaError(VoxpipeError):
    pass


def decode(path: Path) -> np.ndarray:
    """Decode any audio or video file to float32 mono at SAMPLE_RATE."""
    if not path.is_file():
        raise MediaError(f"{path}: file not found")
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except FileNotFoundError:
        raise MediaError("ffmpeg not found; install ffmpeg") from None
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip().splitlines()
        reason = detail[-1] if detail else f"ffmpeg exit code {result.returncode}"
        raise MediaError(f"{path}: cannot decode audio: {reason}")
    samples = np.frombuffer(result.stdout, dtype="<f4").astype(np.float32)
    if samples.size == 0:
        raise MediaError(f"{path}: cannot decode audio: no audio stream")
    return samples


def write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, SAMPLE_RATE, pcm)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_media.py -v`
Expected: 7 passed. If `test_video_without_audio_is_an_error` fails because ffmpeg exits 0 with empty output, the `samples.size == 0` branch must catch it. The message still contains the path.

- [ ] **Step 5: Commit**

```bash
git add src/voxpipe/media.py tests/test_media.py
git commit -m "feat(media): ffmpeg decoding and wav output"
```

---

### Task 11: `voxpipe render` command and golden test

**Files:**
- Create: `src/voxpipe/commands/render.py`, `scripts/update_golden.py`, `tests/golden/mechanicus.npy` (generated)
- Modify: `src/voxpipe/cli.py` (`COMMANDS`)
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `Chain` (Task 7), `load_profile` and `resolve_profile_path` (Task 9), `decode` and `write_wav` (Task 10).
- Produces: `voxpipe render INPUT --profile NAME_OR_PATH [--profiles-dir DIR] [-o OUTPUT]`. The default output is `./<input stem>.voxpipe.wav` (R-23, R-26).

- [ ] **Step 1: Write the golden generator**

`scripts/update_golden.py`:
```python
"""Regenerate tests/golden/mechanicus.npy. Run only when an effect's sound changes on
purpose, and say so in the commit message.

    uv run python scripts/update_golden.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from helpers import voice_like  # noqa: E402

from voxpipe.engine.chain import Chain  # noqa: E402
from voxpipe.profile import load_profile  # noqa: E402

profile = load_profile(ROOT / "tests" / "data" / "mechanicus.json")
out = Chain(profile.chain).process_signal(voice_like())
target = ROOT / "tests" / "golden" / "mechanicus.npy"
target.parent.mkdir(exist_ok=True)
np.save(target, out)
print(f"wrote {target} ({out.size} samples)")
```

- [ ] **Step 2: Write the failing tests**

`tests/test_render.py`:
```python
import shutil
from pathlib import Path

import numpy as np
import pytest
from helpers import voice_like

from voxpipe.cli import main
from voxpipe.engine.chain import Chain
from voxpipe.media import decode, write_wav
from voxpipe.profile import load_profile

DATA = Path(__file__).parent / "data"
GOLDEN = Path(__file__).parent / "golden" / "mechanicus.npy"
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_golden_render_unchanged():
    """Guards the sound of every effect. Regenerate with scripts/update_golden.py
    only when a change in sound is intended."""
    out = Chain(load_profile(DATA / "mechanicus.json").chain).process_signal(voice_like())
    golden = np.load(GOLDEN)
    diff = np.abs(out - golden)
    assert out.shape == golden.shape
    assert np.mean(diff > 1e-4) < 0.001
    assert diff.max() < 0.01


@needs_ffmpeg
def test_render_writes_effected_wav(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "in.wav"
    write_wav(source, voice_like(1.0))
    code = main(["render", str(source), "--profile", str(DATA / "mechanicus.json")])
    assert code == 0
    output = tmp_path / "in.voxpipe.wav"
    rendered = decode(output)
    expected = Chain(load_profile(DATA / "mechanicus.json").chain).process_signal(decode(source))
    np.testing.assert_allclose(rendered, np.clip(expected, -1, 1), atol=2e-3)


@needs_ffmpeg
def test_render_profile_by_name_from_profiles_dir(tmp_path):
    source = tmp_path / "in.wav"
    write_wav(source, voice_like(0.5))
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    shutil.copy(DATA / "mechanicus.json", profiles / "robot.json")
    output = tmp_path / "out" / "x.wav"
    code = main([
        "render", str(source), "--profile", "robot",
        "--profiles-dir", str(profiles), "-o", str(output),
    ])
    assert code == 0
    assert output.is_file()


def test_render_invalid_profile_fails_before_decoding(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    code = main(["render", str(tmp_path / "missing.wav"), "--profile", str(bad)])
    assert code == 1
    assert "not a voxpipe profile" in capsys.readouterr().err
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_render.py -v`
Expected: FAIL. The golden file doesn't exist, and `render` is an invalid argparse choice (SystemExit 2).

- [ ] **Step 4: Write the implementation**

`src/voxpipe/commands/render.py`:
```python
"""`voxpipe render`: apply a profile to an audio or video file (R-23)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voxpipe.audio import SAMPLE_RATE
from voxpipe.engine.chain import Chain
from voxpipe.media import decode, write_wav
from voxpipe.profile import load_profile, resolve_profile_path


def register(subparsers) -> None:
    parser = subparsers.add_parser("render", help="apply a profile to an audio or video file")
    parser.add_argument("input", type=Path, help="any audio or video file ffmpeg can read")
    parser.add_argument("--profile", required=True, help="profile name or path to a .json file")
    parser.add_argument("--profiles-dir", type=Path, default=Path("."),
                        help="where profile names are looked up (default: current directory)")
    parser.add_argument("-o", "--output", type=Path,
                        help="output WAV (default: ./<input>.voxpipe.wav)")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    profile = load_profile(resolve_profile_path(args.profile, args.profiles_dir))
    samples = decode(args.input)
    output = args.output or Path(f"{args.input.stem}.voxpipe.wav")
    write_wav(output, Chain(profile.chain).process_signal(samples))
    print(f"wrote {output} ({samples.size / SAMPLE_RATE:.1f} s, profile '{profile.name}')",
          file=sys.stderr)
    return 0
```

In `src/voxpipe/cli.py`, replace `COMMANDS: tuple = ()` with:
```python
from voxpipe.commands import render

COMMANDS: tuple = (render,)
```
Keep the import with the other imports at the top of the file.

- [ ] **Step 5: Generate the golden file**

Run: `uv run python scripts/update_golden.py`
Expected: `wrote …/tests/golden/mechanicus.npy (96000 samples)`

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -v && uv run ruff check`
Expected: all tests pass; `All checks passed!`

- [ ] **Step 7: Try it by hand**

Run: `uv run voxpipe render missing.wav --profile tests/data/mechanicus.json`
Expected: `voxpipe: error: missing.wav: file not found`, exit 1. Then render a real voice recording:

```bash
uv run voxpipe render ~/some-recording.ogg --profile tests/data/mechanicus.json -o /tmp/mech.wav
```
Listen to `/tmp/mech.wav`.

- [ ] **Step 8: Commit**

```bash
git add src/voxpipe/commands/render.py src/voxpipe/cli.py scripts tests/golden tests/test_render.py
git commit -m "feat: voxpipe render command with golden-file test"
```

---

### Task 12: Real-time budget check (N-02)

**Files:**
- Test: `tests/test_perf.py`

**Interfaces:**
- Consumes: `Chain` (Task 7), `load_profile` (Task 9).

- [ ] **Step 1: Write the test**

`tests/test_perf.py`:
```python
import time
from pathlib import Path

import numpy as np
from helpers import voice_like

from voxpipe.audio import BLOCK_SIZE
from voxpipe.engine.chain import Chain
from voxpipe.profile import load_profile

BUDGET_S = 0.003  # N-02: one 10 ms block in <= 3 ms with the full chain


def test_full_chain_block_fits_realtime_budget():
    chain = Chain(load_profile(Path(__file__).parent / "data" / "mechanicus.json").chain)
    signal = voice_like(5.0)
    blocks = [signal[i : i + BLOCK_SIZE] for i in range(0, signal.size, BLOCK_SIZE)]
    for block in blocks[:50]:  # warm-up
        chain.process_block(block)
    timings = []
    for block in blocks:
        start = time.perf_counter()
        chain.process_block(block)
        timings.append(time.perf_counter() - start)
    assert np.median(timings) < BUDGET_S, f"median {np.median(timings) * 1000:.2f} ms"
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_perf.py -v`
Expected: PASS. If it fails, profile with `uv run python -m cProfile -s cumtime -c "…"` and optimise the slowest effect. Don't raise the budget without asking the user, because it is target N-02.

- [ ] **Step 3: Commit**

```bash
git add tests/test_perf.py
git commit -m "test: real-time budget check for the full effect chain (N-02)"
```

---

## Self-review notes

- **Spec coverage:**
  - C-07 / D-05: block invariance in every effect test, and `test_live_blocks_match_offline_signal`.
  - D-04, D-08, R-15, C-06: Tasks 8–9.
  - R-23, R-26: Task 11.
  - N-02: Task 12.
  - D-09: `audio.py`.
  - Atomic writes: Task 9.
- **Deferred to other plans:** formant shift (noted in Task 6); `list`, `create`, `run`, `clean` (plan 2); `tune` (plan 3).
