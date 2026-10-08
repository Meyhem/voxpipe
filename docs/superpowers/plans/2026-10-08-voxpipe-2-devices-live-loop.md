# voxpipe Plan 2: PipeWire Devices and the Live Loop

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `voxpipe list`, `create`, `clean` and `run`. Together they create the virtual microphone, find devices, and stream the live mic through a profile's effect chain into the virtual mic.

**Architecture:**
- **Device layer.** All PipeWire contact goes through one module, `voxpipe/pw.py`:
  - discovery parses `pw-dump` JSON;
  - the virtual mic is created and destroyed with `pw-cli`;
  - audio flows through `pw-record` and `pw-play` child processes over raw f32 pipes (D-02, D-03).
- **Live loop.** A pure-Python loop over two byte streams, testable with in-memory fakes:
  - a reader thread fills a queue;
  - the main loop processes 480-sample blocks, discards backlog beyond a cap, and counts late blocks.
- **Watchdog.** A device watchdog polls `pw-dump`. When the source or destination node vanishes, it kills the matching child process, which turns "device gone" into EOF or a broken pipe in the loop (R-20, A-07).

**Tech Stack:** Python 3.12, numpy, PipeWire 1.0 CLI tools (`pw-dump`, `pw-cli`, `pw-record`, `pw-play`), pytest.

**Specs:** `design/tech-spec-voxpipe.md`, `design/domain.md`.

**Prerequisite:** Plan 1 is complete. This plan uses `Chain`, `load_profile`, `resolve_profile_path`, `VoxpipeError`, `cli.COMMANDS`, `BLOCK_SIZE` and `SAMPLE_RATE`.

## Global Constraints

- PipeWire is driven only through `pw-dump`, `pw-cli`, `pw-record` and `pw-play`. `pactl` is not available, and there are no native bindings (C-04).
- Parse `pw-dump` JSON only, never human-readable tool output.
- Exactly one virtual mic: `node.name` = `voxpipe.virtual_mic`, `node.description` = `voxpipe virtual microphone`, `media.class` = `Audio/Source/Virtual`, `object.linger = true`. `create` reuses an existing one (R-13, R-17). It doesn't survive a reboot (R-19).
- The device number shown by `list` is the PipeWire object `id`. A device may also be named by `node.name`, or by `node.description` if that is unique (R-08). `pw-record` and `pw-play` `--target` take a node **name** (or serial), never the object id.
- `run` stays in the foreground, and Ctrl+C stops it (C-05). It requires an explicit `--source`, `--destination` and `--profile` (R-03, R-07, R-22). It refuses the same device for both (R-11). It has no monitoring (R-21).
- Loop latency must not drift: discard backlog beyond 3 blocks (N-04). Late blocks are counted, never fatal (N-07).
- Child processes are started with `start_new_session=True`, so the terminal's Ctrl+C reaches only voxpipe, and voxpipe shuts them down itself.
- The interfaces below follow the device-layer seam, so a later move to native bindings touches only `pw.py`.

---

## File Structure

```
src/voxpipe/
  pw.py                 Node, DeviceError, run_tool, parse_nodes, list_nodes, resolve,
                        find_virtual_mic, create_virtual_mic, remove_virtual_mic,
                        open_capture, open_playback, terminate, DeviceWatchdog
  live.py               LiveLoop, LoopStats, StopReason
  commands/
    devices.py          `list`, `create`, `clean`
    run.py              `run`
scripts/
  spike_pipewire.py     manual latency and linking spike (Task 1)
tests/
  data/pw-dump.json     trimmed real pw-dump fixture
  fakes.py              FakeProcess, make_dump()
  test_pw.py
  test_devices_cmd.py
  test_live.py
  test_watchdog.py
  test_run_cmd.py
```

---

### Task 1: PipeWire spike (manual; retires K-01, A-04 and the D-02/D-03 recipe)

Before building on them, confirm these assumptions on this machine:
1. A node created with `pw-cli create-node adapter {…object.linger=true…}` outlives `pw-cli`.
2. `pw-play --target voxpipe.virtual_mic` feeds the virtual source, and apps can record from it.
3. `pw-record -` writes raw f32 with no header. (Already observed: the stream begins with zero bytes, not `RIFF`.)
4. The extra latency of the pipe hop (playback into the virtual mic, then recording from it) is ≤ 35 ms. That leaves room for the 15 ms pitch shifter within N-01's 50 ms.
5. `-P '{ node.dont-reconnect = true }'` stops `pw-record` from silently moving to another mic when its target vanishes.

**Files:**
- Create: `scripts/spike_pipewire.py`
- Modify: `design/tech-spec-voxpipe.md` (record results under D-02, D-03, K-01, A-04)

- [ ] **Step 1: Create the virtual mic by hand and check that it lingers**

```bash
pw-cli create-node adapter '{ factory.name=support.null-audio-sink node.name=voxpipe.virtual_mic node.description="voxpipe virtual microphone" media.class=Audio/Source/Virtual object.linger=true audio.position=[ MONO ] audio.rate=48000 }'
```
```bash
pw-dump | python3 -c "import json,sys; print([(o['id'], o['info']['props']['media.class']) for o in json.load(sys.stdin) if o.get('info',{}).get('props',{}).get('node.name')=='voxpipe.virtual_mic'])"
```
Expected: one tuple `(<id>, 'Audio/Source/Virtual')`, still present after `pw-cli` exited.

- [ ] **Step 2: Write the spike script**

`scripts/spike_pipewire.py`:
```python
"""Manual spike: measure the pw-play -> virtual mic -> pw-record hop latency.

    uv run python scripts/spike_pipewire.py
Requires the virtual mic from Task 1 Step 1 to exist.
"""

import subprocess
import threading
import time

import numpy as np

NAME = "voxpipe.virtual_mic"
RATE, BLOCK = 48_000, 480
FMT = ["--rate", str(RATE), "--channels", "1", "--format", "f32", "--latency", str(BLOCK)]

rec = subprocess.Popen(["pw-record", "--target", NAME, *FMT, "-"], stdout=subprocess.PIPE)
play = subprocess.Popen(["pw-play", "--target", NAME, *FMT, "-"], stdin=subprocess.PIPE)
arrivals: list[float] = []


def reader() -> None:
    while True:
        data = rec.stdout.read(BLOCK * 4)
        if not data:
            return
        if np.max(np.abs(np.frombuffer(data, dtype="<f4"))) > 0.5:
            arrivals.append(time.perf_counter())


threading.Thread(target=reader, daemon=True).start()
silence = np.zeros(BLOCK, dtype="<f4").tobytes()
click = np.ones(BLOCK, dtype="<f4").tobytes()

time.sleep(1.0)  # let both streams link
next_time = time.perf_counter()
sent_at = None
for i in range(300):  # 3 s of writes paced at real time; the click goes out at 1 s
    if i == 100:
        sent_at = time.perf_counter()
    play.stdin.write(click if i == 100 else silence)
    play.stdin.flush()
    next_time += BLOCK / RATE
    time.sleep(max(0.0, next_time - time.perf_counter()))
time.sleep(0.3)
play.stdin.close()
rec.terminate()
play.terminate()
if arrivals:
    print(f"hop latency: {(arrivals[0] - sent_at) * 1000:.1f} ms")
else:
    print("click never arrived: pw-play did not feed the virtual mic (see Step 4)")
```

- [ ] **Step 3: Run the spike three times**

Run: `uv run python scripts/spike_pipewire.py`
Expected: `hop latency: NN.N ms` with NN ≤ 35 on each run.

- [ ] **Step 4: If the click never arrives (pw-play didn't link to the source node)**

Link manually. Start `pw-play` with `--target 0` (don't auto-link), then link with `pw-link`:
```bash
pw-link -o   # find the pw-play output port, e.g. "pw-play:output_MONO"
pw-link pw-play:output_MONO voxpipe.virtual_mic:input_MONO
```
If that works, `open_playback` (Task 5) must use `--target 0` plus `pw-link` after the stream appears. Stop here and update Task 5 before continuing.

- [ ] **Step 5: Check dont-reconnect**

Run `pw-record --target <your mic node.name> --rate 48000 --channels 1 --format f32 -P '{ node.dont-reconnect = true }' /tmp/t.wav`, then unplug the mic (or destroy the virtual mic if you used it as the target).
Expected: recording stops receiving data, or the process exits. In `pw-top` it must **not** move to another mic. Note which happens: the watchdog (Task 5) covers both.

- [ ] **Step 6: Clean up and record the results**

```bash
pw-cli destroy voxpipe.virtual_mic
```
In `design/tech-spec-voxpipe.md`, append to D-03 a "Spike 2026-10-xx" line with the measured hop latency, whether `--target` linking worked, and the dont-reconnect behaviour. Mark A-04 as verified or not.

**If the hop latency is above 35 ms, stop and raise it with the user before Task 2.** D-03 says revisit if N-01 is missed.

- [ ] **Step 7: Commit**

```bash
git add scripts/spike_pipewire.py design/tech-spec-voxpipe.md
git commit -m "spike: verify PipeWire virtual mic recipe and pipe latency"
```

---

### Task 2: Node discovery and identifier resolution

**Files:**
- Create: `src/voxpipe/pw.py`, `tests/data/pw-dump.json`, `tests/fakes.py`
- Test: `tests/test_pw.py`

**Interfaces:**
- Consumes: `VoxpipeError` (plan 1).
- Produces:
  - `VIRTUAL_MIC_NAME = "voxpipe.virtual_mic"`, `VIRTUAL_MIC_DESCRIPTION = "voxpipe virtual microphone"`.
  - `DeviceError(VoxpipeError)`.
  - `Node(id: int, name: str, description: str, media_class: str)`, frozen, with properties `is_microphone` and `is_virtual_mic`.
  - `Runner = Callable[[list[str]], str]`.
  - `run_tool(args: list[str]) -> str`. Returns stdout. Raises `DeviceError` when the tool is missing or exits non-zero.
  - `parse_nodes(dump_json: str) -> list[Node]`. Audio device nodes only (media class starts with `Audio/`), sorted by id.
  - `list_nodes(runner: Runner | None = None) -> list[Node]`. Looks up `run_tool` at call time, so tests can monkeypatch it.
  - `microphones(nodes) -> list[Node]`.
  - `resolve(identifier: str, nodes: Sequence[Node]) -> Node`.
  - `find_virtual_mic(nodes) -> Node | None`.
- Test fakes (`tests/fakes.py`): `make_dump(nodes: list[tuple[int, str, str, str]]) -> str`, where each tuple is (id, name, description, media_class).

- [ ] **Step 1: Capture the fixture**

```bash
pw-dump > /tmp/pw-dump-full.json
uv run python - <<'EOF'
import json
data = json.load(open("/tmp/pw-dump-full.json"))
keep = [o for o in data if o.get("type") == "PipeWire:Interface:Node"][:12]
json.dump(keep, open("tests/data/pw-dump.json", "w"), indent=1)
EOF
```
Check that the fixture contains at least one `Audio/Source`, one `Audio/Sink` and one `Stream/Output/Audio` node. If not, raise the `[:12]` limit.

- [ ] **Step 2: Write the fakes**

`tests/fakes.py`:
```python
"""Test doubles for PipeWire tools and child processes."""

from __future__ import annotations

import io
import json


def make_dump(nodes: list[tuple[int, str, str, str]]) -> str:
    """Minimal pw-dump JSON: (id, node.name, node.description, media.class) per node."""
    return json.dumps(
        [
            {
                "id": node_id,
                "type": "PipeWire:Interface:Node",
                "info": {
                    "props": {
                        "node.name": name,
                        "node.description": description,
                        "media.class": media_class,
                    }
                },
            }
            for node_id, name, description, media_class in nodes
        ]
        + [{"id": 1, "type": "PipeWire:Interface:Core", "info": {}}]
    )


MIC = (57, "alsa_input.usb-HyperX", "HyperX Headset", "Audio/Source")
WEBCAM = (33, "alsa_input.usb-Brio", "Brio 100", "Audio/Source")
SPEAKERS = (56, "alsa_output.usb-HyperX", "HyperX Headset", "Audio/Sink")
VIRTUAL = (120, "voxpipe.virtual_mic", "voxpipe virtual microphone", "Audio/Source/Virtual")
FIREFOX = (94, "Firefox", "Firefox", "Stream/Output/Audio")


class FakeProcess:
    """Stands in for subprocess.Popen with in-memory pipes."""

    def __init__(self, stdout: bytes = b"") -> None:
        self.stdout = io.BytesIO(stdout)
        self.stdin = io.BytesIO()
        self.returncode: int | None = None
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode
```

- [ ] **Step 3: Write the failing tests**

`tests/test_pw.py`:
```python
from pathlib import Path

import pytest
from fakes import FIREFOX, MIC, SPEAKERS, VIRTUAL, WEBCAM, make_dump

from voxpipe import pw
from voxpipe.pw import DeviceError, Node


def test_parse_real_dump_fixture():
    nodes = pw.parse_nodes((Path(__file__).parent / "data" / "pw-dump.json").read_text())
    assert nodes, "fixture should contain audio nodes"
    assert all(n.media_class.startswith("Audio/") for n in nodes)
    assert [n.id for n in nodes] == sorted(n.id for n in nodes)


def test_parse_skips_streams_and_non_nodes():
    nodes = pw.parse_nodes(make_dump([MIC, FIREFOX, SPEAKERS]))
    assert [n.id for n in nodes] == [56, 57]


def test_microphones_and_virtual_flag():
    nodes = pw.parse_nodes(make_dump([MIC, SPEAKERS, VIRTUAL]))
    mics = pw.microphones(nodes)
    assert [n.id for n in mics] == [57, 120]
    assert pw.find_virtual_mic(nodes) == Node(*VIRTUAL)
    assert mics[1].is_virtual_mic and not mics[0].is_virtual_mic


def test_resolve_by_id_name_and_unique_description():
    nodes = pw.parse_nodes(make_dump([MIC, WEBCAM, SPEAKERS, VIRTUAL]))
    assert pw.resolve("57", nodes).name == MIC[1]
    assert pw.resolve("alsa_input.usb-Brio", nodes).id == 33
    assert pw.resolve("Brio 100", nodes).id == 33


def test_resolve_ambiguous_description():
    nodes = pw.parse_nodes(make_dump([MIC, SPEAKERS]))  # both "HyperX Headset"
    with pytest.raises(DeviceError, match="ambiguous"):
        pw.resolve("HyperX Headset", nodes)


def test_resolve_unknown():
    with pytest.raises(DeviceError, match="voxpipe list"):
        pw.resolve("999", pw.parse_nodes(make_dump([MIC])))


def test_list_nodes_uses_run_tool(monkeypatch):
    calls = []

    def fake_run_tool(args):
        calls.append(args)
        return make_dump([MIC])

    monkeypatch.setattr(pw, "run_tool", fake_run_tool)
    assert [n.id for n in pw.list_nodes()] == [57]
    assert calls == [["pw-dump"]]


def test_run_tool_missing_binary():
    with pytest.raises(DeviceError, match="not found"):
        pw.run_tool(["definitely-not-a-pipewire-tool"])


def test_run_tool_nonzero_exit():
    with pytest.raises(DeviceError, match="failed"):
        pw.run_tool(["false"])
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_pw.py -v`
Expected: FAIL with `ImportError: cannot import name 'pw'`

- [ ] **Step 5: Write the implementation**

`src/voxpipe/pw.py`:
```python
"""Device layer: the only module that talks to PipeWire (C-04, D-02, D-03)."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from voxpipe.errors import VoxpipeError

VIRTUAL_MIC_NAME = "voxpipe.virtual_mic"
VIRTUAL_MIC_DESCRIPTION = "voxpipe virtual microphone"
MICROPHONE_CLASSES = ("Audio/Source", "Audio/Source/Virtual")

Runner = Callable[[list[str]], str]


class DeviceError(VoxpipeError):
    pass


@dataclass(frozen=True)
class Node:
    id: int
    name: str
    description: str
    media_class: str

    @property
    def is_microphone(self) -> bool:
        return self.media_class in MICROPHONE_CLASSES

    @property
    def is_virtual_mic(self) -> bool:
        return self.name == VIRTUAL_MIC_NAME


def run_tool(args: list[str]) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        raise DeviceError(f"{args[0]} not found; is PipeWire installed?") from None
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise DeviceError(f"{args[0]} failed: {detail}")
    return result.stdout


def parse_nodes(dump_json: str) -> list[Node]:
    nodes = []
    for obj in json.loads(dump_json):
        if obj.get("type") != "PipeWire:Interface:Node":
            continue
        props = (obj.get("info") or {}).get("props") or {}
        media_class = props.get("media.class", "")
        if not media_class.startswith("Audio/"):
            continue
        name = props.get("node.name", "")
        nodes.append(Node(obj["id"], name, props.get("node.description", name), media_class))
    return sorted(nodes, key=lambda n: n.id)


def list_nodes(runner: Runner | None = None) -> list[Node]:
    return parse_nodes((runner or run_tool)(["pw-dump"]))


def microphones(nodes: Sequence[Node]) -> list[Node]:
    return [n for n in nodes if n.is_microphone]


def find_virtual_mic(nodes: Sequence[Node]) -> Node | None:
    return next((n for n in nodes if n.is_virtual_mic), None)


def resolve(identifier: str, nodes: Sequence[Node]) -> Node:
    """R-08: the number shown by `list` (object id), the node name, or a unique description."""
    if identifier.isdigit():
        matches = [n for n in nodes if n.id == int(identifier)]
    else:
        matches = [n for n in nodes if n.name == identifier]
        if not matches:
            matches = [n for n in nodes if n.description == identifier]
    if len(matches) > 1:
        ids = ", ".join(str(n.id) for n in matches)
        raise DeviceError(f"'{identifier}' is ambiguous (ids {ids}); use the number from `voxpipe list`")
    if not matches:
        raise DeviceError(f"no audio device matches '{identifier}'; see `voxpipe list`")
    return matches[0]
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_pw.py -v`
Expected: 9 passed

- [ ] **Step 7: Commit**

```bash
git add src/voxpipe/pw.py tests/fakes.py tests/data/pw-dump.json tests/test_pw.py
git commit -m "feat(pw): PipeWire node discovery and device resolution"
```

---

### Task 3: Virtual mic lifecycle and the `list`, `create`, `clean` commands

**Files:**
- Modify: `src/voxpipe/pw.py` (append)
- Create: `src/voxpipe/commands/devices.py`
- Modify: `src/voxpipe/cli.py` (`COMMANDS`)
- Test: `tests/test_devices_cmd.py`

**Interfaces:**
- Consumes: Task 2.
- Produces:
  - `create_virtual_mic(runner: Runner | None = None, wait_s: float = 2.0, sleep: Callable[[float], None] = time.sleep) -> tuple[Node, bool]`. Returns (node, created). It reuses an existing mic, creates one otherwise, and polls `pw-dump` until the mic appears or `wait_s` has passed.
  - `remove_virtual_mic(runner: Runner | None = None) -> Node | None`. Returns the removed node, or None when there was nothing to remove. Works even while the mic is in use (R-18).
  - CLI:
    - `voxpipe list` prints a table of microphones, marking the virtual mic.
    - `voxpipe create` prints `created virtual microphone (id N)` or `virtual microphone already exists (id N)`.
    - `voxpipe clean` prints `removed virtual microphone (id N)` or `no virtual microphone to remove`.

- [ ] **Step 1: Write the failing tests**

`tests/test_devices_cmd.py`:
```python
import pytest
from fakes import MIC, SPEAKERS, VIRTUAL, make_dump

from voxpipe import pw
from voxpipe.cli import main
from voxpipe.pw import DeviceError


class FakePipeWire:
    """Scriptable stand-in for pw.run_tool."""

    def __init__(self, nodes, appear_on_create=True):
        self.nodes = list(nodes)
        self.appear_on_create = appear_on_create
        self.calls = []

    def __call__(self, args):
        self.calls.append(args)
        if args == ["pw-dump"]:
            return make_dump(self.nodes)
        if args[:2] == ["pw-cli", "create-node"]:
            if self.appear_on_create:
                self.nodes.append(VIRTUAL)
            return ""
        if args[:2] == ["pw-cli", "destroy"]:
            self.nodes = [n for n in self.nodes if str(n[0]) != args[2]]
            return ""
        raise AssertionError(f"unexpected call {args}")


@pytest.fixture
def fake_pw(monkeypatch):
    def install(nodes, **kwargs):
        fake = FakePipeWire(nodes, **kwargs)
        monkeypatch.setattr(pw, "run_tool", fake)
        return fake

    return install


def test_create_makes_lingering_virtual_source(fake_pw):
    fake = fake_pw([MIC])
    node, created = pw.create_virtual_mic(sleep=lambda s: None)
    assert created and node.id == 120
    create_call = next(c for c in fake.calls if c[:2] == ["pw-cli", "create-node"])
    props = create_call[3]
    for fragment in (
        "factory.name=support.null-audio-sink",
        "node.name=voxpipe.virtual_mic",
        "media.class=Audio/Source/Virtual",
        "object.linger=true",
    ):
        assert fragment in props


def test_create_reuses_existing(fake_pw):
    fake = fake_pw([MIC, VIRTUAL])
    node, created = pw.create_virtual_mic(sleep=lambda s: None)
    assert not created and node.id == 120
    assert not any(c[:2] == ["pw-cli", "create-node"] for c in fake.calls)


def test_create_times_out_when_node_never_appears(fake_pw):
    fake_pw([MIC], appear_on_create=False)
    with pytest.raises(DeviceError, match="did not appear"):
        pw.create_virtual_mic(wait_s=0.3, sleep=lambda s: None)


def test_remove(fake_pw):
    fake = fake_pw([MIC, VIRTUAL])
    assert pw.remove_virtual_mic().id == 120
    assert ["pw-cli", "destroy", "120"] in fake.calls
    assert pw.remove_virtual_mic() is None


def test_list_command(fake_pw, capsys):
    fake_pw([MIC, SPEAKERS, VIRTUAL])
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "57" in out and "HyperX Headset" in out
    assert "alsa_output" not in out  # sinks are not microphones
    assert "voxpipe virtual microphone" in out and "[virtual mic]" in out


def test_create_command_twice(fake_pw, capsys):
    fake_pw([MIC])
    assert main(["create"]) == 0
    assert main(["create"]) == 0
    out = capsys.readouterr().out
    assert "created virtual microphone (id 120)" in out
    assert "already exists (id 120)" in out


def test_clean_command(fake_pw, capsys):
    fake_pw([MIC, VIRTUAL])
    assert main(["clean"]) == 0
    assert main(["clean"]) == 0
    out = capsys.readouterr().out
    assert "removed virtual microphone (id 120)" in out
    assert "no virtual microphone to remove" in out


def test_pipewire_not_running(monkeypatch, capsys):
    def broken(args):
        raise DeviceError("pw-dump failed: failed to connect")

    monkeypatch.setattr(pw, "run_tool", broken)
    assert main(["list"]) == 1
    assert "failed to connect" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_devices_cmd.py -v`
Expected: FAIL with `AttributeError: module 'voxpipe.pw' has no attribute 'create_virtual_mic'`

- [ ] **Step 3: Write the implementation**

Add `import time` to `src/voxpipe/pw.py`'s imports, then append:
```python
_CREATE_PROPS = (
    "{ factory.name=support.null-audio-sink "
    f"node.name={VIRTUAL_MIC_NAME} "
    f'node.description="{VIRTUAL_MIC_DESCRIPTION}" '
    "media.class=Audio/Source/Virtual object.linger=true "
    "audio.position=[ MONO ] audio.rate=48000 }"
)


def create_virtual_mic(
    runner: Runner | None = None,
    wait_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Node, bool]:
    """R-13, R-17: reuse the one virtual mic, or create it and wait until it appears."""
    existing = find_virtual_mic(list_nodes(runner))
    if existing is not None:
        return existing, False
    (runner or run_tool)(["pw-cli", "create-node", "adapter", _CREATE_PROPS])
    interval = 0.1
    for _ in range(max(1, int(wait_s / interval))):
        node = find_virtual_mic(list_nodes(runner))
        if node is not None:
            return node, True
        sleep(interval)
    raise DeviceError("virtual microphone did not appear after creation")


def remove_virtual_mic(runner: Runner | None = None) -> Node | None:
    """R-18: removes the virtual mic even while it is in use."""
    node = find_virtual_mic(list_nodes(runner))
    if node is None:
        return None
    (runner or run_tool)(["pw-cli", "destroy", str(node.id)])
    return node
```

`src/voxpipe/commands/devices.py`:
```python
"""`voxpipe list`, `voxpipe create`, `voxpipe clean`."""

from __future__ import annotations

import argparse

from voxpipe import pw


def register(subparsers) -> None:
    subparsers.add_parser("list", help="show microphones and their ids").set_defaults(
        handler=run_list
    )
    subparsers.add_parser("create", help="create the virtual microphone").set_defaults(
        handler=run_create
    )
    subparsers.add_parser("clean", help="remove the virtual microphone").set_defaults(
        handler=run_clean
    )


def run_list(args: argparse.Namespace) -> int:
    mics = pw.microphones(pw.list_nodes())
    if not mics:
        print("no microphones found")
        return 0
    name_width = max(len(n.name) for n in mics)
    print(f"{'ID':>5}  {'NAME':<{name_width}}  DESCRIPTION")
    for node in mics:
        marker = "  [virtual mic]" if node.is_virtual_mic else ""
        print(f"{node.id:>5}  {node.name:<{name_width}}  {node.description}{marker}")
    return 0


def run_create(args: argparse.Namespace) -> int:
    node, created = pw.create_virtual_mic()
    if created:
        print(f"created virtual microphone (id {node.id})")
    else:
        print(f"virtual microphone already exists (id {node.id})")
    return 0


def run_clean(args: argparse.Namespace) -> int:
    node = pw.remove_virtual_mic()
    if node is None:
        print("no virtual microphone to remove")
    else:
        print(f"removed virtual microphone (id {node.id})")
    return 0
```

In `src/voxpipe/cli.py`, change the import and tuple to:
```python
from voxpipe.commands import devices, render

COMMANDS: tuple = (devices, render)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_devices_cmd.py -v`
Expected: 8 passed

- [ ] **Step 5: Try it on the real system**

```bash
uv run voxpipe list
uv run voxpipe create
uv run voxpipe create
uv run voxpipe list
uv run voxpipe clean
uv run voxpipe clean
```
Expected: the virtual mic appears in the second `list` marked `[virtual mic]`. The second `create` reports it already exists, and the second `clean` reports nothing to remove.

- [ ] **Step 6: Commit**

```bash
git add src/voxpipe/pw.py src/voxpipe/commands/devices.py src/voxpipe/cli.py tests/test_devices_cmd.py
git commit -m "feat: list, create and clean commands for the virtual microphone"
```

---

### Task 4: The live loop

**Files:**
- Create: `src/voxpipe/live.py`
- Test: `tests/test_live.py`

**Interfaces:**
- Consumes: `Chain` (plan 1), `BLOCK_SIZE` and `SAMPLE_RATE` (plan 1).
- Produces:
  - `StopReason(Enum)`: `INTERRUPTED`, `SOURCE_LOST`, `DESTINATION_LOST`.
  - `LoopStats` dataclass: `blocks: int`, `late_blocks: int`, `discarded_blocks: int`, `duration_s: float`.
  - `MAX_BACKLOG = 3`, `INPUT_TIMEOUT_S = 2.0`.
  - `LiveLoop(chain, reader: BinaryIO, writer: BinaryIO, block_size: int = BLOCK_SIZE, input_timeout_s: float = INPUT_TIMEOUT_S)` with `.run(stop: threading.Event) -> StopReason` and `.stats: LoopStats`. `chain` is any object with `process_block(np.ndarray) -> np.ndarray`.
  - Behaviour:
    - EOF on the reader, or no bytes for `input_timeout_s`, means `SOURCE_LOST`.
    - `BrokenPipeError`, `OSError` or `ValueError` on write means `DESTINATION_LOST`.
    - A set `stop` means `INTERRUPTED`.
    - Late blocks (processing time > block duration) are counted; PipeWire plays silence for the gap (N-07).
    - When more than `MAX_BACKLOG` blocks are queued, the oldest are discarded and counted (N-04).

- [ ] **Step 1: Write the failing tests**

`tests/test_live.py`:
```python
import io
import os
import threading
import time

import numpy as np
from helpers import voice_like

from voxpipe.engine.chain import Chain
from voxpipe.engine.entry import ChainEntry
from voxpipe.live import LiveLoop, StopReason

BLOCK = 480
ENTRIES = [ChainEntry("gain", {"gain_db": 6.0}), ChainEntry("reverb", {"room_size": 0.4, "mix": 0.3})]


def to_bytes(x: np.ndarray) -> bytes:
    return x.astype("<f4").tobytes()


def test_processes_all_blocks_then_reports_source_lost():
    x = voice_like(0.5)  # 50 blocks
    writer = io.BytesIO()
    loop = LiveLoop(Chain(ENTRIES), io.BytesIO(to_bytes(x)), writer)
    assert loop.run(threading.Event()) is StopReason.SOURCE_LOST
    out = np.frombuffer(writer.getvalue(), dtype="<f4")
    expected = Chain(ENTRIES).process_signal(x)
    assert loop.stats.blocks + loop.stats.discarded_blocks == 50
    # Without discards the output is exactly the offline render.
    if loop.stats.discarded_blocks == 0:
        np.testing.assert_allclose(out, expected, atol=1e-5)


def test_partial_trailing_block_is_dropped():
    data = to_bytes(np.zeros(BLOCK * 2 + 100, dtype=np.float32))
    writer = io.BytesIO()
    LiveLoop(Chain(ENTRIES), io.BytesIO(data), writer).run(threading.Event())
    assert len(writer.getvalue()) == BLOCK * 2 * 4


class BrokenWriter(io.RawIOBase):
    def writable(self) -> bool:
        return True

    def write(self, data) -> int:
        raise BrokenPipeError


def test_destination_lost():
    loop = LiveLoop(Chain(ENTRIES), io.BytesIO(to_bytes(voice_like(0.1))), BrokenWriter())
    assert loop.run(threading.Event()) is StopReason.DESTINATION_LOST


def test_interrupted_while_waiting_for_input():
    read_fd, write_fd = os.pipe()
    stop = threading.Event()
    loop = LiveLoop(Chain(ENTRIES), os.fdopen(read_fd, "rb", buffering=0), io.BytesIO())
    threading.Timer(0.2, stop.set).start()
    started = time.monotonic()
    assert loop.run(stop) is StopReason.INTERRUPTED
    assert time.monotonic() - started < 1.0
    os.close(write_fd)


def test_input_timeout_counts_as_source_lost():
    read_fd, write_fd = os.pipe()
    loop = LiveLoop(Chain(ENTRIES), os.fdopen(read_fd, "rb", buffering=0), io.BytesIO(),
                    input_timeout_s=0.3)
    assert loop.run(threading.Event()) is StopReason.SOURCE_LOST
    os.close(write_fd)


class SlowChain:
    def process_block(self, block):
        time.sleep(0.02)  # twice the 10 ms block budget
        return block


def test_slow_processing_counts_late_and_discards_backlog():
    data = to_bytes(np.zeros(BLOCK * 20, dtype=np.float32))
    loop = LiveLoop(SlowChain(), io.BytesIO(data), io.BytesIO())
    loop.run(threading.Event())
    assert loop.stats.late_blocks > 0
    assert loop.stats.discarded_blocks > 0
    assert loop.stats.blocks + loop.stats.discarded_blocks == 20
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_live.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voxpipe.live'`

- [ ] **Step 3: Write the implementation**

`src/voxpipe/live.py`:
```python
"""The real-time loop: source bytes -> effect chain -> destination bytes."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import BinaryIO

import numpy as np

from voxpipe.audio import BLOCK_SIZE, SAMPLE_RATE

MAX_BACKLOG = 3  # blocks; anything older is discarded so latency never drifts (N-04)
INPUT_TIMEOUT_S = 2.0
_POLL_S = 0.05


class StopReason(Enum):
    INTERRUPTED = "interrupted"
    SOURCE_LOST = "source lost"
    DESTINATION_LOST = "destination lost"


@dataclass
class LoopStats:
    blocks: int = 0
    late_blocks: int = 0
    discarded_blocks: int = 0
    duration_s: float = 0.0


def _read_exact(reader: BinaryIO, size: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < size:
        chunk = reader.read(size - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


class LiveLoop:
    def __init__(
        self,
        chain,
        reader: BinaryIO,
        writer: BinaryIO,
        block_size: int = BLOCK_SIZE,
        input_timeout_s: float = INPUT_TIMEOUT_S,
    ) -> None:
        self.chain = chain
        self.reader = reader
        self.writer = writer
        self.block_size = block_size
        self.block_bytes = block_size * 4
        self.block_seconds = block_size / SAMPLE_RATE
        self.input_timeout_s = input_timeout_s
        self.stats = LoopStats()
        self._queue: queue.Queue[bytes | None] = queue.Queue()

    def _pump(self) -> None:
        while True:
            data = _read_exact(self.reader, self.block_bytes)
            self._queue.put(data)
            if data is None:
                return

    def run(self, stop: threading.Event) -> StopReason:
        started = time.monotonic()
        try:
            return self._run(stop)
        finally:
            self.stats.duration_s = time.monotonic() - started

    def _run(self, stop: threading.Event) -> StopReason:
        threading.Thread(target=self._pump, daemon=True).start()
        last_input = time.monotonic()
        while not stop.is_set():
            try:
                data = self._queue.get(timeout=_POLL_S)
            except queue.Empty:
                if time.monotonic() - last_input > self.input_timeout_s:
                    return StopReason.SOURCE_LOST
                continue
            last_input = time.monotonic()
            while data is not None and self._queue.qsize() > MAX_BACKLOG:
                self.stats.discarded_blocks += 1
                data = self._queue.get_nowait()
            if data is None:
                return StopReason.SOURCE_LOST

            began = time.perf_counter()
            out = self.chain.process_block(np.frombuffer(data, dtype="<f4"))
            if time.perf_counter() - began > self.block_seconds:
                self.stats.late_blocks += 1
            try:
                self.writer.write(np.asarray(out, dtype="<f4").tobytes())
                self.writer.flush()
            except (BrokenPipeError, OSError, ValueError):
                return StopReason.DESTINATION_LOST
            self.stats.blocks += 1
        return StopReason.INTERRUPTED
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_live.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/voxpipe/live.py tests/test_live.py
git commit -m "feat: real-time live loop with backlog cap and late-block accounting"
```

---

### Task 5: Child processes and the device watchdog

**Files:**
- Modify: `src/voxpipe/pw.py` (append)
- Test: `tests/test_watchdog.py`

**Interfaces:**
- Consumes: `Node`, `list_nodes`, `DeviceError` (Task 2), `BLOCK_SIZE` and `SAMPLE_RATE` (plan 1).
- Produces:
  - `capture_command(node: Node) -> list[str]`, `playback_command(node: Node) -> list[str]`.
  - `open_capture(node: Node) -> subprocess.Popen`. stdout is a raw f32 pipe.
  - `open_playback(node: Node) -> subprocess.Popen`. stdin is a raw f32 pipe.
  - `terminate(*processes) -> None`. Terminates, waits 1 s, then kills.
  - `DeviceWatchdog(source: Node, destination: Node, on_lost: Callable[[Node], None], lister: Callable[[], list[Node]] | None = None, interval_s: float = 1.0)`. A `threading.Thread` with `.stop()` and `.lost: Node | None`. It calls `on_lost(node)` once, for the first watched node that disappears, then exits. A `DeviceError` from the lister is ignored (transient).

  If the Task 1 spike required `pw-link`, change `playback_command` and `open_playback` here to match what was recorded in the tech spec.

- [ ] **Step 1: Write the failing tests**

`tests/test_watchdog.py`:
```python
import threading

from fakes import MIC, VIRTUAL

from voxpipe import pw
from voxpipe.pw import DeviceError, Node

SOURCE, DEST = Node(*MIC), Node(*VIRTUAL)


def test_capture_and_playback_commands():
    capture = pw.capture_command(SOURCE)
    assert capture[:3] == ["pw-record", "--target", SOURCE.name]
    assert capture[-1] == "-"
    assert "node.dont-reconnect = true" in " ".join(capture)
    for flag, value in (("--rate", "48000"), ("--channels", "1"), ("--format", "f32"), ("--latency", "480")):
        assert capture[capture.index(flag) + 1] == value
    playback = pw.playback_command(DEST)
    assert playback[:3] == ["pw-play", "--target", "voxpipe.virtual_mic"]
    assert playback[-1] == "-"


def run_watchdog(snapshots):
    lost = []
    done = threading.Event()
    it = iter(snapshots)

    def lister():
        item = next(it, snapshots[-1])
        if isinstance(item, Exception):
            raise item
        return item

    def on_lost(node):
        lost.append(node)
        done.set()

    dog = pw.DeviceWatchdog(SOURCE, DEST, on_lost, lister=lister, interval_s=0.01)
    dog.start()
    done.wait(1.0)
    dog.stop()
    dog.join(1.0)
    return lost, dog


def test_reports_source_loss():
    lost, dog = run_watchdog([[SOURCE, DEST], [DEST]])
    assert lost == [SOURCE] and dog.lost == SOURCE


def test_reports_destination_loss():
    lost, _ = run_watchdog([[SOURCE, DEST], [SOURCE]])
    assert lost == [DEST]


def test_ignores_transient_errors():
    lost, _ = run_watchdog([DeviceError("busy"), [SOURCE, DEST], [SOURCE]])
    assert lost == [DEST]


def test_stop_without_loss():
    dog = pw.DeviceWatchdog(SOURCE, DEST, lambda n: None, lister=lambda: [SOURCE, DEST],
                            interval_s=0.01)
    dog.start()
    dog.stop()
    dog.join(1.0)
    assert not dog.is_alive() and dog.lost is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_watchdog.py -v`
Expected: FAIL with `AttributeError: module 'voxpipe.pw' has no attribute 'capture_command'`

- [ ] **Step 3: Write the implementation**

Add `import threading` and `from voxpipe.audio import BLOCK_SIZE, SAMPLE_RATE` to `src/voxpipe/pw.py`'s imports, then append:
```python
_STREAM_FORMAT = [
    "--rate", str(SAMPLE_RATE), "--channels", "1", "--format", "f32",
    "--latency", str(BLOCK_SIZE),
]
# Never silently fall back to another device when the target disappears (R-20).
_NO_RECONNECT = ["-P", "{ node.dont-reconnect = true }"]


def capture_command(node: Node) -> list[str]:
    return ["pw-record", "--target", node.name, *_STREAM_FORMAT, *_NO_RECONNECT, "-"]


def playback_command(node: Node) -> list[str]:
    return ["pw-play", "--target", node.name, *_STREAM_FORMAT, *_NO_RECONNECT, "-"]


def _spawn(command: list[str], **pipes) -> subprocess.Popen:
    try:
        # New session: the terminal's Ctrl+C must reach only voxpipe, which then
        # shuts children down itself instead of seeing them die as "device lost".
        return subprocess.Popen(command, stderr=subprocess.DEVNULL, start_new_session=True,
                                **pipes)
    except FileNotFoundError:
        raise DeviceError(f"{command[0]} not found; is PipeWire installed?") from None


def open_capture(node: Node) -> subprocess.Popen:
    return _spawn(capture_command(node), stdout=subprocess.PIPE)


def open_playback(node: Node) -> subprocess.Popen:
    return _spawn(playback_command(node), stdin=subprocess.PIPE)


def terminate(*processes) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


class DeviceWatchdog(threading.Thread):
    """Polls PipeWire; when a watched node vanishes, reports it once and exits."""

    def __init__(
        self,
        source: Node,
        destination: Node,
        on_lost: Callable[[Node], None],
        lister: Callable[[], list[Node]] | None = None,
        interval_s: float = 1.0,
    ) -> None:
        super().__init__(daemon=True)
        self.watched = (source, destination)
        self.on_lost = on_lost
        self.lister = lister or list_nodes
        self.interval_s = interval_s
        self.lost: Node | None = None
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            try:
                present = {node.id for node in self.lister()}
            except DeviceError:
                continue
            for node in self.watched:
                if node.id not in present:
                    self.lost = node
                    self.on_lost(node)
                    return
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_watchdog.py tests/test_pw.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/voxpipe/pw.py tests/test_watchdog.py
git commit -m "feat(pw): capture/playback processes and device watchdog"
```

---

### Task 6: `voxpipe run`

**Files:**
- Create: `src/voxpipe/commands/run.py`
- Modify: `src/voxpipe/cli.py` (`COMMANDS`)
- Test: `tests/test_run_cmd.py`

**Interfaces:**
- Consumes:
  - `pw.list_nodes`, `resolve`, `find_virtual_mic`, `open_capture`, `open_playback`, `terminate`, `DeviceWatchdog` (Tasks 2–5);
  - `LiveLoop`, `StopReason` (Task 4);
  - `Chain`, `load_profile`, `resolve_profile_path` (plan 1).
- Produces: `voxpipe run --source ID|NAME --destination ID|NAME --profile NAME|PATH [--profiles-dir DIR]`.
  - Exit 0 after Ctrl+C. Exit 1 when a device is lost or on any error.
  - On stop, prints a summary line to stderr: `stopped (<reason>) after S.s s: B blocks, L late, D discarded`.

- [ ] **Step 1: Write the failing tests**

`tests/test_run_cmd.py`:
```python
import shutil
from pathlib import Path

import numpy as np
import pytest
from fakes import MIC, SPEAKERS, VIRTUAL, FakeProcess
from helpers import voice_like

from voxpipe import pw
from voxpipe.cli import main
from voxpipe.pw import Node

FIXTURE = Path(__file__).parent / "data" / "mechanicus.json"


class NullWatchdog:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        pass


@pytest.fixture
def system(monkeypatch):
    """Fake PipeWire with a mic, speakers and the virtual mic."""
    state = {"nodes": [Node(*MIC), Node(*SPEAKERS), Node(*VIRTUAL)], "processes": {}}
    audio = voice_like(0.3).astype("<f4").tobytes()

    def open_capture(node):
        state["processes"]["capture"] = FakeProcess(stdout=audio)
        return state["processes"]["capture"]

    def open_playback(node):
        state["processes"]["playback"] = FakeProcess()
        state["playback_target"] = node
        return state["processes"]["playback"]

    monkeypatch.setattr(pw, "list_nodes", lambda runner=None: list(state["nodes"]))
    monkeypatch.setattr(pw, "open_capture", open_capture)
    monkeypatch.setattr(pw, "open_playback", open_playback)
    monkeypatch.setattr(pw, "DeviceWatchdog", NullWatchdog)
    return state


def test_run_streams_audio_until_source_ends(system, capsys):
    code = main(["run", "--source", "57", "--destination", "voxpipe.virtual_mic",
                 "--profile", str(FIXTURE)])
    assert code == 1  # source EOF counts as source lost
    written = system["processes"]["playback"].stdin.getvalue()
    assert len(written) > 0 and np.any(np.frombuffer(written, dtype="<f4") != 0)
    assert system["playback_target"].id == 120
    err = capsys.readouterr().err
    assert "stopped (source lost)" in err
    assert "chain latency" in err


def test_run_profile_by_name(system, tmp_path):
    shutil.copy(FIXTURE, tmp_path / "robot.json")
    code = main(["run", "--source", "57", "--destination", "120", "--profile", "robot",
                 "--profiles-dir", str(tmp_path)])
    assert code == 1


def test_same_source_and_destination_refused(system, capsys):
    code = main(["run", "--source", "57", "--destination", "57", "--profile", str(FIXTURE)])
    assert code == 1
    assert "same device" in capsys.readouterr().err
    assert "capture" not in system["processes"]


def test_missing_virtual_mic_hint(system, capsys):
    system["nodes"] = [Node(*MIC), Node(*SPEAKERS)]
    code = main(["run", "--source", "57", "--destination", "voxpipe.virtual_mic",
                 "--profile", str(FIXTURE)])
    assert code == 1
    assert "voxpipe create" in capsys.readouterr().err


def test_bad_profile_fails_before_audio(system, tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text('{"format": "voxpipe-profile", "version": 7}')
    code = main(["run", "--source", "57", "--destination", "120", "--profile", str(bad)])
    assert code == 1
    assert "newer" in capsys.readouterr().err
    assert "capture" not in system["processes"]


def test_profile_is_required(system):
    with pytest.raises(SystemExit) as exc:
        main(["run", "--source", "57", "--destination", "120"])
    assert exc.value.code == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_run_cmd.py -v`
Expected: FAIL with `argument command: invalid choice: 'run'` (SystemExit 2)

- [ ] **Step 3: Write the implementation**

`src/voxpipe/commands/run.py`:
```python
"""`voxpipe run`: the live effect loop (R-03, R-07, R-11, R-20, R-21, R-22)."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path

from voxpipe import pw
from voxpipe.audio import SAMPLE_RATE
from voxpipe.engine.chain import Chain
from voxpipe.live import LiveLoop, StopReason
from voxpipe.profile import load_profile, resolve_profile_path
from voxpipe.pw import DeviceError, Node


def register(subparsers) -> None:
    parser = subparsers.add_parser("run", help="run the live effect loop")
    parser.add_argument("--source", required=True, help="microphone id or name (see `list`)")
    parser.add_argument("--destination", required=True,
                        help="destination id or name, normally voxpipe.virtual_mic")
    parser.add_argument("--profile", required=True, help="profile name or path to a .json file")
    parser.add_argument("--profiles-dir", type=Path, default=Path("."),
                        help="where profile names are looked up (default: current directory)")
    parser.set_defaults(handler=run)


def _resolve_devices(source_arg: str, destination_arg: str) -> tuple[Node, Node]:
    nodes = pw.list_nodes()
    source = pw.resolve(source_arg, nodes)
    try:
        destination = pw.resolve(destination_arg, nodes)
    except DeviceError as exc:
        if pw.find_virtual_mic(nodes) is None:
            raise DeviceError(f"{exc}; virtual microphone not found, run `voxpipe create`") from None
        raise
    if source.id == destination.id:
        raise DeviceError("source and destination are the same device (would feed back)")
    return source, destination


def run(args: argparse.Namespace) -> int:
    profile = load_profile(resolve_profile_path(args.profile, args.profiles_dir))
    chain = Chain(profile.chain)
    source, destination = _resolve_devices(args.source, args.destination)

    capture = pw.open_capture(source)
    playback = pw.open_playback(destination)
    processes = {source.id: capture, destination.id: playback}
    watchdog = pw.DeviceWatchdog(source, destination, lambda node: processes[node.id].kill())
    stop = threading.Event()
    previous_handler = signal.signal(signal.SIGINT, lambda *_: stop.set())
    loop = LiveLoop(chain, capture.stdout, playback.stdin)

    latency_ms = chain.latency_samples / SAMPLE_RATE * 1000
    print(f"running: {source.name} -> {destination.name}, profile '{profile.name}' "
          f"(chain latency {latency_ms:.1f} ms); Ctrl+C to stop", file=sys.stderr)
    watchdog.start()
    try:
        reason = loop.run(stop)
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        watchdog.stop()
        pw.terminate(capture, playback)

    s = loop.stats
    print(f"stopped ({reason.value}) after {s.duration_s:.1f} s: {s.blocks} blocks, "
          f"{s.late_blocks} late, {s.discarded_blocks} discarded", file=sys.stderr)
    return 0 if reason is StopReason.INTERRUPTED else 1
```

In `src/voxpipe/cli.py`:
```python
from voxpipe.commands import devices, render, run

COMMANDS: tuple = (devices, run, render)
```

`playback.stdin` writes may block if PipeWire stops consuming. The watchdog then kills `pw-play`, and the blocked write raises `BrokenPipeError`, which the loop already handles.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -v && uv run ruff check`
Expected: all pass; `All checks passed!`

- [ ] **Step 5: Manual end-to-end check (hardware)**

```bash
uv run voxpipe create
uv run voxpipe list
uv run voxpipe run --source <your mic id> --destination voxpipe.virtual_mic --profile tests/data/mechanicus.json
```
While it runs:
1. In another terminal, run `pw-record --target voxpipe.virtual_mic /tmp/check.wav`, speak, press Ctrl+C, and play `/tmp/check.wav`. Expected: your effected voice.
2. Select "voxpipe virtual microphone" as the input in Telegram's settings and test a call.
3. Press Ctrl+C in the `run` terminal. Expected: `stopped (interrupted) …`, exit 0.
4. Start `run` again, then run `uv run voxpipe clean` in another terminal. Expected: `run` stops within about 1 s with `stopped (destination lost)`, exit 1.
5. Start `run` again (after `create`), then unplug the source mic. Expected: `stopped (source lost)`, exit 1. It must not keep running on another mic.

Record any failures as new tasks. Don't patch silently.

- [ ] **Step 6: Commit**

```bash
git add src/voxpipe/commands/run.py src/voxpipe/cli.py tests/test_run_cmd.py
git commit -m "feat: voxpipe run live effect loop"
```

---

## Self-review notes

- **Spec coverage:**
  - R-08 (Task 2).
  - R-13, R-17, R-18, R-19 (Task 3; reboot loss follows from PipeWire state, not voxpipe).
  - R-03, R-07, R-11, R-22 (Task 6).
  - R-20, A-07 (Tasks 4–5).
  - R-21: no monitoring path exists.
  - N-04, N-07 (Task 4).
  - C-04, C-05 (Tasks 2–6).
  - K-01 and A-04 are retired by the Task 1 spike.
- **Known gap:** `--verbose` per-second stats (tech spec section 10) aren't implemented. The exit summary covers dropouts. Add it as a follow-up if needed.
