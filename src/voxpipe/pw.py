"""Device layer: the only module that talks to PipeWire (C-04, D-02, D-03)."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from voxpipe.audio import SAMPLE_RATE
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


# --latency 120 (2.5 ms) gave a ~10 ms record/play hop; 480 gave ~32 ms (spike, D-03).
# --volume 1.0: WirePlumber otherwise restored a 0.185 stream volume (-14 dB).
STREAM_LATENCY = 120
_STREAM_FORMAT = [
    "--rate", str(SAMPLE_RATE), "--channels", "1", "--format", "f32",
    "--latency", str(STREAM_LATENCY), "--volume", "1.0",
]
PLAYBACK_STREAM_NAME = "voxpipe-out"
# Never silently fall back to another device when the target disappears (R-20).
_NO_RECONNECT = ["-P", "{ node.dont-reconnect = true }"]


def capture_command(node: Node) -> list[str]:
    return ["pw-record", "--target", node.name, *_STREAM_FORMAT, *_NO_RECONNECT, "-"]


def playback_command() -> list[str]:
    # `pw-play --target <source/virtual node>` does not link (it falls back to the default
    # sink), so start unlinked (--target 0) under a fixed name and link by hand.
    props = f"{{ node.name={PLAYBACK_STREAM_NAME} node.dont-reconnect = true }}"
    return ["pw-play", "--target", "0", *_STREAM_FORMAT, "-P", props, "-"]


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


def link_playback(
    node: Node,
    runner: Runner | None = None,
    wait_s: float = 3.0,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Link our playback stream's output to every input port of `node`."""
    run = runner or run_tool
    out_port = f"{PLAYBACK_STREAM_NAME}:output_MONO"
    interval = 0.05
    for _ in range(max(1, int(wait_s / interval))):
        outputs = run(["pw-link", "-o"]).splitlines()
        inputs = [p for p in run(["pw-link", "-i"]).splitlines() if p.startswith(f"{node.name}:")]
        if out_port in outputs and inputs:
            for port in inputs:
                run(["pw-link", out_port, port])
            return
        sleep(interval)
    raise DeviceError(f"could not link audio output to {node.name}")


def open_playback(node: Node) -> subprocess.Popen:
    process = _spawn(playback_command(), stdin=subprocess.PIPE)
    try:
        link_playback(node)
    except BaseException:
        terminate(process)
        raise
    return process


def terminate(*processes) -> None:
    for process in processes:
        if isinstance(process, subprocess.Popen) and process.stdin is not None:
            try:
                process.stdin.close()  # flush errors from a dead reader are expected here
            except OSError:
                pass
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
