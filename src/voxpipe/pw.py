"""Device layer: the only module that talks to PipeWire (C-04, D-02, D-03)."""

from __future__ import annotations

import json
import subprocess
import time
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
