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
