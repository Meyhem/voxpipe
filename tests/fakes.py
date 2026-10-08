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
