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
