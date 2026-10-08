"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from voxpipe import __version__
from voxpipe.commands import render
from voxpipe.errors import VoxpipeError

COMMANDS: tuple = (render,)


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
