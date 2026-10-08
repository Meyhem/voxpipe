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
