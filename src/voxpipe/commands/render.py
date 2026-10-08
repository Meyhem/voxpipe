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
