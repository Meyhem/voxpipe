"""`voxpipe tune`: fit effect settings to sample pairs and save a profile (R-06)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from voxpipe import __version__
from voxpipe.errors import VoxpipeError
from voxpipe.profile import Profile, Provenance, now_iso, save_profile
from voxpipe.tuning.objective import OBJECTIVE_VERSION, prepare_pair
from voxpipe.tuning.samples import find_pairs
from voxpipe.tuning.search import TuneSettings, tune

EXIT_INTERRUPTED = 130


def register(subparsers) -> None:
    parser = subparsers.add_parser("tune", help="fit effect settings to sample pairs")
    parser.add_argument("--name", required=True, help="profile name (saved as <name>.json)")
    parser.add_argument("--samples", type=Path, default=Path("."),
                        help="folder with <id>-target / <id>-reference files (default: .)")
    parser.add_argument("--profiles-dir", type=Path, default=Path("."),
                        help="where the profile is written (default: current directory)")
    parser.add_argument("--evaluations", type=int, default=TuneSettings.evaluations,
                        help="search budget (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=TuneSettings.seed)
    parser.add_argument("--workers", type=int, default=0, help="processes (default: all CPUs)")
    parser.set_defaults(handler=run)


def _say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class _Progress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.last = 0.0

    def __call__(self, done: int, score: float) -> None:
        now = time.monotonic()
        if now - self.last >= 1.0 or done >= self.total:
            self.last = now
            _say(f"  {min(done, self.total)}/{self.total} evaluations, best score {score:.1f}")


def run(args: argparse.Namespace) -> int:
    if Path(args.name).name != args.name or args.name in ("", ".", ".."):
        raise VoxpipeError("--name must be a plain file name without directories")
    pairs, warnings = find_pairs(args.samples)
    prepared = []
    for pair in pairs:
        data, pair_warnings = prepare_pair(pair)
        warnings.extend(pair_warnings)
        prepared.append(data)
    for warning in warnings:
        _say(f"warning: {warning}")

    _say(f"tuning '{args.name}' on {len(prepared)} pair(s), {args.evaluations} evaluations "
         "(Ctrl+C saves the best so far)")
    settings = TuneSettings(evaluations=args.evaluations, seed=args.seed, workers=args.workers)
    result = tune(prepared, settings, progress=_Progress(args.evaluations))

    profile = Profile(
        name=args.name,
        created=now_iso(),
        provenance=Provenance(
            tool_version=__version__,
            seed=args.seed,
            pairs=tuple(p.id for p in prepared),
            score=round(result.score, 2),
            partial=result.partial,
            duration_s=round(result.duration_s, 1),
            objective_version=OBJECTIVE_VERSION,
        ),
        chain=tuple(result.entries),
    )
    path = args.profiles_dir / f"{args.name}.json"
    save_profile(profile, path)
    _say(f"saved {path} (score {result.score:.1f}{', partial' if result.partial else ''})")
    return EXIT_INTERRUPTED if result.partial else 0
