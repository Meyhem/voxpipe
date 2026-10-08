"""Sample pairs: <id>-target.* plus <id>-reference.* (R-09, R-16)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from voxpipe.errors import VoxpipeError

_PATTERN = re.compile(r"^(?P<id>.+)-(?P<role>target|reference)$")


class SampleError(VoxpipeError):
    pass


@dataclass(frozen=True)
class SamplePair:
    id: str
    target: Path
    reference: Path


def find_pairs(folder: Path) -> tuple[list[SamplePair], list[str]]:
    if not folder.is_dir():
        raise SampleError(f"{folder}: samples folder not found")
    found: dict[str, dict[str, Path]] = {}
    for path in sorted(folder.iterdir()):
        if path.name.startswith(".") or not path.is_file():
            continue
        match = _PATTERN.match(path.stem)
        if match is None:
            continue
        roles = found.setdefault(match["id"], {})
        if match["role"] in roles:
            raise SampleError(
                f"pair '{match['id']}' has more than one {match['role']} file: "
                f"{roles[match['role']].name}, {path.name}"
            )
        roles[match["role"]] = path

    pairs, warnings = [], []
    for pair_id in sorted(found):
        roles = found[pair_id]
        if "target" in roles and "reference" in roles:
            pairs.append(SamplePair(pair_id, roles["target"], roles["reference"]))
        else:
            missing = "reference" if "target" in roles else "target"
            warnings.append(f"skipping pair '{pair_id}': no {missing} recording")
    if not pairs:
        raise SampleError(f"{folder}: no complete sample pairs (<id>-target + <id>-reference)")
    return pairs, warnings
