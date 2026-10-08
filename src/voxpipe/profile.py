"""Profiles: strict, versioned JSON effect configurations (D-04, D-08, R-15)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match

from voxpipe import __version__
from voxpipe.engine.entry import ChainEntry
from voxpipe.engine.registry import EFFECTS
from voxpipe.errors import VoxpipeError
from voxpipe.profile_schema import CURRENT_VERSION, FORMAT, envelope_schema, params_schema

# Maps a version to the function that upgrades a profile dict from it to version + 1.
MIGRATIONS: dict[int, Callable[[dict], dict]] = {}


class ProfileError(VoxpipeError):
    pass


@dataclass(frozen=True)
class Provenance:
    tool_version: str
    seed: int | None
    pairs: tuple[str, ...]
    score: float | None
    partial: bool
    duration_s: float | None
    objective_version: int | None

    @classmethod
    def manual(cls) -> Provenance:
        return cls(__version__, None, (), None, False, None, None)


@dataclass(frozen=True)
class Profile:
    name: str
    created: str
    provenance: Provenance
    chain: tuple[ChainEntry, ...]


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _format_path(parts: Iterable[str | int]) -> str:
    out = "profile"
    for part in parts:
        out += f"[{part}]" if isinstance(part, int) else f".{part}"
    return out


@cache
def _envelope_validator() -> Draft202012Validator:
    return Draft202012Validator(envelope_schema())


@cache
def _params_validator(effect: str) -> Draft202012Validator:
    return Draft202012Validator(params_schema(EFFECTS[effect]))


def _check(validator: Draft202012Validator, data: object, prefix: list[str | int]) -> None:
    error = best_match(validator.iter_errors(data))
    if error is not None:
        raise ProfileError(f"{_format_path([*prefix, *error.absolute_path])}: {error.message}")


def _migrate(data: dict) -> dict:
    while data["version"] < CURRENT_VERSION:
        data = MIGRATIONS[data["version"]](data)
    return data


def parse_profile(data: object) -> Profile:
    if not isinstance(data, dict):
        raise ProfileError("profile: expected a JSON object")
    if data.get("format") != FORMAT:
        raise ProfileError("profile: not a voxpipe profile (missing or wrong 'format')")
    version = data.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ProfileError("profile.version: must be a positive integer")
    if version > CURRENT_VERSION:
        raise ProfileError(
            f"profile.version: version {version} is newer than this voxpipe supports "
            f"({CURRENT_VERSION}); upgrade voxpipe"
        )
    data = _migrate(data)
    _check(_envelope_validator(), data, [])
    for index, item in enumerate(data["chain"]):
        _check(_params_validator(item["effect"]), item["params"], ["chain", index, "params"])

    prov = data["provenance"]
    return Profile(
        name=data["name"],
        created=data["created"],
        provenance=Provenance(
            tool_version=prov["tool_version"],
            seed=prov["seed"],
            pairs=tuple(prov["pairs"]),
            score=prov["score"],
            partial=prov["partial"],
            duration_s=prov["duration_s"],
            objective_version=prov["objective_version"],
        ),
        chain=tuple(
            ChainEntry(item["effect"], {k: float(v) for k, v in item["params"].items()})
            for item in data["chain"]
        ),
    )


def to_dict(profile: Profile) -> dict:
    prov = profile.provenance
    return {
        "format": FORMAT,
        "version": CURRENT_VERSION,
        "name": profile.name,
        "created": profile.created,
        "provenance": {
            "tool_version": prov.tool_version,
            "seed": prov.seed,
            "pairs": list(prov.pairs),
            "score": prov.score,
            "partial": prov.partial,
            "duration_s": prov.duration_s,
            "objective_version": prov.objective_version,
        },
        "chain": [{"effect": e.effect, "params": dict(e.params)} for e in profile.chain],
    }
