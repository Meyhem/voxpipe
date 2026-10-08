"""JSON schemas for profiles. Effect param schemas are generated from Param
declarations, so the engine and the schema cannot drift apart (D-08)."""

from voxpipe.engine.effect import Effect
from voxpipe.engine.registry import EFFECTS

FORMAT = "voxpipe-profile"
CURRENT_VERSION = 1

_NULLABLE_NUMBER = {"type": ["number", "null"]}


def envelope_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["format", "version", "name", "created", "provenance", "chain"],
        "properties": {
            "format": {"const": FORMAT},
            "version": {"const": CURRENT_VERSION},
            "name": {"type": "string", "minLength": 1},
            "created": {"type": "string", "minLength": 1},
            "provenance": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "tool_version", "seed", "pairs", "score",
                    "partial", "duration_s", "objective_version",
                ],
                "properties": {
                    "tool_version": {"type": "string"},
                    "seed": {"type": ["integer", "null"]},
                    "pairs": {"type": "array", "items": {"type": "string"}},
                    "score": _NULLABLE_NUMBER,
                    "partial": {"type": "boolean"},
                    "duration_s": {"type": ["number", "null"], "minimum": 0},
                    "objective_version": {"type": ["integer", "null"]},
                },
            },
            "chain": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["effect", "params"],
                    "properties": {
                        "effect": {"enum": sorted(EFFECTS)},
                        "params": {"type": "object"},
                    },
                },
            },
        },
    }


def params_schema(effect_cls: type[Effect]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [p.name for p in effect_cls.params],
        "properties": {
            p.name: {"type": "number", "minimum": p.minimum, "maximum": p.maximum}
            for p in effect_cls.params
        },
    }
