import copy
import json
from pathlib import Path

import pytest

from voxpipe import profile as profile_module
from voxpipe.engine.entry import ChainEntry
from voxpipe.profile import (
    ProfileError,
    Provenance,
    load_profile,
    parse_profile,
    resolve_profile_path,
    save_profile,
    to_dict,
)

FIXTURE = Path(__file__).parent / "data" / "mechanicus.json"


@pytest.fixture
def data() -> dict:
    return json.loads(FIXTURE.read_text())


def error_for(data: object) -> str:
    with pytest.raises(ProfileError) as exc:
        parse_profile(data)
    return str(exc.value)


def test_valid_fixture_parses(data):
    profile = parse_profile(data)
    assert profile.name == "mechanicus-fixture"
    assert profile.chain[0] == ChainEntry("gain", {"gain_db": 3.0})
    assert profile.provenance == Provenance(
        tool_version="0.1.0", seed=None, pairs=(), score=None,
        partial=False, duration_s=None, objective_version=None,
    )


def test_round_trip(data):
    assert to_dict(parse_profile(data)) == data


def test_not_an_object():
    assert "expected a JSON object" in error_for([1, 2])


def test_wrong_format(data):
    data["format"] = "something-else"
    assert "not a voxpipe profile" in error_for(data)


def test_newer_version_refused(data):
    data["version"] = 2
    assert "newer" in error_for(data)


def test_bad_version(data):
    data["version"] = "1"
    assert "profile.version" in error_for(data)


def test_missing_param_names_field(data):
    del data["chain"][0]["params"]["gain_db"]
    message = error_for(data)
    assert "profile.chain[0].params" in message
    assert "gain_db" in message


def test_out_of_range_names_field(data):
    data["chain"][4]["params"]["freq_hz"] = 5000.0
    message = error_for(data)
    assert "profile.chain[4].params.freq_hz" in message
    assert "maximum" in message


def test_unknown_param_rejected(data):
    data["chain"][0]["params"]["volume"] = 1.0
    assert "volume" in error_for(data)


def test_unknown_effect_rejected(data):
    data["chain"][0]["effect"] = "flanger"
    assert "profile.chain[0].effect" in error_for(data)


def test_unknown_top_level_key_rejected(data):
    data["comment"] = "hi"
    assert "comment" in error_for(data)


def test_boolean_is_not_a_number(data):
    data["chain"][0]["params"]["gain_db"] = True
    assert "profile.chain[0].params.gain_db" in error_for(data)


def test_missing_provenance_field(data):
    broken = copy.deepcopy(data)
    del broken["provenance"]["seed"]
    assert "profile.provenance" in error_for(broken)


def test_empty_chain_rejected(data):
    data["chain"] = []
    assert "profile.chain" in error_for(data)


def test_load_fixture():
    assert load_profile(FIXTURE).name == "mechanicus-fixture"


def test_load_missing_file(tmp_path):
    with pytest.raises(ProfileError, match="profile not found"):
        load_profile(tmp_path / "nope.json")


def test_load_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(ProfileError, match="invalid JSON"):
        load_profile(path)


def test_load_error_is_prefixed_with_path(tmp_path, data):
    data["version"] = 9
    path = tmp_path / "future.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ProfileError, match=str(path)):
        load_profile(path)


def test_save_then_load_round_trip(tmp_path):
    original = load_profile(FIXTURE)
    path = tmp_path / "sub" / "copy.json"
    save_profile(original, path)
    assert load_profile(path) == original
    assert list(path.parent.iterdir()) == [path]  # no temp file left behind


def test_failed_save_keeps_old_file(tmp_path, monkeypatch):
    path = tmp_path / "keep.json"
    path.write_text("old")

    def explode(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(profile_module.json, "dump", explode)
    with pytest.raises(RuntimeError):
        save_profile(load_profile(FIXTURE), path)
    assert path.read_text() == "old"
    assert list(tmp_path.iterdir()) == [path]


def test_resolve_profile_path(tmp_path):
    assert resolve_profile_path("robot", tmp_path) == tmp_path / "robot.json"
    assert resolve_profile_path("robot.json", tmp_path) == Path("robot.json")
    assert resolve_profile_path("dir/robot", tmp_path) == Path("dir/robot")
