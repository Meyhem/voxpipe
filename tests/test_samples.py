import pytest

from voxpipe.tuning.samples import SampleError, SamplePair, find_pairs


def touch(folder, *names):
    for name in names:
        (folder / name).write_bytes(b"")


def test_pairs_matched_by_id_any_extension(tmp_path):
    touch(tmp_path, "02-target.ogg", "02-reference.wav", "01-target.wav", "01-reference.mp3")
    pairs, warnings = find_pairs(tmp_path)
    assert pairs == [
        SamplePair("01", tmp_path / "01-target.wav", tmp_path / "01-reference.mp3"),
        SamplePair("02", tmp_path / "02-target.ogg", tmp_path / "02-reference.wav"),
    ]
    assert warnings == []


def test_incomplete_pairs_skipped_with_warning(tmp_path):
    touch(tmp_path, "a-target.wav", "a-reference.wav", "b-target.wav", "c-reference.wav")
    pairs, warnings = find_pairs(tmp_path)
    assert [p.id for p in pairs] == ["a"]
    assert any("'b'" in w and "reference" in w for w in warnings)
    assert any("'c'" in w and "target" in w for w in warnings)


def test_ignores_hidden_and_unrelated_files(tmp_path):
    touch(tmp_path, "x-target.wav", "x-reference.wav", ".x-target.wav", "notes.txt", "profile.json")
    pairs, warnings = find_pairs(tmp_path)
    assert [p.id for p in pairs] == ["x"] and warnings == []


def test_id_may_contain_dashes(tmp_path):
    touch(tmp_path, "tech-priest-1-target.wav", "tech-priest-1-reference.wav")
    assert find_pairs(tmp_path)[0][0].id == "tech-priest-1"


def test_duplicate_role_is_an_error(tmp_path):
    touch(tmp_path, "a-target.wav", "a-target.mp3", "a-reference.wav")
    with pytest.raises(SampleError, match="more than one target"):
        find_pairs(tmp_path)


def test_no_complete_pairs(tmp_path):
    touch(tmp_path, "a-target.wav")
    with pytest.raises(SampleError, match="no complete sample pairs"):
        find_pairs(tmp_path)


def test_missing_folder(tmp_path):
    with pytest.raises(SampleError, match="not found"):
        find_pairs(tmp_path / "nope")
