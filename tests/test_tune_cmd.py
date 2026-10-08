import shutil

import pytest

from helpers import voice_like
from voxpipe.cli import main
from voxpipe.engine.chain import Chain
from voxpipe.engine.entry import ChainEntry
from voxpipe.engine.registry import TUNING_CHAIN
from voxpipe.media import write_wav
from voxpipe.profile import load_profile
from voxpipe.tuning import search
from voxpipe.tuning.objective import OBJECTIVE_VERSION

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture
def samples(tmp_path):
    folder = tmp_path / "samples"
    folder.mkdir()
    reference = voice_like(1.0)
    target = Chain([ChainEntry("gain", {"gain_db": -6.0})]).process_signal(reference)
    write_wav(folder / "01-reference.wav", reference)
    write_wav(folder / "01-target.wav", target)
    write_wav(folder / "02-target.wav", target)  # incomplete pair
    return folder


def run_tune(samples, profiles, *extra):
    return main([
        "tune", "--name", "robot", "--samples", str(samples), "--profiles-dir", str(profiles),
        "--evaluations", "48", "--workers", "1", *extra,
    ])


def test_tune_saves_valid_profile(samples, tmp_path, capsys):
    profiles = tmp_path / "profiles"
    assert run_tune(samples, profiles) == 0
    profile = load_profile(profiles / "robot.json")
    assert profile.name == "robot"
    assert [e.effect for e in profile.chain] == list(TUNING_CHAIN)
    prov = profile.provenance
    assert prov.pairs == ("01",) and prov.seed == 1 and prov.objective_version == OBJECTIVE_VERSION
    assert prov.partial is False and 0 < prov.score <= 100
    err = capsys.readouterr().err
    assert "skipping pair '02'" in err
    assert "saved" in err and "robot.json" in err


def test_tune_overwrites_existing_profile(samples, tmp_path):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "robot.json").write_text("old")
    assert run_tune(samples, profiles) == 0
    assert load_profile(profiles / "robot.json").name == "robot"


def test_tune_is_deterministic(samples, tmp_path):
    run_tune(samples, tmp_path / "a")
    run_tune(samples, tmp_path / "b")
    a = load_profile(tmp_path / "a" / "robot.json")
    b = load_profile(tmp_path / "b" / "robot.json")
    assert a.chain == b.chain and a.provenance.score == b.provenance.score


def test_interrupt_saves_partial_and_exits_130(samples, tmp_path, monkeypatch):
    real_tune = search.tune

    def interrupting_tune(pairs, settings, progress=None, template=None):
        def stop_after_first(done, score):
            raise KeyboardInterrupt

        return real_tune(pairs, settings, progress=stop_after_first, template=template)

    monkeypatch.setattr("voxpipe.commands.tune.tune", interrupting_tune)
    assert run_tune(samples, tmp_path / "p") == 130
    assert load_profile(tmp_path / "p" / "robot.json").provenance.partial is True


def test_no_pairs_is_an_error(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run_tune(empty, tmp_path) == 1
    assert "no complete sample pairs" in capsys.readouterr().err


def test_name_must_be_a_plain_file_name(samples, tmp_path, capsys):
    code = main(["tune", "--name", "../evil", "--samples", str(samples),
                 "--profiles-dir", str(tmp_path)])
    assert code == 1
    assert "--name" in capsys.readouterr().err
