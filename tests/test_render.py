import shutil
from pathlib import Path

import numpy as np
import pytest

from helpers import voice_like
from voxpipe.cli import main
from voxpipe.engine.chain import Chain
from voxpipe.media import decode, write_wav
from voxpipe.profile import load_profile

DATA = Path(__file__).parent / "data"
GOLDEN = Path(__file__).parent / "golden" / "mechanicus.npy"
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_golden_render_unchanged():
    """Guards the sound of every effect. Regenerate with scripts/update_golden.py
    only when a change in sound is intended."""
    out = Chain(load_profile(DATA / "mechanicus.json").chain).process_signal(voice_like())
    golden = np.load(GOLDEN)
    diff = np.abs(out - golden)
    assert out.shape == golden.shape
    assert np.mean(diff > 1e-4) < 0.001
    assert diff.max() < 0.01


@needs_ffmpeg
def test_render_writes_effected_wav(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "in.wav"
    write_wav(source, voice_like(1.0))
    code = main(["render", str(source), "--profile", str(DATA / "mechanicus.json")])
    assert code == 0
    output = tmp_path / "in.voxpipe.wav"
    rendered = decode(output)
    expected = Chain(load_profile(DATA / "mechanicus.json").chain).process_signal(decode(source))
    np.testing.assert_allclose(rendered, np.clip(expected, -1, 1), atol=2e-3)


@needs_ffmpeg
def test_render_profile_by_name_from_profiles_dir(tmp_path):
    source = tmp_path / "in.wav"
    write_wav(source, voice_like(0.5))
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    shutil.copy(DATA / "mechanicus.json", profiles / "robot.json")
    output = tmp_path / "out" / "x.wav"
    code = main([
        "render", str(source), "--profile", "robot",
        "--profiles-dir", str(profiles), "-o", str(output),
    ])
    assert code == 0
    assert output.is_file()


def test_render_invalid_profile_fails_before_decoding(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    code = main(["render", str(tmp_path / "missing.wav"), "--profile", str(bad)])
    assert code == 1
    assert "not a voxpipe profile" in capsys.readouterr().err
