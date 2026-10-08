import shutil
from pathlib import Path

import numpy as np
import pytest

from fakes import MIC, SPEAKERS, VIRTUAL, FakeProcess
from helpers import voice_like
from voxpipe import pw
from voxpipe.cli import main
from voxpipe.pw import Node

FIXTURE = Path(__file__).parent / "data" / "mechanicus.json"


class NullWatchdog:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        pass


@pytest.fixture
def system(monkeypatch):
    """Fake PipeWire with a mic, speakers and the virtual mic."""
    state = {"nodes": [Node(*MIC), Node(*SPEAKERS), Node(*VIRTUAL)], "processes": {}}
    audio = voice_like(0.3).astype("<f4").tobytes()

    def open_capture(node):
        state["processes"]["capture"] = FakeProcess(stdout=audio)
        return state["processes"]["capture"]

    def open_playback(node):
        state["processes"]["playback"] = FakeProcess()
        state["playback_target"] = node
        return state["processes"]["playback"]

    monkeypatch.setattr(pw, "list_nodes", lambda runner=None: list(state["nodes"]))
    monkeypatch.setattr(pw, "open_capture", open_capture)
    monkeypatch.setattr(pw, "open_playback", open_playback)
    monkeypatch.setattr(pw, "DeviceWatchdog", NullWatchdog)
    return state


def test_run_streams_audio_until_source_ends(system, capsys):
    code = main(["run", "--source", "57", "--destination", "voxpipe.virtual_mic",
                 "--profile", str(FIXTURE)])
    assert code == 1  # source EOF counts as source lost
    written = system["processes"]["playback"].stdin.getvalue()
    assert len(written) > 0 and np.any(np.frombuffer(written, dtype="<f4") != 0)
    assert system["playback_target"].id == 120
    err = capsys.readouterr().err
    assert "stopped (source lost)" in err
    assert "chain latency" in err


def test_run_profile_by_name(system, tmp_path):
    shutil.copy(FIXTURE, tmp_path / "robot.json")
    code = main(["run", "--source", "57", "--destination", "120", "--profile", "robot",
                 "--profiles-dir", str(tmp_path)])
    assert code == 1


def test_same_source_and_destination_refused(system, capsys):
    code = main(["run", "--source", "57", "--destination", "57", "--profile", str(FIXTURE)])
    assert code == 1
    assert "same device" in capsys.readouterr().err
    assert "capture" not in system["processes"]


def test_missing_virtual_mic_hint(system, capsys):
    system["nodes"] = [Node(*MIC), Node(*SPEAKERS)]
    code = main(["run", "--source", "57", "--destination", "voxpipe.virtual_mic",
                 "--profile", str(FIXTURE)])
    assert code == 1
    assert "voxpipe create" in capsys.readouterr().err


def test_bad_profile_fails_before_audio(system, tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text('{"format": "voxpipe-profile", "version": 7}')
    code = main(["run", "--source", "57", "--destination", "120", "--profile", str(bad)])
    assert code == 1
    assert "newer" in capsys.readouterr().err
    assert "capture" not in system["processes"]


def test_profile_is_required(system):
    with pytest.raises(SystemExit) as exc:
        main(["run", "--source", "57", "--destination", "120"])
    assert exc.value.code == 2
