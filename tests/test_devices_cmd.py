import pytest

from fakes import MIC, SPEAKERS, VIRTUAL, make_dump
from voxpipe import pw
from voxpipe.cli import main
from voxpipe.pw import DeviceError


class FakePipeWire:
    """Scriptable stand-in for pw.run_tool."""

    def __init__(self, nodes, appear_on_create=True):
        self.nodes = list(nodes)
        self.appear_on_create = appear_on_create
        self.calls = []

    def __call__(self, args):
        self.calls.append(args)
        if args == ["pw-dump"]:
            return make_dump(self.nodes)
        if args[:2] == ["pw-cli", "create-node"]:
            if self.appear_on_create:
                self.nodes.append(VIRTUAL)
            return ""
        if args[:2] == ["pw-cli", "destroy"]:
            self.nodes = [n for n in self.nodes if str(n[0]) != args[2]]
            return ""
        raise AssertionError(f"unexpected call {args}")


@pytest.fixture
def fake_pw(monkeypatch):
    def install(nodes, **kwargs):
        fake = FakePipeWire(nodes, **kwargs)
        monkeypatch.setattr(pw, "run_tool", fake)
        return fake

    return install


def test_create_makes_lingering_virtual_source(fake_pw):
    fake = fake_pw([MIC])
    node, created = pw.create_virtual_mic(sleep=lambda s: None)
    assert created and node.id == 120
    create_call = next(c for c in fake.calls if c[:2] == ["pw-cli", "create-node"])
    props = create_call[3]
    for fragment in (
        "factory.name=support.null-audio-sink",
        "node.name=voxpipe.virtual_mic",
        "media.class=Audio/Source/Virtual",
        "object.linger=true",
    ):
        assert fragment in props


def test_create_reuses_existing(fake_pw):
    fake = fake_pw([MIC, VIRTUAL])
    node, created = pw.create_virtual_mic(sleep=lambda s: None)
    assert not created and node.id == 120
    assert not any(c[:2] == ["pw-cli", "create-node"] for c in fake.calls)


def test_create_times_out_when_node_never_appears(fake_pw):
    fake_pw([MIC], appear_on_create=False)
    with pytest.raises(DeviceError, match="did not appear"):
        pw.create_virtual_mic(wait_s=0.3, sleep=lambda s: None)


def test_remove(fake_pw):
    fake = fake_pw([MIC, VIRTUAL])
    assert pw.remove_virtual_mic().id == 120
    assert ["pw-cli", "destroy", "120"] in fake.calls
    assert pw.remove_virtual_mic() is None


def test_list_command(fake_pw, capsys):
    fake_pw([MIC, SPEAKERS, VIRTUAL])
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "57" in out and "HyperX Headset" in out
    assert "alsa_output" not in out  # sinks are not microphones
    assert "voxpipe virtual microphone" in out and "[virtual mic]" in out


def test_create_command_twice(fake_pw, capsys):
    fake_pw([MIC])
    assert main(["create"]) == 0
    assert main(["create"]) == 0
    out = capsys.readouterr().out
    assert "created virtual microphone (id 120)" in out
    assert "already exists (id 120)" in out


def test_clean_command(fake_pw, capsys):
    fake_pw([MIC, VIRTUAL])
    assert main(["clean"]) == 0
    assert main(["clean"]) == 0
    out = capsys.readouterr().out
    assert "removed virtual microphone (id 120)" in out
    assert "no virtual microphone to remove" in out


def test_pipewire_not_running(monkeypatch, capsys):
    def broken(args):
        raise DeviceError("pw-dump failed: failed to connect")

    monkeypatch.setattr(pw, "run_tool", broken)
    assert main(["list"]) == 1
    assert "failed to connect" in capsys.readouterr().err
