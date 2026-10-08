from pathlib import Path

import pytest

from fakes import FIREFOX, MIC, SPEAKERS, VIRTUAL, WEBCAM, make_dump
from voxpipe import pw
from voxpipe.pw import DeviceError, Node


def test_parse_real_dump_fixture():
    nodes = pw.parse_nodes((Path(__file__).parent / "data" / "pw-dump.json").read_text())
    assert nodes, "fixture should contain audio nodes"
    assert all(n.media_class.startswith("Audio/") for n in nodes)
    assert [n.id for n in nodes] == sorted(n.id for n in nodes)


def test_parse_skips_streams_and_non_nodes():
    nodes = pw.parse_nodes(make_dump([MIC, FIREFOX, SPEAKERS]))
    assert [n.id for n in nodes] == [56, 57]


def test_microphones_and_virtual_flag():
    nodes = pw.parse_nodes(make_dump([MIC, SPEAKERS, VIRTUAL]))
    mics = pw.microphones(nodes)
    assert [n.id for n in mics] == [57, 120]
    assert pw.find_virtual_mic(nodes) == Node(*VIRTUAL)
    assert mics[1].is_virtual_mic and not mics[0].is_virtual_mic


def test_resolve_by_id_name_and_unique_description():
    nodes = pw.parse_nodes(make_dump([MIC, WEBCAM, SPEAKERS, VIRTUAL]))
    assert pw.resolve("57", nodes).name == MIC[1]
    assert pw.resolve("alsa_input.usb-Brio", nodes).id == 33
    assert pw.resolve("Brio 100", nodes).id == 33


def test_resolve_ambiguous_description():
    nodes = pw.parse_nodes(make_dump([MIC, SPEAKERS]))  # both "HyperX Headset"
    with pytest.raises(DeviceError, match="ambiguous"):
        pw.resolve("HyperX Headset", nodes)


def test_resolve_unknown():
    with pytest.raises(DeviceError, match="voxpipe list"):
        pw.resolve("999", pw.parse_nodes(make_dump([MIC])))


def test_list_nodes_uses_run_tool(monkeypatch):
    calls = []

    def fake_run_tool(args):
        calls.append(args)
        return make_dump([MIC])

    monkeypatch.setattr(pw, "run_tool", fake_run_tool)
    assert [n.id for n in pw.list_nodes()] == [57]
    assert calls == [["pw-dump"]]


def test_run_tool_missing_binary():
    with pytest.raises(DeviceError, match="not found"):
        pw.run_tool(["definitely-not-a-pipewire-tool"])


def test_run_tool_nonzero_exit():
    with pytest.raises(DeviceError, match="failed"):
        pw.run_tool(["false"])
