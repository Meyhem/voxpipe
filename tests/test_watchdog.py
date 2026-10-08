import threading

import pytest

from fakes import MIC, VIRTUAL
from voxpipe import pw
from voxpipe.pw import DeviceError, Node

SOURCE, DEST = Node(*MIC), Node(*VIRTUAL)


def test_capture_and_playback_commands():
    capture = pw.capture_command(SOURCE)
    assert capture[:3] == ["pw-record", "--target", SOURCE.name]
    assert capture[-1] == "-"
    assert "node.dont-reconnect = true" in " ".join(capture)
    for flag, value in (
        ("--rate", "48000"),
        ("--channels", "1"),
        ("--format", "f32"),
        ("--latency", "120"),
        ("--volume", "1.0"),
    ):
        assert capture[capture.index(flag) + 1] == value
    playback = pw.playback_command()
    assert playback[:3] == ["pw-play", "--target", "0"]
    assert "node.name=voxpipe-out" in " ".join(playback)
    assert playback[playback.index("--volume") + 1] == "1.0"
    assert playback[-1] == "-"


def test_link_playback_links_every_input_port():
    sink = Node(56, "alsa_output.usb-HyperX", "HyperX Headset", "Audio/Sink")
    calls = []

    def runner(args):
        calls.append(args)
        if args == ["pw-link", "-o"]:
            return "voxpipe-out:output_MONO\nother:output_FL\n"
        if args == ["pw-link", "-i"]:
            return "alsa_output.usb-HyperX:playback_FL\nalsa_output.usb-HyperX:playback_FR\nx:in\n"
        return ""

    pw.link_playback(sink, runner=runner)
    links = [c for c in calls if len(c) == 3]
    assert links == [
        ["pw-link", "voxpipe-out:output_MONO", "alsa_output.usb-HyperX:playback_FL"],
        ["pw-link", "voxpipe-out:output_MONO", "alsa_output.usb-HyperX:playback_FR"],
    ]


def test_link_playback_times_out_when_stream_never_appears():
    with pytest.raises(DeviceError, match="could not link"):
        pw.link_playback(DEST, runner=lambda args: "", wait_s=0.1, sleep=lambda s: None)


def run_watchdog(snapshots):
    lost = []
    done = threading.Event()
    it = iter(snapshots)

    def lister():
        item = next(it, snapshots[-1])
        if isinstance(item, Exception):
            raise item
        return item

    def on_lost(node):
        lost.append(node)
        done.set()

    dog = pw.DeviceWatchdog(SOURCE, DEST, on_lost, lister=lister, interval_s=0.01)
    dog.start()
    done.wait(1.0)
    dog.stop()
    dog.join(1.0)
    return lost, dog


def test_reports_source_loss():
    lost, dog = run_watchdog([[SOURCE, DEST], [DEST]])
    assert lost == [SOURCE] and dog.lost == SOURCE


def test_reports_destination_loss():
    lost, _ = run_watchdog([[SOURCE, DEST], [SOURCE]])
    assert lost == [DEST]


def test_ignores_transient_errors():
    lost, _ = run_watchdog([DeviceError("busy"), [SOURCE, DEST], [SOURCE]])
    assert lost == [DEST]


def test_stop_without_loss():
    dog = pw.DeviceWatchdog(SOURCE, DEST, lambda n: None, lister=lambda: [SOURCE, DEST],
                            interval_s=0.01)
    dog.start()
    dog.stop()
    dog.join(1.0)
    assert not dog.is_alive() and dog.lost is None
