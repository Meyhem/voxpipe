import argparse

import pytest

from voxpipe import __version__, cli
from voxpipe.errors import VoxpipeError


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"voxpipe {__version__}"


def test_voxpipe_error_is_printed_and_returns_1(monkeypatch, capsys):
    def failing(args: argparse.Namespace) -> int:
        raise VoxpipeError("boom")

    class FakeCommand:
        @staticmethod
        def register(subparsers) -> None:
            subparsers.add_parser("fail").set_defaults(handler=failing)

    monkeypatch.setattr(cli, "COMMANDS", (FakeCommand,))
    assert cli.main(["fail"]) == 1
    assert capsys.readouterr().err == "voxpipe: error: boom\n"
