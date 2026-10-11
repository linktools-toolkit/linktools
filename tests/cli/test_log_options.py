#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Global log display flags work through the real command parser."""

from unittest.mock import Mock, call

import pytest

from linktools.cli import BaseCommandGroup, subcommand, subcommand_argument


class _Command(BaseCommandGroup):
    def __init__(self) -> None:
        self.execute = Mock(return_value=0)

    @subcommand("up")
    @subcommand_argument("names", nargs="*")
    def up(self, names: "list[str]") -> int:
        return self.execute(names=names)


@pytest.fixture
def command(monkeypatch: pytest.MonkeyPatch) -> _Command:
    result = _Command()
    manager = result.environ.logging
    monkeypatch.setattr(manager, "get_handler", lambda: object())
    monkeypatch.setattr(manager, "set_show_time", Mock())
    monkeypatch.setattr(manager, "set_show_level", Mock())
    return result


@pytest.mark.parametrize("flag,setting,expected", [
    ("--time", "time", True),
    ("--no-time", "time", False),
    ("--level", "level", True),
    ("--no-level", "level", False),
])
@pytest.mark.parametrize("after_subcommand", [False, True])
def test_log_options_apply_before_or_after_subcommand(
    command: _Command, flag: str, setting: str, expected: bool, after_subcommand: bool,
) -> None:
    args = ["up", "nginx", flag] if after_subcommand else [flag, "up", "nginx"]

    assert command(args) == 0

    command.execute.assert_called_once_with(names=["nginx"])
    getattr(command.environ.logging, "set_show_" + setting).assert_called_once_with(expected)
    other = "level" if setting == "time" else "time"
    getattr(command.environ.logging, "set_show_" + other).assert_not_called()


@pytest.mark.parametrize("setting", ["time", "level"])
def test_log_options_preserve_order_across_subcommand(command: _Command, setting: str) -> None:
    assert command(["--" + setting, "up", "nginx", "--no-" + setting, "--" + setting]) == 0

    command.execute.assert_called_once_with(names=["nginx"])
    assert getattr(command.environ.logging, "set_show_" + setting).call_args_list == [
        call(True), call(False), call(True),
    ]


def test_log_options_leave_display_unchanged_when_omitted(command: _Command) -> None:
    assert command(["up", "nginx"]) == 0

    command.execute.assert_called_once_with(names=["nginx"])
    command.environ.logging.set_show_time.assert_not_called()
    command.environ.logging.set_show_level.assert_not_called()
