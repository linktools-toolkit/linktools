#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command ordering is an explicit CLI contract: fixed by an ``order=``
value on every visible subcommand, never by decorator declaration position,
Mixin MRO, or wrapper registration order."""
import subprocess
import sys

from linktools.cli import CommandParser, SubCommandWrapper
import linktools.cntr.__main__ as cntr_main
from linktools.cntr.commands._order import CONFIG_COMMAND_ORDER, REPO_COMMAND_ORDER, ROOT_COMMAND_ORDER
from linktools.cntr.commands.compose import ComposeCommand
from linktools.cntr.commands.config import ConfigCommand
from linktools.cntr.commands.exec_ import ExecCommand
from linktools.cntr.commands.plan import PlanCommand
from linktools.cntr.commands.repo import RepoCommand


def _help_text(*argv):
    result = subprocess.run(
        [sys.executable, "-m", "linktools.cntr", *argv, "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _assert_in_order(text, names):
    offsets = [text.index(name) for name in names]
    assert offsets == sorted(offsets), f"expected {names} in order, got offsets {offsets} in:\n{text}"


def test_root_help_matches_exact_order():
    text = _help_text()
    _assert_in_order(text, list(ROOT_COMMAND_ORDER.keys()))


def test_config_help_matches_exact_order():
    text = _help_text("config")
    _assert_in_order(text, list(CONFIG_COMMAND_ORDER.keys()))


def test_repo_help_matches_exact_order():
    text = _help_text("repo")
    _assert_in_order(text, list(REPO_COMMAND_ORDER.keys()))


def test_root_order_values_are_unique():
    assert len(set(ROOT_COMMAND_ORDER.values())) == len(ROOT_COMMAND_ORDER)


def test_config_order_values_are_unique():
    assert len(set(CONFIG_COMMAND_ORDER.values())) == len(CONFIG_COMMAND_ORDER)


def test_repo_order_values_are_unique():
    assert len(set(REPO_COMMAND_ORDER.values())) == len(REPO_COMMAND_ORDER)


def test_status_mixin_position_does_not_affect_order():
    """StatusCommands is mixed in before BaseCommandGroup in the MRO, but
    `status` must still land at its declared order (right after `list`),
    not wherever MRO resolution happens to place it."""
    text = _help_text()
    idx_list = text.index("list")
    idx_status = text.index("status")
    idx_add = text.index("add")
    assert idx_list < idx_status < idx_add


def test_wrapper_registration_order_does_not_affect_final_order():
    """init_subcommands() lists wrappers as [exec, compose, plan, config,
    repo], but the final help order (exec, compose, plan, config, repo) is
    driven by ROOT_COMMAND_ORDER, not by that list's position."""
    subcommands = cntr_main.Command().init_subcommands()
    wrapped_names = [type(sub.command).__name__ for sub in subcommands[1:]]
    assert set(wrapped_names) == {
        "ExecCommand", "ComposeCommand", "PlanCommand", "ConfigCommand", "RepoCommand",
    }

    text = _help_text()
    _assert_in_order(text, ["exec", "compose", "plan", "config", "repo"])


def test_registration_list_order_is_genuinely_ignored_when_reversed(monkeypatch):
    """Directly adversarial: register the exact same wrappers in the exact
    reverse of ROOT_COMMAND_ORDER's intent and confirm the rendered help
    order is unaffected -- proving `sort=True` truly drives the order,
    rather than the test above merely coinciding with an already-sorted
    declaration list."""
    command = cntr_main.Command()

    def reversed_registration():
        return [
            command,
            SubCommandWrapper(RepoCommand(), order=ROOT_COMMAND_ORDER["repo"]),
            SubCommandWrapper(ConfigCommand(), order=ROOT_COMMAND_ORDER["config"]),
            SubCommandWrapper(PlanCommand(), order=ROOT_COMMAND_ORDER["plan"]),
            SubCommandWrapper(ComposeCommand(), order=ROOT_COMMAND_ORDER["compose"]),
            SubCommandWrapper(ExecCommand(), order=ROOT_COMMAND_ORDER["exec"]),
        ]

    monkeypatch.setattr(command, "init_subcommands", reversed_registration)
    parser = CommandParser(command=command)
    command.init_arguments(parser)
    text = parser.format_help()
    _assert_in_order(text, list(ROOT_COMMAND_ORDER.keys()))


def test_lock_and_diff_commands_no_longer_exist():
    text = _help_text()
    assert "lock" not in text.split()
    assert "diff" not in text.split()
