#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command ordering is an explicit CLI contract: fixed by an ``order=``
value on every visible subcommand, never by decorator declaration position,
Mixin MRO, or wrapper registration order."""
import subprocess
import sys

import linktools.cntr.__main__ as cntr_main
from linktools.cntr.commands._order import CONFIG_COMMAND_ORDER, REPO_COMMAND_ORDER, ROOT_COMMAND_ORDER


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
        "LockCommand", "DiffCommand",
    }

    text = _help_text()
    _assert_in_order(text, ["exec", "compose", "plan", "config", "repo"])
