# -*- coding: utf-8 -*-
"""Tests for the CLI exit-code mapping."""
import pytest

from linktools.cli.exitcodes import (
    exit_code_for, EXIT_SUCCESS, EXIT_USER_INPUT, EXIT_CONFIG, EXIT_NETWORK,
    EXIT_TOOL, EXIT_INTERNAL, EXIT_INTERRUPT,
)
from linktools.cli.command import CommandError
from linktools.errors import (
    LinktoolsError, ConfigError, DownloadError, GitError, SSHError, ToolError,
    CacheError, EnvironmentError,
)


def test_user_input_for_command_error():
    assert exit_code_for(CommandError("bad arg")) == EXIT_USER_INPUT


def test_config_error_maps_to_3():
    assert exit_code_for(ConfigError("x")) == EXIT_CONFIG


@pytest.mark.parametrize("exc", [DownloadError("x"), GitError("x"), SSHError("x")])
def test_network_remote_errors_map_to_4(exc):
    assert exit_code_for(exc) == EXIT_NETWORK


def test_tool_error_maps_to_5():
    assert exit_code_for(ToolError("x")) == EXIT_TOOL


def test_other_linktools_error_is_internal():
    # Cache / Environment are not user-input/config/network/tool -> internal.
    assert exit_code_for(CacheError("x")) == EXIT_INTERNAL
    assert exit_code_for(EnvironmentError("x")) == EXIT_INTERNAL
    assert exit_code_for(LinktoolsError("x")) == EXIT_INTERNAL


def test_plain_exception_is_internal():
    assert exit_code_for(ValueError("x")) == EXIT_INTERNAL


def test_keyboard_interrupt_maps_to_130():
    assert exit_code_for(KeyboardInterrupt()) == EXIT_INTERRUPT


def test_exit_codes_match_spec_table():
    assert EXIT_SUCCESS == 0
    assert EXIT_USER_INPUT == 2
    assert EXIT_CONFIG == 3
    assert EXIT_NETWORK == 4
    assert EXIT_TOOL == 5
    assert EXIT_INTERNAL == 10
    assert EXIT_INTERRUPT == 130
