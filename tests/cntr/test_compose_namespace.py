#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compose command namespace: root up/restart/down and `compose` subcommands
must dispatch through the exact same ComposeOperations implementation, and
ComposeSelection must enforce its selection/validation rules."""
import sys

import pytest

import linktools.cntr.__main__ as cntr_main
import linktools.cntr.commands._shared as cntr_shared
from linktools.cntr.commands.compose.group import ComposeCommand
from linktools.cntr.commands.config import ConfigCommand
from linktools.cntr.container import ContainerError
from linktools.cntr.lifecycle.dispatcher import LifecycleDispatcher
from linktools.cntr.lifecycle.hooks import HookRegistry
from linktools.cntr.operations.compose import ComposeSelection

_PROXY_KEYS = ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
               "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")


def _record(manager, monkeypatch):
    recorded = []

    def fake(containers, *args, privilege=None, **kwargs):
        recorded.append((tuple(containers), args))

        class _Proc:
            def check_call(self):
                return 0

        return _Proc()

    monkeypatch.setattr(manager.runtime, "create_docker_compose_process", fake)
    monkeypatch.setattr(LifecycleDispatcher, "_invoke_callback", lambda self, func, context=None: None)
    monkeypatch.setattr(HookRegistry, "call", lambda self, phase, context=None, reverse=False: None)
    return recorded


def test_root_and_compose_up_dispatch_identical_args(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)

    recorded_root = _record(fresh_manager, monkeypatch)
    cntr_main.command.on_command_up(names=["portainer"], build=True, pull=False)

    recorded_compose = _record(fresh_manager, monkeypatch)
    ComposeCommand().on_command_up(names=["portainer"], build=True, pull=False)

    assert [args for _, args in recorded_root] == [args for _, args in recorded_compose]


def test_root_and_compose_restart_dispatch_identical_args(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)

    recorded_root = _record(fresh_manager, monkeypatch)
    cntr_main.command.on_command_restart(names=["portainer"], build=True, pull=False)

    recorded_compose = _record(fresh_manager, monkeypatch)
    ComposeCommand().on_command_restart(names=["portainer"], build=True, pull=False)

    assert [args for _, args in recorded_root] == [args for _, args in recorded_compose]


def test_root_and_compose_down_dispatch_identical_args(monkeypatch, fresh_manager):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)

    recorded_root = _record(fresh_manager, monkeypatch)
    cntr_main.command.on_command_down(names=None)

    recorded_compose = _record(fresh_manager, monkeypatch)
    ComposeCommand().on_command_down(names=None)

    assert [args for _, args in recorded_root] == [args for _, args in recorded_compose]


def test_compose_selection_rejects_unknown_container(fresh_manager):
    with pytest.raises(ContainerError):
        fresh_manager.compose_operations.select(names=["does-not-exist"])


def test_compose_selection_full_when_no_names(fresh_manager):
    selection = fresh_manager.compose_operations.select()
    assert selection.full is True
    assert selection.services == ()
    assert set(selection.target_containers) == set(selection.project_containers)


def test_compose_selection_with_dependencies_expands_targets(fresh_manager):
    selection = fresh_manager.compose_operations.select(names=["authelia"], with_dependencies=True)
    names = {c.name for c in selection.target_containers}
    assert {"authelia", "nginx", "lldap"} <= names


def test_compose_selection_without_dependencies_keeps_single_target(fresh_manager):
    selection = fresh_manager.compose_operations.select(names=["authelia"], with_dependencies=False)
    assert [c.name for c in selection.target_containers] == ["authelia"]


def test_compose_selection_services_are_stably_deduped(fresh_manager):
    project = fresh_manager.compose_operations.select().project_containers
    nginx = next(c for c in project if c.name == "nginx")
    lldap = next(c for c in project if c.name == "lldap")
    # Force an overlapping service name between two distinct target containers.
    nginx.__dict__["services"] = {"shared": {}, "nginx-only": {}}
    lldap.__dict__["services"] = {"shared": {}, "lldap-only": {}}

    selection = fresh_manager.compose_operations.select(names=["nginx", "lldap"])
    assert selection.services == ("shared", "nginx-only", "lldap-only")


def test_compose_config_uses_full_file_set_with_filtered_services(monkeypatch, fresh_manager):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    recorded = _record(fresh_manager, monkeypatch)

    fresh_manager.compose_operations.config(names=["portainer"])

    assert len(recorded) == 1
    containers, args = recorded[0]
    # Full installed project's --file set, not just the selected container.
    assert set(c.name for c in containers) == set(c.name for c in fresh_manager.installed_state.get())
    assert args[0] == "config"
    assert "portainer" in args


def test_compose_validate_uses_quiet_flag(monkeypatch, fresh_manager):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    recorded = _record(fresh_manager, monkeypatch)

    fresh_manager.compose_operations.validate(names=["portainer"])

    assert recorded[0][1][:2] == ("config", "--quiet")


def test_legacy_bare_config_warns_on_stderr_only(monkeypatch, fresh_manager, capsys):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    command = ConfigCommand()
    monkeypatch.setattr(command, "parse_subcommand", lambda args: None)
    monkeypatch.setattr(fresh_manager.compose_operations, "config", lambda *a, **k: 0)

    command.run(args=None)

    captured = capsys.readouterr()
    assert "deprecated" in captured.err
    assert "deprecated" not in captured.out


def test_compose_command_bare_shows_help_without_executing(fresh_manager, monkeypatch, capsys):
    """A bare `ct-cntr compose` (new namespace, no compatibility baggage) must
    only print subcommand help -- unlike the legacy bare `config`, it never
    had prior forwarding behavior to preserve."""
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    called = []
    monkeypatch.setattr(fresh_manager.compose_operations, "up", lambda *a, **k: called.append(1))
    monkeypatch.setattr(fresh_manager.compose_operations, "config", lambda *a, **k: called.append(1))

    command = ComposeCommand()
    monkeypatch.setattr(command, "parse_subcommand", lambda args: None)
    command.print_subcommands = lambda *a, **k: None
    command.run(args=None)

    assert called == []
