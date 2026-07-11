#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``ct-cntr compose`` is a leaf command (not a command group) that renders
the final resolved Docker Compose model; the old compose up/restart/down/
status/config/validate subcommands are gone -- those live only at the root
and under ``ct-cntr compose``'s own --check/--format flags."""
import subprocess
import sys

import pytest

from linktools.cli import BaseCommand, BaseCommandGroup
from linktools.cntr.commands.compose import ComposeCommand
from linktools.cntr.container import ContainerError
from linktools.cntr.lifecycle.dispatcher import LifecycleDispatcher
from linktools.cntr.lifecycle.hooks import HookRegistry
import linktools.cntr.commands._shared as cntr_shared


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


class _Args:
    def __init__(self, names=None, with_dependencies=False, output_format=None, check=False):
        self.names = names or []
        self.with_dependencies = with_dependencies
        self.output_format = output_format
        self.check = check


def test_compose_is_a_leaf_command_not_a_group():
    command = ComposeCommand()
    assert isinstance(command, BaseCommand)
    assert not isinstance(command, BaseCommandGroup)


def test_compose_has_no_lifecycle_or_inspect_subcommands():
    for name in ("on_command_up", "on_command_restart", "on_command_down",
                "on_command_status", "on_command_config", "on_command_validate"):
        assert not hasattr(ComposeCommand, name)


def test_compose_no_names_renders_full_project(monkeypatch, fresh_manager):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    recorded = _record(fresh_manager, monkeypatch)

    ComposeCommand().run(_Args())

    assert len(recorded) == 1
    containers, args = recorded[0]
    assert set(c.name for c in containers) == set(c.name for c in fresh_manager.installed_state.get())
    assert args == ("config",)


def test_compose_one_container_converts_to_service(monkeypatch, fresh_manager):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    recorded = _record(fresh_manager, monkeypatch)

    ComposeCommand().run(_Args(names=["portainer"]))

    containers, args = recorded[0]
    # Full installed project's --file set, not just the selected container.
    assert set(c.name for c in containers) == set(c.name for c in fresh_manager.installed_state.get())
    assert args[0] == "config"
    assert "portainer" in args


def test_compose_multi_container_stably_dedupes_services(monkeypatch, fresh_manager):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    recorded = _record(fresh_manager, monkeypatch)

    project = fresh_manager.compose_operations.select().project_containers
    nginx = next(c for c in project if c.name == "nginx")
    lldap = next(c for c in project if c.name == "lldap")
    nginx.__dict__["services"] = {"shared": {}, "nginx-only": {}}
    lldap.__dict__["services"] = {"shared": {}, "lldap-only": {}}

    ComposeCommand().run(_Args(names=["nginx", "lldap"]))

    _, args = recorded[0]
    assert args == ("config", "shared", "nginx-only", "lldap-only")


def test_compose_with_dependencies_expands_selection(monkeypatch, fresh_manager):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    recorded = _record(fresh_manager, monkeypatch)

    ComposeCommand().run(_Args(names=["authelia"], with_dependencies=True))

    _, args = recorded[0]
    assert "nginx" in args and "lldap" in args


def test_compose_format_json_is_forwarded(monkeypatch, fresh_manager):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    recorded = _record(fresh_manager, monkeypatch)

    ComposeCommand().run(_Args(output_format="json"))

    _, args = recorded[0]
    assert "--format" in args and "json" in args


def test_compose_check_uses_quiet_flag(monkeypatch, fresh_manager):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    recorded = _record(fresh_manager, monkeypatch)

    ComposeCommand().run(_Args(check=True))

    _, args = recorded[0]
    assert args[:2] == ("config", "--quiet")


def test_compose_check_and_format_are_mutually_exclusive(fresh_manager, monkeypatch):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    with pytest.raises(ContainerError):
        ComposeCommand().run(_Args(check=True, output_format="json"))


def test_bare_config_shows_help_and_never_executes_compose():
    """Bare ``ct-cntr config`` is a real command group now: no bare-command
    fallback to Compose rendering, no deprecation warning, and no Compose
    YAML on stdout (a fresh interpreter, so a rendered YAML document
    would prove Compose actually ran)."""
    result = subprocess.run(
        [sys.executable, "-m", "linktools.cntr", "config"],
        capture_output=True, text=True,
    )
    assert "deprecated" not in result.stderr
    assert "deprecated" not in result.stdout
    assert "services:" not in result.stdout
