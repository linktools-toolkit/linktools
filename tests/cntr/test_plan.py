#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ExecutionPlanner remains read-only and reuses real compose arg builders."""
import json
import os
import subprocess
import sys

import pytest

import linktools.cntr.__main__ as cntr_main
import linktools.cntr.commands._shared as cntr_shared
from linktools.cntr.container import ContainerError
from linktools.cntr.execution.model import ExecutionPlan
from linktools.cntr.lifecycle.dispatcher import LifecycleDispatcher
from linktools.cntr.lifecycle.hooks import HookRegistry
from linktools.cntr.runtime.compose import ComposeOptions
from linktools.cntr.runtime.structured import CommandResult

_PROXY_KEYS = ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
               "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
_SECRET_PROXY_URL = "http://user:super-secret-password@proxy.example:8080"


@pytest.fixture(autouse=True)
def _no_real_processes(monkeypatch, fresh_manager):
    def fail(*args, **kwargs):
        raise AssertionError("Plan must not execute a compose apply command")

    monkeypatch.setattr(fresh_manager.compose_runner, "build", fail)
    monkeypatch.setattr(fresh_manager.compose_runner, "pull", fail)
    monkeypatch.setattr(fresh_manager.compose_runner, "up", fail)
    monkeypatch.setattr(fresh_manager.compose_runner, "stop", fail)
    monkeypatch.setattr(fresh_manager.compose_runner, "down", fail)
    monkeypatch.setattr(fresh_manager.runtime, "create_docker_process", lambda *a, **k: object())
    monkeypatch.setattr(
        fresh_manager.structured_runner,
        "execute_text",
        lambda *a, **k: CommandResult(
            args=(), returncode=0, stdout="", stderr="", duration=0.0
        ),
    )


def test_plan_never_invokes_hooks(fresh_manager, monkeypatch):
    calls = []
    monkeypatch.setattr(
        LifecycleDispatcher,
        "_invoke_callback",
        lambda self, func, context=None: calls.append(1),
    )
    monkeypatch.setattr(
        HookRegistry,
        "call",
        lambda self, phase, context=None, reverse=False: calls.append(1),
    )

    fresh_manager.planner.plan("up")

    assert calls == []


def test_plan_never_writes_state_lock_or_artifact_index(fresh_manager, monkeypatch):
    calls = []
    monkeypatch.setattr(fresh_manager.running_state, "mark_started", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(fresh_manager.running_state, "mark_stopped", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(fresh_manager.artifact_index, "record", lambda *a, **k: calls.append(1))

    fresh_manager.planner.plan("up")
    fresh_manager.planner.plan("down")

    assert calls == []


def test_plan_up_full(fresh_manager):
    plan = fresh_manager.planner.plan("up")
    assert isinstance(plan, ExecutionPlan)
    assert plan.action == "up"
    assert plan.full is True
    assert plan.targets == ()
    expected = {c.name for c in fresh_manager.prepare_installed_containers()}
    assert set(plan.resolved_containers) == expected
    assert expected
    assert [c.phase for c in plan.commands] == ["up"]


def test_plan_resolved_containers_include_dependencies_of_a_partial_target(fresh_manager):
    plan = fresh_manager.planner.plan("up", names=["authelia"])
    assert plan.targets == ("authelia",)
    assert {"nginx", "lldap", "authelia"} <= set(plan.resolved_containers)


def test_plan_up_partial_matches_real_selection(fresh_manager):
    plan = fresh_manager.planner.plan("up", names=["portainer"])
    assert plan.full is False
    assert plan.targets == ("portainer",)
    selection = fresh_manager.compose_operations.select(["portainer"])
    assert set(plan.services) == set(selection.services)


def test_plan_restart_includes_stop_and_up(fresh_manager):
    plan = fresh_manager.planner.plan("restart", names=["portainer"])
    assert [c.phase for c in plan.commands] == ["stop", "up"]


def test_plan_down_includes_down_command(fresh_manager):
    plan = fresh_manager.planner.plan("down", names=["portainer"])
    assert [c.phase for c in plan.commands] == ["down"]


def test_plan_pull_keeps_compose_side_pull_disabled(fresh_manager):
    plan = fresh_manager.planner.plan("up", names=["portainer"], pull=True)
    assert [c.phase for c in plan.commands] == ["up"]
    up_command = plan.commands[0]
    assert "never" in up_command.args
    assert "always" not in up_command.args


def test_plan_rejects_unsupported_action(fresh_manager):
    with pytest.raises(ContainerError):
        fresh_manager.planner.plan("bogus-action")


def test_plan_commands_reuse_real_arg_builder(fresh_manager, monkeypatch):
    recorded_options = []
    real_up_args = fresh_manager.compose_runner.up_args

    def spy_up_args(options):
        recorded_options.append(options)
        return real_up_args(options)

    monkeypatch.setattr(fresh_manager.compose_runner, "up_args", spy_up_args)
    plan = fresh_manager.planner.plan("up", names=["portainer"])
    assert recorded_options
    assert "up" in plan.commands[0].args


def test_plan_artifact_change_states(fresh_manager):
    plan = fresh_manager.planner.plan("up")
    nginx_artifact = next(
        a for a in plan.artifacts if a.container == "nginx" and a.kind == "compose"
    )
    assert nginx_artifact.change == "added"
    assert nginx_artifact.old_sha256 is None
    assert len(nginx_artifact.new_sha256) == 64

    fresh_manager.containers["nginx"].get_docker_compose_file()
    plan2 = fresh_manager.planner.plan("up")
    nginx_artifact2 = next(
        a for a in plan2.artifacts if a.container == "nginx" and a.kind == "compose"
    )
    assert nginx_artifact2.change == "unchanged"
    assert nginx_artifact2.old_sha256 == nginx_artifact2.new_sha256


def test_plan_does_not_write_real_artifact_file(fresh_manager):
    compose_dir = fresh_manager.data_path / "compose"
    fresh_manager.planner.plan("up")
    assert not os.path.exists(compose_dir)


def test_plan_hooks_are_described_not_executed(fresh_manager):
    plan = fresh_manager.planner.plan("up", names=["portainer"])
    assert {h.name for h in plan.hooks}


def test_plan_preflight_passes_by_default(fresh_manager):
    plan = fresh_manager.planner.plan("up", names=["portainer"])
    assert plan.preflight == "passed"


def test_plan_preflight_failure_is_reported(fresh_manager, monkeypatch):
    monkeypatch.setattr(
        fresh_manager.structured_runner,
        "execute_text",
        lambda *a, **k: CommandResult(
            args=(), returncode=1, stdout="", stderr="boom", duration=0.0
        ),
    )
    plan = fresh_manager.planner.plan("up", names=["portainer"])
    assert plan.preflight == "failed"
    assert any("preflight" in warning.lower() for warning in plan.warnings)


def test_plan_down_skips_preflight(fresh_manager):
    plan = fresh_manager.planner.plan("down", names=["portainer"])
    assert plan.preflight == "skipped"


def test_dry_run_and_plan_command_share_one_model(fresh_manager, monkeypatch):
    recorded = []
    real_plan = fresh_manager.planner.plan

    def spy_plan(action, **kwargs):
        result = real_plan(action, **kwargs)
        recorded.append(result)
        return result

    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    monkeypatch.setattr(fresh_manager.planner, "plan", spy_plan)

    cntr_main.command.on_command_up(names=["portainer"], pull=False, dry_run=True)
    from linktools.cntr.commands.plan import maybe_dry_run
    maybe_dry_run(
        fresh_manager,
        cntr_main.command.logger,
        "up",
        names=["portainer"],
        pull=False,
        dry_run=True,
    )

    assert len(recorded) == 2
    assert recorded[0] == recorded[1]


@pytest.mark.parametrize("action", ["up", "restart", "down"])
def test_dry_run_json_is_exposed_by_each_lifecycle_command(
    fresh_manager, monkeypatch, capsys, action
):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    getattr(cntr_main.command, "on_command_" + action)(dry_run=True, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == action
    assert "commands" in payload


@pytest.mark.parametrize("action", ["up", "restart", "down"])
def test_json_without_dry_run_fails_before_side_effects(fresh_manager, monkeypatch, action):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    calls = []
    monkeypatch.setattr(
        fresh_manager.compose_operations,
        action,
        lambda *args, **kwargs: calls.append(action),
    )
    with pytest.raises(ContainerError, match="--json requires --dry-run"):
        getattr(cntr_main.command, "on_command_" + action)(as_json=True)
    assert calls == []


def test_cntr_help_has_no_removed_plan_command():
    result = subprocess.run(
        [sys.executable, "-m", "linktools.cntr.__main__", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "plan" not in result.stdout


def test_plan_command_includes_full_docker_compose_prefix(fresh_manager):
    fresh_manager.env_config.set("DOCKER_TYPE", "docker-rootless")
    plan = fresh_manager.planner.plan("up", names=["portainer"])
    up_command = plan.commands[0]
    assert up_command.args[:2] == ("docker", "compose")
    assert up_command.display_args[:2] == ("docker", "compose")
    assert "--file" in up_command.args
    assert "--project-name" in up_command.args


def test_plan_display_args_show_sudo_for_rootful_docker(fresh_manager, monkeypatch):
    fresh_manager.env_config.set("DOCKER_TYPE", "docker")
    monkeypatch.setattr(type(fresh_manager), "system", "linux")
    monkeypatch.setattr(type(fresh_manager), "uid", 1000)
    plan = fresh_manager.planner.plan("up", names=["portainer"])
    up_command = plan.commands[0]
    assert up_command.privilege is True
    assert up_command.display_args[0] == "sudo"
    assert "sudo" not in up_command.args


def test_plan_display_args_no_sudo_for_rootless(fresh_manager):
    fresh_manager.env_config.set("DOCKER_TYPE", "docker-rootless")
    plan = fresh_manager.planner.plan("up", names=["portainer"])
    up_command = plan.commands[0]
    assert up_command.privilege is False
    assert "sudo" not in up_command.display_args


def test_plan_respects_configured_docker_host(fresh_manager, monkeypatch):
    fresh_manager.env_config.set("DOCKER_TYPE", "docker")
    original_get = fresh_manager.env_config.get

    def fake_get(key, type=None, default=None):
        if key == "DOCKER_HOST":
            return "tcp://10.0.0.1:2376"
        return original_get(key, type=type, default=default)

    monkeypatch.setattr(fresh_manager.env_config, "get", fake_get)
    plan = fresh_manager.planner.plan("up", names=["portainer"])
    up_command = plan.commands[0]
    assert "-H" in up_command.args
    assert "tcp://10.0.0.1:2376" in up_command.args


def test_plan_compose_file_order_matches_container_order_not_sorted(fresh_manager):
    plan = fresh_manager.planner.plan("up")
    project_containers = fresh_manager.prepare_installed_containers()
    expected_order = [c.name for c in project_containers if c.docker_compose]
    actual_order = [os.path.basename(path).rsplit(".", 1)[0] for path in plan.compose_files]
    assert actual_order == expected_order
    assert actual_order != sorted(actual_order)


def test_plan_up_command_file_args_match_display_order(fresh_manager):
    plan = fresh_manager.planner.plan("up")
    up_command = plan.commands[0]
    file_positions = [index for index, arg in enumerate(up_command.args) if arg == "--file"]
    files_in_command = [up_command.args[index + 1] for index in file_positions]
    assert tuple(files_in_command) == plan.compose_files


def test_plan_never_injects_proxy_secrets(fresh_manager, monkeypatch):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("http_proxy", _SECRET_PROXY_URL)

    plan = fresh_manager.planner.plan("up", names=["portainer"])

    for command in plan.commands:
        joined = " ".join(command.args + command.display_args)
        assert "super-secret-password" not in joined
        assert _SECRET_PROXY_URL not in joined


def test_plan_json_never_contains_raw_proxy_secret(fresh_manager, monkeypatch):
    from linktools.cntr.commands.plan import _plan_to_dict

    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("http_proxy", _SECRET_PROXY_URL)

    data = _plan_to_dict(fresh_manager.planner.plan("up", names=["portainer"]))
    for command in data["commands"]:
        assert "args" not in command
    payload = json.dumps(data)
    assert "super-secret-password" not in payload
    assert _SECRET_PROXY_URL not in payload


def test_plan_text_render_never_contains_raw_proxy_secret(fresh_manager, monkeypatch):
    from linktools.cntr.commands.plan import render_plan

    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("http_proxy", _SECRET_PROXY_URL)

    class _CollectingLogger:
        def __init__(self):
            self.messages = []

        def info(self, msg):
            self.messages.append(str(msg))

    logger = _CollectingLogger()
    render_plan(logger, fresh_manager.planner.plan("up", names=["portainer"]))
    text = "\n".join(logger.messages)
    assert "super-secret-password" not in text
    assert _SECRET_PROXY_URL not in text


def test_plan_up_command_matches_runtime_builder_exactly(fresh_manager):
    plan = fresh_manager.planner.plan("up", names=["portainer"])
    selection = fresh_manager.compose_operations.select(["portainer"])
    options = ComposeOptions(
        remove_orphans=selection.full,
        services=list(selection.services),
    )
    expected_tail = tuple(fresh_manager.compose_runner.up_args(options))
    up_command = plan.commands[0]
    assert up_command.args[-len(expected_tail):] == expected_tail


def test_plan_restart_up_command_matches_runtime_builder_exactly(fresh_manager):
    plan = fresh_manager.planner.plan("restart", names=["portainer"])
    selection = fresh_manager.compose_operations.select(["portainer"])
    options = ComposeOptions(
        remove_orphans=selection.full,
        services=list(selection.services),
    )
    expected_tail = tuple(fresh_manager.compose_runner.up_args(options))
    up_command = next(command for command in plan.commands if command.phase == "up")
    assert up_command.args[-len(expected_tail):] == expected_tail
