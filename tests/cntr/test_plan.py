#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ExecutionPlanner: a plan must never execute a Docker write
operation, run a lifecycle hook, or write a generated artifact/state/lock
file -- and must reuse the same selection/arg-building logic real commands
use."""
import os

import pytest

import linktools.cntr.__main__ as cntr_main
import linktools.cntr.commands._shared as cntr_shared
from linktools.cntr.container import ContainerError
from linktools.cntr.execution.model import ExecutionPlan
from linktools.cntr.lifecycle.dispatcher import LifecycleDispatcher
from linktools.cntr.lifecycle.hooks import HookRegistry
from linktools.cntr.runtime.structured import CommandResult

_PROXY_KEYS = ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
               "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
_SECRET_PROXY_URL = "http://user:super-secret-password@proxy.example:8080"


@pytest.fixture(autouse=True)
def _no_real_processes(monkeypatch, fresh_manager):
    """Plan must never spawn a real docker/docker-compose process for the
    actual command paths; only the preflight is allowed to (and that's
    checked separately with its own explicit stub)."""

    def fail(*a, **k):
        raise AssertionError("Plan must not create a real compose apply process")

    monkeypatch.setattr(fresh_manager.compose_runner, "build", fail)
    monkeypatch.setattr(fresh_manager.compose_runner, "up", fail)
    monkeypatch.setattr(fresh_manager.compose_runner, "stop", fail)
    monkeypatch.setattr(fresh_manager.compose_runner, "down", fail)
    # Preflight is expected to attempt a process; this sandbox has no real
    # `docker` binary, so stub both process creation and execution to be
    # inert/successful by default (overridden per-test where needed).
    monkeypatch.setattr(fresh_manager.runtime, "create_docker_process", lambda *a, **k: object())
    monkeypatch.setattr(
        fresh_manager.structured_runner, "execute_text",
        lambda *a, **k: CommandResult(args=(), returncode=0, stdout="", stderr="", duration=0.0),
    )


def test_plan_never_invokes_hooks(fresh_manager, monkeypatch):
    calls = []
    monkeypatch.setattr(LifecycleDispatcher, "_invoke_callback", lambda self, func, context=None: calls.append(1))
    monkeypatch.setattr(HookRegistry, "call", lambda self, phase, context=None, reverse=False: calls.append(1))

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
    assert expected  # sanity: the builtin fixture actually installs containers
    assert "up" in [c.phase for c in plan.commands]


def test_plan_resolved_containers_include_dependencies_of_a_partial_target(fresh_manager):
    """`resolved_containers` is always the full dependency-resolved project
    (needed for the complete --file set), even for a partial `targets`
    selection that itself does not auto-expand to its dependencies."""
    plan = fresh_manager.planner.plan("up", names=["authelia"])
    assert plan.targets == ("authelia",)
    # authelia depends on nginx/lldap; those must still appear in the full
    # resolved project set, without being pulled into `targets` itself.
    assert {"nginx", "lldap", "authelia"} <= set(plan.resolved_containers)
    assert plan.targets == ("authelia",)


def test_plan_up_partial_matches_real_selection(fresh_manager):
    plan = fresh_manager.planner.plan("up", names=["portainer"])
    assert plan.full is False
    assert plan.targets == ("portainer",)
    selection = fresh_manager.compose_operations.select(["portainer"])
    assert set(plan.services) == set(selection.services)


def test_plan_restart_includes_stop_and_up(fresh_manager):
    plan = fresh_manager.planner.plan("restart", names=["portainer"])
    phases = [c.phase for c in plan.commands]
    assert phases == ["stop", "build", "up"]


def test_plan_down_includes_down_command(fresh_manager):
    plan = fresh_manager.planner.plan("down", names=["portainer"])
    assert [c.phase for c in plan.commands] == ["down"]


def test_plan_build_false_omits_build_command(fresh_manager):
    plan = fresh_manager.planner.plan("up", names=["portainer"], build=False)
    assert "build" not in [c.phase for c in plan.commands]
    up_command = next(c for c in plan.commands if c.phase == "up")
    assert "--pull" in up_command.args or "missing" in up_command.args


def test_plan_pull_true_uses_always(fresh_manager):
    plan = fresh_manager.planner.plan("up", names=["portainer"], build=False, pull=True)
    up_command = next(c for c in plan.commands if c.phase == "up")
    assert "always" in up_command.args


def test_plan_rejects_unsupported_action(fresh_manager):
    with pytest.raises(ContainerError):
        fresh_manager.planner.plan("bogus-action")


def test_plan_commands_reuse_real_arg_builder(fresh_manager, monkeypatch):
    """Plan's build/up args must come from the exact same ComposeRunner
    builder the real path uses, not a re-implementation."""
    from linktools.cntr.runtime.compose import ComposeOptions
    recorded_options = []
    real_up_args = fresh_manager.compose_runner.up_args

    def spy_up_args(options):
        recorded_options.append(options)
        return real_up_args(options)

    monkeypatch.setattr(fresh_manager.compose_runner, "up_args", spy_up_args)
    plan = fresh_manager.planner.plan("up", names=["portainer"], build=False, pull=False)
    up_command = next(c for c in plan.commands if c.phase == "up")
    assert recorded_options  # up_args was actually consulted
    assert "up" in up_command.args


def test_plan_artifact_change_states(fresh_manager):
    plan = fresh_manager.planner.plan("up")
    nginx_artifact = next(a for a in plan.artifacts if a.container == "nginx" and a.kind == "compose")
    assert nginx_artifact.change == "added"
    assert nginx_artifact.old_sha256 is None
    assert len(nginx_artifact.new_sha256) == 64

    # Actually write it once, then re-plan: unchanged.
    fresh_manager.containers["nginx"].get_docker_compose_file()
    plan2 = fresh_manager.planner.plan("up")
    nginx_artifact2 = next(a for a in plan2.artifacts if a.container == "nginx" and a.kind == "compose")
    assert nginx_artifact2.change == "unchanged"
    assert nginx_artifact2.old_sha256 == nginx_artifact2.new_sha256


def test_plan_does_not_write_real_artifact_file(fresh_manager, tmp_path):
    import os
    compose_dir = fresh_manager.data_path / "compose"
    fresh_manager.planner.plan("up")
    assert not os.path.exists(compose_dir)


def test_plan_hooks_are_described_not_executed(fresh_manager):
    plan = fresh_manager.planner.plan("up", names=["portainer"])
    # mkdir/chown/chmod hooks from template rendering should show up as planned hooks.
    names = {h.name for h in plan.hooks}
    assert names  # at least the built-in hooks registered during prepare


def test_plan_preflight_passes_by_default(fresh_manager):
    plan = fresh_manager.planner.plan("up", names=["portainer"])
    assert plan.preflight == "passed"


def test_plan_preflight_failure_is_reported(fresh_manager, monkeypatch):
    monkeypatch.setattr(
        fresh_manager.structured_runner, "execute_text",
        lambda *a, **k: CommandResult(args=(), returncode=1, stdout="", stderr="boom", duration=0.0),
    )
    plan = fresh_manager.planner.plan("up", names=["portainer"])
    assert plan.preflight == "failed"
    assert any("preflight" in w.lower() for w in plan.warnings)


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

    cntr_main.command.on_command_up(names=["portainer"], build=False, pull=False, dry_run=True)
    from linktools.cntr.commands.plan import PlanCommand
    PlanCommand().on_command_up(names=["portainer"], build=False, pull=False, as_json=False)

    assert len(recorded) == 2
    assert recorded[0] == recorded[1]


# -- Command exactness/redaction (review issues #1-#4) -----------------------

def test_plan_command_includes_full_docker_compose_prefix(fresh_manager, monkeypatch):
    monkeypatch.setattr(fresh_manager, "container_type", "docker-rootless")
    plan = fresh_manager.planner.plan("up", names=["portainer"], build=False)
    up_command = next(c for c in plan.commands if c.phase == "up")
    assert up_command.args[:2] == ("docker", "compose")
    assert up_command.display_args[:2] == ("docker", "compose")
    assert "--file" in up_command.args
    assert "--project-name" in up_command.args


def test_plan_display_args_show_sudo_for_rootful_docker(fresh_manager, monkeypatch):
    monkeypatch.setattr(fresh_manager, "container_type", "docker")
    monkeypatch.setattr(fresh_manager, "system", "linux")
    monkeypatch.setattr(fresh_manager, "uid", 1000)
    plan = fresh_manager.planner.plan("up", names=["portainer"], build=False)
    up_command = next(c for c in plan.commands if c.phase == "up")
    assert up_command.privilege is True
    assert up_command.display_args[0] == "sudo"
    # args (for test/structural comparison) never includes sudo.
    assert "sudo" not in up_command.args


def test_plan_display_args_no_sudo_for_rootless(fresh_manager, monkeypatch):
    monkeypatch.setattr(fresh_manager, "container_type", "docker-rootless")
    plan = fresh_manager.planner.plan("up", names=["portainer"], build=False)
    up_command = next(c for c in plan.commands if c.phase == "up")
    assert up_command.privilege is False
    assert "sudo" not in up_command.display_args


def test_plan_respects_configured_docker_host(fresh_manager, monkeypatch):
    monkeypatch.setattr(fresh_manager, "container_type", "docker")
    monkeypatch.setattr(fresh_manager.env_config, "get",
                        lambda key, type=None, default=None:
                        "tcp://10.0.0.1:2376" if key == "DOCKER_HOST" else default)
    plan = fresh_manager.planner.plan("up", names=["portainer"], build=False)
    up_command = next(c for c in plan.commands if c.phase == "up")
    assert "-H" in up_command.args
    assert "tcp://10.0.0.1:2376" in up_command.args


def test_plan_compose_file_order_matches_container_order_not_sorted(fresh_manager):
    plan = fresh_manager.planner.plan("up")
    project_containers = fresh_manager.prepare_installed_containers()
    expected_order = [c.name for c in project_containers if c.docker_compose]
    actual_order = [os.path.basename(p).rsplit(".", 1)[0] for p in plan.compose_files]
    assert actual_order == expected_order
    # The actual regression: real container install order != alphabetical
    # name order for this fixture's builtins, so sorting would have changed it.
    assert actual_order != sorted(actual_order)


def test_plan_up_command_file_args_match_display_order(fresh_manager):
    plan = fresh_manager.planner.plan("up")
    up_command = next(c for c in plan.commands if c.phase == "up")
    file_positions = [i for i, a in enumerate(up_command.args) if a == "--file"]
    files_in_command = [up_command.args[i + 1] for i in file_positions]
    assert tuple(files_in_command) == plan.compose_files


def test_plan_up_includes_proxy_build_args_but_redacted(fresh_manager, monkeypatch):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("http_proxy", _SECRET_PROXY_URL)

    plan = fresh_manager.planner.plan("up", names=["portainer"], build=True)
    build_command = next(c for c in plan.commands if c.phase == "build")

    assert "--build-arg" in build_command.args
    joined = " ".join(build_command.args)
    assert "super-secret-password" not in joined
    assert _SECRET_PROXY_URL not in joined
    assert any(a == "http_proxy=***" for a in build_command.args)


def test_plan_restart_never_includes_proxy_build_args(fresh_manager, monkeypatch):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("http_proxy", _SECRET_PROXY_URL)

    plan = fresh_manager.planner.plan("restart", names=["portainer"], build=True)
    build_command = next(c for c in plan.commands if c.phase == "build")

    assert "--build-arg" not in build_command.args
    assert "super-secret-password" not in " ".join(build_command.args)


def test_plan_json_never_contains_raw_proxy_secret(fresh_manager, monkeypatch):
    import json
    from linktools.cntr.commands.plan import _plan_to_dict

    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("http_proxy", _SECRET_PROXY_URL)

    plan = fresh_manager.planner.plan("up", names=["portainer"], build=True)
    data = _plan_to_dict(plan)
    for command in data["commands"]:
        assert "args" not in command  # only display_args is ever serialized
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

    plan = fresh_manager.planner.plan("up", names=["portainer"], build=True)
    logger = _CollectingLogger()
    render_plan(logger, plan)
    text = "\n".join(logger.messages)
    assert "super-secret-password" not in text
    assert _SECRET_PROXY_URL not in text


def test_plan_up_command_matches_runtime_builder_exactly(fresh_manager, monkeypatch):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    plan = fresh_manager.planner.plan("up", names=["portainer"], build=True, pull=False)
    selection = fresh_manager.compose_operations.select(["portainer"])
    options = fresh_manager.compose_operations.build_options("up", selection, True, False)
    expected_tail = tuple(str(a) for a in fresh_manager.compose_runner.build_args(options))
    build_command = next(c for c in plan.commands if c.phase == "build")
    assert build_command.args[-len(expected_tail):] == expected_tail


def test_plan_restart_options_match_real_restart_options(fresh_manager):
    """The exact bug this guards against: Planner previously built its own
    ComposeOptions for restart and forgot include_proxy_build_args=False,
    letting a restart Plan show proxy args a real restart never sends."""
    selection = fresh_manager.compose_operations.select(["portainer"])
    plan_options = fresh_manager.compose_operations.build_options("restart", selection, True, False)
    assert plan_options.include_proxy_build_args is False
    assert plan_options.emit_default_pull is False


def test_plan_up_options_match_real_up_options(fresh_manager):
    selection = fresh_manager.compose_operations.select(["portainer"])
    plan_options = fresh_manager.compose_operations.build_options("up", selection, True, False)
    assert plan_options.include_proxy_build_args is True
    assert plan_options.emit_default_pull is True


def test_plan_restart_build_tail_matches_real_restart_dispatch(fresh_manager, monkeypatch):
    """End-to-end: the real ComposeOperations.restart() build-phase argv and
    the Plan's restart build-command argv must have an identical tail, since
    both now come from the same build_options()/build_args() call."""
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    recorded = []

    def fake(containers, *args, privilege=None, **kwargs):
        recorded.append(args)

        class _Proc:
            def check_call(self):
                return 0

        return _Proc()

    # Undo this module's autouse fail-stubs: this test exercises the real
    # restart() dispatch path on purpose, unlike every other test here.
    from linktools.cntr.runtime.compose import ComposeRunner
    monkeypatch.setattr(fresh_manager.compose_runner, "stop",
                        ComposeRunner.stop.__get__(fresh_manager.compose_runner, ComposeRunner))
    monkeypatch.setattr(fresh_manager.compose_runner, "build",
                        ComposeRunner.build.__get__(fresh_manager.compose_runner, ComposeRunner))
    monkeypatch.setattr(fresh_manager.compose_runner, "up",
                        ComposeRunner.up.__get__(fresh_manager.compose_runner, ComposeRunner))
    monkeypatch.setattr(fresh_manager.runtime, "create_docker_compose_process", fake)
    monkeypatch.setattr(LifecycleDispatcher, "_invoke_callback", lambda self, func, context=None: None)
    monkeypatch.setattr(HookRegistry, "call", lambda self, phase, context=None, reverse=False: None)

    fresh_manager.compose_operations.restart(names=["portainer"], build=True, pull=False)
    real_build_args = next(args for args in recorded if args and args[0] == "build")

    plan = fresh_manager.planner.plan("restart", names=["portainer"], build=True, pull=False)
    plan_build_command = next(c for c in plan.commands if c.phase == "build")

    assert plan_build_command.args[-len(real_build_args):] == real_build_args
