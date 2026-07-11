#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ExecutionPlanner (Spec Part V): a plan must never execute a Docker write
operation, run a lifecycle hook, or write a generated artifact/state/lock
file -- and must reuse the same selection/arg-building logic real commands
use."""
import pytest

import linktools.cntr.__main__ as cntr_main
import linktools.cntr.commands._shared as cntr_shared
from linktools.cntr.container import ContainerError
from linktools.cntr.execution.model import ExecutionPlan
from linktools.cntr.lifecycle.dispatcher import LifecycleDispatcher
from linktools.cntr.lifecycle.hooks import HookRegistry
from linktools.cntr.runtime.structured import CommandResult


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
