#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Running-state consistency.

RunningStateStore owns RUNNING_CONTAINERS, updated only after a successful
compose run (partial up/down adjust just the target names; full down clears
it). These tests cover the store directly and via the CLI down/up flows.
"""
import pytest

import linktools.cntr.__main__ as cntr_main
import linktools.cntr.commands._shared as cntr_shared
from linktools.cntr.context import OperationContext
from linktools.cntr.errors import ContainerError
from linktools.cntr.lifecycle.dispatcher import LifecycleDispatcher
from linktools.cntr.lifecycle.hooks import HookRegistry
from types import SimpleNamespace
from linktools.cntr.runtime.images import ImagePlan
from linktools.cntr.state.running import RuntimeStateUnavailable

_PROXY_KEYS = ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
               "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")


def _record(manager, monkeypatch, fail=False):
    from _harness import stub_generated_runtime
    stub_generated_runtime(manager, monkeypatch)
    def fake(containers, *args, privilege=None, **kwargs):
        class _Proc:
            def check_call(self):
                if fail:
                    raise RuntimeError("compose failed")
                return 0
        return _Proc()

    monkeypatch.setattr(manager.runtime, "create_docker_process", fake)
    monkeypatch.setattr(manager.compose_runner, "final_model", lambda context: {"services": {
        name: {"image": name + ":current"} for container in context.project_containers for name in container.services}})
    monkeypatch.setattr(
        manager.image_preparer,
        "plan",
        lambda model, services=(), force_pull=False, refresh_services=(): ImagePlan(
            build=(), pull=(), targets=tuple(services),
            images={spec["image"]: SimpleNamespace(image_id="sha256:local-" + spec["image"])
                    for spec in model["services"].values()}),
    )
    monkeypatch.setattr(LifecycleDispatcher, "_invoke_callback", lambda self, func, context=None: None)
    monkeypatch.setattr(HookRegistry, "call", lambda self, phase, context=None, reverse=False: None)


def _partial_ctx(manager, name):
    ctx = OperationContext()
    ctx.actions = ["up"]
    ctx.project_containers = manager.installed_state.get(resolve=True)
    ctx.target_containers = [c for c in ctx.project_containers if c.name == name]
    ctx.is_full_project = False
    return ctx


def test_get_actual_with_no_containers_returns_empty(fresh_manager, monkeypatch):
    from linktools.cntr.runtime.inspect import ProjectRuntimeState

    def inspect(containers):
        assert list(containers) == []
        return ProjectRuntimeState(project=fresh_manager.project_name, services=(), backend="docker")

    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", inspect)
    assert fresh_manager.running_state.get_actual([]) == []


def test_get_actual_raises_unavailable_when_runtime_query_fails(fresh_manager, monkeypatch):
    from linktools.cntr.runtime.inspect import RuntimeInspectionUnavailable

    def unavailable(containers):
        raise RuntimeInspectionUnavailable("docker runtime unavailable")

    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", unavailable)
    container = fresh_manager.containers["nginx"]
    with pytest.raises(RuntimeStateUnavailable):
        fresh_manager.running_state.get_actual([container])


def test_get_effective_falls_back_to_persisted(fresh_manager, monkeypatch):
    from linktools.cntr.runtime.inspect import RuntimeInspectionUnavailable

    def unavailable(containers):
        raise RuntimeInspectionUnavailable("docker runtime unavailable")

    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", unavailable)
    fresh_manager.running_state._set(["portainer"])
    container = fresh_manager.containers["nginx"]
    assert fresh_manager.running_state.get_effective([container]) == ["portainer"]


def test_get_actual_treats_output_error_as_unavailable_too(fresh_manager, monkeypatch):
    from linktools.cntr.runtime.inspect import RuntimeInspectionOutputError

    def raise_error(containers):
        raise RuntimeInspectionOutputError("docker inspect output root is not a list")

    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", raise_error)
    container = fresh_manager.containers["nginx"]
    with pytest.raises(RuntimeStateUnavailable):
        fresh_manager.running_state.get_actual([container])

    fresh_manager.running_state._set(["portainer"])
    assert fresh_manager.running_state.get_effective([container]) == ["portainer"]


def test_mark_started_partial_adds_targets(fresh_manager):
    fresh_manager.running_state._set(["nginx"])
    fresh_manager.running_state.mark_started(_partial_ctx(fresh_manager, "portainer"))
    assert set(fresh_manager.running_state.get_persisted()) == {"nginx", "portainer"}


def test_mark_stopped_partial_removes_targets(fresh_manager):
    fresh_manager.running_state._set(["nginx", "portainer"])
    fresh_manager.running_state.mark_stopped(_partial_ctx(fresh_manager, "portainer"))
    assert set(fresh_manager.running_state.get_persisted()) == {"nginx"}


def test_mark_started_full_writes_target_set(fresh_manager):
    fresh_manager.running_state._set(["stale"])
    ctx = OperationContext()
    ctx.actions = ["up"]
    ctx.project_containers = fresh_manager.installed_state.get(resolve=False)
    ctx.target_containers = ctx.project_containers
    ctx.is_full_project = True
    fresh_manager.running_state.mark_started(ctx)
    persisted = set(fresh_manager.running_state.get_persisted())
    assert persisted == {c.name for c in ctx.project_containers}
    assert "stale" not in persisted


def test_mark_stopped_full_clears(fresh_manager):
    fresh_manager.running_state._set(["nginx", "portainer"])
    ctx = OperationContext()
    ctx.actions = ["down"]
    ctx.project_containers = fresh_manager.installed_state.get(resolve=False)
    ctx.target_containers = ctx.project_containers
    ctx.is_full_project = True
    fresh_manager.running_state.mark_stopped(ctx)
    assert fresh_manager.running_state.get_persisted() == []


def test_cli_partial_up_includes_required_dependencies(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    _record(fresh_manager, monkeypatch)
    cntr_main.command.on_command_up(names=["portainer"], pull=False)
    running = set(fresh_manager.running_state.get_persisted())
    assert {"portainer", "nginx"} <= running


def test_cli_partial_down_marks_target_stopped(monkeypatch, fresh_manager):
    fresh_manager.running_state._set(["portainer"])
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    _record(fresh_manager, monkeypatch)
    cntr_main.command.on_command_down(names=["portainer"])
    assert "portainer" not in fresh_manager.running_state.get_persisted()


def test_cli_full_down_clears_running(monkeypatch, fresh_manager):
    fresh_manager.running_state._set(["nginx", "portainer", "flare"])
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    _record(fresh_manager, monkeypatch)
    cntr_main.command.on_command_down(names=None)
    assert fresh_manager.running_state.get_persisted() == []


def test_cli_failed_up_does_not_mark_running(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    fresh_manager.running_state._set([])
    _record(fresh_manager, monkeypatch, fail=True)
    # A failed cold start followed by a failed cleanup reports both errors.
    with pytest.raises(ContainerError, match="Operation failed.*recovery failed"):
        cntr_main.command.on_command_up(names=["portainer"], pull=False)
    assert fresh_manager.running_state.get_persisted() == []


def test_remove_drops_only_the_given_names(fresh_manager):
    fresh_manager.running_state._set(["nginx", "portainer", "stale"])
    fresh_manager.running_state.remove(["stale"])
    assert set(fresh_manager.running_state.get_persisted()) == {"nginx", "portainer"}


def test_remove_is_a_noop_for_names_not_present(fresh_manager):
    fresh_manager.running_state._set(["nginx"])
    fresh_manager.running_state.remove(["does-not-exist"])
    assert fresh_manager.running_state.get_persisted() == ["nginx"]


def test_manager_get_running_containers_and_load_dump_wrappers_are_gone(fresh_manager):
    assert not hasattr(fresh_manager, "get_running_containers")
    assert not hasattr(fresh_manager, "_load_running_containers")
    assert not hasattr(fresh_manager, "_dump_running_containers")


def test_dispatcher_reconciles_removed_container_out_of_running_state(fresh_manager, monkeypatch):
    fresh_manager.running_state._set(["nginx", "flare"])

    ctx = OperationContext()
    ctx.actions = ["up"]
    ctx.project_containers = [c for c in fresh_manager.containers.values() if c.name != "flare"]
    ctx.target_containers = ctx.project_containers
    ctx.is_full_project = True

    fresh_manager.lifecycle.reconcile_removed(ctx)

    assert "flare" not in fresh_manager.running_state.get_persisted()
    assert "nginx" in fresh_manager.running_state.get_persisted()


@pytest.mark.parametrize("action", ["up", "restart"])
def test_successful_start_records_config_only_target(monkeypatch, fresh_manager, action):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    monkeypatch.setattr(fresh_manager.containers["portainer"], "services", {})
    _record(fresh_manager, monkeypatch)

    getattr(cntr_main.command, "on_command_" + action)(names=["portainer"], pull=False)

    assert {"portainer", "nginx"} <= set(fresh_manager.running_state.get_persisted())


@pytest.mark.parametrize("previous", [False, True])
def test_failed_start_preserves_config_only_state(monkeypatch, fresh_manager, previous):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    monkeypatch.setattr(fresh_manager.containers["portainer"], "services", {})
    fresh_manager.running_state._set(["portainer"] if previous else [])
    _record(fresh_manager, monkeypatch)

    def failed(context, services):
        raise RuntimeError("file publication failed")

    monkeypatch.setattr("linktools.cntr.artifacts.publish_prepared_files", failed)
    with pytest.raises(ContainerError, match="publishing file references failed"):
        cntr_main.command.on_command_up(names=["portainer"], pull=False)

    assert ("portainer" in fresh_manager.running_state.get_persisted()) == previous


def test_partial_start_records_config_dependency_but_not_unselected_config(monkeypatch, fresh_manager, tmp_path):
    from linktools.cntr.container import BaseContainer
    unused = BaseContainer(fresh_manager, tmp_path, name="unused")
    monkeypatch.setitem(fresh_manager.containers, "unused", unused)
    fresh_manager.installed_state.add("unused")
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    monkeypatch.setattr(fresh_manager.containers["flare"], "services", {})
    monkeypatch.setattr(type(fresh_manager.containers["portainer"]), "dependencies", property(lambda self: ["flare"]))
    _record(fresh_manager, monkeypatch)

    cntr_main.command.on_command_up(names=["portainer"], pull=False)

    running = fresh_manager.running_state.get_persisted()
    assert "flare" in running
    assert "portainer" in running
    assert "unused" not in running


def test_unrunnable_config_only_selection_stays_pending(monkeypatch, fresh_manager):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    target = fresh_manager.containers["portainer"]
    monkeypatch.setattr(target, "services", {})
    monkeypatch.setattr(type(target), "dependencies", property(lambda self: []))
    for container in fresh_manager.containers.values():
        monkeypatch.setattr(container, "services", {})
    _record(fresh_manager, monkeypatch)

    with pytest.raises(ContainerError, match="No runnable service"):
        cntr_main.command.on_command_up(names=["portainer"], pull=False)

    assert "portainer" not in fresh_manager.running_state.get_persisted()


@pytest.mark.parametrize("previous", [False, True])
@pytest.mark.parametrize("failure", ["compose", "after-start"])
def test_failed_lifecycle_preserves_config_only_state(monkeypatch, fresh_manager, previous, failure):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    monkeypatch.setattr(fresh_manager.containers["portainer"], "services", {})
    fresh_manager.running_state._set(["portainer"] if previous else [])
    _record(fresh_manager, monkeypatch, fail=failure == "compose")
    if failure == "after-start":
        def failed(self, func, context=None):
            if func.__name__ == "on_started":
                raise RuntimeError("after-start failed")
        monkeypatch.setattr(LifecycleDispatcher, "_invoke_callback", failed)

    with pytest.raises(ContainerError):
        cntr_main.command.on_command_up(names=["portainer"], pull=False)

    assert ("portainer" in fresh_manager.running_state.get_persisted()) == previous


def test_full_start_and_down_record_config_only_state(monkeypatch, fresh_manager, caplog):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    monkeypatch.setattr(fresh_manager.containers["portainer"], "services", {})
    _record(fresh_manager, monkeypatch)
    cntr_main.command.on_command_up(names=None, pull=False)
    assert "portainer" in fresh_manager.running_state.get_persisted()

    with caplog.at_level("INFO"):
        cntr_main.command.on_command_list(names=["portainer"])
    assert "pending install" not in caplog.text
    assert "[*] portainer" in caplog.text

    cntr_main.command.on_command_down(names=None)
    assert fresh_manager.running_state.get_persisted() == []


def test_partial_down_preserves_applied_config_only_target(monkeypatch, fresh_manager):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    monkeypatch.setattr(fresh_manager.containers["portainer"], "services", {})
    fresh_manager.running_state._set(["portainer", "nginx"])
    _record(fresh_manager, monkeypatch)

    cntr_main.command.on_command_down(names=["portainer", "nginx"])

    assert "portainer" in fresh_manager.running_state.get_persisted()
    assert "nginx" not in fresh_manager.running_state.get_persisted()
