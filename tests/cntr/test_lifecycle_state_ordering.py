#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persisted running state follows successful runtime changes before post-hooks."""
import pytest

from linktools.cntr.lifecycle.dispatcher import LifecycleDispatcher
from linktools.cntr.lifecycle.hooks import HookPhase, HookRegistry
from types import SimpleNamespace
from linktools.cntr.runtime.images import ImagePlan
from linktools.types import MISSING


def _neutralize_runtime(manager, monkeypatch):
    from _harness import configure_builtin_inputs
    configure_builtin_inputs(manager)
    from _harness import stub_generated_runtime
    stub_generated_runtime(manager, monkeypatch)
    manager.config.set("NGINX_ROOT_DOMAIN", "example.test")
    monkeypatch.setattr(manager.compose_operations, "_check_external_requirements", lambda *args, **kwargs: None)
    from linktools.cntr.runtime.inspect import ProjectRuntimeState
    monkeypatch.setattr(manager.docker_inspector, "get_project_state", lambda containers:
                        ProjectRuntimeState(manager.project_name, (), "docker"))
    monkeypatch.setattr(manager.compose_operations, "start_selection",
                        lambda selection, **kwargs: selection)
    monkeypatch.setattr(manager.compose_runner, "apply_services",
                        lambda context, services: None)
    monkeypatch.setattr(manager.compose_runner, "wait_service_ready",
                        lambda context, service, **kwargs: True)
    monkeypatch.setattr(manager.image_preparer, "image_id",
                        lambda image: "sha256:fixture-image")

    monkeypatch.setattr(manager.compose_runner, "final_model", lambda context: {"services": {
        name: {"image": name + ":local"} for container in context.project_containers for name in container.services}})
    monkeypatch.setattr(
        manager.image_preparer,
        "plan",
        lambda model, services=(), force_pull=False, **kwargs: ImagePlan(
            build=(), pull=(), targets=tuple(services),
            images={spec["image"]: SimpleNamespace(image_id="sha256:local-" + spec["image"])
                    for spec in model["services"].values()}),
    )
    monkeypatch.setattr(manager.compose_runner, "build", lambda context, options: None)
    monkeypatch.setattr(manager.compose_runner, "pull", lambda context, services: None)
    monkeypatch.setattr(manager.compose_runner, "stop", lambda context, services: None)
    monkeypatch.setattr(manager.compose_runner, "down", lambda context, services: None)


def _fail_lifecycle_callback(monkeypatch, name):
    original = LifecycleDispatcher._invoke_callback

    def patched(self, func, context=MISSING):
        if getattr(func, "__name__", "") == name:
            raise RuntimeError(f"{name} boom")
        return original(self, func, context)

    monkeypatch.setattr(LifecycleDispatcher, "_invoke_callback", patched)


def _fail_hook_phase(monkeypatch, phase):
    def patched(self, called_phase, context=None, reverse=False):
        if called_phase == phase:
            raise RuntimeError(f"{phase.value} boom")

    monkeypatch.setattr(HookRegistry, "call", patched)


def test_up_marks_started_before_on_started_hook_failure(fresh_manager, monkeypatch):
    _neutralize_runtime(fresh_manager, monkeypatch)
    _fail_lifecycle_callback(monkeypatch, "on_started")

    with pytest.raises(Exception, match="after-start callback failed: on_started boom"):
        fresh_manager.compose_operations.up(names=["portainer"])

    assert "portainer" in fresh_manager.running_state.get_persisted()


def test_up_marks_started_before_after_start_hook_failure(fresh_manager, monkeypatch):
    _neutralize_runtime(fresh_manager, monkeypatch)
    _fail_hook_phase(monkeypatch, HookPhase.AFTER_START)

    with pytest.raises(Exception, match="after-start callback failed: after-start boom"):
        fresh_manager.compose_operations.up(names=["portainer"])

    assert "portainer" in fresh_manager.running_state.get_persisted()


def test_down_marks_stopped_before_on_stopped_hook_failure(fresh_manager, monkeypatch):
    _neutralize_runtime(fresh_manager, monkeypatch)
    fresh_manager.running_state._mutate(lambda current: {"portainer"})
    _fail_lifecycle_callback(monkeypatch, "on_stopped")

    with pytest.raises(RuntimeError, match="on_stopped boom"):
        fresh_manager.compose_operations.down(names=["portainer"])

    assert "portainer" not in fresh_manager.running_state.get_persisted()


def test_down_marks_stopped_before_after_stop_hook_failure(fresh_manager, monkeypatch):
    _neutralize_runtime(fresh_manager, monkeypatch)
    fresh_manager.running_state._mutate(lambda current: {"portainer"})
    _fail_hook_phase(monkeypatch, HookPhase.AFTER_STOP)

    with pytest.raises(RuntimeError, match="after-stop boom"):
        fresh_manager.compose_operations.down(names=["portainer"])

    assert "portainer" not in fresh_manager.running_state.get_persisted()


def test_restart_build_failure_keeps_running_targets(fresh_manager, monkeypatch):
    _neutralize_runtime(fresh_manager, monkeypatch)
    fresh_manager.running_state._mutate(lambda current: {"portainer"})
    monkeypatch.setattr(
        fresh_manager.image_preparer,
        "plan",
        lambda model, services=(), force_pull=False, **kwargs: ImagePlan(
            build=tuple(services), pull=(), targets=tuple(services)
        , images={spec["image"]: SimpleNamespace(image_id="sha256:local-" + spec["image"]) for spec in model["services"].values()}),
    )

    def broken_build(context, options):
        raise RuntimeError("build boom")

    monkeypatch.setattr(fresh_manager.compose_runner, "build", broken_build)

    with pytest.raises(RuntimeError, match="build boom"):
        fresh_manager.compose_operations.restart(names=["portainer"])

    assert "portainer" in fresh_manager.running_state.get_persisted()


def test_restart_stop_success_up_failure_still_marks_stopped(fresh_manager, monkeypatch):
    _neutralize_runtime(fresh_manager, monkeypatch)
    fresh_manager.running_state._mutate(lambda current: {"portainer"})

    def broken_up(context, service, recreate=False):
        raise RuntimeError("up boom")

    monkeypatch.setattr(fresh_manager.compose_runner, "apply_services", broken_up)

    with pytest.raises(RuntimeError, match="up boom"):
        fresh_manager.compose_operations.restart(names=["portainer"])

    assert "portainer" not in fresh_manager.running_state.get_persisted()


def test_restart_full_success_marks_started(fresh_manager, monkeypatch):
    _neutralize_runtime(fresh_manager, monkeypatch)
    fresh_manager.running_state._mutate(lambda current: {"portainer"})

    fresh_manager.compose_operations.restart(names=["portainer"])

    assert "portainer" in fresh_manager.running_state.get_persisted()


def test_exec_up_marks_started_before_on_started_hook_failure(fresh_manager, monkeypatch):
    _neutralize_runtime(fresh_manager, monkeypatch)
    _fail_lifecycle_callback(monkeypatch, "on_started")
    container = fresh_manager.containers["portainer"]

    with pytest.raises(Exception, match="after-start callback failed: on_started boom"):
        container.on_exec_up()

    assert "portainer" in fresh_manager.running_state.get_persisted()


def test_exec_down_marks_stopped_before_on_stopped_hook_failure(fresh_manager, monkeypatch):
    _neutralize_runtime(fresh_manager, monkeypatch)
    fresh_manager.running_state._mutate(lambda current: {"portainer"})
    _fail_lifecycle_callback(monkeypatch, "on_stopped")
    container = fresh_manager.containers["portainer"]

    with pytest.raises(RuntimeError, match="on_stopped boom"):
        container.on_exec_down()

    assert "portainer" not in fresh_manager.running_state.get_persisted()


def test_partial_up_preserves_a_container_removed_from_installed_set(fresh_manager, monkeypatch):
    _neutralize_runtime(fresh_manager, monkeypatch)
    fresh_manager.running_state._mutate(lambda current: {"portainer", "safeline"})
    fresh_manager.installed_state.remove("safeline")
    fresh_manager.config.set("NGINX_WAF_ENABLE", False)

    fresh_manager.compose_operations.up(names=["portainer"])

    assert "safeline" in fresh_manager.running_state.get_persisted()
    assert "portainer" in fresh_manager.running_state.get_persisted()


def test_on_removed_hook_does_not_fire_for_named_operation(fresh_manager, monkeypatch):
    _neutralize_runtime(fresh_manager, monkeypatch)
    fresh_manager.running_state._mutate(lambda current: {"portainer", "safeline"})
    fresh_manager.installed_state.remove("safeline")
    fresh_manager.config.set("NGINX_WAF_ENABLE", False)

    calls = []
    original = LifecycleDispatcher._invoke_callback

    def patched(self, func, context=MISSING):
        if getattr(func, "__name__", "") == "on_removed":
            calls.append(1)
            return None
        return original(self, func, context)

    monkeypatch.setattr(LifecycleDispatcher, "_invoke_callback", patched)

    fresh_manager.compose_operations.up(names=["portainer"])

    assert calls == []
