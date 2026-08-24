#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persisted running state must reflect a successful runtime change
immediately, not wait for post-hooks to finish (review P1-09): if
on_started/AFTER_START (or on_stopped/AFTER_STOP) then raises, the command
still fails, but running_state must already show what's actually running --
not lag behind it. Covers both the root (ComposeOperations) and exec
(_container/actions) entry points, which must behave identically.
"""
import pytest

from linktools.cntr.lifecycle.dispatcher import LifecycleDispatcher
from linktools.cntr.lifecycle.hooks import HookPhase, HookRegistry
from linktools.types import MISSING


def _neutralize_runtime(manager, monkeypatch):
    monkeypatch.setattr(manager.compose_runner, "build", lambda context, options: None)
    monkeypatch.setattr(manager.compose_runner, "up", lambda context, options: None)
    monkeypatch.setattr(manager.compose_runner, "stop", lambda context, services: None)
    monkeypatch.setattr(manager.compose_runner, "down", lambda context, services: None)


def _fail_lifecycle_callback(monkeypatch, name):
    """Raise from a specific on_check/on_starting/on_started/... callback,
    passing every other one through unchanged."""
    original = LifecycleDispatcher._invoke_callback

    def patched(self, func, context=MISSING):
        if getattr(func, "__name__", "") == name:
            raise RuntimeError(f"{name} boom")
        return original(self, func, context)

    monkeypatch.setattr(LifecycleDispatcher, "_invoke_callback", patched)


def _fail_hook_phase(monkeypatch, phase):
    """Raise from HookRegistry.call for a specific phase, passing every
    other phase through as a no-op (no hooks registered in these tests
    anyway, so a no-op is equivalent to the real call)."""
    def patched(self, called_phase, context=None, reverse=False):
        if called_phase == phase:
            raise RuntimeError(f"{phase} boom")

    monkeypatch.setattr(HookRegistry, "call", patched)


# -- root entry point (ComposeOperations) ------------------------------------

def test_up_marks_started_before_on_started_hook_failure(fresh_manager, monkeypatch):
    _neutralize_runtime(fresh_manager, monkeypatch)
    _fail_lifecycle_callback(monkeypatch, "on_started")

    with pytest.raises(RuntimeError, match="on_started boom"):
        fresh_manager.compose_operations.up(names=["portainer"])

    assert "portainer" in fresh_manager.running_state.get_persisted()


def test_up_marks_started_before_after_start_hook_failure(fresh_manager, monkeypatch):
    _neutralize_runtime(fresh_manager, monkeypatch)
    _fail_hook_phase(monkeypatch, HookPhase.AFTER_START)

    with pytest.raises(RuntimeError, match="AFTER_START boom"):
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

    with pytest.raises(RuntimeError, match="AFTER_STOP boom"):
        fresh_manager.compose_operations.down(names=["portainer"])

    assert "portainer" not in fresh_manager.running_state.get_persisted()


def test_restart_stop_success_build_failure_still_marks_stopped(fresh_manager, monkeypatch):
    _neutralize_runtime(fresh_manager, monkeypatch)
    fresh_manager.running_state._mutate(lambda current: {"portainer"})

    def broken_build(context, options):
        raise RuntimeError("build boom")

    monkeypatch.setattr(fresh_manager.compose_runner, "build", broken_build)

    with pytest.raises(RuntimeError, match="build boom"):
        fresh_manager.compose_operations.restart(names=["portainer"])

    # stop succeeded, build failed -- the target is actually stopped, not
    # still running from before restart began.
    assert "portainer" not in fresh_manager.running_state.get_persisted()


def test_restart_stop_success_up_failure_still_marks_stopped(fresh_manager, monkeypatch):
    _neutralize_runtime(fresh_manager, monkeypatch)
    fresh_manager.running_state._mutate(lambda current: {"portainer"})

    def broken_up(context, options):
        raise RuntimeError("up boom")

    monkeypatch.setattr(fresh_manager.compose_runner, "up", broken_up)

    with pytest.raises(RuntimeError, match="up boom"):
        fresh_manager.compose_operations.restart(names=["portainer"])

    assert "portainer" not in fresh_manager.running_state.get_persisted()


def test_restart_full_success_marks_started(fresh_manager, monkeypatch):
    _neutralize_runtime(fresh_manager, monkeypatch)
    fresh_manager.running_state._mutate(lambda current: {"portainer"})

    fresh_manager.compose_operations.restart(names=["portainer"])

    assert "portainer" in fresh_manager.running_state.get_persisted()


# -- exec entry point (_container/actions.py) must match root ----------------

def test_exec_up_marks_started_before_on_started_hook_failure(fresh_manager, monkeypatch):
    _neutralize_runtime(fresh_manager, monkeypatch)
    _fail_lifecycle_callback(monkeypatch, "on_started")
    container = fresh_manager.containers["portainer"]

    with pytest.raises(RuntimeError, match="on_started boom"):
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


# -- partial lifecycle also reconciles removed containers (P1-10) -----------

def test_partial_up_reconciles_a_container_removed_from_installed_set(fresh_manager, monkeypatch):
    """A container that was running, then removed from the installed set,
    must be dropped from running_state even by a PARTIAL up/down of some
    other container -- not only a full-project operation."""
    _neutralize_runtime(fresh_manager, monkeypatch)
    fresh_manager.running_state._mutate(lambda current: {"portainer", "safeline"})
    fresh_manager.installed_state.remove("safeline")

    fresh_manager.compose_operations.up(names=["portainer"])

    assert "safeline" not in fresh_manager.running_state.get_persisted()
    assert "portainer" in fresh_manager.running_state.get_persisted()


def test_on_removed_hook_fires_exactly_once_per_removed_container(fresh_manager, monkeypatch):
    _neutralize_runtime(fresh_manager, monkeypatch)
    fresh_manager.running_state._mutate(lambda current: {"portainer", "safeline"})
    fresh_manager.installed_state.remove("safeline")

    calls = []
    original = LifecycleDispatcher._invoke_callback

    def patched(self, func, context=MISSING):
        if getattr(func, "__name__", "") == "on_removed":
            calls.append(1)
            return None
        return original(self, func, context)

    monkeypatch.setattr(LifecycleDispatcher, "_invoke_callback", patched)

    fresh_manager.compose_operations.up(names=["portainer"])

    assert calls == [1]
