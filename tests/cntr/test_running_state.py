#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Running-state consistency.

RunningStateStore owns RUNNING_CONTAINERS, updated only after a successful
compose run (partial up/down adjust just the target names; full down clears
it). These tests cover the store directly and via the CLI down/up flows
(process replaced by a recorder, hooks neutralized).
"""
import pytest

import linktools.cntr.__main__ as cntr_main
import linktools.cntr.commands._shared as cntr_shared
from linktools.cntr.context import EventContext
from linktools.cntr.lifecycle.dispatcher import LifecycleDispatcher
from linktools.cntr.lifecycle.hooks import HookRegistry
from linktools.cntr.state.running import RuntimeStateUnavailable

_PROXY_KEYS = ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
               "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")


def _record(manager, monkeypatch, fail=False):
    def fake(containers, *args, privilege=None, **kwargs):
        class _Proc:
            def check_call(self):
                if fail:
                    raise RuntimeError("compose failed")
                return 0
        return _Proc()
    monkeypatch.setattr(manager.runtime, "create_docker_compose_process", fake)
    monkeypatch.setattr(LifecycleDispatcher, "_invoke_callback", lambda self, func, context=None: None)
    monkeypatch.setattr(HookRegistry, "call", lambda self, phase, context=None, reverse=False: None)


def _partial_ctx(manager, name):
    ctx = EventContext()
    ctx.commands = ["up"]
    ctx.containers = manager.installed_state.get(resolve=True)
    ctx.target_containers = [c for c in ctx.containers if c.name == name]
    ctx.is_full_containers = False
    return ctx


# --- RunningStateStore unit tests ---

def test_get_actual_with_no_containers_returns_empty(fresh_manager):
    # Nothing to build a --file set from -- trivially nothing running, not
    # an unavailable/unqueryable runtime.
    assert fresh_manager.running_state.get_actual([]) == []


def test_get_actual_raises_unavailable_when_docker_binary_is_missing(fresh_manager):
    # This sandbox has no `docker` binary; querying a real container's actual
    # state must degrade to RuntimeStateUnavailable, not crash.
    container = fresh_manager.containers["nginx"]
    with pytest.raises(RuntimeStateUnavailable):
        fresh_manager.running_state.get_actual([container])


def test_get_effective_falls_back_to_persisted(fresh_manager):
    fresh_manager.running_state._set(["portainer"])
    container = fresh_manager.containers["nginx"]
    assert fresh_manager.running_state.get_effective([container]) == ["portainer"]


def test_get_actual_treats_output_error_as_unavailable_too(fresh_manager, monkeypatch):
    """Unlike the explicit `ct-cntr status` command, `list`'s get_actual/
    get_effective must never crash on a structurally invalid response --
    it degrades to persisted state the same as an unqueryable runtime."""
    from linktools.cntr.runtime.inspect import RuntimeInspectionOutputError

    def raise_error(containers, allow_sudo_prompt=False):
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
    ctx = EventContext()
    ctx.commands = ["up"]
    ctx.containers = fresh_manager.installed_state.get(resolve=False)
    ctx.target_containers = ctx.containers
    ctx.is_full_containers = True
    fresh_manager.running_state.mark_started(ctx)
    persisted = set(fresh_manager.running_state.get_persisted())
    assert persisted == {c.name for c in ctx.containers}
    assert "stale" not in persisted


def test_mark_stopped_full_clears(fresh_manager):
    fresh_manager.running_state._set(["nginx", "portainer"])
    ctx = EventContext()
    ctx.commands = ["down"]
    ctx.containers = fresh_manager.installed_state.get(resolve=False)
    ctx.target_containers = ctx.containers
    ctx.is_full_containers = True
    fresh_manager.running_state.mark_stopped(ctx)
    assert fresh_manager.running_state.get_persisted() == []


# --- CLI integration: marks fire after a successful run ---

def test_cli_partial_up_marks_only_target(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    _record(fresh_manager, monkeypatch)
    cntr_main.command.on_command_up(names=["portainer"], build=False, pull=False)
    running = set(fresh_manager.running_state.get_persisted())
    assert "portainer" in running
    assert "nginx" not in running


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
    with pytest.raises(RuntimeError):
        cntr_main.command.on_command_up(names=["portainer"], build=False, pull=False)
    assert fresh_manager.running_state.get_persisted() == []


# -- RunningStateStore.remove() --------------------------------------------

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
    """A container that's known (still registered/loadable) but no longer
    in the installed/full-project set must be dropped from the persisted
    running set once notify_remove sees a full-project context, via
    RunningStateStore.remove() -- not the deleted Manager wrapper."""
    from linktools.cntr.context import EventContext
    fresh_manager.running_state._set(["nginx", "flare"])

    ctx = EventContext()
    ctx.commands = ["up"]
    # "flare" is still a known container (fresh_manager.containers), but is
    # no longer part of the current full-project set being installed.
    ctx.containers = [c for c in fresh_manager.containers.values() if c.name != "flare"]
    ctx.target_containers = ctx.containers
    ctx.is_full_containers = True

    with fresh_manager.lifecycle.notify_remove(ctx):
        pass

    assert "flare" not in fresh_manager.running_state.get_persisted()
    assert "nginx" in fresh_manager.running_state.get_persisted()
