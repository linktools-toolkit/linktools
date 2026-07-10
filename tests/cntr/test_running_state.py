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
    monkeypatch.setattr(manager, "create_docker_compose_process", fake)
    monkeypatch.setattr(manager, "_callback", lambda *a, **k: None)


def _partial_ctx(manager, name):
    ctx = EventContext()
    ctx.commands = ["up"]
    ctx.containers = manager.get_installed_containers(resolve=True)
    ctx.target_containers = [c for c in ctx.containers if c.name == name]
    ctx.is_full_containers = False
    return ctx


# --- RunningStateStore unit tests ---

def test_get_actual_raises_unavailable(fresh_manager):
    with pytest.raises(RuntimeStateUnavailable):
        fresh_manager.running_state.get_actual([])


def test_get_effective_falls_back_to_persisted(fresh_manager):
    fresh_manager.running_state._set(["portainer"])
    assert fresh_manager.running_state.get_effective([]) == ["portainer"]


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
    ctx.containers = fresh_manager.get_installed_containers(resolve=False)
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
    ctx.containers = fresh_manager.get_installed_containers(resolve=False)
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
