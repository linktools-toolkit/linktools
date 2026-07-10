#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-container exec up/restart/down/config routing.

Drives the BaseContainer exec subcommands end-to-end with
create_docker_compose_process replaced by a recorder and lifecycle hooks
neutralized, then asserts the recorded docker-compose args (no default
--pull flags; pull=True uniform).
"""
_PROXY_KEYS = ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
               "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")


def _record(manager, monkeypatch):
    recorded = []

    def fake(containers, *args, privilege=None, **kwargs):
        recorded.append(args)

        class _Proc:
            def check_call(self):
                return 0

        return _Proc()

    monkeypatch.setattr(manager, "create_docker_compose_process", fake)
    monkeypatch.setattr(manager, "_callback", lambda *a, **k: None)
    return recorded


def test_exec_up_omits_default_pull_flags(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    recorded = _record(fresh_manager, monkeypatch)
    fresh_manager.containers["portainer"].on_exec_up(build=True, pull=False)
    # exec never emitted --pull=false / --pull missing.
    assert ("build", "portainer") in recorded
    assert ("up", "--detach", "--no-build", "portainer") in recorded


def test_exec_up_pull_true_uses_always(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    recorded = _record(fresh_manager, monkeypatch)
    fresh_manager.containers["portainer"].on_exec_up(build=False, pull=True)
    assert ("up", "--detach", "--no-build", "--pull", "always", "portainer") in recorded


def test_exec_restart_records_stop_then_build_then_up(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    recorded = _record(fresh_manager, monkeypatch)
    fresh_manager.containers["portainer"].on_exec_restart(build=True, pull=False)
    assert recorded[0] == ("stop", "portainer")
    assert ("build", "portainer") in recorded
    assert ("up", "--detach", "--no-build", "portainer") in recorded


def test_exec_down_records_down_with_service(monkeypatch, fresh_manager):
    recorded = _record(fresh_manager, monkeypatch)
    fresh_manager.containers["portainer"].on_exec_down()
    assert ("down", "portainer") in recorded


def test_exec_config_records_config_with_service(monkeypatch, fresh_manager):
    recorded = _record(fresh_manager, monkeypatch)
    fresh_manager.containers["portainer"].on_exec_config()
    assert ("config", "portainer") in recorded
