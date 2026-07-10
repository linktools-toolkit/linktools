#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI up/restart/down routing through ComposeRunner (refactor spec Phase 2).

Drives the real Command methods end-to-end with create_docker_compose_process
replaced by a recorder and lifecycle hooks neutralized, then asserts the
recorded docker-compose arguments match each command's pre-refactor line.
"""
import linktools.cntr.__main__ as cntr_main

_PROXY_KEYS = ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
               "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")


def _record(manager, monkeypatch):
    """Replace create_docker_compose_process with a recorder; neutralize hooks."""
    recorded = []

    def fake(containers, *args, privilege=None, **kwargs):
        recorded.append(args)

        class _Proc:
            def check_call(self):
                return 0

        return _Proc()

    monkeypatch.setattr(manager, "create_docker_compose_process", fake)
    # Avoid running real mkdir/chown hooks during the routing check.
    monkeypatch.setattr(manager, "_callback", lambda *a, **k: None)
    return recorded


def test_cli_up_partial_records_exact_args(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cntr_main, "manager", fresh_manager)
    recorded = _record(fresh_manager, monkeypatch)

    cntr_main.command.on_command_up(names=["portainer"], build=True, pull=False)

    assert ("build", "--pull=false", "portainer") in recorded
    assert ("up", "--detach", "--no-build", "--pull", "missing", "portainer") in recorded


def test_cli_restart_partial_omits_default_pull(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cntr_main, "manager", fresh_manager)
    recorded = _record(fresh_manager, monkeypatch)

    cntr_main.command.on_command_restart(names=["portainer"], build=True, pull=False)

    assert ("stop", "portainer") in recorded
    # restart sets emit_default_pull=False -> no --pull=false / --pull missing.
    assert ("build", "portainer") in recorded
    assert ("up", "--detach", "--no-build", "portainer") in recorded


def test_cli_down_full_records_down(monkeypatch, fresh_manager):
    monkeypatch.setattr(cntr_main, "manager", fresh_manager)
    recorded = _record(fresh_manager, monkeypatch)

    cntr_main.command.on_command_down(names=None)

    # Full down runs `docker compose down` with no service args.
    assert ("down",) in recorded


def test_cli_up_pull_true_uses_always(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cntr_main, "manager", fresh_manager)
    recorded = _record(fresh_manager, monkeypatch)

    cntr_main.command.on_command_up(names=["portainer"], build=False, pull=True)

    assert ("build", "--pull", "portainer") not in recorded  # build=False
    assert ("up", "--detach", "--no-build", "--pull", "always", "portainer") in recorded
