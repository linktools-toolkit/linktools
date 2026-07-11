#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI up/restart/down routing through ComposeRunner.

Drives the real Command methods end-to-end with create_docker_compose_process
replaced by a recorder and lifecycle hooks neutralized, then asserts the
recorded docker-compose arguments for each command.
"""
import linktools.cntr.__main__ as cntr_main
import linktools.cntr.commands._shared as cntr_shared
from linktools.cntr.lifecycle.dispatcher import LifecycleDispatcher
from linktools.cntr.lifecycle.hooks import HookRegistry

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

    monkeypatch.setattr(manager.runtime, "create_docker_compose_process", fake)
    # Avoid running real mkdir/chown hooks and on_check/on_starting/... calls
    # during the routing check.
    monkeypatch.setattr(LifecycleDispatcher, "_invoke_callback", lambda self, func, context=None: None)
    monkeypatch.setattr(HookRegistry, "call", lambda self, phase, context=None, reverse=False: None)
    return recorded


def test_cli_up_partial_records_exact_args(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    recorded = _record(fresh_manager, monkeypatch)

    cntr_main.command.on_command_up(names=["portainer"], build=True, pull=False)

    assert ("build", "--pull=false", "portainer") in recorded
    assert ("up", "--detach", "--no-build", "--pull", "missing", "portainer") in recorded


def test_cli_restart_partial_omits_default_pull(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    recorded = _record(fresh_manager, monkeypatch)

    cntr_main.command.on_command_restart(names=["portainer"], build=True, pull=False)

    assert ("stop", "portainer") in recorded
    # restart sets emit_default_pull=False -> no --pull=false / --pull missing.
    assert ("build", "portainer") in recorded
    assert ("up", "--detach", "--no-build", "portainer") in recorded


def test_cli_down_full_records_down(monkeypatch, fresh_manager):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    recorded = _record(fresh_manager, monkeypatch)

    cntr_main.command.on_command_down(names=None)

    # Full down runs `docker compose down` with no service args.
    assert ("down",) in recorded


def test_cli_up_pull_true_uses_always(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    recorded = _record(fresh_manager, monkeypatch)

    cntr_main.command.on_command_up(names=["portainer"], build=False, pull=True)

    assert ("build", "--pull", "portainer") not in recorded  # build=False
    assert ("up", "--detach", "--no-build", "--pull", "always", "portainer") in recorded


def test_only_one_manager_singleton_backs_the_cli():
    import linktools.cntr.__main__ as main_module
    from linktools.cntr.manager import ContainerManager

    assert isinstance(main_module.command, main_module.Command)
    assert isinstance(main_module.manager, ContainerManager)


def test_root_command_mounts_subcommands_in_order():
    import linktools.cntr.__main__ as main_module

    subcommands = main_module.Command().init_subcommands()
    wrapped_names = [type(sub.command).__name__ for sub in subcommands[1:]]
    assert wrapped_names == [
        "ExecCommand", "ConfigCommand", "RepoCommand", "ComposeCommand", "PlanCommand", "LockCommand", "DiffCommand",
    ]
