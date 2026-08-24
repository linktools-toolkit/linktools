#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI up/restart/down routing through ComposeRunner.

Drives the real Command methods end-to-end with runtime image preparation and
create_docker_compose_process replaced by deterministic fakes, then asserts the
recorded docker-compose arguments for each command.
"""
import linktools.cntr.__main__ as cntr_main
import linktools.cntr.commands._shared as cntr_shared
from linktools.cntr.lifecycle.dispatcher import LifecycleDispatcher
from linktools.cntr.lifecycle.hooks import HookRegistry
from linktools.cntr.runtime.images import ImagePlan

_PROXY_KEYS = ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
               "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")


def _record(manager, monkeypatch, image_plan=None):
    """Replace runtime-dependent preparation with recorders; neutralize hooks."""
    recorded = []

    def fake(containers, *args, privilege=None, **kwargs):
        recorded.append(args)

        class _Proc:
            def check_call(self):
                return 0

        return _Proc()

    if image_plan is None:
        image_plan = ImagePlan((), (), ("portainer",))
    monkeypatch.setattr(manager.runtime, "create_docker_compose_process", fake)
    monkeypatch.setattr(manager.compose_runner, "final_model", lambda context: {"services": {}})
    monkeypatch.setattr(
        manager.image_preparer,
        "plan",
        lambda model, services=(), force_pull=False: image_plan,
    )
    monkeypatch.setattr(LifecycleDispatcher, "_invoke_callback", lambda self, func, context=None: None)
    monkeypatch.setattr(HookRegistry, "call", lambda self, phase, context=None, reverse=False: None)
    return recorded


def test_cli_up_partial_records_image_build_then_up(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    recorded = _record(
        fresh_manager,
        monkeypatch,
        ImagePlan(("portainer",), (), ("portainer",)),
    )

    cntr_main.command.on_command_up(names=["portainer"], pull=False)

    assert ("build", "portainer") in recorded
    assert ("up", "--detach", "--no-build", "--pull", "never", "portainer") in recorded


def test_cli_restart_partial_uses_stop_then_up(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    recorded = _record(fresh_manager, monkeypatch)

    cntr_main.command.on_command_restart(names=["portainer"], pull=False)

    assert ("stop", "portainer") in recorded
    assert not any(args and args[0] in ("build", "pull") for args in recorded)
    assert ("up", "--detach", "--no-build", "--pull", "never", "portainer") in recorded


def test_cli_down_full_records_down(monkeypatch, fresh_manager):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    recorded = _record(fresh_manager, monkeypatch)

    cntr_main.command.on_command_down(names=None)

    assert ("down",) in recorded


def test_cli_up_force_pull_routes_explicit_pull_before_up(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    recorded = _record(
        fresh_manager,
        monkeypatch,
        ImagePlan((), ("portainer",), ("portainer",)),
    )

    cntr_main.command.on_command_up(names=["portainer"], pull=True)

    assert ("pull", "--ignore-buildable", "portainer") in recorded
    assert ("up", "--detach", "--no-build", "--pull", "never", "portainer") in recorded


def test_only_one_manager_singleton_backs_the_cli():
    import linktools.cntr.__main__ as main_module
    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.manager import ContainerManager

    assert isinstance(main_module.command, main_module.Command)
    assert isinstance(cntr_shared.manager, ContainerManager)


def test_root_command_mounts_subcommands_in_order():
    import linktools.cntr.__main__ as main_module

    subcommands = main_module.Command().init_subcommands()
    wrapped_names = [type(sub.command).__name__ for sub in subcommands[1:]]
    assert wrapped_names == [
        "ExecCommand", "ComposeCommand", "ConfigCommand", "RepoCommand",
    ]
