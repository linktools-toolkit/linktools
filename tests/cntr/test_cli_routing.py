#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI up/restart/down routing through image preparation and ComposeRunner."""
import pytest

import linktools.cntr.__main__ as cntr_main
import linktools.cntr.commands._shared as cntr_shared
from linktools.cntr.artifacts import AppliedServiceModels
from linktools.cntr.errors import ContainerError
from linktools.cntr.lifecycle.dispatcher import LifecycleDispatcher
from linktools.cntr.lifecycle.hooks import HookRegistry
from types import SimpleNamespace
from linktools.cntr.runtime.images import ImagePlan
from linktools.cntr.runtime.inspect import ProjectRuntimeState

_PROXY_KEYS = ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
               "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")


def _ready_dependencies(manager, monkeypatch) -> None:
    from _harness import configure_builtin_inputs
    configure_builtin_inputs(manager)
    services = {name for container in manager.containers.values() if container.name != "portainer"
                for name in container.services}
    model = {"services": {name: {"image": name + ":current"}
                          for container in manager.containers.values() for name in container.services}}
    actual = ProjectRuntimeState(manager.project_name, tuple(
        SimpleNamespace(service=name, state="running", health="healthy", exit_code=None, labels={},
                        image_id="sha256:local-" + name + ":current") for name in sorted(services)), "docker")
    monkeypatch.setattr(manager.docker_inspector, "get_project_state", lambda containers, **kwargs: actual)
    AppliedServiceModels(manager, model).record(services)


def _record(manager, monkeypatch):
    from _harness import stub_generated_runtime
    stub_generated_runtime(manager, monkeypatch)
    _ready_dependencies(manager, monkeypatch)
    recorded = []

    def fake(*args, privilege=None, **kwargs):
        recorded.append(tuple(args[next(i for i, value in enumerate(args)
                                        if value in ("up", "down", "stop", "pull", "build", "config")):]))

        class _Proc:
            def check_call(self):
                return 0

        return _Proc()

    def fake_plan(model, services=(), force_pull=False, refresh_services=(), include_dependencies=True):
        targets = tuple(services)
        images = {spec["image"]: SimpleNamespace(image_id="sha256:local-" + spec["image"])
                  for spec in model["services"].values()}
        if force_pull:
            return ImagePlan(build=(), pull=targets, targets=targets, images=images)
        return ImagePlan(build=targets, pull=(), targets=targets, images=images)

    monkeypatch.setattr(manager.runtime, "create_docker_process", fake)
    monkeypatch.setattr(manager.compose_runner, "final_model", lambda context: {"services": {
        name: {"image": name + ":current"} for container in context.project_containers for name in container.services}})
    monkeypatch.setattr(manager.image_preparer, "plan", fake_plan)
    monkeypatch.setattr(LifecycleDispatcher, "_invoke_callback", lambda self, func, context=None: None)
    monkeypatch.setattr(HookRegistry, "call", lambda self, phase, context=None, reverse=False: None)
    return recorded


def test_cli_up_partial_records_exact_args(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    recorded = _record(fresh_manager, monkeypatch)

    cntr_main.command.on_command_up(names=["portainer"], pull=False)

    assert any(cmd[0] == "build" and "portainer" in cmd and "nginx" not in cmd for cmd in recorded)
    assert {"portainer"} == {service for cmd in recorded if cmd[0] == "up" for service in cmd[cmd.index("--no-deps") + 1:]}


def test_cli_restart_partial_records_stop_build_and_up(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    recorded = _record(fresh_manager, monkeypatch)

    cntr_main.command.on_command_restart(names=["portainer"], pull=False)

    assert any(cmd[0] == "build" and "portainer" in cmd for cmd in recorded)
    assert ("stop", "portainer") in recorded
    assert {"portainer"} == {service for cmd in recorded if cmd[0] == "up" for service in cmd[cmd.index("--no-deps") + 1:]}
    assert next(i for i, cmd in enumerate(recorded) if cmd[0] == "build") < recorded.index(("stop", "portainer"))


def test_restart_config_only_target_requires_explicit_consumer(monkeypatch, fresh_manager):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    monkeypatch.setattr(fresh_manager.containers["portainer"], "services", {})
    recorded = _record(fresh_manager, monkeypatch)

    with pytest.raises(ContainerError, match="[Nn]o runnable service"):
        cntr_main.command.on_command_restart(names=["portainer"])

    assert recorded == []


def test_restart_mixed_config_and_service_targets_stops_only_real_service(monkeypatch, fresh_manager):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    monkeypatch.setattr(fresh_manager.containers["portainer"], "services", {})
    recorded = _record(fresh_manager, monkeypatch)

    cntr_main.command.on_command_restart(names=["portainer", "nginx"])

    stops = [args for args in recorded if args[0] == "stop"]
    assert stops == [("stop", "nginx")]


def test_cli_down_full_records_down(monkeypatch, fresh_manager):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    recorded = _record(fresh_manager, monkeypatch)

    cntr_main.command.on_command_down(names=None)

    assert ("down",) in recorded


def test_cli_up_pull_true_routes_through_image_preparation(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    recorded = _record(fresh_manager, monkeypatch)

    cntr_main.command.on_command_up(names=["portainer"], pull=True)

    assert any(cmd[:2] == ("pull", "--ignore-buildable") and "portainer" in cmd for cmd in recorded)
    assert not any(cmd[0] == "build" for cmd in recorded)
    assert {"portainer"} == {service for cmd in recorded if cmd[0] == "up" for service in cmd[cmd.index("--no-deps") + 1:]}


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
