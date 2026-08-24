#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ComposeRunner argument assembly for strict image preparation.

Build and pull decisions happen before ``compose up``. The final up command is
always ``--no-build --pull never``; build ``--pull`` and proxy args belong only
to the explicit build phase.
"""
import pytest

from linktools.cntr.container import ContainerError
from linktools.cntr.context import EventContext
from linktools.cntr.runtime.compose import ComposeOptions, ComposeRunner

_PROXY_KEYS = ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
               "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")


@pytest.fixture(autouse=True)
def _no_proxy_env(monkeypatch):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)


def _ctx(manager, target_names=None, is_full=False):
    ctx = EventContext()
    ctx.commands = ["up"]
    ctx.containers = manager.installed_state.get(resolve=True)
    if is_full:
        ctx.target_containers = ctx.containers
        ctx.is_full_containers = True
    else:
        ctx.target_containers = [c for c in ctx.containers if c.name in (target_names or [])]
        ctx.is_full_containers = False
    return ctx


def test_collect_services_full_is_empty(fresh_manager):
    runner = fresh_manager.compose_runner
    assert runner.collect_services(_ctx(fresh_manager, is_full=True)) == []


def test_collect_services_partial_collects_target_services(fresh_manager):
    runner = fresh_manager.compose_runner
    assert runner.collect_services(_ctx(fresh_manager, ["portainer"])) == ["portainer"]


def test_collect_services_no_services_raises(fresh_manager):
    runner = fresh_manager.compose_runner
    ctx = EventContext()
    ctx.commands = ["up"]
    ctx.containers = []
    ctx.target_containers = []
    ctx.is_full_containers = False
    with pytest.raises(ContainerError):
        runner.collect_services(ctx)


def test_build_without_pull_and_strict_up_args(fresh_manager):
    runner = fresh_manager.compose_runner
    opts = ComposeOptions(pull=False, services=["portainer"])
    assert runner.build_args(opts) == ["build", "portainer"]
    assert runner.up_args(opts) == [
        "up", "--detach", "--no-build", "--pull", "never", "portainer",
    ]


def test_full_up_adds_remove_orphans(fresh_manager):
    runner = fresh_manager.compose_runner
    opts = ComposeOptions(remove_orphans=True, services=[])
    assert runner.up_args(opts) == [
        "up", "--detach", "--no-build", "--pull", "never", "--remove-orphans",
    ]


def test_build_pull_true_does_not_change_strict_up(fresh_manager):
    runner = fresh_manager.compose_runner
    opts = ComposeOptions(pull=True, services=["portainer"])
    assert runner.build_args(opts) == ["build", "--pull", "portainer"]
    assert runner.up_args(opts) == [
        "up", "--detach", "--no-build", "--pull", "never", "portainer",
    ]


def test_emit_default_pull_is_compatibility_only(fresh_manager):
    runner = fresh_manager.compose_runner
    disabled = ComposeOptions(emit_default_pull=False, services=["portainer"])
    enabled = ComposeOptions(emit_default_pull=True, services=["portainer"])
    assert runner.build_args(disabled) == runner.build_args(enabled)
    assert runner.up_args(disabled) == runner.up_args(enabled)


def test_pull_args_are_explicit_and_ignore_buildable(fresh_manager):
    assert fresh_manager.compose_runner.pull_args(["one", "two"]) == [
        "pull", "--ignore-buildable", "one", "two",
    ]


def test_proxy_build_args_preserve_lower_and_upper(monkeypatch):
    runner = ComposeRunner(manager=None)
    monkeypatch.setenv("http_proxy", "http://lower")
    monkeypatch.setenv("HTTPS_PROXY", "http://upper")
    pairs = runner.collect_proxy_build_args()
    assert pairs == ["--build-arg", "http_proxy=http://lower",
                     "--build-arg", "HTTPS_PROXY=http://upper"]


def test_build_args_include_proxy_build_args_by_default(fresh_manager, monkeypatch):
    monkeypatch.setenv("http_proxy", "http://proxy")
    runner = fresh_manager.compose_runner
    opts = ComposeOptions(services=["portainer"])
    assert runner.build_args(opts) == [
        "build", "--build-arg", "http_proxy=http://proxy", "portainer",
    ]


def test_root_restart_can_omit_proxy_build_args(fresh_manager, monkeypatch):
    monkeypatch.setenv("http_proxy", "http://proxy")
    runner = fresh_manager.compose_runner
    opts = ComposeOptions(services=["portainer"], include_proxy_build_args=False)
    assert runner.build_args(opts) == ["build", "portainer"]


def test_build_and_up_route_args_through_process(fresh_manager, monkeypatch):
    recorded = []

    def fake_create(containers, *args, privilege=None, **kwargs):
        recorded.append(args)

        class _Proc:
            def check_call(self):
                return 0

        return _Proc()

    monkeypatch.setattr(fresh_manager.runtime, "create_docker_compose_process", fake_create)
    runner = fresh_manager.compose_runner
    ctx = _ctx(fresh_manager, ["portainer"])
    opts = ComposeOptions(services=["portainer"])
    runner.build(ctx, opts)
    runner.up(ctx, opts)
    assert recorded[0] == ("build", "portainer")
    assert recorded[1] == (
        "up", "--detach", "--no-build", "--pull", "never", "portainer",
    )
