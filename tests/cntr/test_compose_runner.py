#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ComposeRunner argument assembly.

Image preparation owns pull/build decisions. ComposeRunner only assembles the
explicit build, pull, up, stop/down, and config commands it is asked to run.
"""
import pytest

from linktools.cntr.container import ContainerError
from linktools.cntr.context import OperationContext
from linktools.cntr.runtime.compose import ComposeOptions, ComposeRunner

_PROXY_KEYS = ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
               "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")


@pytest.fixture(autouse=True)
def _no_proxy_env(monkeypatch):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)


def _ctx(manager, target_names=None, is_full=False):
    ctx = OperationContext()
    ctx.actions = ["up"]
    ctx.project_containers = manager.installed_state.get(resolve=True)
    if is_full:
        ctx.target_containers = ctx.project_containers
        ctx.is_full_project = True
    else:
        ctx.target_containers = [c for c in ctx.project_containers if c.name in (target_names or [])]
        ctx.is_full_project = False
    return ctx


def test_collect_services_full_is_empty(fresh_manager):
    runner = fresh_manager.compose_runner
    assert runner.collect_services(_ctx(fresh_manager, is_full=True)) == []


def test_collect_services_partial_collects_target_services(fresh_manager):
    runner = fresh_manager.compose_runner
    assert runner.collect_services(_ctx(fresh_manager, ["portainer"])) == ["portainer"]


def test_collect_services_no_services_raises(fresh_manager):
    runner = fresh_manager.compose_runner
    ctx = OperationContext()
    ctx.actions = ["up"]
    ctx.project_containers = []
    ctx.target_containers = []
    ctx.is_full_project = False
    with pytest.raises(ContainerError):
        runner.collect_services(ctx)


def test_up_always_disables_compose_side_build_and_pull(fresh_manager):
    runner = fresh_manager.compose_runner
    opts = ComposeOptions(services=["portainer"])
    assert runner.up_args(opts) == [
        "up", "--detach", "--no-build", "--pull", "never", "portainer",
    ]


def test_up_full_adds_remove_orphans(fresh_manager):
    runner = fresh_manager.compose_runner
    opts = ComposeOptions(remove_orphans=True, services=[])
    assert runner.up_args(opts) == [
        "up", "--detach", "--no-build", "--pull", "never", "--remove-orphans",
    ]


def test_build_pull_flag_is_explicit(fresh_manager):
    runner = fresh_manager.compose_runner
    assert runner.build_args(ComposeOptions(pull=False, services=["portainer"])) == [
        "build", "portainer",
    ]
    assert runner.build_args(ComposeOptions(pull=True, services=["portainer"])) == [
        "build", "--pull", "portainer",
    ]


def test_pull_args_ignore_buildable_services(fresh_manager):
    assert fresh_manager.compose_runner.pull_args(["portainer"]) == [
        "pull", "--ignore-buildable", "portainer",
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


def test_build_args_can_omit_proxy_build_args(fresh_manager, monkeypatch):
    monkeypatch.setenv("http_proxy", "http://proxy")
    runner = fresh_manager.compose_runner
    opts = ComposeOptions(services=["portainer"], include_proxy_build_args=False)
    assert runner.build_args(opts) == ["build", "portainer"]


def test_build_and_apply_route_args_through_process(fresh_manager, monkeypatch):
    recorded = []

    class _Proc:
        def check_call(self):
            return 0

    def fake_docker(*args, privilege=None, **kwargs):
        recorded.append(args[args.index("build") if "build" in args else args.index("up"):])
        return _Proc()

    monkeypatch.setattr(fresh_manager.runtime, "create_docker_process", fake_docker)
    runner = fresh_manager.compose_runner
    ctx = _ctx(fresh_manager, ["portainer"])
    ctx.compose_model = {"services": {"portainer": {"image": "portainer:local"}}}
    opts = ComposeOptions(services=["portainer"])
    runner.build(ctx, opts)
    runner.apply_service(ctx, "portainer")
    assert recorded[0] == ("build", "portainer")
    assert recorded[1] == (
        "up", "--detach", "--no-build", "--pull", "never", "--no-deps", "portainer",
    )
