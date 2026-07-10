#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ComposeRunner argument assembly (refactor spec Phase 2).

Verifies the unified compose command builder reproduces each path's exact
pre-refactor command line: CLI ``up`` emits --pull=false / --pull missing when
not pulling; ``restart`` and ``exec`` emit nothing; pull=True is uniform; proxy
build args preserve both cases.
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
    ctx.containers = manager.get_installed_containers(resolve=True)
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
    # portainer has a single service named "portainer".
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


def test_cli_up_no_pull_emits_default_pull_flags(fresh_manager):
    runner = fresh_manager.compose_runner
    opts = ComposeOptions(pull=False, emit_default_pull=True, services=["portainer"])
    assert runner.build_args(opts) == ["build", "--pull=false", "portainer"]
    assert runner.up_args(opts) == ["up", "--detach", "--no-build", "--pull", "missing", "portainer"]


def test_cli_up_full_adds_remove_orphans(fresh_manager):
    runner = fresh_manager.compose_runner
    opts = ComposeOptions(pull=False, emit_default_pull=True, remove_orphans=True, services=[])
    assert runner.up_args(opts) == ["up", "--detach", "--no-build", "--pull", "missing", "--remove-orphans"]


def test_pull_true_is_uniform(fresh_manager):
    runner = fresh_manager.compose_runner
    opts = ComposeOptions(pull=True, emit_default_pull=True, services=["portainer"])
    assert runner.build_args(opts) == ["build", "--pull", "portainer"]
    assert runner.up_args(opts) == ["up", "--detach", "--no-build", "--pull", "always", "portainer"]


def test_restart_and_exec_omit_default_pull_flags(fresh_manager):
    # restart and exec set emit_default_pull=False -> no --pull=false / --pull missing.
    runner = fresh_manager.compose_runner
    opts = ComposeOptions(pull=False, emit_default_pull=False, services=["portainer"])
    assert runner.build_args(opts) == ["build", "portainer"]
    assert runner.up_args(opts) == ["up", "--detach", "--no-build", "portainer"]


def test_proxy_build_args_preserve_lower_and_upper(monkeypatch):
    # Build a runner without the autouse proxy clearing for this case.
    from linktools.cntr.runtime.compose import ComposeRunner
    runner = ComposeRunner(manager=None)
    monkeypatch.setenv("http_proxy", "http://lower")
    monkeypatch.setenv("HTTPS_PROXY", "http://upper")
    pairs = runner.collect_proxy_build_args()
    assert pairs == ["--build-arg", "http_proxy=http://lower",
                     "--build-arg", "HTTPS_PROXY=http://upper"]


def test_build_args_include_proxy_build_args_by_default(fresh_manager, monkeypatch):
    # CLI `up` and both `exec up`/`exec restart` include proxy build-args.
    monkeypatch.setenv("http_proxy", "http://proxy")
    runner = fresh_manager.compose_runner
    opts = ComposeOptions(pull=False, emit_default_pull=True, services=["portainer"])
    assert runner.build_args(opts) == [
        "build", "--pull=false", "--build-arg", "http_proxy=http://proxy", "portainer",
    ]


def test_cli_restart_omits_proxy_build_args(fresh_manager, monkeypatch):
    # CLI `restart` never included proxy build-args pre-refactor -- unlike
    # `up`/`exec up`/`exec restart`, which all still do (the two tests above).
    monkeypatch.setenv("http_proxy", "http://proxy")
    runner = fresh_manager.compose_runner
    opts = ComposeOptions(pull=False, emit_default_pull=False, services=["portainer"],
                          include_proxy_build_args=False)
    assert runner.build_args(opts) == ["build", "portainer"]


def test_build_and_up_route_args_through_process(fresh_manager, monkeypatch):
    recorded = []

    def fake_create(containers, *args, privilege=None, **kwargs):
        recorded.append(args)

        class _Proc:
            def check_call(self):
                return 0

        return _Proc()

    monkeypatch.setattr(fresh_manager, "create_docker_compose_process", fake_create)
    runner = fresh_manager.compose_runner
    ctx = _ctx(fresh_manager, ["portainer"])
    opts = ComposeOptions(build=True, pull=False, emit_default_pull=True, services=["portainer"])
    runner.build(ctx, opts)
    runner.up(ctx, opts)
    assert recorded[0] == ("build", "--pull=false", "portainer")
    assert recorded[1] == ("up", "--detach", "--no-build", "--pull", "missing", "portainer")
