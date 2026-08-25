#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-container exec up/restart/down/config routing."""
from linktools.cntr.lifecycle.dispatcher import LifecycleDispatcher
from linktools.cntr.lifecycle.hooks import HookRegistry
from linktools.cntr.runtime.images import ImagePlan

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

    def fake_plan(model, services=(), force_pull=False):
        targets = tuple(services)
        if force_pull:
            return ImagePlan(build=(), pull=targets, targets=targets)
        return ImagePlan(build=targets, pull=(), targets=targets)

    monkeypatch.setattr(manager.runtime, "create_docker_compose_process", fake)
    monkeypatch.setattr(manager.compose_runner, "final_model", lambda context: {"services": {}})
    monkeypatch.setattr(manager.image_preparer, "plan", fake_plan)
    monkeypatch.setattr(LifecycleDispatcher, "_invoke_callback", lambda self, func, context=None: None)
    monkeypatch.setattr(HookRegistry, "call", lambda self, phase, context=None, reverse=False: None)
    return recorded


def test_exec_up_prepares_images_then_starts(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    recorded = _record(fresh_manager, monkeypatch)
    fresh_manager.containers["portainer"].on_exec_up(pull=False)
    assert ("build", "portainer") in recorded
    assert ("up", "--detach", "--no-build", "--pull", "never", "portainer") in recorded


def test_exec_up_pull_true_routes_through_image_preparation(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    recorded = _record(fresh_manager, monkeypatch)
    fresh_manager.containers["portainer"].on_exec_up(pull=True)
    assert ("pull", "--ignore-buildable", "portainer") in recorded
    assert ("up", "--detach", "--no-build", "--pull", "never", "portainer") in recorded


def test_exec_restart_records_stop_build_then_up(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    recorded = _record(fresh_manager, monkeypatch)
    fresh_manager.containers["portainer"].on_exec_restart(pull=False)
    assert recorded[0] == ("stop", "portainer")
    assert ("build", "portainer") in recorded
    assert ("up", "--detach", "--no-build", "--pull", "never", "portainer") in recorded


def test_exec_down_records_down_with_service(monkeypatch, fresh_manager):
    recorded = _record(fresh_manager, monkeypatch)
    fresh_manager.containers["portainer"].on_exec_down()
    assert ("down", "portainer") in recorded


def test_exec_config_records_config_with_service(monkeypatch, fresh_manager):
    recorded = _record(fresh_manager, monkeypatch)
    fresh_manager.containers["portainer"].on_exec_config()
    assert ("config", "portainer") in recorded
