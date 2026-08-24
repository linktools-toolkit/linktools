#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-container exec up/restart/down/config routing.

Drives the BaseContainer exec subcommands with runtime-dependent image
preparation replaced by a recorder, then locks the compose routing contract.
"""
from linktools.cntr.lifecycle.dispatcher import LifecycleDispatcher
from linktools.cntr.lifecycle.hooks import HookRegistry

_PROXY_KEYS = ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
               "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")


def _record(manager, monkeypatch):
    recorded = []
    prepared = []

    def fake(containers, *args, privilege=None, **kwargs):
        recorded.append(args)

        class _Proc:
            def check_call(self):
                return 0

        return _Proc()

    def prepare(context, model, services=(), force_pull=False):
        prepared.append((tuple(services), force_pull))
        return None

    monkeypatch.setattr(manager.runtime, "create_docker_compose_process", fake)
    monkeypatch.setattr(manager.compose_runner, "final_model", lambda context: {"services": {}})
    monkeypatch.setattr(manager.image_preparer, "execute", prepare)
    monkeypatch.setattr(LifecycleDispatcher, "_invoke_callback", lambda self, func, context=None: None)
    monkeypatch.setattr(HookRegistry, "call", lambda self, phase, context=None, reverse=False: None)
    return recorded, prepared


def test_exec_up_prepares_images_then_uses_strict_up(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    recorded, prepared = _record(fresh_manager, monkeypatch)

    fresh_manager.containers["portainer"].on_exec_up(pull=False)

    assert prepared == [(('portainer',), False)]
    assert recorded == [("up", "--detach", "--no-build", "--pull", "never", "portainer")]


def test_exec_up_pull_true_forces_image_preparation(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    recorded, prepared = _record(fresh_manager, monkeypatch)

    fresh_manager.containers["portainer"].on_exec_up(pull=True)

    assert prepared == [(('portainer',), True)]
    assert recorded == [("up", "--detach", "--no-build", "--pull", "never", "portainer")]


def test_exec_restart_records_stop_prepare_then_up(monkeypatch, fresh_manager):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    recorded, prepared = _record(fresh_manager, monkeypatch)

    fresh_manager.containers["portainer"].on_exec_restart(pull=False)

    assert prepared == [(('portainer',), False)]
    assert recorded == [
        ("stop", "portainer"),
        ("up", "--detach", "--no-build", "--pull", "never", "portainer"),
    ]


def test_exec_down_records_down_with_service(monkeypatch, fresh_manager):
    recorded, prepared = _record(fresh_manager, monkeypatch)
    fresh_manager.containers["portainer"].on_exec_down()
    assert prepared == []
    assert recorded == [("down", "portainer")]


def test_exec_config_records_config_with_service(monkeypatch, fresh_manager):
    recorded, prepared = _record(fresh_manager, monkeypatch)
    fresh_manager.containers["portainer"].on_exec_config()
    assert prepared == []
    assert recorded == [("config", "portainer")]
