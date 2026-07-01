#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.support.hooks import HookEvent, HookRegistry


def test_hook_registry_dispatches_registered_handler():
    registry = HookRegistry()
    seen = []

    def handler(**kwargs):
        seen.append(kwargs)

    registry.on(HookEvent.AGENT_START, handler)
    registry.fire(HookEvent.AGENT_START, agent="worker")

    assert seen == [{"agent": "worker"}]


def test_hook_registry_swallows_handler_exceptions():
    registry = HookRegistry()

    def bad_handler(**kwargs):
        raise RuntimeError("boom")

    registry.on(HookEvent.AGENT_END, bad_handler)
    registry.fire(HookEvent.AGENT_END)  # must not raise
