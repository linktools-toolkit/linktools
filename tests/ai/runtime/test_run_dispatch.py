#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RunDispatcher (plan §4.1 "消除隐式循环"): AgentEngine.dispatch() adapts run()
to the narrow Protocol, and the build kernel's _LateBoundRunDispatcher defers
binding to the real dispatcher until it exists."""

import asyncio

import pytest

from linktools.ai._runtime.build import _LateBoundRunDispatcher
from linktools.ai.run.dispatch import RunDispatchRequest


class _FakeDispatcher:
    async def dispatch(self, request):
        return ("dispatched", request)


def test_late_bound_dispatcher_raises_before_bind():
    handle = _LateBoundRunDispatcher()

    async def _run():
        with pytest.raises(RuntimeError, match="before bind"):
            await handle.dispatch(RunDispatchRequest(agent=None, input=None, context=None))

    asyncio.run(_run())


def test_late_bound_dispatcher_delegates_after_bind():
    handle = _LateBoundRunDispatcher()
    target = _FakeDispatcher()
    handle.bind(target)
    request = RunDispatchRequest(agent="agent", input="input", context="context")

    async def _run():
        return await handle.dispatch(request)

    result = asyncio.run(_run())
    assert result == ("dispatched", request)
