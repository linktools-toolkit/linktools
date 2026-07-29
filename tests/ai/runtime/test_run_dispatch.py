#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RunDispatcher: the build kernel's LateBoundRunDispatcher defers binding to
the real dispatcher until it exists. Covers both steps of the two-step
contract -- ``open_child`` (pure id allocation) and ``dispatch`` (lifecycle)."""

import asyncio

import pytest

from linktools.ai.execution.dispatch import ChildRunHandle, RunDispatchRequest
from linktools.ai.runtime.dispatcher import LateBoundRunDispatcher


def _handle() -> ChildRunHandle:
    return ChildRunHandle(
        run_id="child-1",
        session_id="session-1",
        root_run_id="root-1",
        parent_run_id="parent-1",
        parent_session_id=None,
        user_id=None,
        tenant_id=None,
        session_needs_create=True,
    )


class _FakeDispatcher:
    async def open_child(self, parent_context, session_policy, metadata):
        return ("open_child", parent_context, session_policy, metadata)

    async def dispatch(self, request):
        return ("dispatched", request)


def test_open_child_raises_before_bind():
    handle = LateBoundRunDispatcher()

    async def _run():
        with pytest.raises(RuntimeError, match="before bind"):
            await handle.open_child(None, None, {})

    asyncio.run(_run())


def test_dispatch_raises_before_bind():
    handle = LateBoundRunDispatcher()

    async def _run():
        with pytest.raises(RuntimeError, match="before bind"):
            await handle.dispatch(
                RunDispatchRequest(
                    compiled_agent=None, input=None, handle=_handle()
                )
            )

    asyncio.run(_run())


def test_open_child_and_dispatch_delegate_after_bind():
    handle = LateBoundRunDispatcher()
    handle.bind(_FakeDispatcher())
    request = RunDispatchRequest(
        compiled_agent="agent", input="input", handle=_handle()
    )

    async def _run():
        opened = await handle.open_child("parent", "policy", {"k": "v"})
        dispatched = await handle.dispatch(request)
        return opened, dispatched

    opened, dispatched = asyncio.run(_run())
    assert opened == ("open_child", "parent", "policy", {"k": "v"})
    assert dispatched == ("dispatched", request)
