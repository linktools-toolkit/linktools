#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/middleware/test_loop_guard.py"""
import pytest

from linktools.ai.errors import ToolDeniedError
from linktools.ai.middleware.loop_guard import LoopGuardMiddleware
from linktools.ai.policy.engine import ToolRequest


@pytest.mark.asyncio
async def test_allows_first_two_failures_then_blocks_third_repeat():
    middleware = LoopGuardMiddleware(max_repeats=3)
    request = ToolRequest(tool_name="terminal", arguments={"command": "flaky"})

    for _ in range(2):
        await middleware.before_tool(context=None, request=request)
        await middleware.after_tool(context=None, request=request, result={"error": "failed"})

    await middleware.before_tool(context=None, request=request)  # 3rd attempt still allowed
    await middleware.after_tool(context=None, request=request, result={"error": "failed"})

    with pytest.raises(ToolDeniedError):
        await middleware.before_tool(context=None, request=request)  # 4th attempt blocked


@pytest.mark.asyncio
async def test_success_clears_the_failure_counter():
    middleware = LoopGuardMiddleware(max_repeats=3)
    request = ToolRequest(tool_name="terminal", arguments={"command": "flaky"})

    await middleware.before_tool(context=None, request=request)
    await middleware.after_tool(context=None, request=request, result={"error": "failed"})
    await middleware.before_tool(context=None, request=request)
    await middleware.after_tool(context=None, request=request, result={"ok": True})  # success clears it

    for _ in range(3):
        await middleware.before_tool(context=None, request=request)
        await middleware.after_tool(context=None, request=request, result={"error": "failed"})
    # Counter was cleared by the success above, so we're back to only 3 consecutive
    # failures here -- the 4th call is the one that should be blocked, not this loop's calls.
    with pytest.raises(ToolDeniedError):
        await middleware.before_tool(context=None, request=request)


@pytest.mark.asyncio
async def test_different_arguments_are_tracked_independently():
    middleware = LoopGuardMiddleware(max_repeats=1)
    request_a = ToolRequest(tool_name="terminal", arguments={"command": "a"})
    request_b = ToolRequest(tool_name="terminal", arguments={"command": "b"})

    await middleware.before_tool(context=None, request=request_a)
    await middleware.after_tool(context=None, request=request_a, result={"error": "failed"})

    await middleware.before_tool(context=None, request=request_b)  # different args, not blocked
