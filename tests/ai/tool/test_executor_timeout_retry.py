#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio

import pytest

from linktools.ai.policy.engine import PolicyEngine, ToolContext, ToolRequest
from linktools.ai.tool.executor import ToolExecutor


def test_execute_timeout_raises_asyncio_timeout_error_when_handler_exceeds_timeout():
    """Case 1: handler sleeps longer than ``timeout`` -> asyncio.TimeoutError
    propagates after retries are exhausted."""
    executor = ToolExecutor(policy=PolicyEngine(rules=()))

    async def _handler(seconds: float) -> str:
        await asyncio.sleep(seconds)
        return "done"

    async def _run():
        return await executor.execute(
            ToolRequest(tool_name="slow", arguments={"seconds": 0.5}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
            timeout=0.05,
        )

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(_run())


def test_execute_max_retries_succeeds_after_failures():
    """Case 2: handler fails twice then succeeds on the third attempt -> the
    successful result is returned."""
    executor = ToolExecutor(policy=PolicyEngine(rules=()))
    calls = {"n": 0}

    async def _handler() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError(f"transient {calls['n']}")
        return "ok"

    async def _run():
        return await executor.execute(
            ToolRequest(tool_name="flaky", arguments={}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
            max_retries=2,
        )

    assert asyncio.run(_run()) == "ok"
    assert calls["n"] == 3


def test_execute_max_retries_raises_after_all_attempts_fail():
    """Case 3: handler always fails -> its exception is raised after
    ``max_retries + 1`` total attempts."""
    executor = ToolExecutor(policy=PolicyEngine(rules=()))
    calls = {"n": 0}

    async def _handler() -> str:
        calls["n"] += 1
        raise RuntimeError(f"fail {calls['n']}")

    async def _run():
        return await executor.execute(
            ToolRequest(tool_name="broken", arguments={}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
            max_retries=1,
        )

    with pytest.raises(RuntimeError, match="fail 2"):
        asyncio.run(_run())
    assert calls["n"] == 2


def test_execute_default_no_timeout_no_retry_runs_handler_once():
    """Case 4 (regression): default timeout=None / max_retries=0 -> handler
    invoked exactly once and its result returned (current behavior)."""
    executor = ToolExecutor(policy=PolicyEngine(rules=()))
    calls = {"n": 0}

    async def _handler(value: int) -> int:
        calls["n"] += 1
        return value * 2

    async def _run():
        return await executor.execute(
            ToolRequest(tool_name="double", arguments={"value": 21}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
        )

    assert asyncio.run(_run()) == 42
    assert calls["n"] == 1
