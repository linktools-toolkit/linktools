#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool-level idempotency tests for ToolExecutor.execute (spec section 27,
basic in-process form). Same idempotency_key -> handler invoked once and the
second call returns the cached result; different keys -> handler invoked
twice; no cache (default) -> no caching, today's behavior."""
import asyncio

import pytest

from linktools.ai.policy.engine import PolicyEngine, ToolContext, ToolRequest
from linktools.ai.tool.executor import ToolExecutor


def test_same_idempotency_key_calls_handler_once_and_returns_cached_result():
    """Two execute() calls sharing an idempotency_key: the handler runs once;
    the second call returns the cached result without re-invoking the handler."""
    cache: "dict[tuple[str, str], object]" = {}
    executor = ToolExecutor(policy=PolicyEngine(rules=()), idempotency_cache=cache)
    calls = {"n": 0}

    async def _handler(value: int) -> int:
        calls["n"] += 1
        return value * 2

    async def _run():
        first = await executor.execute(
            ToolRequest(tool_name="double", arguments={"value": 21}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
            idempotency_key="op-1",
        )
        second = await executor.execute(
            ToolRequest(tool_name="double", arguments={"value": 21}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
            idempotency_key="op-1",
        )
        return first, second

    first, second = asyncio.run(_run())
    assert calls["n"] == 1, "handler must run exactly once for a repeated idempotency key"
    assert first == 42 and second == 42, "both calls return the (cached) result"
    assert ("double", "op-1") in cache


def test_different_idempotency_keys_call_handler_twice():
    """Two execute() calls with DIFFERENT keys: handler runs twice (no cross-key
    cache hit)."""
    cache: "dict[tuple[str, str], object]" = {}
    executor = ToolExecutor(policy=PolicyEngine(rules=()), idempotency_cache=cache)
    calls = {"n": 0}

    async def _handler(value: int) -> int:
        calls["n"] += 1
        return value * 2

    async def _run():
        a = await executor.execute(
            ToolRequest(tool_name="double", arguments={"value": 1}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
            idempotency_key="key-a",
        )
        b = await executor.execute(
            ToolRequest(tool_name="double", arguments={"value": 2}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
            idempotency_key="key-b",
        )
        return a, b

    a, b = asyncio.run(_run())
    assert calls["n"] == 2, "different keys must not collide in the cache"
    assert a == 2 and b == 4


def test_no_idempotency_cache_disables_caching_and_handler_runs_each_call():
    """Default idempotency_cache=None: no caching -- the handler runs on every
    call even when an idempotency_key is supplied (key is simply ignored).
    This preserves today's pre-idempotency behavior."""
    executor = ToolExecutor(policy=PolicyEngine(rules=()))
    calls = {"n": 0}

    async def _handler(value: int) -> int:
        calls["n"] += 1
        return value

    async def _run():
        await executor.execute(
            ToolRequest(tool_name="echo", arguments={"value": 1}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
            idempotency_key="ignored",
        )
        await executor.execute(
            ToolRequest(tool_name="echo", arguments={"value": 2}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
            idempotency_key="ignored",
        )

    asyncio.run(_run())
    assert calls["n"] == 2, "without a cache the handler must run on every call"


def test_failed_handler_is_not_cached_so_retry_re_invokes_it():
    """A handler that raises is NOT cached: a later call with the same key
    re-invokes the handler (the cache only records successes). This is what
    makes the idempotency cache safe for retry -- a transient failure does not
    poison the key."""
    cache: "dict[tuple[str, str], object]" = {}
    executor = ToolExecutor(policy=PolicyEngine(rules=()), idempotency_cache=cache)
    calls = {"n": 0}

    async def _handler() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return "ok"

    async def _run():
        first_failed = False
        try:
            await executor.execute(
                ToolRequest(tool_name="flaky", arguments={}),
                ToolContext(run_id="r1", session_id="s1"),
                _handler,
                idempotency_key="k",
            )
        except RuntimeError:
            first_failed = True
        assert first_failed, "first call must propagate the handler error"
        assert ("flaky", "k") not in cache, "a failed call must not populate the cache"
        second = await executor.execute(
            ToolRequest(tool_name="flaky", arguments={}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
            idempotency_key="k",
        )
        return second

    result = asyncio.run(_run())
    assert result == "ok"
    assert calls["n"] == 2, "the failed call did not short-circuit the retry"
    assert ("flaky", "k") in cache, "the successful retry is now cached"


def test_cache_key_is_namespaced_by_tool_name():
    """The same idempotency_key under two different tool_names does NOT collide:
    cache_key is (tool_name, idempotency_key)."""
    cache: "dict[tuple[str, str], object]" = {}
    executor = ToolExecutor(policy=PolicyEngine(rules=()), idempotency_cache=cache)

    async def _add_a() -> str:
        return "from-a"

    async def _add_b() -> str:
        return "from-b"

    async def _run():
        await executor.execute(
            ToolRequest(tool_name="tool_a", arguments={}),
            ToolContext(run_id="r1", session_id="s1"),
            _add_a,
            idempotency_key="shared",
        )
        await executor.execute(
            ToolRequest(tool_name="tool_b", arguments={}),
            ToolContext(run_id="r1", session_id="s1"),
            _add_b,
            idempotency_key="shared",
        )

    asyncio.run(_run())
    assert ("tool_a", "shared") in cache
    assert ("tool_b", "shared") in cache
