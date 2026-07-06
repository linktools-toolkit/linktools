#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool-level idempotency tests for ToolExecutor.execute (review doc §11).

ToolExecutor now consults a persistent IdempotencyStore instead of an
in-process dict. Same (scope, key) + same request hash -> handler runs
once and the cached result is returned on subsequent calls; same (scope,
key) + different request hash -> IdempotencyConflictError; no store
(default None) -> no caching, the legacy per-call behavior."""
import asyncio

import pytest

from linktools.ai.errors import IdempotencyConflictError, IdempotencyInProgressError
from linktools.ai.policy.engine import PolicyEngine, ToolContext, ToolRequest
from linktools.ai.storage.file.idempotency import FileIdempotencyStore
from linktools.ai.tool.executor import ToolExecutor


def _file_store(tmp_path) -> FileIdempotencyStore:
    return FileIdempotencyStore(root=tmp_path / "idem")


# ---------------------------------------------------------------------------
# 1. Same idempotency_key + same arguments: handler runs once; the second
#    call returns the cached COMPLETED result without re-invoking the handler.
# ---------------------------------------------------------------------------


def test_same_idempotency_key_calls_handler_once_and_returns_cached_result(tmp_path):
    store = _file_store(tmp_path)
    executor = ToolExecutor(policy=PolicyEngine(rules=()), idempotency_store=store)
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


# ---------------------------------------------------------------------------
# 2. Same idempotency_key + DIFFERENT arguments -> IdempotencyConflictError
#    (request_hash includes the arguments, so the second reserve sees a hash
#    mismatch). Spec §11.3 -- the hash covers tool_name + normalized args +
#    scope.
# ---------------------------------------------------------------------------


def test_same_key_with_different_args_raises_conflict(tmp_path):
    store = _file_store(tmp_path)
    executor = ToolExecutor(policy=PolicyEngine(rules=()), idempotency_store=store)
    calls = {"n": 0}

    async def _handler(value: int) -> int:
        calls["n"] += 1
        return value

    async def _run():
        await executor.execute(
            ToolRequest(tool_name="echo", arguments={"value": 1}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
            idempotency_key="shared-key",
        )
        # Same key, different arguments -> the second reserve sees a hash
        # mismatch and raises before the handler is invoked.
        with pytest.raises(IdempotencyConflictError):
            await executor.execute(
                ToolRequest(tool_name="echo", arguments={"value": 2}),
                ToolContext(run_id="r1", session_id="s1"),
                _handler,
                idempotency_key="shared-key",
            )

    asyncio.run(_run())
    assert calls["n"] == 1, "the conflicting call must not invoke the handler"


# ---------------------------------------------------------------------------
# 3. Default idempotency_store=None: no caching -- the handler runs on every
#    call even when an idempotency_key is supplied (key is simply ignored).
# ---------------------------------------------------------------------------


def test_no_idempotency_store_disables_caching_and_handler_runs_each_call():
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
    assert calls["n"] == 2, "without a store the handler must run on every call"


# ---------------------------------------------------------------------------
# 4. Different idempotency_keys do not collide: handler runs once per key.
# ---------------------------------------------------------------------------


def test_different_idempotency_keys_do_not_collide(tmp_path):
    store = _file_store(tmp_path)
    executor = ToolExecutor(policy=PolicyEngine(rules=()), idempotency_store=store)
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
    assert calls["n"] == 2, "different keys must not collide"
    assert a == 2 and b == 4


# ---------------------------------------------------------------------------
# 5. Same idempotency_key under DIFFERENT run_id does not collide (scope is
#    part of the (scope, key) primary key) -- so the same key can be reused
#    across different runs without conflict.
# ---------------------------------------------------------------------------


def test_same_key_under_different_run_id_does_not_collide(tmp_path):
    store = _file_store(tmp_path)
    executor = ToolExecutor(policy=PolicyEngine(rules=()), idempotency_store=store)
    calls = {"n": 0}

    async def _handler(value: int) -> int:
        calls["n"] += 1
        return value * 3

    async def _run():
        a = await executor.execute(
            ToolRequest(tool_name="triple", arguments={"value": 5}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
            idempotency_key="shared",
        )
        b = await executor.execute(
            ToolRequest(tool_name="triple", arguments={"value": 5}),
            ToolContext(run_id="r2", session_id="s1"),
            _handler,
            idempotency_key="shared",
        )
        return a, b

    a, b = asyncio.run(_run())
    assert calls["n"] == 2
    assert a == 15 and b == 15


# ---------------------------------------------------------------------------
# 6. Failed handler -> record is FAILED with the error string; a retry with
#    the same key proceeds (FAILED allows retry per §11.2) and on success the
#    record is overwritten with COMPLETED.
# ---------------------------------------------------------------------------


def test_failed_then_succeed_re_invokes_handler_and_eventually_completes(tmp_path):
    store = _file_store(tmp_path)
    executor = ToolExecutor(policy=PolicyEngine(rules=()), idempotency_store=store)
    calls = {"n": 0}

    async def _handler() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return "ok"

    async def _run():
        with pytest.raises(RuntimeError):
            await executor.execute(
                ToolRequest(tool_name="flaky", arguments={}),
                ToolContext(run_id="r1", session_id="s1"),
                _handler,
                idempotency_key="k",
            )
        # The failed reservation is persisted as FAILED.
        failed = await store.get("r1", "k")
        assert failed is not None
        assert failed.status.value == "failed"
        assert "transient" in (failed.error or "")
        # Retry: same key + same args -> reserve returns the FAILED record,
        # executor falls through to re-invoke the handler, and on success
        # overwrites the record with COMPLETED.
        result = await executor.execute(
            ToolRequest(tool_name="flaky", arguments={}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
            idempotency_key="k",
        )
        return result

    result = asyncio.run(_run())
    assert result == "ok"
    assert calls["n"] == 2
    final = asyncio.run(store.get("r1", "k"))
    assert final is not None
    assert final.status.value == "completed"
    assert final.result == "ok"


# ---------------------------------------------------------------------------
# 7. RESERVED record (in-progress call) -> IdempotencyInProgressError. We
#    seed the store with a RESERVED record directly, simulating a concurrent
#    in-flight call, then assert the executor refuses to re-invoke the handler.
# ---------------------------------------------------------------------------


def test_reserved_record_blocks_second_call_with_in_progress_error(tmp_path):
    store = _file_store(tmp_path)
    executor = ToolExecutor(policy=PolicyEngine(rules=()), idempotency_store=store)
    calls = {"n": 0}

    async def _handler() -> str:
        calls["n"] += 1
        return "should-not-reach"

    async def _run():
        # Seed a RESERVED record with the EXACT request hash the executor will
        # compute (same tool_name, same arguments, same scope) so reserve()
        # returns the existing record instead of raising conflict.
        from linktools.ai.tool.idempotency import compute_request_hash

        expected_hash = compute_request_hash("tool-x", {"a": 1}, "r1")
        seeded = await store.reserve("r1", "key-r", expected_hash)
        assert seeded is None, "fixture seed: first reserve is fresh"
        # Now the executor hits the RESERVED record and raises.
        with pytest.raises(IdempotencyInProgressError):
            await executor.execute(
                ToolRequest(tool_name="tool-x", arguments={"a": 1}),
                ToolContext(run_id="r1", session_id="s1"),
                _handler,
                idempotency_key="key-r",
            )

    asyncio.run(_run())
    assert calls["n"] == 0, "the in-progress reservation must NOT invoke the handler"


# ---------------------------------------------------------------------------
# 8. IdempotencyStore survives "process restart": a fresh ToolExecutor wired
#    to the same on-disk store sees the previously-completed record. (§11.1:
#    "禁止仅使用进程内字典" -- the whole point of persistence.)
# ---------------------------------------------------------------------------


def test_completed_record_survives_executor_replacement(tmp_path):
    store = _file_store(tmp_path)
    executor_a = ToolExecutor(policy=PolicyEngine(rules=()), idempotency_store=store)
    calls = {"n": 0}

    async def _handler(value: int) -> int:
        calls["n"] += 1
        return value + 1

    async def _first_run():
        return await executor_a.execute(
            ToolRequest(tool_name="inc", arguments={"value": 41}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
            idempotency_key="op-9",
        )

    result_a = asyncio.run(_first_run())
    assert result_a == 42
    assert calls["n"] == 1

    # New executor, same on-disk store: the second call is a cache hit -- the
    # handler is NOT invoked.
    executor_b = ToolExecutor(policy=PolicyEngine(rules=()), idempotency_store=store)

    async def _second_run():
        return await executor_b.execute(
            ToolRequest(tool_name="inc", arguments={"value": 41}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
            idempotency_key="op-9",
        )

    result_b = asyncio.run(_second_run())
    assert result_b == 42
    assert calls["n"] == 1, "the new executor must hit the persisted cache"
