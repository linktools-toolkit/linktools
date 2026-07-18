#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Executor execution/commit separation (§9.1 / §9.3): a Handler that succeeds
must never be re-invoked because the fenced result commit failed.

The legacy executor ran ``complete()`` inside the Handler retry loop, so a
commit failure was caught like a transient handler error and the Handler was
re-run -- re-executing any side effect. The §9 fix records an EXECUTED receipt
between the Handler and the commit, and moves the commit out of the retry loop.
"""

import asyncio

import pytest

from linktools.ai.errors import ToolCommitError
from linktools.ai.policy.engine import PolicyEngine
from linktools.ai.storage.file.idempotency import FileIdempotencyStore
from linktools.ai.tool.executor import ToolExecutor, ToolRequest, ToolContext
from linktools.ai.tool.idempotency import IdempotencyStatus
from linktools.ai.tool.models import ToolDescriptor
from linktools.ai.tool.policy import EffectiveToolPolicy


_DESC = ToolDescriptor(
    name="charge", source="test", category="misc", risk="high", mutating=True
)
_POLICY = EffectiveToolPolicy()


class _FailingStore:
    """Wraps a real FileIdempotencyStore. ``fail_complete`` / ``fail_mark``
    inject a commit-phase failure to prove the Handler is not re-run."""

    def __init__(self, inner, *, fail_complete=False, fail_mark=False):
        self._inner = inner
        self.fail_complete = fail_complete
        self.fail_mark = fail_mark
        self.mark_calls = 0
        self.complete_calls = 0

    def __getattr__(self, name):
        # Delegate anything not overridden (e.g. internal helpers) to the real
        # store so the state machine behaves exactly as in production.
        return getattr(self._inner, name)

    async def claim(self, **kw):
        return await self._inner.claim(**kw)

    async def get(self, scope, key):
        return await self._inner.get(scope, key)

    async def mark_executed(self, claim, result):
        self.mark_calls += 1
        if self.fail_mark:
            raise RuntimeError("mark_executed injected failure")
        return await self._inner.mark_executed(claim, result)

    async def complete(self, claim, result):
        self.complete_calls += 1
        if self.fail_complete:
            raise RuntimeError("complete injected failure")
        return await self._inner.complete(claim, result)

    async def fail(self, claim, error):
        return await self._inner.fail(claim, error)

    async def mark_unknown(self, claim):
        return await self._inner.mark_unknown(claim)


def _executor(store):
    return ToolExecutor(policy=PolicyEngine(rules=()), idempotency_store=store)


def _execute(executor, store, run_id, key, handler):
    return asyncio.run(
        executor.execute(
            ToolRequest(tool_name="charge", arguments={}),
            ToolContext(run_id=run_id, session_id="s1"),
            handler,
            descriptor=_DESC,
            effective_policy=_POLICY,
            idempotency_key=key,
        )
    )


def test_handler_runs_once_when_complete_fails(tmp_path):
    # §9.1: a commit failure after the Handler returned must never re-invoke it.
    calls = {"n": 0}

    async def handler(**kwargs):
        calls["n"] += 1
        return {"charged": True}

    inner = FileIdempotencyStore(root=tmp_path)
    store = _FailingStore(inner, fail_complete=True)
    executor = _executor(store)

    with pytest.raises(ToolCommitError):
        _execute(executor, store, "r1", "k1", handler)
    assert calls["n"] == 1, "Handler was re-invoked after a commit failure"
    # The receipt landed and was left EXECUTED for recovery (not UNKNOWN).
    rec = asyncio.run(store.get("r1", "k1"))
    assert rec.status is IdempotencyStatus.EXECUTED
    assert rec.result == {"charged": True}
def test_handler_runs_once_when_mark_executed_fails(tmp_path):
    calls = {"n": 0}

    async def handler(**kwargs):
        calls["n"] += 1
        return {"charged": True}

    inner = FileIdempotencyStore(root=tmp_path)
    store = _FailingStore(inner, fail_mark=True)
    executor = _executor(store)

    with pytest.raises(ToolCommitError):
        _execute(executor, store, "r2", "k2", handler)
    assert calls["n"] == 1
    # No receipt stored -> outcome unknowable -> UNKNOWN.
    rec = asyncio.run(store.get("r2", "k2"))
    assert rec.status is IdempotencyStatus.UNKNOWN


def test_successful_execution_marked_executed_then_completed(tmp_path):
    async def handler(**kwargs):
        return {"ok": 1}

    store = _FailingStore(FileIdempotencyStore(root=tmp_path))
    executor = _executor(store)

    result = _execute(executor, store, "r3", "k3", handler)
    assert result == {"ok": 1}
    assert store.mark_calls == 1
    assert store.complete_calls == 1
    rec = asyncio.run(store.get("r3", "k3"))
    assert rec.status is IdempotencyStatus.COMPLETED


def test_handler_failure_marks_failed_without_receipt(tmp_path):
    async def handler(**kwargs):
        raise RuntimeError("boom")

    store = _FailingStore(FileIdempotencyStore(root=tmp_path))
    executor = _executor(store)

    with pytest.raises(RuntimeError):
        _execute(executor, store, "r4", "k4", handler)
    # Handler never succeeded -> no EXECUTED receipt written.
    assert store.mark_calls == 0
    rec = asyncio.run(store.get("r4", "k4"))
    assert rec.status is IdempotencyStatus.FAILED


def test_executed_record_replays_without_re_running_handler(tmp_path):
    # §9.6 / §9.7 precondition: an EXECUTED record (crash between mark_executed
    # and complete) is safe to replay on a later claim -- the Handler is NOT
    # invoked again; the stored receipt is returned.
    from linktools.ai.tool.idempotency import compute_request_hash

    inner = FileIdempotencyStore(root=tmp_path)
    real_hash = compute_request_hash("charge", {}, "r5")
    cr = asyncio.run(
        inner.claim(scope="r5", key="k5", request_hash=real_hash, owner_id="w1")
    )
    asyncio.run(inner.mark_executed(cr.claim, {"charged": True}))

    calls = {"n": 0}

    async def handler(**kwargs):
        calls["n"] += 1
        return {"charged": True}

    store = _FailingStore(inner)
    executor = _executor(store)
    result = asyncio.run(
        executor.execute(
            ToolRequest(tool_name="charge", arguments={}),
            ToolContext(run_id="r5", session_id="s1"),
            handler,
            descriptor=_DESC,
            effective_policy=_POLICY,
            idempotency_key="k5",
        )
    )
    assert result == {"charged": True}
    assert calls["n"] == 0, "Handler was re-invoked on an EXECUTED record"
    assert store.mark_calls == 0
    assert store.complete_calls == 0


def test_mark_unknown_rejected_on_executed_preserves_receipt(tmp_path):
    # mark_unknown accepts RESERVED only. An EXECUTED receipt already holds a
    # recoverable result; downgrading it to UNKNOWN would make it inaccessible
    # (a later claim CONFLICTs instead of replaying). The receipt must survive.
    from linktools.ai.errors import LostIdempotencyClaimError

    inner = FileIdempotencyStore(root=tmp_path)
    cr = asyncio.run(
        inner.claim(scope="r6", key="k6", request_hash="h", owner_id="w1")
    )
    asyncio.run(inner.mark_executed(cr.claim, {"charged": True}))
    # mark_unknown on the EXECUTED record is rejected (fence miss, not a no-op
    # overwrite) -- the receipt stays EXECUTED + replayable.
    with pytest.raises(LostIdempotencyClaimError):
        asyncio.run(inner.mark_unknown(cr.claim))
    rec = asyncio.run(inner.get("r6", "k6"))
    assert rec.status is IdempotencyStatus.EXECUTED
    assert rec.result == {"charged": True}
