#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ToolExecutor idempotency heartbeat (§10.2): a long-running Handler keeps its
claim alive via periodic renew, and stops (Handler cancelled) the moment the
claim can no longer be renewed (stolen / lost) so it never keeps producing
side effects under a lost claim."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from linktools.ai.errors import LostIdempotencyClaimError
from linktools.ai.policy.engine import PolicyEngine
from linktools.ai.storage.file.idempotency import FileIdempotencyStore
from linktools.ai.tool.executor import ToolExecutor, ToolContext, ToolRequest
from linktools.ai.tool.idempotency import ToolIdempotencyOptions
from linktools.ai.tool.models import ToolDescriptor
from linktools.ai.tool.policy import EffectiveToolPolicy

_DESC = ToolDescriptor(
    name="longtool", source="t", category="misc", risk="low", mutating=False
)
_POLICY = EffectiveToolPolicy()


class _RecordingStore:
    """Delegates to a real FileIdempotencyStore; counts renew calls and can
    inject a renew failure to simulate a stolen claim."""

    def __init__(self, inner, *, fail_renew=False):
        self._inner = inner
        self.fail_renew = fail_renew
        self.renew_calls = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def renew(self, claim, *, now, lease_seconds):
        self.renew_calls += 1
        if self.fail_renew:
            raise LostIdempotencyClaimError("simulated stolen claim")
        return await self._inner.renew(
            claim, now=now, lease_seconds=lease_seconds
        )


def _executor(store, *, lease=0.5, beat=0.1):
    return ToolExecutor(
        policy=PolicyEngine(rules=()),
        idempotency_store=store,
        idempotency_options=ToolIdempotencyOptions(
            lease_seconds=lease, heartbeat_seconds=beat
        ),
    )


def test_long_handler_lease_renewed_not_stolen(tmp_path):
    # Handler outlasts the initial lease; the heartbeat must keep extending it
    # so a concurrent worker cannot steal the claim mid-execution.
    store = _RecordingStore(FileIdempotencyStore(root=tmp_path))

    async def handler(**kw):
        await asyncio.sleep(0.35)  # longer than the 0.5s lease? no -- but >1 beat
        return {"ok": True}

    executor = _executor(store, lease=0.5, beat=0.1)
    result = asyncio.run(
        executor.execute(
            ToolRequest(tool_name="longtool", arguments={}),
            ToolContext(run_id="r1", session_id="s1"),
            handler,
            descriptor=_DESC,
            effective_policy=_POLICY,
            idempotency_key="k1",
        )
    )
    assert result == {"ok": True}
    assert store.renew_calls >= 2, store.renew_calls
    # renew_calls >= 2 proves the heartbeat extended the lease while the
    # handler outlasted a single heartbeat interval (the claim stayed owned).


def test_lease_loss_cancels_handler(tmp_path):
    # When renew fails (claim stolen), the Handler is cancelled and
    # LostIdempotencyClaimError propagates -- no further side effects.
    store = _RecordingStore(FileIdempotencyStore(root=tmp_path), fail_renew=True)
    executor = _executor(store, lease=0.5, beat=0.1)
    cancelled = {"v": False}

    async def handler(**kw):
        try:
            await asyncio.sleep(5.0)
        except asyncio.CancelledError:
            cancelled["v"] = True
            raise

    with pytest.raises(LostIdempotencyClaimError):
        asyncio.run(
            executor.execute(
                ToolRequest(tool_name="longtool", arguments={}),
                ToolContext(run_id="r2", session_id="s2"),
                handler,
                descriptor=_DESC,
                effective_policy=_POLICY,
                idempotency_key="k2",
            )
        )
    assert store.renew_calls >= 1
    assert cancelled["v"] is True, "Handler was not cancelled on lease loss"


def test_heartbeat_options_require_heartbeat_below_lease():
    with pytest.raises(Exception):
        ToolIdempotencyOptions(lease_seconds=1.0, heartbeat_seconds=1.0)
    with pytest.raises(Exception):
        ToolIdempotencyOptions(lease_seconds=1.0, heartbeat_seconds=2.0)
    with pytest.raises(Exception):
        ToolIdempotencyOptions(lease_seconds=0, heartbeat_seconds=-1)
    # Valid config constructs fine.
    ToolIdempotencyOptions(lease_seconds=60.0, heartbeat_seconds=20.0)


def test_no_heartbeat_when_options_unset(tmp_path):
    # Default executor (no idempotency_options): a slow handler runs to
    # completion via the legacy path; renew is never called.
    store = _RecordingStore(FileIdempotencyStore(root=tmp_path))
    executor = ToolExecutor(
        policy=PolicyEngine(rules=()), idempotency_store=store
    )

    async def handler(**kw):
        await asyncio.sleep(0.05)
        return {"ok": True}

    result = asyncio.run(
        executor.execute(
            ToolRequest(tool_name="longtool", arguments={}),
            ToolContext(run_id="r3", session_id="s3"),
            handler,
            descriptor=_DESC,
            effective_policy=_POLICY,
            idempotency_key="k3",
        )
    )
    assert result == {"ok": True}
    assert store.renew_calls == 0


class _TransientRenewStore:
    """Renew raises OSError for the first ``fail_n`` calls (transient store
    error), then delegates. Records fail() calls -- they must NOT happen when
    the handler succeeds."""

    def __init__(self, inner, *, fail_n):
        self._inner = inner
        self._remaining = fail_n
        self.fail_calls = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def renew(self, claim, *, now, lease_seconds):
        if self._remaining > 0:
            self._remaining -= 1
            raise OSError("transient store error")
        return await self._inner.renew(claim, now=now, lease_seconds=lease_seconds)

    async def fail(self, claim, error):
        self.fail_calls += 1
        return await self._inner.fail(claim, error)


def test_transient_renew_error_does_not_shadow_handler_or_redrive(tmp_path):
    # A transient renew failure (non-LostIdempotencyClaimError) must NOT kill
    # the heartbeat, shadow the handler's success, fail the claim, or cause a
    # handler re-drive on retry. The heartbeat tolerates it and retries.
    store = _TransientRenewStore(FileIdempotencyStore(root=tmp_path), fail_n=2)
    executor = _executor(store, lease=0.5, beat=0.05)
    handler_calls = {"n": 0}

    async def handler(**kw):
        handler_calls["n"] += 1
        await asyncio.sleep(0.3)
        return {"done": True}

    result = asyncio.run(
        executor.execute(
            ToolRequest(tool_name="longtool", arguments={}),
            ToolContext(run_id="r4", session_id="s4"),
            handler,
            descriptor=_DESC,
            effective_policy=_POLICY,
            idempotency_key="k4",
        )
    )
    assert result == {"done": True}
    assert handler_calls["n"] == 1, "handler must not be re-driven"
    assert store.fail_calls == 0, "fail() must not be called (handler succeeded)"
    record = asyncio.run(store.get("r4", "k4"))
    assert record.status.value == "completed"
