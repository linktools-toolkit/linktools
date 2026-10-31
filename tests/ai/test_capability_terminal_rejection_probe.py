#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Uncaught capability non-admission settles the canonical failed execution."""

import json

import pytest

from linktools.ai.core import CommitPolicy, ExecutionStatus
from linktools.ai.errors import ErrorCode

from .test_capability_atomic_writes import _Buffer, _Capability, _opened

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("mode", ("terminal", "batch"))
@pytest.mark.parametrize("accepted_prefix", (False, True))
@pytest.mark.parametrize("hook", ("before_run", "after_run"))
async def test_uncaught_capability_buffer_rejection_settles_failed(tmp_path, monkeypatch, accepted_prefix, hook, mode):
    accepted = []
    rejected = []

    class Rejecting(_Buffer):
        async def append(self, scope, identity, key, content, *, wait):
            assert wait is False
            if json.loads(content)["key"] == "rejected":
                rejected.append(identity)
                return False
            result = await super().append(scope, identity, key, content, wait=wait)
            if result:
                accepted.append(identity)
            return result

    original = getattr(_Capability, hook)

    async def run_hook(self, ctx, **kwargs):
        result = await original(self, ctx, **kwargs)
        if self.id == "audit":
            if accepted_prefix:
                await self.submit("accepted")
            await self.submit("rejected")
        return result

    monkeypatch.setattr(_Capability, hook, run_hook)
    buffer = Rejecting()
    policy = (CommitPolicy.terminal(buffer=buffer) if mode == "terminal" else
              CommitPolicy.batch(buffer=buffer, batch_size=128, flush_interval=60))
    async with _opened(tmp_path, monkeypatch, backend="memory", policies=(policy, policy)) as opened:
        execution = await opened.runtime.agents.get().start("question")
        opened.executions.append(execution)
        opened.release.set()
        result = (await execution.wait(timeout_seconds=2)).result
        assert result.status is ExecutionStatus.FAILED
        stored = await opened.runtime._state.execution.executions.get(
            execution.execution_id, tenant_id=opened.runtime.tenant_id,
        )
        assert stored.status is result.status and stored.error_code == ErrorCode.STORAGE_CONFLICT.value
        assert stored.result.output is None
        assert opened.ready.qsize() == (0 if hook == "before_run" else 1)
        facts = await opened.probe.facts("audit")
        assert len(facts) == int(accepted_prefix)
        assert all(fact.data["key"] == "accepted" for fact in facts)
        assert (await execution.wait(timeout_seconds=2)).result.status is result.status
        assert opened.ready.qsize() == (0 if hook == "before_run" else 1)
    assert await buffer.pending() == 0
    acknowledged = [identity for _scope, identities in buffer.acknowledged for identity in identities]
    assert sorted(acknowledged) == sorted(accepted)
    assert len(accepted) == int(accepted_prefix)
    assert len(rejected) == 1 and not set(rejected).intersection(acknowledged)
