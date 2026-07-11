#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Subagent max_concurrency enforces a per-ref semaphore (contract)."""

import asyncio
import pytest

from linktools.ai.run.identity import ParentRunIdentity
from linktools.ai.subagent.models import SubagentResult
from linktools.ai.subagent.toolset import build_subagent_toolset


class _ConcExecutor:
    """Records the high-water mark of concurrently in-flight executions."""
    def __init__(self):
        self.in_flight = 0
        self.max_seen = 0

    async def execute(self, **kw):
        self.in_flight += 1
        self.max_seen = max(self.max_seen, self.in_flight)
        await asyncio.sleep(0.02)
        self.in_flight -= 1
        return SubagentResult(agent_id=kw["agent_spec"].id, session_id="s", run_id="r",
                              status="succeeded")


def _toolset(max_concurrency, executor):
    return build_subagent_toolset(
        allowed_names={"a"}, subagent_provider=None, entrypoint_resolver=None,
        executor=executor, depth_provider=lambda: 0, max_depth=3,
        timeout_seconds=None, max_concurrency=max_concurrency,
        parent=ParentRunIdentity(run_id="run-1", root_run_id="run-1", session_id="sess-1"),
    )


class _Spec:
    id = "a"


@pytest.mark.asyncio
async def test_max_concurrency_one_serializes(monkeypatch):
    # Force _resolve_spec to return immediately without a real provider.
    import linktools.ai.subagent.toolset as ts
    monkeypatch.setattr(ts, "_resolve_spec", lambda *a, **k: _async_none())
    executor = _ConcExecutor()
    call = _toolset(1, executor).tools["call_subagent"].function
    await asyncio.gather(*[call("a", "t") for _ in range(4)])
    assert executor.max_seen == 1


@pytest.mark.asyncio
async def test_max_concurrency_two_allows_pair(monkeypatch):
    import linktools.ai.subagent.toolset as ts
    monkeypatch.setattr(ts, "_resolve_spec", lambda *a, **k: _async_none())
    executor = _ConcExecutor()
    call = _toolset(2, executor).tools["call_subagent"].function
    await asyncio.gather(*[call("a", "t") for _ in range(4)])
    assert executor.max_seen == 2


async def _async_none():
    return _Spec()
