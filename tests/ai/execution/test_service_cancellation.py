#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cancel() must interrupt a run that is currently suspended inside a model
call, not just flip a status flag nothing in the hung coroutine is checking.
CancellationToken.raise_if_cancelled() is only awaited BETWEEN execution
points -- a model call that never returns would hang forever if cancel()
only set the token. ExecutionService routes the driving coroutine through a
RunController-tracked asyncio.Task specifically so cancel() can also call
task.cancel(), which injects CancelledError at whatever await point the
model call is currently suspended on."""

import asyncio
from datetime import timedelta

import pytest

from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.execution import service as execution_service
from linktools.ai.execution.domain import RunStatus
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.runtime import LocalDirectoryStorage, build_runtime
from tests.ai.fakes.model import make_hanging_router, make_raising_router


def _spec() -> AgentSpec:
    return AgentSpec("agent", "agent", ModelPolicy(primary="test-model"), PromptSpec("answer"))


@pytest.mark.asyncio
async def test_cancel_interrupts_a_hung_model_call(tmp_path):
    started = asyncio.Event()
    runtime = build_runtime(storage=LocalDirectoryStorage(tmp_path), model_resolver=make_hanging_router(started))

    run_task = asyncio.ensure_future(runtime.run(_spec(), "hi", session_id="s", run_id="r1", tenant_id="t1"))
    await asyncio.wait_for(started.wait(), timeout=5)

    await runtime.cancel("r1", tenant_id="t1")

    # Before the RunController wiring, cancel() only set a token nothing
    # inside the hung `await Event().wait()` was checking -- this would hang
    # forever. task.cancel() makes it resolve promptly.
    result = await asyncio.wait_for(run_task, timeout=5)
    assert result is None

    record = await runtime.execution.store.get_run("r1")
    assert record.status is RunStatus.CANCELLED
    await runtime.aclose()


@pytest.mark.asyncio
async def test_unexpected_model_error_aborts_the_run_with_persisted_error(tmp_path):
    runtime = build_runtime(storage=LocalDirectoryStorage(tmp_path), model_resolver=make_raising_router(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        await runtime.run(_spec(), "hi", session_id="s", run_id="r1", tenant_id="t1")

    record = await runtime.execution.store.get_run("r1")
    assert record.status is RunStatus.FAILED
    assert record.error.error_type == "RuntimeError"
    assert record.error.message == "boom"
    assert record.lease.owner is None
    await runtime.aclose()


@pytest.mark.asyncio
async def test_heartbeat_renews_the_lease_of_a_long_running_execution(tmp_path, monkeypatch):
    # The claim's lease would otherwise expire mid-run (default duration 5
    # minutes) and let another worker mistake a still-RUNNING execution for
    # an abandoned one. Shrink the interval so the test doesn't wait minutes.
    monkeypatch.setattr(execution_service, "_HEARTBEAT_INTERVAL", timedelta(milliseconds=10))
    started = asyncio.Event()
    runtime = build_runtime(storage=LocalDirectoryStorage(tmp_path), model_resolver=make_hanging_router(started))

    run_task = asyncio.ensure_future(runtime.run(_spec(), "hi", session_id="s", run_id="r1", tenant_id="t1"))
    await asyncio.wait_for(started.wait(), timeout=5)

    record = await runtime.execution.store.get_run("r1")
    first_expiry = record.lease.expires_at
    for _ in range(50):
        await asyncio.sleep(0.01)
        record = await runtime.execution.store.get_run("r1")
        if record.lease.expires_at > first_expiry:
            break
    assert record.lease.expires_at > first_expiry

    await runtime.cancel("r1", tenant_id="t1")
    await asyncio.wait_for(run_task, timeout=5)
    await runtime.aclose()
