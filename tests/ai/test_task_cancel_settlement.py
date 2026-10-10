#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded graph cancellation preserves control and authoritative outcome ownership."""

import asyncio
from dataclasses import replace

import pytest

from linktools.ai.core import TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.task import TaskGraphState

from .test_unified_wait_contract import _Bundle


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf"), True, "1"])
async def test_invalid_budget_precedes_control(timeout) -> None:
    bundle = _Bundle()
    with pytest.raises(AIError) as caught:
        await bundle.graph_run.cancel(settle_timeout_seconds=timeout)
    assert caught.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert not bundle.runtime._cancel_settlements


@pytest.mark.asyncio
async def test_zero_budget_does_not_activate_or_submit_control() -> None:
    bundle = _Bundle()
    with pytest.raises(AIError) as caught:
        await bundle.graph_run.cancel(settle_timeout_seconds=0)
    assert caught.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    assert caught.value.safe_details == {
        "phase": "cancel_settlement", "graph_id": "graph", "reason": "deadline_unknown",
    }
    assert not bundle.runtime._cancel_settlements


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED])
async def test_settlement_returns_actual_terminal_status(status) -> None:
    bundle = _Bundle()
    requests = []

    async def cancel(graph_id, request, *, admission_guard=None):
        if admission_guard is not None:
            admission_guard()
        requests.append(request)
        bundle.graph.result = TaskGraphState(graph_id, status, (), (), 1)

    bundle.graph.cancel = cancel
    result = await bundle.graph_run.cancel(idempotency_key="intent", force=True, settle_timeout_seconds=1)
    assert result.status is status
    assert requests[0].idempotency_key == "intent"
    assert requests[0].force is True
    assert not bundle.runtime._cancel_settlements


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [TaskStatus.RUNNING, TaskStatus.WAITING, TaskStatus.RECOVERY_REQUIRED])
async def test_settlement_rejects_nonterminal_wait_stops(status) -> None:
    bundle = _Bundle()
    bundle.graph.result = TaskGraphState("graph", status, (), (), 1)

    async def cancel(graph_id, request, *, admission_guard=None):
        if admission_guard is not None:
            admission_guard()
        return None

    bundle.graph.cancel = cancel
    result = await bundle.graph_run.cancel()
    assert result.status is status
    with pytest.raises(AIError) as caught:
        await bundle.graph_run.cancel(settle_timeout_seconds=0.01)
    assert caught.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    assert caught.value.safe_details["reason"] == "deadline_nonterminal"
    assert caught.value.safe_details["graph_status"] == status.value
    assert bundle.graph.wait_calls == 0
    await asyncio.sleep(0)
    await bundle.runtime.close()


@pytest.mark.asyncio
async def test_pending_control_remains_owned_and_closes_only_after_completion() -> None:
    bundle = _Bundle()
    release = asyncio.Event()
    started = asyncio.Event()

    async def cancel(graph_id, request, *, admission_guard=None):
        if admission_guard is not None:
            admission_guard()
        started.set()
        await release.wait()

    bundle.graph.cancel = cancel
    with pytest.raises(AIError) as caught:
        await bundle.graph_run.cancel(settle_timeout_seconds=0.01)
    assert started.is_set()
    assert caught.value.safe_details["reason"] == "control_pending"
    assert caught.value.safe_details["cleanup_pending"] is True
    assert bundle.graph.state_calls == 0
    with pytest.raises(AIError):
        await bundle.runtime.close()
    assert not bundle.storage_closed
    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await bundle.runtime.close()
    assert bundle.storage_closed
    assert bundle.graph.state_calls == 0


@pytest.mark.asyncio
async def test_late_control_failure_is_retained_by_runtime() -> None:
    bundle = _Bundle()
    release = asyncio.Event()
    error = AIError(ErrorCode.AUTHORIZATION_DENIED)

    async def cancel(graph_id, request, *, admission_guard=None):
        if admission_guard is not None:
            admission_guard()
        await release.wait()
        raise error

    bundle.graph.cancel = cancel
    with pytest.raises(AIError):
        await bundle.graph_run.cancel(settle_timeout_seconds=0.01)
    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    with pytest.raises(AIError) as caught:
        await bundle.runtime.close()
    assert caught.value is error
    assert not bundle.storage_closed


@pytest.mark.asyncio
async def test_activation_timeout_does_not_submit_later() -> None:
    bundle = _Bundle()
    release = asyncio.Event()
    entered = asyncio.Event()

    class Engine:
        async def _activate_graph(self, graph_id, principal):
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()

    run = replace(bundle.graph_run, _engine=Engine())
    with pytest.raises(AIError) as caught:
        await run.cancel(settle_timeout_seconds=0.01)
    assert entered.is_set()
    assert caught.value.safe_details["reason"] == "deadline_unknown"
    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not bundle.runtime._cancel_settlements
    await bundle.runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["control", "state"])
async def test_control_and_authority_errors_are_not_reclassified(phase) -> None:
    bundle = _Bundle()
    error = AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def cancel(graph_id, request, *, admission_guard=None):
        if admission_guard is not None:
            admission_guard()
        if phase == "control":
            raise error

    async def state(graph_id, *, principal):
        raise error

    bundle.graph.cancel = cancel
    bundle.graph.state = state
    with pytest.raises(AIError) as caught:
        await bundle.graph_run.cancel(settle_timeout_seconds=1)
    assert caught.value is error
    assert not bundle.runtime._cancel_settlements


@pytest.mark.asyncio
async def test_caller_cancellation_does_not_cancel_accepted_control() -> None:
    bundle = _Bundle()
    started = asyncio.Event()
    release = asyncio.Event()

    async def cancel(graph_id, request, *, admission_guard=None):
        if admission_guard is not None:
            admission_guard()
        started.set()
        await release.wait()

    bundle.graph.cancel = cancel
    pending = asyncio.create_task(bundle.graph_run.cancel(settle_timeout_seconds=1))
    await started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert bundle.runtime._cancel_settlements
    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await bundle.runtime.close()


@pytest.mark.asyncio
async def test_caller_cancellation_during_activation_never_submits_later() -> None:
    bundle = _Bundle()
    entered = asyncio.Event()
    release = asyncio.Event()

    class Engine:
        async def _activate_graph(self, graph_id, principal):
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()

    run = replace(bundle.graph_run, _engine=Engine())
    pending = asyncio.create_task(run.cancel(settle_timeout_seconds=10))
    await entered.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not bundle.runtime._cancel_settlements
    await bundle.runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["deadline", "caller"])
@pytest.mark.parametrize("swallow", [False, True])
async def test_authorization_before_handoff_cannot_admit_after_stop(stop, swallow) -> None:
    from linktools.ai.task._service_impl import DefaultTaskGraphService

    bundle = _Bundle()
    service = object.__new__(DefaultTaskGraphService)
    entered = asyncio.Event()
    release = asyncio.Event()
    finalizers = []

    async def authorize(*args, **kwargs):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            if not swallow:
                raise
            await release.wait()

    async def finalize(*args):
        finalizers.append(args)

    service._authorize_graph = authorize
    service._cancel_finalizer = finalize
    bundle.graph.cancel = service.cancel
    pending = asyncio.create_task(bundle.graph_run.cancel(
        settle_timeout_seconds=0.02 if stop == "deadline" else 10,
    ))
    await entered.wait()
    if stop == "caller":
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    else:
        with pytest.raises(AIError) as caught:
            await pending
        assert caught.value.safe_details["reason"] == "deadline_unknown"
    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert finalizers == []
    assert not bundle.runtime._cancel_settlements
    await bundle.runtime.close()


@pytest.mark.asyncio
async def test_real_control_handoff_retains_finalizer_after_deadline() -> None:
    from linktools.ai.task._service_impl import DefaultTaskGraphService

    bundle = _Bundle()
    service = object.__new__(DefaultTaskGraphService)
    entered = asyncio.Event()
    release = asyncio.Event()
    completed = asyncio.Event()

    async def authorize(*args, **kwargs):
        return None

    async def finalize(*args):
        entered.set()
        await release.wait()
        completed.set()

    service._authorize_graph = authorize
    service._cancel_finalizer = finalize
    bundle.graph.cancel = service.cancel
    with pytest.raises(AIError) as caught:
        await bundle.graph_run.cancel(settle_timeout_seconds=0.02)
    assert caught.value.safe_details["reason"] == "control_pending"
    assert entered.is_set()
    release.set()
    await asyncio.wait_for(completed.wait(), 1)
    for _ in range(4):
        await asyncio.sleep(0)
    assert not bundle.runtime._cancel_settlements
    assert bundle.graph.state_calls == 0
    await bundle.runtime.close()


@pytest.mark.asyncio
async def test_budget_expired_before_scheduling_does_not_start_activation() -> None:
    bundle = _Bundle()
    activated = []

    class Engine:
        async def _activate_graph(self, graph_id, principal):
            activated.append(graph_id)

    run = replace(bundle.graph_run, _engine=Engine())
    with pytest.raises(AIError) as caught:
        await run.cancel(settle_timeout_seconds=1e-100)
    assert caught.value.safe_details["reason"] == "deadline_unknown"
    assert not activated
    assert not bundle.runtime._cancel_settlements
