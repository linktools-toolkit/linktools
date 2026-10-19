#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pending execution consumers defer terminal truth to the shared source."""

from linktools.ai.runtime.state import RuntimeState
import asyncio
import json
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import replace

import pytest
from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    Principal,
    StopReason,
    TenantAuthorizationPolicy,
    UsageMetrics,
    canonical_json_bytes,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import CancelExecutionRequest, RuntimeDomain, RuntimeStorage
from linktools.ai.runtime._event import LiveExecutionEventBroker
from linktools.ai.runtime._execution import (
    DefaultExecutionService,
    _ExecutionRuntimeBridge,
)
from linktools.ai.runtime._pending_coordination import _PendingTurnCoordination
from linktools.ai.runtime._pending_turn import _PendingExecutionTurns
from linktools.ai.runtime.state._codec import encode_domain
from linktools.ai.runtime.state._contracts import ResultRecord
from linktools.ai.storage import InMemoryCacheBackend, StoredPayload

from .test_history_projection_conformance import _record

_PRINCIPAL = Principal("principal", "tenant", "service")


class _RemotePendingTurns(_PendingExecutionTurns):
    def __init__(self, coordination):
        super().__init__()
        self.coordination = coordination
        self.cancel_requested = asyncio.Event()

    async def read_remote_execution(self, execution_id, *, tenant_id):
        return await self.coordination.read_execution(execution_id, tenant_id=tenant_id)

    async def cancel_remote(self, execution_id, *, tenant_id):
        result = await self.coordination.cancel(execution_id, tenant_id=tenant_id)
        self.cancel_requested.set()
        return result


@asynccontextmanager
async def _consumer(*, authorization=None):
    state = RuntimeStorage.in_memory()
    state = RuntimeState(state)
    await state.initialize(namespace="remote-pending", tenant_id="tenant")
    cache = InMemoryCacheBackend(max_bytes=1024 * 1024)
    coordination = _PendingTurnCoordination(cache, cache, namespace="remote-pending", tenant_id="tenant")
    owned = await coordination.claim(
        session_key="session", request_key="request", request_digest="digest",
        execution_id="execution", owner_id="owner", lease_seconds=30,
    )
    record = _record(ExecutionStatus.STARTED, 1)
    await coordination.publish(record, owned)
    pending = _RemotePendingTurns(coordination)
    bridge = _ExecutionRuntimeBridge()
    bridge.pending_turns = pending
    service = DefaultExecutionService(
        state.execution, state.object_store(RuntimeDomain.EXECUTION),
        authorization or TenantAuthorizationPolicy("tenant"), sessions=state.conversation.sessions,
        catalog=None, compiler=None, runtime_bridge=bridge,
        live_broker=LiveExecutionEventBroker(), history_reader=None,
    )
    try:
        yield state, service, pending, record, coordination, owned, cache
    finally:
        await service.preflight_close()
        await state.close()


def _terminal(record, status=ExecutionStatus.SUCCEEDED):
    cancelled = status is ExecutionStatus.CANCELLED
    return replace(
        record, status=status, revision=record.revision + 1,
        error_code=ErrorCode.EXECUTION_CANCELLED.value if cancelled else None,
        result=ResultRecord(
            output=None if cancelled else StoredPayload.inline_json({"answer": "source"}),
            stop_reason=StopReason.CANCELLED if cancelled else StopReason.END_TURN,
            usage=UsageMetrics(), created_at=record.updated_at,
        ),
    )


@pytest.mark.asyncio
async def test_remote_pending_inspect_result_and_wait_follow_source_commit() -> None:
    async with _consumer() as (state, service, pending, record, coordination, owned, _cache):
        assert await state.execution.executions.get("execution", tenant_id="tenant") is None
        assert (await service.inspect("execution", principal=_PRINCIPAL)).status is ExecutionStatus.STARTED
        with pytest.raises(AIError) as raised:
            await service.result("execution", principal=_PRINCIPAL)
        assert raised.value.code is ErrorCode.EXECUTION_NOT_READY
        with pytest.raises(AIError) as raised:
            await service.wait("execution", principal=_PRINCIPAL, timeout_seconds=0.01)
        assert raised.value.code is ErrorCode.EXECUTION_WAIT_TIMEOUT
        waiting = asyncio.create_task(service.wait("execution", principal=_PRINCIPAL, timeout_seconds=3))
        await asyncio.sleep(0)
        await state.execution.executions.create_with_history_head(_terminal(record))
        assert await coordination.release(owned.lease)
        result = await waiting
        assert result.status is ExecutionStatus.SUCCEEDED
        assert result.output == {"answer": "source"}
        assert not pending.cancel_requested.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ("inspect", "result", "wait", "cancel", "_cancel", "cancel_authorized"))
async def test_source_terminal_winner_takes_priority_over_lingering_remote_pending(method) -> None:
    async with _consumer() as (state, service, pending, record, _coordination, _owned, _cache):
        await state.execution.executions.create_with_history_head(_terminal(record))
        if "cancel" in method:
            result = await getattr(service, method)("execution", CancelExecutionRequest(_PRINCIPAL, method))
            assert result.cancelled is False
            assert not pending.cancel_requested.is_set()
        else:
            result = await getattr(service, method)("execution", principal=_PRINCIPAL)
            assert result.status is ExecutionStatus.SUCCEEDED
            if method != "inspect":
                assert result.output == {"answer": "source"}


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ("inspect", "result", "wait", "cancel", "list_children"))
@pytest.mark.parametrize("principal", (Principal("other", "tenant", "service"), Principal("principal", "other", "service")))
async def test_pending_remote_observation_preserves_owner_and_tenant_authorization(method, principal) -> None:
    async with _consumer() as (_state, service, pending, _record, _coordination, _owned, _cache):
        with pytest.raises(AIError) as raised:
            if method == "cancel":
                await service.cancel("execution", CancelExecutionRequest(principal, "denied"))
            else:
                await getattr(service, method)("execution", principal=principal)
        assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert not pending.cancel_requested.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("principal", (_PRINCIPAL, Principal("other", "tenant", "service")))
async def test_inspect_refreshes_stale_source_miss_after_pending_owner_releases(monkeypatch, principal) -> None:
    async with _consumer() as (state, service, _pending, record, coordination, owned, _cache):
        await state.execution.executions.create_with_history_head(_terminal(record))
        await coordination.release(owned.lease)
        repository = state.execution.executions
        get, read = repository.get, repository.state_store.read
        source_scope = ContextVar("source_scope", default=False)

        async def strong_read(callback):
            token = source_scope.set(True)
            try:
                return await read(callback)
            finally:
                source_scope.reset(token)

        async def observed_get(execution_id, *, tenant_id):
            if source_scope.get():
                return await get(execution_id, tenant_id=tenant_id)
            return None

        monkeypatch.setattr(repository.state_store, "read", strong_read)
        monkeypatch.setattr(repository, "get", observed_get)
        view = await service.inspect("execution", principal=principal)
        assert view.status is ExecutionStatus.SUCCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ("cancel", "_cancel", "cancel_authorized"))
@pytest.mark.parametrize("terminal_status", (ExecutionStatus.SUCCEEDED, ExecutionStatus.CANCELLED))
async def test_remote_cancel_waits_for_source_outcome_instead_of_confirming_cache_bit(method, terminal_status) -> None:
    async with _consumer() as (state, service, pending, record, coordination, owned, _cache):
        cancelling = asyncio.create_task(getattr(service, method)(
            "execution", CancelExecutionRequest(_PRINCIPAL, "remote-cancel"),
        ))
        try:
            await asyncio.wait_for(pending.cancel_requested.wait(), 2)
            current = await coordination.renew(owned.lease, lease_seconds=30)
            assert current.cancel_requested
            assert not cancelling.done()
            assert await state.execution.executions.get("execution", tenant_id="tenant") is None
            await state.execution.executions.create_with_history_head(_terminal(record, terminal_status))
            await coordination.release(owned.lease)
            outcome = await asyncio.wait_for(cancelling, 2)
            assert outcome.cancelled is (terminal_status is ExecutionStatus.CANCELLED)
        finally:
            if not cancelling.done():
                await coordination.release(owned.lease)
                await asyncio.gather(cancelling, return_exceptions=True)


@pytest.mark.asyncio
async def test_remote_cancel_reports_not_ready_when_owner_is_lost_without_source() -> None:
    async with _consumer() as (_state, service, pending, _record, coordination, owned, _cache):
        cancelling = asyncio.create_task(service.cancel("execution", CancelExecutionRequest(_PRINCIPAL, "lost")))
        await asyncio.wait_for(pending.cancel_requested.wait(), 2)
        await coordination.release(owned.lease)
        with pytest.raises(AIError) as raised:
            await asyncio.wait_for(cancelling, 2)
        assert raised.value.code is ErrorCode.EXECUTION_NOT_READY


@pytest.mark.asyncio
async def test_remote_cancel_coordination_failure_propagates(monkeypatch) -> None:
    async with _consumer() as (_state, service, _pending, _record, coordination, _owned, _cache):
        async def fail(*args, **kwargs):
            raise OSError("coordination unavailable")

        monkeypatch.setattr(coordination, "cancel", fail)
        with pytest.raises(OSError, match="coordination unavailable"):
            await service.cancel("execution", CancelExecutionRequest(_PRINCIPAL, "failed"))


@pytest.mark.asyncio
async def test_terminal_cache_payload_never_supplies_a_successful_result() -> None:
    async with _consumer() as (_state, service, _pending, record, coordination, owned, cache):
        key = coordination._record_key(owned)
        envelope = json.loads(await cache.get(key))
        envelope["record"] = encode_domain(_terminal(record))
        await cache.put(key, canonical_json_bytes(envelope))
        with pytest.raises(AIError) as raised:
            await service.result("execution", principal=_PRINCIPAL)
        assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ("inspect", "result", "wait", "cancel", "list_children"))
async def test_remote_authorization_cannot_bypass_canonical_source_policy(method) -> None:
    seeded = False
    policy = TenantAuthorizationPolicy("tenant")

    class Authorization:
        async def authorize(self, principal, action, resource):
            nonlocal seeded
            await policy.authorize(principal, action, resource)
            if resource.owner_principal_id is None and principal.principal_id != record.principal_id:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            if not seeded:
                seeded = True
                await state.execution.executions.create_with_history_head(_terminal(record))

    async with _consumer(authorization=Authorization()) as (state, service, pending, record, coordination, owned, _cache):
        other = Principal("other", "tenant", "service")
        await coordination.publish(replace(record, principal_id="other"), owned)
        with pytest.raises(AIError) as raised:
            if method == "cancel":
                await service.cancel("execution", CancelExecutionRequest(other, "conflicting-owner"))
            else:
                await getattr(service, method)("execution", principal=other)
        assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert not pending.cancel_requested.is_set()


@pytest.mark.asyncio
async def test_strong_result_refreshes_source_when_owner_commits_during_remote_read(monkeypatch) -> None:
    authorized = False
    policy = TenantAuthorizationPolicy("tenant")

    class Authorization:
        async def authorize(self, principal, action, resource):
            nonlocal authorized
            await policy.authorize(principal, action, resource)
            authorized = True

    async with _consumer(authorization=Authorization()) as (state, service, pending, record, coordination, owned, _cache):
        read_remote = pending.read_remote_execution

        async def commit_during_read(execution_id, *, tenant_id):
            if authorized:
                await state.execution.executions.create_with_history_head(_terminal(record))
                await coordination.release(owned.lease)
            return await read_remote(execution_id, tenant_id=tenant_id)

        monkeypatch.setattr(pending, "read_remote_execution", commit_during_read)
        result = await service.result("execution", principal=_PRINCIPAL)
        assert result.status is ExecutionStatus.SUCCEEDED
        assert result.output == {"answer": "source"}


@pytest.mark.asyncio
async def test_remote_pending_children_use_source_after_parent_promotion() -> None:
    async with _consumer() as (state, service, _pending, record, _coordination, _owned, _cache):
        assert await service.list_children("execution", principal=_PRINCIPAL) == ()
        child = replace(
            record, execution_id="child", parent_execution_id="execution",
            lineage_kind=ExecutionLineageKind.SUBAGENT, parent_invocation_id="call",
        )
        await state.execution.executions.create_with_history_head(record)
        await state.execution.executions.create_with_history_head(child)
        children = await service.list_children("execution", principal=_PRINCIPAL)
        assert tuple(item.execution_id for item in children) == ("child",)
