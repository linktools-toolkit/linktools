#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pending preview ownership never substitutes for canonical execution events."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from dataclasses import replace
from types import SimpleNamespace
from typing import TypeVar

import pytest

from linktools.ai.core import (
    AuthorizationAction, ExecutionStatus, Page, Principal, ResourceKind, ResourceRef,
    canonical_json_bytes,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._event import DefaultEventService, LiveExecutionEventBroker
from linktools.ai.runtime._shared_stream import (
    SharedStreamPublisher, shared_stream_positions, stream_key,
)
from linktools.ai.runtime.service_api import ExecutionEvent, ExecutionStreamEvent
from linktools.ai.runtime.state._contracts import ExecutionRecord
from linktools.ai.storage import InMemoryStreamBackend, StreamBatch, StreamWriter

from .test_runtime_stream_order import _execution


_T = TypeVar("_T")
_PRINCIPAL = Principal(principal_id="reader", tenant_id="tenant")


class _PendingStore:
    def __init__(self) -> None:
        self.pending = replace(_execution(), event_seq=1, conversation=True)
        self.execution = None
        self.head = None
        self.owner = "pending-owner"
        self.records = ()
        self.pending_read = asyncio.Event()
        self.state_store = self

    async def read(self, callback: Callable[[object], Awaitable[_T]]) -> _T:
        return await callback(None)

    async def get(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        assert execution_id == "execution" and tenant_id == "tenant"
        return self.execution

    async def get_history_head(self, execution_id: str, *, tenant_id: str) -> SimpleNamespace | None:
        assert execution_id == "execution" and tenant_id == "tenant"
        return self.head

    async def get_header(self, execution_id: str, *, tenant_id: str) -> ResourceRef | None:
        record = await self.get(execution_id, tenant_id=tenant_id)
        return None if record is None else ResourceRef(
            ResourceKind.EXECUTION, execution_id, tenant_id,
            owner_principal_id=record.principal_id,
        )

    async def list(
        self, execution_id: str, *, tenant_id: str, after_event_seq: int, limit: int,
    ) -> Page[ExecutionEvent]:
        assert execution_id == "execution" and tenant_id == "tenant"
        return Page(tuple(value for value in self.records if value.event_seq > after_event_seq)[:limit], None)

    async def read_pending(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        assert execution_id == "execution" and tenant_id == "tenant"
        self.pending_read.set()
        return self.pending

    async def read_stream_owner(
        self, execution_id: str, *, tenant_id: str,
    ) -> tuple[ExecutionRecord, int, str] | None:
        assert execution_id == "execution" and tenant_id == "tenant"
        if self.pending is None or self.pending.agent_run_seq != 1 or self.owner is None:
            return None
        return self.pending, 2, self.owner

    def activate(self) -> None:
        self.pending = replace(self.pending, agent_run_seq=1)

    def promote(self) -> None:
        self.execution = self.pending
        self.pending = None
        self.head = SimpleNamespace(
            producer_generation=2, producer_claim_id=self.owner, state=SimpleNamespace(value="open"),
        )

    def commit(self) -> None:
        self.execution = replace(
            self.pending or self.execution, status=ExecutionStatus.SUCCEEDED, event_seq=2,
        )
        self.pending = None
        self.head = SimpleNamespace(
            producer_generation=2, producer_claim_id=self.owner, state=SimpleNamespace(value="sealed"),
        )
        self.records = (
            ExecutionEvent("execution", 1, "EXECUTION_STARTED", {"source": "canonical SQL"}),
            ExecutionEvent("execution", 2, "EXECUTION_SUCCEEDED", {"source": "canonical SQL"}),
        )


class _Authorization:
    def __init__(self, *, allow: bool = True) -> None:
        self.allow = allow
        self.actions = []

    async def authorize(
        self, principal: Principal, action: AuthorizationAction, resource: ResourceRef,
    ) -> None:
        self.actions.append((action, resource))
        if not self.allow:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)


class _Streams(InMemoryStreamBackend):
    def __init__(self) -> None:
        super().__init__(max_bytes=1024 * 1024)
        self.read_entered = asyncio.Event()
        self.opens = []

    async def open_stream(self, key: bytes, producer: bytes) -> StreamWriter:
        self.opens.append(key)
        return await super().open_stream(key, producer)

    async def read_stream(
        self, key: bytes, after: int, *, limit: int, wait_seconds: float,
    ) -> StreamBatch:
        self.read_entered.set()
        return await super().read_stream(key, after, limit=limit, wait_seconds=wait_seconds)


def _service(
    store: _PendingStore, backend: _Streams, *, authorization: _Authorization | None = None,
) -> DefaultEventService:
    publisher = SharedStreamPublisher(backend, "scope", "tenant")
    return DefaultEventService(
        store, store, authorization or _Authorization(), lambda *args, **kwargs: None,
        LiveExecutionEventBroker(shared=publisher),
        pending_execution=store.read_pending, pending_stream=store.read_stream_owner,
    )


async def _writer(backend: _Streams) -> StreamWriter:
    return await backend.open_stream(stream_key("scope", "tenant", "execution", 2, "pending-owner"), b"owner")


async def _delta(writer: StreamWriter, sequence: int, text: str) -> None:
    await writer.append(sequence, canonical_json_bytes({
        "kind": "delta", "type": "ASSISTANT_TEXT_DELTA", "text": text,
    }))


async def _collect(stream: AsyncIterator[ExecutionStreamEvent]) -> list[ExecutionStreamEvent]:
    return [item async for item in stream]


@pytest.mark.asyncio
async def test_pending_remote_watch_bootstraps_before_activation_and_confirms_sql_terminal() -> None:
    store, backend, authorization = _PendingStore(), _Streams(), _Authorization()
    service = _service(store, backend, authorization=authorization)
    with shared_stream_positions({}):
        async with aclosing(service.stream("execution", principal=_PRINCIPAL)) as stream:
            first_read = asyncio.create_task(anext(stream))
            try:
                await asyncio.wait_for(store.pending_read.wait(), 2)
                await asyncio.sleep(.02)
                assert not first_read.done() and backend.opens == []
                store.activate()
                writer = await _writer(backend)
                await _delta(writer, 1, "pending preview")
                first = await asyncio.wait_for(first_read, 3)
                assert first.event_type == "ASSISTANT_TEXT_DELTA"
                assert first.durable_seq is None
                assert store.execution is None and store.records == ()
                store.commit()
                await writer.seal(2, canonical_json_bytes({"kind": "event", "durable": 2}))
                remaining = await asyncio.wait_for(_collect(stream), 3)
                assert [item.durable_seq for item in remaining] == [1, 2]
                assert [item.event_type for item in remaining] == ["EXECUTION_STARTED", "EXECUTION_SUCCEEDED"]
                assert all(item.payload == {"source": "canonical SQL"} for item in remaining)
                assert len(backend.opens) == 1
            finally:
                if not first_read.done():
                    first_read.cancel()
                    await asyncio.gather(first_read, return_exceptions=True)
    assert [action for action, _ in authorization.actions] == [
        AuthorizationAction.EVENT_READ, AuthorizationAction.EXECUTION_READ,
    ]
    assert all(resource.owner_principal_id == store.execution.principal_id
               for _, resource in authorization.actions)


@pytest.mark.asyncio
async def test_closing_pending_remote_reader_preserves_owner_and_resume_position() -> None:
    store, backend = _PendingStore(), _Streams()
    store.activate()
    service = _service(store, backend)
    publisher = SharedStreamPublisher(backend, "scope", "tenant")
    release = asyncio.Event()

    async def produce() -> None:
        publisher.admit("execution", 2, "pending-owner")
        publisher.delta("execution", "ASSISTANT_TEXT_DELTA", "first", False)
        await release.wait()
        publisher.delta("execution", "ASSISTANT_TEXT_DELTA", "last", False)
        store.commit()
        publisher.event("execution", "EXECUTION_SUCCEEDED", {}, 2)
        publisher.complete("execution")

    owner = asyncio.create_task(produce())
    try:
        with shared_stream_positions({}):
            async with aclosing(service.stream("execution", principal=_PRINCIPAL)) as first_stream:
                first = await asyncio.wait_for(anext(first_stream), 3)
                assert first.payload["text"] == "first"
        assert not owner.done()
        assert publisher.producers and not publisher.closed
        with shared_stream_positions({"execution": first.position}):
            async with aclosing(service.stream("execution", principal=_PRINCIPAL)) as resumed:
                pending = asyncio.create_task(_collect(resumed))
                try:
                    backend.read_entered.clear()
                    await asyncio.wait_for(backend.read_entered.wait(), 2)
                    release.set()
                    values = await asyncio.wait_for(pending, 3)
                finally:
                    if not pending.done():
                        pending.cancel()
                        await asyncio.gather(pending, return_exceptions=True)
                assert [value.payload["text"] for value in values
                        if value.event_type == "ASSISTANT_TEXT_DELTA"] == ["last"]
                assert values[-1].event_type == "EXECUTION_SUCCEEDED"
                assert values[-1].durable_seq == 2
                assert all(value.event_type != "STREAM_RESET" for value in values)
        assert len(backend.opens) == 1
    finally:
        release.set()
        await owner
        await publisher.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ("lost", "replaced", "sql_replaced"))
async def test_pending_owner_change_resets_buffered_preview_before_durable_fallback(change: str) -> None:
    store, backend = _PendingStore(), _Streams()
    store.activate()
    service = _service(store, backend)
    writer = await _writer(backend)
    await _delta(writer, 1, "first")
    await _delta(writer, 2, "stale")
    with shared_stream_positions({}):
        async with aclosing(service.stream("execution", principal=_PRINCIPAL)) as stream:
            assert (await anext(stream)).payload["text"] == "first"
            if change == "sql_replaced":
                store.promote()
                store.head.producer_claim_id = "replacement"
            else:
                store.owner = None if change == "lost" else "replacement"
            await asyncio.sleep(1.01)
            reset = await asyncio.wait_for(anext(stream), 2)
            assert reset.event_type == "STREAM_RESET" and reset.durable_seq is None
            store.commit()
            remaining = await asyncio.wait_for(_collect(stream), 2)
            assert [item.durable_seq for item in remaining] == [1, 2]
            assert all(item.event_type != "ASSISTANT_TEXT_DELTA" for item in remaining)


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor_kind", ("wrong_owner", "beyond_tail", "durable_only"))
async def test_pending_invalid_cursor_resets_without_using_pending_event_sequence(cursor_kind: str) -> None:
    store, backend = _PendingStore(), _Streams()
    store.activate()
    service = _service(store, backend)
    writer = await _writer(backend)
    await _delta(writer, 1, "unconfirmed")
    key = stream_key("scope", "tenant", "execution", 2, "pending-owner").hex()
    positions = ({} if cursor_kind == "durable_only" else {
        "execution": ("wrong" if cursor_kind == "wrong_owner" else key, 20),
    })
    after = 1 if cursor_kind == "durable_only" else 0
    with shared_stream_positions(positions):
        async with aclosing(service.stream("execution", principal=_PRINCIPAL, after_event_seq=after)) as stream:
            reset = await asyncio.wait_for(anext(stream), 2)
            assert reset.event_type == "STREAM_RESET" and reset.durable_seq is None
            store.commit()
            remaining = await asyncio.wait_for(_collect(stream), 2)
            assert [item.durable_seq for item in remaining] == list(range(after + 1, 3))


@pytest.mark.asyncio
async def test_pending_promotion_keeps_same_shared_owner_until_sql_terminal() -> None:
    store, backend = _PendingStore(), _Streams()
    store.activate()
    service = _service(store, backend)
    writer = await _writer(backend)
    await _delta(writer, 1, "pending")
    await _delta(writer, 2, "promoted")
    with shared_stream_positions({}):
        async with aclosing(service.stream("execution", principal=_PRINCIPAL)) as stream:
            assert (await anext(stream)).payload["text"] == "pending"
            store.promote()
            await asyncio.sleep(1.01)
            promoted = await asyncio.wait_for(anext(stream), 2)
            assert promoted.payload["text"] == "promoted" and promoted.durable_seq is None
            store.commit()
            await writer.seal(3, canonical_json_bytes({"kind": "event", "durable": 2}))
            remaining = await asyncio.wait_for(_collect(stream), 2)
            assert [item.durable_seq for item in remaining] == [1, 2]


@pytest.mark.asyncio
async def test_pending_watch_authorizes_before_reading_shared_preview() -> None:
    store, backend = _PendingStore(), _Streams()
    store.activate()
    service = _service(store, backend, authorization=_Authorization(allow=False))
    with shared_stream_positions({}):
        async with aclosing(service.stream("execution", principal=_PRINCIPAL)) as stream:
            with pytest.raises(AIError) as failure:
                await anext(stream)
            assert failure.value.code is ErrorCode.AUTHORIZATION_DENIED
    assert not backend.read_entered.is_set()


@pytest.mark.asyncio
async def test_pending_durable_only_observer_waits_for_canonical_events() -> None:
    store, backend = _PendingStore(), _Streams()
    service = _service(store, backend)
    async with aclosing(service.stream("execution", principal=_PRINCIPAL)) as stream:
        first_read = asyncio.create_task(anext(stream))
        try:
            await asyncio.wait_for(store.pending_read.wait(), 2)
            await asyncio.sleep(.02)
            assert not first_read.done()
            store.commit()
            first = await asyncio.wait_for(first_read, 3)
            remaining = await asyncio.wait_for(_collect(stream), 2)
            assert [first.durable_seq] + [item.durable_seq for item in remaining] == [1, 2]
            assert not backend.read_entered.is_set()
        finally:
            if not first_read.done():
                first_read.cancel()
                await asyncio.gather(first_read, return_exceptions=True)


@pytest.mark.asyncio
async def test_lost_pending_execution_resets_then_reports_recovery_required() -> None:
    store, backend = _PendingStore(), _Streams()
    store.activate()
    service = _service(store, backend)
    writer = await _writer(backend)
    await _delta(writer, 1, "first")
    await _delta(writer, 2, "stale")
    with shared_stream_positions({}):
        async with aclosing(service.stream("execution", principal=_PRINCIPAL)) as stream:
            assert (await anext(stream)).payload["text"] == "first"
            store.pending = None
            await asyncio.sleep(1.01)
            assert (await anext(stream)).event_type == "STREAM_RESET"
            with pytest.raises(AIError) as failure:
                await anext(stream)
            assert failure.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
            assert failure.value.safe_details == {"reason": "pending_execution_unavailable"}


@pytest.mark.asyncio
async def test_pending_coordination_errors_propagate_without_becoming_a_miss() -> None:
    store, backend = _PendingStore(), _Streams()
    store.activate()
    service = _service(store, backend)

    async def unavailable(*args: object, **kwargs: object) -> None:
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

    service._pending_stream = unavailable
    with shared_stream_positions({}):
        async with aclosing(service.stream("execution", principal=_PRINCIPAL)) as stream:
            with pytest.raises(AIError) as failure:
                await anext(stream)
            assert failure.value.code is ErrorCode.STORAGE_UNAVAILABLE
    assert not backend.read_entered.is_set()


@pytest.mark.asyncio
async def test_pending_owner_disappearing_after_authorization_is_recovery_boundary() -> None:
    store, backend = _PendingStore(), _Streams()
    store.activate()
    service = _service(store, backend)

    async def read_then_lose(execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        pending = await store.read_pending(execution_id, tenant_id=tenant_id)
        store.pending = None
        return pending

    service._pending_execution = read_then_lose
    with shared_stream_positions({}):
        async with aclosing(service.stream("execution", principal=_PRINCIPAL)) as stream:
            assert (await anext(stream)).event_type == "STREAM_RESET"
            with pytest.raises(AIError) as failure:
                await anext(stream)
            assert failure.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
