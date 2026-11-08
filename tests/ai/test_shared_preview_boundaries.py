#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared preview batching preserves the positions delivered to remote readers."""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import replace

import pytest

from linktools.ai.core import (
    ExecutionDeltaType, JsonValue, Page, Principal, ResourceKind, ResourceRef,
    TenantAuthorizationPolicy,
)
from linktools.ai.execution import ExecutionEvent, ExecutionRecord, ExecutionStreamEvent
from linktools.ai.execution.events import DefaultEventService, ExecutionDelta, LiveExecutionEventBroker
from linktools.ai.execution.stream import (
    SharedStreamEvent, SharedStreamPublisher, decode_stream_record,
    shared_stream_positions, stream_key,
)
from linktools.ai.storage import InMemoryStreamBackend

from .test_runtime_stream_order import _execution


_POSITION: Mapping[str, JsonValue] = {
    "agent_run_seq": 1, "model_request_seq": 1, "stream_attempt_seq": 1,
    "part_index": 0, "part_start": False,
}
_PRINCIPAL = Principal("reader", "tenant", kind="service")


class _ExecutionReader:
    def __init__(self) -> None:
        self.execution = replace(
            _execution(), principal_id="reader", agent_run_seq=1,
            producer_generation=1, producer_claim_id="claim",
        )

    async def get(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord:
        assert execution_id == "execution" and tenant_id == "tenant"
        return self.execution

    async def get_header(self, execution_id: str, *, tenant_id: str) -> ResourceRef:
        execution = await self.get(execution_id, tenant_id=tenant_id)
        return ResourceRef(
            ResourceKind.EXECUTION, execution_id, tenant_id,
            owner_principal_id=execution.principal_id,
        )

    async def list(
        self, execution_id: str, *, tenant_id: str, after_event_seq: int, limit: int,
    ) -> Page[ExecutionEvent]:
        await self.get(execution_id, tenant_id=tenant_id)
        return Page((), None)


def _delta(
    text: str, position: Mapping[str, JsonValue] | None = _POSITION, *,
    event_type: ExecutionDeltaType = ExecutionDeltaType.ASSISTANT_TEXT_DELTA,
    truncated: bool = False,
) -> ExecutionDelta:
    return ExecutionDelta("execution", event_type, text, truncated, position=position)


async def _publish(
    deltas: Sequence[ExecutionDelta],
) -> tuple[tuple[dict[str, JsonValue], ...], DefaultEventService]:
    backend = InMemoryStreamBackend(max_bytes=2 * 1024 * 1024)
    publisher = SharedStreamPublisher(backend, "scope", "tenant")
    broker = LiveExecutionEventBroker(shared=publisher)
    broker.register_local_producer("execution", 0)
    broker.admit_shared_producer("execution", 1, "claim")
    try:
        for delta in deltas:
            broker.publish(delta)
        publisher.complete("execution")
    finally:
        await publisher.close()
    batch = await backend.read_stream(
        stream_key("scope", "tenant", "execution", 1, "claim"), 0,
        limit=100, wait_seconds=0,
    )
    assert batch.sealed and not batch.gap
    records = tuple(decode_stream_record(payload) for _, payload in batch.records)
    reader = _ExecutionReader()
    service = DefaultEventService(
        reader, reader, TenantAuthorizationPolicy(), lambda *args, **kwargs: None,
        LiveExecutionEventBroker(shared=publisher),
    )
    return records, service


async def _read(
    service: DefaultEventService, count: int, *, position: tuple[str, int] | None = None,
) -> tuple[ExecutionStreamEvent, ...]:
    with shared_stream_positions({} if position is None else {"execution": position}):
        stream = service.stream("execution", principal=_PRINCIPAL)
        try:
            return tuple([await asyncio.wait_for(anext(stream), 1) for _ in range(count)])
        finally:
            await stream.aclose()


async def _assert_distinct(deltas: Sequence[ExecutionDelta]) -> None:
    records, service = await _publish(deltas)
    expected = [
        {"kind": "delta", "type": delta.event_type.value, "text": delta.content,
         "truncated": delta.stream_truncated,
         **({"position": dict(delta.position)} if delta.position is not None else {})}
        for delta in deltas
    ]
    assert [record for record in records if record["kind"] == "delta"] == expected
    events = await _read(service, len(deltas))
    assert [(event.event_type, event.durable_seq, event.payload) for event in events] == [
        (delta.event_type.value, None,
         {"text": delta.content, "stream_truncated": delta.stream_truncated, **(delta.position or {})})
        for delta in deltas
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("start_text", ("", "start"))
async def test_shared_preview_preserves_part_start_before_continuation(start_text: str) -> None:
    await _assert_distinct((
        _delta(start_text, {**_POSITION, "part_start": True}), _delta("tail"),
    ))


@pytest.mark.asyncio
async def test_shared_preview_preserves_repeated_part_start() -> None:
    position = {**_POSITION, "part_start": True}
    await _assert_distinct((_delta("first", position), _delta("second", position)))


@pytest.mark.asyncio
@pytest.mark.parametrize("coordinate", ("agent_run_seq", "model_request_seq", "stream_attempt_seq", "part_index"))
async def test_shared_preview_preserves_changed_coordinate(coordinate: str) -> None:
    await _assert_distinct((_delta("first"), _delta("second", {**_POSITION, coordinate: 2})))


@pytest.mark.asyncio
@pytest.mark.parametrize("position_added", (False, True))
async def test_shared_preview_preserves_position_presence(position_added: bool) -> None:
    before, after = (None, _POSITION) if position_added else (_POSITION, None)
    await _assert_distinct((_delta("first", before), _delta("second", after)))


@pytest.mark.asyncio
async def test_shared_preview_does_not_normalize_omitted_start_flag() -> None:
    position = {key: value for key, value in _POSITION.items() if key != "part_start"}
    await _assert_distinct((_delta("first", position), _delta("second")))


@pytest.mark.asyncio
@pytest.mark.parametrize("position", (None, {"part_index": 0}, _POSITION))
async def test_shared_preview_coalesces_same_position_and_preserves_truncation(
    position: Mapping[str, JsonValue] | None,
) -> None:
    records, service = await _publish((
        _delta("世界", position), _delta("tail", position, truncated=True),
    ))
    assert [record for record in records if record["kind"] == "delta"] == [{
        "kind": "delta", "type": "ASSISTANT_TEXT_DELTA", "text": "世界tail", "truncated": True,
        **({"position": dict(position)} if position is not None else {}),
    }]
    event, = await _read(service, 1)
    assert event.payload == {"text": "世界tail", "stream_truncated": True, **(position or {})}


@pytest.mark.asyncio
async def test_shared_preview_separates_text_and_thinking() -> None:
    await _assert_distinct((
        _delta("text"), _delta("thinking", event_type=ExecutionDeltaType.ASSISTANT_THINKING_DELTA),
    ))


@pytest.mark.asyncio
async def test_shared_preview_overflow_seals_gap_and_ignores_later_deltas() -> None:
    records, service = await _publish((_delta("X" * (1024 * 1024)), _delta("later")))
    assert records == ({"kind": "gap"},)
    event, = await _read(service, 1)
    assert event.event_type == "STREAM_RESET"
    assert event.payload == {"stream_truncated": True}


@pytest.mark.asyncio
async def test_shared_preview_resumes_after_empty_part_start_without_repeating_marker() -> None:
    _, service = await _publish((
        _delta("", {**_POSITION, "part_start": True}), _delta("tail"),
    ))
    first, = await _read(service, 1)
    assert isinstance(first, SharedStreamEvent)
    assert first.payload == {"text": "", "stream_truncated": False, **_POSITION, "part_start": True}
    resumed, = await _read(service, 1, position=first.position)
    assert resumed.payload == {"text": "tail", "stream_truncated": False, **_POSITION}
