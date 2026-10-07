#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step event wire values survive durable archives and portable snapshots."""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeDomain, RuntimeStorage, SnapshotLimits
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder
from linktools.ai.runtime.state._codec import _decode_step_envelope, _encode_step_envelope
from linktools.ai.runtime.state._step_contracts import AgentRunRecord, StepEvent, StepEventType
from linktools.ai.storage import InMemoryObjectStore

_EVENT_TYPES: tuple[StepEventType, ...] = (
    "AGENT_RUN_STARTED",
    "AGENT_RUN_SUCCEEDED",
    "AGENT_RUN_INTERRUPTED",
    "AGENT_RUN_FAILED",
    "MODEL_REQUEST_STARTED",
    "MODEL_REQUEST_SUCCEEDED",
    "MODEL_REQUEST_FAILED",
    "MODEL_REQUEST_CANCELLED",
    "TOOL_CALL_STARTED",
    "TOOL_CALL_SUCCEEDED",
    "TOOL_CALL_FAILED",
)
_NOW = datetime(2026, 9, 20, tzinfo=timezone.utc)


@pytest.mark.parametrize("event_type", _EVENT_TYPES)
def test_step_event_wire_round_trips_event_type(event_type: StepEventType) -> None:
    event = StepEvent("run", event_type, 1, timestamp=_NOW)
    encoded = _encode_step_envelope(event)
    assert encoded["value"]["payload"]["fields"]["event_type"] == event_type
    assert _decode_step_envelope(encoded) == event


@pytest.mark.parametrize(
    "invalid_type",
    ("run_completed", "model_request_cancelled", "MODEL_REQUEST_COMPLETED", "UNKNOWN"),
)
def test_step_event_wire_rejects_unsupported_types(invalid_type: str) -> None:
    encoded = _encode_step_envelope(StepEvent("run", "AGENT_RUN_SUCCEEDED", 1))
    encoded["value"]["payload"]["fields"]["event_type"] = invalid_type
    with pytest.raises(AIError) as raised:
        _decode_step_envelope(encoded)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_step_event_wire_requires_event_type_field() -> None:
    encoded = _encode_step_envelope(StepEvent("run", "AGENT_RUN_SUCCEEDED", 1))
    fields = encoded["value"]["payload"]["fields"]
    fields["kind"] = fields.pop("event_type")
    with pytest.raises(AIError) as raised:
        _decode_step_envelope(encoded)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_step_event_wire_accepts_additive_metadata_fields() -> None:
    event = StepEvent("run", "MODEL_REQUEST_SUCCEEDED", 1)
    encoded = deepcopy(_encode_step_envelope(event))
    encoded["value"]["payload"]["fields"]["future_note"] = "diagnostic"
    assert _decode_step_envelope(encoded) == event


def _storage(backend: str, root: Path) -> RuntimeStorage:
    if backend == "filesystem":
        return RuntimeStorage.filesystem(root)
    return RuntimeStorage.sqlite(root / "state.db")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
async def test_step_event_archive_and_idempotency_survive_snapshot(
    backend: str,
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime"
    state = _storage(backend, root)
    await state.initialize(namespace="steps", tenant_id="tenant")
    try:
        archive = state.run_store.read_store(RuntimeDomain.RECOVERY)
        run = AgentRunRecord("run", started_at=_NOW)
        recorder = AgentRunRecorder(archive, execution_id=None, agent_run_id="run")
        await recorder.register_agent_run(run)
        for event_type in _EVENT_TYPES:
            await recorder.record_event(event_type, 1, timestamp=_NOW)
        events = await archive.list_events(agent_run_id="run")
        assert [event.event_type for event in events] == list(_EVENT_TYPES)
        assert [event.idempotency_key for event in events] == [
            f"{index}:1:{event_type}:" for index, event_type in enumerate(_EVENT_TYPES)
        ]
        resumed = AgentRunRecorder(archive, execution_id=None, agent_run_id="run")
        await resumed.register_agent_run(run)
        await resumed.record_event("AGENT_RUN_SUCCEEDED", 2, timestamp=_NOW)
        events = await archive.list_events(agent_run_id="run")
        assert events[-1].idempotency_key == "11:2:AGENT_RUN_SUCCEEDED:"
        assert [event.event_index for event in events] == list(range(12))
    finally:
        await state.close()

    objects = InMemoryObjectStore("snapshot")
    limits = SnapshotLimits(max_entries=100, max_bytes=1024 * 1024)
    reader = _storage(backend, root)
    await reader.initialize(namespace="steps", tenant_id="tenant", read_only=True)
    try:
        reference = await reader.export_snapshot(object_store=objects, limits=limits)
    finally:
        await reader.close()
    restored_root = tmp_path / "restored"
    await RuntimeStorage.restore_snapshot(
        reference, object_store=objects, root=restored_root, limits=limits
    )
    restored = RuntimeStorage.from_root(restored_root)
    await restored.initialize(namespace="steps", tenant_id="tenant")
    try:
        restored_archive = restored.run_store.read_store(RuntimeDomain.RECOVERY)
        assert await restored_archive.list_events(agent_run_id="run") == events
        recorder = AgentRunRecorder(restored_archive, execution_id=None, agent_run_id="run")
        await recorder.register_agent_run(run)
        await recorder.record_event("AGENT_RUN_STARTED", 3, timestamp=_NOW)
        resumed_events = await restored_archive.list_events(agent_run_id="run")
        assert resumed_events[:-1] == events
        assert resumed_events[-1].idempotency_key == "12:3:AGENT_RUN_STARTED:"
        assert resumed_events[-1].event_index == 12
    finally:
        await restored.close()
