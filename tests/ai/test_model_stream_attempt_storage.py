#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Failed stream attempts are trace evidence, never winning conversation history."""

import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from linktools.ai.core import ExecutionStatus, HmacCursorSigner, UsageMetrics, agent_conversation_id, agent_run_id
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelClientRequest
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.runtime._history_projection import StepExecutionHistoryReader
from linktools.ai.runtime._model_interaction import (
    StagedContextProjection, StagedContextSpan, StagedModelInteraction,
    StagedStreamAttemptRecord, build_inline_context_projection,
)
from linktools.ai.runtime.state import RuntimeDomain, RuntimeState, SnapshotLimits
from linktools.ai.runtime.state._archive import PreparedArchive
from linktools.ai.runtime.state._codec import (
    _CURRENT_CODEC, _decode_domain, _decode_step_envelope,
    _encode_persisted_domain, _encode_step_envelope,
)
from linktools.ai.runtime.state._contracts import ModelInteractionRecord
from linktools.ai.runtime.state._step_contracts import AgentRunRecord
from linktools.ai.storage import InMemoryObjectStore, ObjectRef

from .test_evaluation_retention import Offline
from .test_history_projection_conformance import _record

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
_RUN = agent_run_id(namespace="stream-observation", tenant_id="tenant", execution_id="execution", agent_run_seq=1)
_CONVERSATION = agent_conversation_id(namespace="stream-observation", tenant_id="tenant", execution_id="execution")


def _run() -> AgentRunRecord:
    return AgentRunRecord(_RUN, agent_conversation_id=_CONVERSATION, metadata={"agent_run_seq": "1"})


def _storage(backend: str, root: Path) -> RuntimeStorage:
    if backend == "memory":
        return RuntimeStorage.in_memory()
    if backend == "filesystem":
        return RuntimeStorage.filesystem(root / "state")
    return RuntimeStorage.sqlite(root / "state.db")



async def _object_refs(state: RuntimeState) -> tuple[tuple[RuntimeDomain, ObjectRef], ...]:
    store = state.archive_object_store(RuntimeDomain.TRACE)
    return tuple([(RuntimeDomain.TRACE, ObjectRef(store.store_id, item.key, item.digest, item.size))
                  async for item in store.list_objects()])


def _staged(*, large: bool = False):
    payloads = {}

    def intern(raw: bytes) -> tuple[str, int]:
        digest = hashlib.sha256(raw).hexdigest()
        payloads[digest] = raw
        return digest, len(raw)

    request = (ModelRequest(parts=[UserPromptPart("synthetic request")]),)
    winner = (ModelResponse(parts=[TextPart("complete winner")]),)
    partial = (ModelResponse(parts=[TextPart("failed partial " + ("\n".join(hashlib.sha256(str(i).encode()).hexdigest() for i in range(8000)) if large else ""))], state="incomplete"),)
    failed = StagedStreamAttemptRecord(
        stream_attempt_seq=1, status="FAILED", started_at=_NOW, finished_at=_NOW,
        duration_ns=13, client_request_start=0, client_request_end=2,
        response_context=build_inline_context_projection(partial, intern),
        usage=UsageMetrics(1, 0, 10, 3), error_code=ErrorCode.MODEL_API_ERROR.value,
        error_type="ModelAPIError", retry_scheduled=True,
    )
    succeeded = replace(failed, stream_attempt_seq=2, status="SUCCEEDED", duration_ns=17,
                        client_request_start=2, client_request_end=2, response_context=None,
                        usage=UsageMetrics(1, 0, 11, 5), error_code=None, error_type=None, retry_scheduled=False)
    interaction = StagedModelInteraction(
        agent_run_id=_RUN, step_index=1, model_request_seq=1, purpose="agent", output_retry_index=None,
        model={"model_name": "synthetic"}, request_context=build_inline_context_projection(request, intern),
        request_envelope_digest=intern(b'{"streaming":true}')[0],
        response_context=build_inline_context_projection(winner, intern),
        status="SUCCEEDED", error_code=None, duration_ns=30, usage=UsageMetrics(2, 0, 21, 8),
        started_at=_NOW, finished_at=_NOW,
        client_requests=(ModelClientRequest("synthetic", None), ModelClientRequest("synthetic", None)),
        stream_attempts=(failed, succeeded),
    )
    return interaction, payloads, request, winner, partial


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_stream_attempts_round_trip_relocate_and_omit_content(tmp_path: Path, backend: str) -> None:
    state = RuntimeState(_storage(backend, tmp_path))
    await state.initialize(namespace="stream-observation", tenant_id="tenant")
    await state.execution.executions.create_with_history_head(_record(ExecutionStatus.STARTED, 1))
    staged, payloads, request, winner, partial = _staged()
    prepared = PreparedArchive()
    try:
        trace = state.run_store.read_store(RuntimeDomain.TRACE)
        run = _run()
        records = await trace.prepare_interactions(run, (staged,), payloads.__getitem__, prepared=prepared)
        record, = records
        assert _decode_step_envelope(_encode_step_envelope(record)) == record
        await trace.sync_projection(run, events=(), checkpoints=(), interactions=records,
                                    execution_id="execution", archive=prepared)
        resolved = await trace.resolve_model_interactions(records, include_client_requests=True)
        assert resolved == [(request, winner, b'{"streaming":true}', staged.client_requests, (partial, None))]
        assert await trace.resolve_model_interaction(record, include_client_requests=True) == resolved[0]
        assert [message async for message in trace.iter_messages(agent_run_id=_RUN)] == []
        hidden = await trace.resolve_model_interaction(record, include_content=False)
        assert hidden == (None, None, None, None, (None, None))
        reader = StepExecutionHistoryReader(
            namespace="stream-observation", executions=state.execution.executions, store=trace,
            cursor_signer=HmacCursorSigner("synthetic", b"synthetic-cursor-key"),
        )
        page = await reader.model_interactions("execution", tenant_id="tenant", cursor=None, limit=10)
        item, = page.items
        assert item.response["parts"][0]["content"] == "complete winner"
        assert item.stream_attempts[0].response["parts"][0]["content"] == partial[0].parts[0].content
        assert item.stream_attempts[1].response is None
        assert item.stream_attempts[0].client_request_end == 2
        page = await reader.model_interactions("execution", tenant_id="tenant", cursor=None, limit=10,
                                               include_content=False)
        light, = page.items
        assert all(attempt.response is None for attempt in light.stream_attempts)
        assert light.stream_attempts[0].usage == staged.stream_attempts[0].usage
        for domain in (RuntimeDomain.CHECKPOINT,):
            target = state.run_store.read_store(domain)
            relocated = await target.prepare_relocated_interactions(records, resolved, prepared=prepared)
            assert all(attempt.response_context is None for attempt in relocated[0].stream_attempts)
            projected = await target.resolve_model_interactions(relocated, include_client_requests=True)
            assert projected[0][-1] == (None, None)
            restored = await trace.prepare_relocated_interactions(relocated, projected, prepared=prepared)
            assert all(attempt.response_context is None for attempt in restored[0].stream_attempts)
        with pytest.raises(AIError) as changed:
            await trace.sync_projection(run, events=(), checkpoints=(),
                interactions=(replace(record, stream_attempts=()),), execution_id="execution", archive=prepared)
        assert changed.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


def test_stream_attempt_metadata_requires_ordered_actual_ranges_and_separate_partial() -> None:
    interaction, _, _, _, _ = _staged()
    first, winner = interaction.stream_attempts
    for changes in ({"stream_attempt_seq": 0}, {"client_request_end": -1},
                    {"status": "SUCCEEDED"}, {"status": "CANCELLED"}, {"duration_ns": True},
                    {"response_context": StagedContextProjection((StagedContextSpan(0, 1),))}):
        with pytest.raises((ValueError, TypeError)):
            replace(first, **changes)
    for attempts in ((winner, first), (first, replace(winner, stream_attempt_seq=3)),
                     (first, replace(winner, client_request_start=1)),
                     (first, replace(winner, client_request_end=3))):
        with pytest.raises(ValueError):
            replace(interaction, stream_attempts=attempts)


@pytest.mark.asyncio
@pytest.mark.parametrize("observed", (False, True))
async def test_old_missing_attempts_and_observed_empty_remain_distinct(observed: bool) -> None:
    interaction, payloads, _, _, _ = _staged()
    state = RuntimeState(RuntimeStorage.in_memory())
    await state.initialize(namespace="stream-observation", tenant_id="tenant")
    try:
        archive = state.run_store.read_store(RuntimeDomain.TRACE)
        record, = await archive.prepare_interactions(_run(),
            (replace(interaction, stream_attempts=() if observed else None),), payloads.__getitem__,
            prepared=PreparedArchive())
        wire = _encode_persisted_domain(record)
        assert ("stream_attempts" in wire["fields"]) is observed
        assert _decode_domain(wire, ModelInteractionRecord, _CURRENT_CODEC, persisted=True) == record
        wire["fields"].pop("stream_attempts", None)
        legacy = _decode_domain(wire, ModelInteractionRecord, _CURRENT_CODEC, persisted=True)
        assert legacy.stream_attempts is None
        wire["fields"]["stream_attempts"] = {"$tuple": ["invalid"]}
        with pytest.raises(AIError):
            _decode_domain(wire, ModelInteractionRecord, _CURRENT_CODEC, persisted=True)
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
async def test_failed_partial_survives_snapshot_and_owned_cleanup(tmp_path: Path, backend: str) -> None:
    staged, payloads, request, winner, partial = _staged(large=True)
    state = RuntimeState(_storage(backend, tmp_path))
    await state.initialize(namespace="stream-observation", tenant_id="tenant")
    await state.execution.executions.create_with_history_head(_record(ExecutionStatus.STARTED, 1))
    try:
        archive = state.run_store.read_store(RuntimeDomain.TRACE)
        prepared = PreparedArchive()
        run = _run()
        records = await archive.prepare_interactions(run, (staged,), payloads.__getitem__, prepared=prepared)
        await archive.sync_projection(run, events=(), checkpoints=(), interactions=records,
                                     execution_id="execution", archive=prepared)
        refs = await _object_refs(state)
        assert refs
        assert (await state.purge_unreferenced_objects(refs, exclusive=Offline())).retained == refs
    finally:
        await state.close()
    objects = InMemoryObjectStore("snapshot")
    limits = SnapshotLimits(max_entries=100, max_bytes=2 * 1024 * 1024)
    reader = RuntimeState(_storage(backend, tmp_path))
    await reader.initialize(namespace="stream-observation", tenant_id="tenant", read_only=True)
    try:
        snapshot = await reader.export_snapshot(object_store=objects, limits=limits)
    finally:
        await reader.close()
    root = tmp_path / "restored"
    await RuntimeState.restore_snapshot(snapshot, object_store=objects, root=root, limits=limits)
    restored = RuntimeState(RuntimeStorage.from_root(root))
    await restored.initialize(namespace="stream-observation", tenant_id="tenant")
    try:
        archive = restored.run_store.read_store(RuntimeDomain.TRACE)
        records = await archive.list_model_interactions(agent_run_id=_RUN)
        resolved = await archive.resolve_model_interactions(records)
        assert resolved == [(request, winner, b'{"streaming":true}', None, (partial, None))]
        assert [message async for message in archive.iter_messages(agent_run_id=_RUN)] == []
        refs = await _object_refs(restored)
        await archive.release_agent_run(_RUN, execution_id="execution")
        assert (await restored.purge_unreferenced_objects(refs, exclusive=Offline())).deleted == refs
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_successful_continuation_segments_share_one_canonical_response() -> None:
    staged, payloads, request, response, partial = _staged()
    failed, succeeded = staged.stream_attempts
    first = replace(succeeded, stream_attempt_seq=1, client_request_start=0, client_request_end=0,
                    usage=UsageMetrics(1, 0, 5, 2))
    staged = replace(staged, stream_attempts=(first, replace(failed, stream_attempt_seq=2),
                                             replace(succeeded, stream_attempt_seq=3)),
                     usage=UsageMetrics(3, 0, 26, 10))
    state = RuntimeState(RuntimeStorage.in_memory())
    await state.initialize(namespace="stream-observation", tenant_id="tenant")
    await state.execution.executions.create_with_history_head(_record(ExecutionStatus.STARTED, 1))
    try:
        archive = state.run_store.read_store(RuntimeDomain.TRACE)
        prepared = PreparedArchive()
        record, = await archive.prepare_interactions(_run(), (staged,), payloads.__getitem__, prepared=prepared)
        await archive.sync_projection(_run(), events=(), checkpoints=(), interactions=(record,),
                                     execution_id="execution", archive=prepared)
        recovered = _decode_step_envelope(_encode_step_envelope(record))
        assert [attempt.status for attempt in recovered.stream_attempts] == ["SUCCEEDED", "FAILED", "SUCCEEDED"]
        resolved = await archive.resolve_model_interaction(recovered)
        assert resolved == (request, response, b'{"streaming":true}', None, (None, partial, None))
        assert recovered.usage == UsageMetrics(3, 0, 26, 10)
    finally:
        await state.close()
