#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unpublished projections carry their own immutable content dependencies."""

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelClientRequest, ModelClientResponse
from linktools.ai.model.messages import encode_model_messages
from linktools.ai.persistence._model_interaction import (
    StagedContextInline, StagedContextProjection, StagedContextSpan,
    StagedModelInteraction, StagedStreamAttemptRecord,
)
from linktools.ai.persistence._projection_payload import decode_projection_payload, encode_projection_payload
from linktools.ai.persistence._step_archive import CapturedExecutionProjection, StagingAgentRunStore, _ProjectionOffset
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint, AgentRunRecord, StepEvent
from linktools.ai.persistence._transcript_staging import StagedTranscript


def _capture(store: StagingAgentRunStore | None = None) -> tuple[StagingAgentRunStore, CapturedExecutionProjection]:
    store = StagingAgentRunStore() if store is None else store
    store.register_agent_run_local(store.get_agent_run_local("run") or AgentRunRecord("run", metadata={"key": "original"}))
    request = ModelRequest(parts=[UserPromptPart("input")])
    response = ModelResponse(parts=[TextPart("output")])
    store.stage_transcript("run", StagedTranscript((request, response), ModelResponse(parts=[TextPart("pending")]), ("part:0",)))
    store.append_event_local(StepEvent("run", "MODEL_REQUEST_FAILED", 1))
    store.save_checkpoint_local(AgentRunCheckpoint("run", 1, [request, response], transcript_message_count_before=0))
    envelope, _ = store.intern_payload("run", b'{"request":true}')
    digest, size = store.intern_payload("run", encode_model_messages((response,)))
    store.intern_payload("run", b"not selected")
    now = datetime.now(timezone.utc)
    context = StagedContextProjection((StagedContextInline(digest, size),))
    store.stage_model_interaction(StagedModelInteraction(
        "run", 1, 1, "agent", None, {"model": "test"},
        StagedContextProjection((StagedContextSpan(0, 1),)), envelope, context,
        "FAILED", "request_failed", 1, None, now, now,
        client_requests=(ModelClientRequest(
            "responses", b"request-body", response=ModelClientResponse(b"response-body", 500, "json", None, True),
        ),),
        stream_attempts=(StagedStreamAttemptRecord(
            1, "FAILED", now, now, 1, 0, 1, context, None, "request_failed", None, False,
        ),),
    ))
    return store, _projection(store)


def _projection(store: StagingAgentRunStore, offset: _ProjectionOffset | None = None) -> CapturedExecutionProjection:
    captured = store.capture_projection_local("run", offset or _ProjectionOffset())
    assert captured is not None
    return CapturedExecutionProjection(
        captured.run, captured.events, captured.checkpoints,
        captured.base_event_offset, captured.base_checkpoint_offset,
        captured.target_event_offset, captured.target_checkpoint_offset,
        captured.interactions, captured.base_interaction_offset, captured.target_interaction_offset,
        captured.transcript, captured.base_message_index, captured.target_message_index, 1,
        captured.payloads, captured.local_messages,
    )


def test_projection_bytes_restore_without_staging_payload_mirror() -> None:
    store, captured = _capture()
    content = encode_projection_payload(captured)
    store._payloads.clear()
    store._transcripts.clear()
    captured.run.metadata["key"] = "changed"
    restored = decode_projection_payload(content)
    assert restored.run.metadata == {"key": "original"}
    assert restored.events == captured.events
    assert list(restored.checkpoints) == list(captured.checkpoints)
    assert restored.interactions == captured.interactions
    assert restored.local_messages == tuple(captured.local_messages)
    assert dict(restored.payloads) == dict(captured.payloads)
    assert hashlib.sha256(b"not selected").hexdigest() not in dict(restored.payloads)
    assert restored.interactions[0].client_requests[0].body == b"request-body"
    assert restored.interactions[0].client_requests[0].response.body == b"response-body"
    assert encode_projection_payload(restored) == content


@pytest.mark.parametrize("damage", ("digest", "frontier", "version", "run"))
def test_projection_payload_rejects_corrupt_content(damage: str) -> None:
    _, captured = _capture()
    if damage == "digest":
        captured = replace(captured, payloads=((captured.payloads[0][0], b"wrong"),))
    elif damage == "frontier":
        captured = replace(captured, target_event_offset=20)
    elif damage == "run":
        captured = replace(captured, run=replace(captured.run, agent_run_id="other"))
    content = encode_projection_payload(captured)
    if damage == "version":
        value = json.loads(content)
        value["format"] = "trace-projection-v2"
        content = json.dumps(value).encode()
    with pytest.raises(AIError) as error:
        decode_projection_payload(content)
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_source_materialization_uses_only_captured_payloads() -> None:
    from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
    from .test_step_projection import _execution

    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="captured-payload", tenant_id="tenant")
    try:
        await state.execution.executions.create_with_history_head(_execution())
        staging, projection = _capture(state.run_store._staging)
        staging.stage_transcript("run", StagedTranscript(projection.local_messages))
        await state.run_store.register_agent_run(projection.run, execution_id="execution")
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        await archive.register_agent_run(projection.run, execution_id="execution")
        await state.run_store.append_event(projection.events[0], execution_id="execution")
        captured = await state.run_store.capture_execution_projection("run")
        assert captured is not None
        projection, flight = captured
        restored = decode_projection_payload(encode_projection_payload(projection))
        staging._payloads.clear()
        staging._transcripts.clear()
        await state.run_store.commit_captured_execution_projection(restored, flight, execution_id="execution")
        observed = await archive.list_model_interactions(agent_run_id="run")
        assert len(observed) == 1
        request, response, envelope, clients, partials = await archive.resolve_model_interaction(
            observed[0], include_client_requests=True,
        )
        assert request[0].parts[0].content == "input"
        assert response[0].parts[0].content == "output"
        assert clients[0].body == b"request-body"
        assert clients[0].response.body == b"response-body"
        assert partials[0][0].parts[0].content == "output"
    finally:
        await state.close()


def test_projection_tail_retains_context_before_source_frontier() -> None:
    store, _ = _capture()
    projection = _projection(store, _ProjectionOffset(events=1, checkpoints=1, transcript_messages=1, interactions=1))
    restored = decode_projection_payload(encode_projection_payload(projection))
    assert restored.base_message_index == 1
    assert restored.target_message_index == 2
    assert len(restored.transcript.messages) == 1
    assert restored.transcript.messages[0].parts[0].content == "output"
    assert restored.local_messages[0].parts[0].content == "input"
    assert not restored.events and not restored.checkpoints


def test_old_projection_cannot_acknowledge_new_interaction_content() -> None:
    _, complete = _capture()
    store = StagingAgentRunStore()
    store.register_agent_run_local(complete.run)
    store.stage_transcript("run", complete.transcript)
    for _, content in complete.payloads:
        store.intern_payload("run", content)
    running = replace(
        complete.interactions[0], status="RUNNING", response_context=None,
        error_code=None, duration_ns=None, finished_at=None, client_requests=None, stream_attempts=None,
    )
    store.stage_model_interaction(running)
    first = decode_projection_payload(encode_projection_payload(_projection(store)))
    store.stage_model_interaction(complete.interactions[0])
    second = decode_projection_payload(encode_projection_payload(_projection(store)))
    assert first.interactions[0].model_request_seq == second.interactions[0].model_request_seq
    assert first.interactions[0].status == "RUNNING"
    assert second.interactions[0].status == "FAILED"
    store.acknowledge_interactions_local("run", first.interactions)
    assert _projection(store).interactions == second.interactions
    store.acknowledge_interactions_local("run", second.interactions)
    assert not _projection(store).interactions
