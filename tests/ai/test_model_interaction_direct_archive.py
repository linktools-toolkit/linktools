#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct responses retain content without weakening prepared-request records."""

import hashlib
from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model.messages import encode_model_messages
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence.codec import _decode_step_envelope, _encode_step_envelope
from linktools.ai.persistence.contracts import ContextProjection, InlineContextBlock, ModelInteractionRecord, RuntimePayloadRef
from linktools.ai.persistence.runs import AgentRunCheckpoint, AgentRunRecord, InMemoryStepArchive, RuntimeAgentRunStore, StagedModelInteraction, StagingAgentRunStore
from linktools.ai.persistence._model_interaction import StagedContextProjection, build_inline_context_projection
from linktools.ai.persistence._model_interaction_store import ModelInteractionInMemoryStepArchive
from linktools.ai.storage import StoredPayload

from .test_step_archive_read_boundaries import _archive


def _record(sequence: int, *, direct: bool) -> ModelInteractionRecord:
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    def projection(message):
        return ContextProjection((InlineContextBlock(RuntimePayloadRef(
            StoredPayload.inline_bytes(encode_model_messages((message,))), PersistenceDomain.TRACE,
        )),))
    return ModelInteractionRecord(
        agent_run_id="run", step_index=sequence, model_request_seq=sequence,
        purpose="agent", output_retry_index=None, model={"route_id": "default"},
        request_context=None if direct else projection(ModelRequest(parts=[UserPromptPart(f"request-{sequence}", timestamp=timestamp)], timestamp=timestamp)),
        request_envelope=None if direct else RuntimePayloadRef(StoredPayload.inline_bytes(b"{}"), PersistenceDomain.TRACE),
        response_context=projection(ModelResponse(parts=[TextPart(f"response-{sequence}")], timestamp=timestamp)),
        status="SUCCEEDED", error_code=None, duration_ns=17, usage=None,
        started_at=timestamp, finished_at=timestamp,
    )


def test_request_shape_round_trips_without_classification_fields() -> None:
    ordinary = _record(1, direct=False)
    direct = _record(2, direct=True)
    for record in (ordinary, direct):
        wire = _encode_step_envelope(record)
        assert "is_direct_result" not in wire["value"]["payload"]["fields"]
        assert not hasattr(record, "is_direct_result")
        assert _decode_step_envelope(wire) == record
    ordinary_wire = _encode_step_envelope(ordinary)
    assert hashlib.sha256(canonical_json_bytes(ordinary_wire)).hexdigest() == "2142ca8c5c2e8acdb55ee082b46da796ac8b22344f52afeeaa6647e70000e60e"
    # Both omitted request fields mean no observed preparation; a partial pair is invalid.
    for field in ("request_context", "request_envelope", "response_context"):
        wire = _encode_step_envelope(ordinary)
        wire["value"]["payload"]["fields"][field] = None
        with pytest.raises(AIError) as raised:
            _decode_step_envelope(wire)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("staged", (False, True))
def test_direct_result_contract_rejects_fabricated_preparation(staged: bool) -> None:
    direct = _record(1, direct=True)
    if staged:
        values = {field.name: getattr(direct, field.name) for field in fields(direct)
                  if field.name != "request_envelope"}
        values.update(request_envelope_digest=None, response_context=StagedContextProjection(()))
        direct = StagedModelInteraction(**values)
        preparation = {"request_context": StagedContextProjection(()), "request_envelope_digest": "a" * 64}
    else:
        ordinary = _record(1, direct=False)
        preparation = {"request_context": ordinary.request_context, "request_envelope": ordinary.request_envelope}
    assert replace(direct, **preparation).request_context is not None
    for change in (
        {"request_context": preparation["request_context"]},
        {key: value for key, value in preparation.items() if key != "request_context"},
        {"response_context": None},
        {"status": "RUNNING", "duration_ns": None, "finished_at": None},
        {"status": "INTERRUPTED", "duration_ns": None, "finished_at": None},
    ):
        with pytest.raises(ValueError):
            replace(direct, **change)
    with pytest.raises(FrozenInstanceError):
        direct.status = "FAILED"

    accepted = {
        "fact": "accepted", "attachment_id": "a" * 64, "source": "input",
        "media_type": "text/plain", "size": 17, "digest": "b" * 64,
        "content_key": "c" * 64, "position": 0, "call_id": None, "input_identifier": None,
    }
    assert replace(direct, attachments=(accepted,)).attachments == (accepted,)
    included = {**accepted, "fact": "included_in_request"}
    running_metadata = replace(direct, status="RUNNING", response_context=None,
                               duration_ns=None, finished_at=None, attachments=(included,))
    assert running_metadata.attachments == (included,)
    if not staged:
        assert _decode_step_envelope(_encode_step_envelope(running_metadata)) == running_metadata
    with pytest.raises(ValueError):
        replace(direct, attachments=({**accepted, "fact": "included_in_request"},))


def test_published_preparation_cannot_disappear_from_successor() -> None:
    terminal = _record(1, direct=False)
    prepared = replace(terminal, status="RUNNING", response_context=None,
                       duration_ns=None, finished_at=None)
    prepared.validate_successor(terminal)
    without_request = replace(terminal, request_context=None, request_envelope=None)
    # The terminal shape is legitimate on its own, not after this published request.
    assert _decode_step_envelope(_encode_step_envelope(without_request)) == without_request
    with pytest.raises(AIError) as raised:
        prepared.validate_successor(without_request)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_mixed_direct_and_prepared_archive_reads_keep_response_positions(tmp_path: Path, backend: str) -> None:
    async with _archive(tmp_path, backend) as (_state, archive, run):
        records = tuple(_record(sequence, direct=sequence in (1, 3)) for sequence in range(1, 5))
        await archive.sync_prepared_projection(run, events=(), checkpoints=(), interactions=records, execution_id="execution", producer_generation=1)
        page = await archive.list_model_interactions(agent_run_id="run", after_model_request_seq=1, limit=2)
        assert [value.model_request_seq for value in page] == [2, 3]
        assert [value.request_context is None for value in page] == [False, True]
        batch = await archive.resolve_model_interactions(records)
        singles = [await archive.resolve_model_interaction(value) for value in records]
        assert batch == singles
        for sequence, (request, response, envelope, clients, streams) in enumerate(batch, 1):
            assert response[0].parts[0].content == f"response-{sequence}"
            assert (request is None) == (sequence in (1, 3))
            assert envelope == (None if sequence in (1, 3) else b"{}")
            if request is not None:
                assert request[0].parts[0].content == f"request-{sequence}"
            assert clients is None and streams is None
        assert await archive.resolve_model_interactions(records, include_content=False) == [(None, None, None, None, None)] * 4


@pytest.mark.asyncio
async def test_staged_and_prepared_direct_response_round_trip() -> None:
    staging = StagingAgentRunStore()
    archive = ModelInteractionInMemoryStepArchive(PersistenceDomain.TRACE)
    store = RuntimeAgentRunStore(staging, conversation_archive=InMemoryStepArchive(PersistenceDomain.CONVERSATION), execution_archive=None, recovery_archive=None)
    await store.initialize()
    await archive.initialize()
    try:
        run = AgentRunRecord("run")
        await store.register_agent_run(run)
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        def intern(payload: bytes) -> tuple[str, int]:
            return staging.intern_payload("run", payload)
        values = []
        for sequence in range(1, 5):
            direct = sequence in (1, 3)
            response = ModelResponse(parts=[TextPart(f"response-{sequence}")], timestamp=now)
            request = ModelRequest(parts=[UserPromptPart(f"request-{sequence}", timestamp=now)], timestamp=now)
            staged = StagedModelInteraction(
                agent_run_id="run", step_index=sequence, model_request_seq=sequence, purpose="agent", output_retry_index=None,
                model={}, request_context=None if direct else build_inline_context_projection((request,), intern),
                request_envelope_digest=None if direct else intern(b"{}")[0],
                response_context=build_inline_context_projection((response,), intern),
                status="SUCCEEDED", error_code=None, duration_ns=0, usage=None,
                started_at=now, finished_at=now,
            )
            store.stage_model_interaction(staged)
            values.append(staged)
        staged_batch = await store.resolve_model_interactions(values)
        assert staged_batch == [await store.resolve_model_interaction(value) for value in values]
        durable = await archive.prepare_interactions(run, values, lambda digest: staging.staged_payload("run", digest))
        assert [value.request_context is None for value in durable] == [True, False, True, False]
        assert await archive.resolve_model_interaction(durable[0]) == staged_batch[0]
        with pytest.raises(AIError) as raised:
            await archive.resolve_model_interaction(durable[1])
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        await archive.sync_projection(
            run, events=(), checkpoints=(AgentRunCheckpoint("run", 1, [response]),), interactions=durable,
        )
        durable_batch = await archive.resolve_model_interactions(durable)
        assert durable_batch == [await archive.resolve_model_interaction(value) for value in durable]
        assert durable_batch == staged_batch
        for sequence, (request, response, envelope, _clients, _streams) in enumerate(durable_batch, 1):
            assert response[0].parts[0].content == f"response-{sequence}"
            assert (request is None) == (sequence in (1, 3))
            assert envelope == (None if sequence in (1, 3) else b"{}")
    finally:
        await store.preflight_close()
        await store.close()
        await archive.close()


@pytest.mark.asyncio
async def test_running_page_resolves_after_terminal_archive_releases_staging() -> None:
    staging = StagingAgentRunStore()
    archive = ModelInteractionInMemoryStepArchive(PersistenceDomain.TRACE)
    store = RuntimeAgentRunStore(staging, conversation_archive=InMemoryStepArchive(PersistenceDomain.CONVERSATION),
                                 execution_archive=archive, recovery_archive=None)
    await store.initialize()
    try:
        run = AgentRunRecord("run")
        await store.register_agent_run(run)
        terminal = _record(1, direct=False)
        values = {field.name: getattr(terminal, field.name) for field in fields(terminal)
                  if field.name != "request_envelope"}
        request = ModelRequest(parts=[UserPromptPart("request-1", timestamp=terminal.started_at)],
                               timestamp=terminal.started_at)
        values.update(request_context=build_inline_context_projection((request,),
                          lambda payload: staging.intern_payload("run", payload)),
                      request_envelope_digest=staging.intern_payload("run", b"{}")[0],
                      response_context=None, status="RUNNING", duration_ns=None, finished_at=None)
        store.stage_model_interaction(StagedModelInteraction(**values))
        page = await staging.list_model_interactions(agent_run_id="run", after_model_request_seq=0, limit=10)
        before = await store.resolve_model_interactions(page)
        response = ModelResponse(parts=[TextPart("response-1")], timestamp=terminal.started_at)
        await archive.sync_projection(run, events=(),
                                      checkpoints=(AgentRunCheckpoint("run", 1, [request, response]),),
                                      interactions=(terminal,))
        await staging.release_agent_run("run")
        assert await store.resolve_model_interactions(page) == before
        assert before[0][1] is None
        assert (await archive.resolve_model_interaction(terminal))[1] is not None
    finally:
        await store.preflight_close()
        await store.close()
