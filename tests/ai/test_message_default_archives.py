#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Omitted message defaults do not replace original archive-byte integrity."""

import asyncio
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolReturnPart, UserPromptPart

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionInputContext
from linktools.ai.inputs import ExecutionContextArchive
from linktools.ai.persistence import Persistence, PersistenceDomain
from linktools.ai.persistence._archive import PreparedArchive
from linktools.ai.persistence.contracts import ContextProjection
from linktools.ai.persistence import _history as history_module
from linktools.ai.model.messages import encode_model_messages

from .test_incremental_projection_boundaries import _storage

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _messages():
    return (
        ModelRequest(parts=[UserPromptPart("prompt", timestamp=_NOW)], timestamp=_NOW),
        ModelResponse(parts=[TextPart("answer")], timestamp=_NOW),
    )


def _omitted_payload(messages):
    values = json.loads(encode_model_messages(messages))
    for value in values:
        value.pop("instructions", None)
        value.pop("model_name", None)
        for part in value["parts"]:
            if part["part_kind"] == "text":
                part.pop("id")
    return canonical_json_bytes(values)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_transcript_and_archived_projection_verify_original_omitted_field_bytes(tmp_path, monkeypatch, backend):
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="message-defaults", tenant_id="tenant")
    try:
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        repository = archive.transcript_repository
        await repository.create_head("run")
        bundle = PreparedArchive()
        with monkeypatch.context() as writer:
            writer.setattr(history_module, "encode_model_messages", _omitted_payload)
            prepared = await repository.prepare_observation("run", _messages(), first_message_index=0, pending=None, pending_keys=())
            projection = await repository.prepare_model_projection(_messages(), bundle)
            await archive.state_store.mutate(lambda tx: repository.commit_observation(tx, prepared))
            content_archive = await repository.content_archive()
            await content_archive.store.mutate(lambda tx: content_archive.publish(tx, bundle))
        assert await repository.load_messages("run") == _messages()
        for start, end in ((0, 1), (1, 2)):
            assert tuple([message async for message in repository.iter_message_range("run", start=start, end=end)]) == _messages()[start:end]
        restored = await repository.load_projected_context("run", projection)
        assert restored.model_messages() == _messages()
        assert projection.items[0].raw_digest == hashlib.sha256(_omitted_payload(_messages())).hexdigest()
        assert _omitted_payload(_messages()) != encode_model_messages(_messages())
        bad_projection = replace(projection, items=(replace(projection.items[0], raw_digest="0" * 64),))
        with pytest.raises(AIError) as caught:
            await repository.load_projected_context("run", bad_projection)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        bad_chunk = replace(prepared.chunks[0], raw_digest="0" * 64)
        with pytest.raises(AIError) as caught:
            await repository._decode_chunk_ranges(((bad_chunk, 0, bad_chunk.message_count),))
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_execution_context_preserves_omitted_source_bytes_and_semantic_digest(tmp_path, backend):
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="message-defaults", tenant_id="tenant")
    try:
        service = ExecutionContextArchive(
            state.execution.executions.state_store, state.archive_object_store(PersistenceDomain.EXECUTION),
            namespace="message-defaults", tenant_id="tenant",
        )
        context = ExecutionInputContext(_omitted_payload(_messages()), session_metadata={"label": "retained"})
        prepared = await service.prepare(context)
        await prepared.archive.store.mutate(lambda tx: prepared.archive.publish(tx, prepared.records))
        restored = await service.load(prepared.reference)
        assert restored == context
        assert restored.history == context.history
        assert restored.digest == context.digest
        assert restored.model_messages() == _messages()
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
@pytest.mark.parametrize("pending_kind", ("response", "request"))
async def test_pending_omitted_defaults_reopen_without_accepting_changed_content(tmp_path, monkeypatch, backend, pending_kind):
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="message-defaults", tenant_id="tenant")
    archive = state.run_store.read_store(PersistenceDomain.TRACE)
    repository = archive.transcript_repository
    await repository.create_head("run")
    pending = _messages()[1] if pending_kind == "response" else ModelRequest(
        parts=[ToolReturnPart("tool", "answer", tool_call_id="call", timestamp=_NOW)], timestamp=_NOW,
    )
    pending_keys = ("part:0",) if pending_kind == "response" else ("tool_result:call",)
    with monkeypatch.context() as writer:
        writer.setattr(history_module, "encode_model_messages", _omitted_payload)
        prepared = await repository.prepare_observation("run", (), first_message_index=0, pending=pending, pending_keys=pending_keys)
        await archive.state_store.mutate(lambda tx: repository.commit_observation(tx, prepared))
    await state.close()
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="message-defaults", tenant_id="tenant")
    try:
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        repository = archive.transcript_repository
        head = await repository.get_head("run")
        chunks, keys = await archive.state_store.read(lambda tx: repository.capture_pending_in_transaction(tx, head))
        restored = await repository.read_pending_message(head.pending, chunks)
        assert keys == pending_keys and restored == pending
        part = restored.parts[0]
        changes = [replace(part, content="changed")]
        if pending_kind == "response":
            changes.extend((replace(part, id="changed"), replace(part, provider_details={"changed": True})))
        else:
            changes.extend((replace(part, tool_call_id="changed"), replace(part, metadata={"changed": True}),
                            replace(part, timestamp=_NOW + timedelta(seconds=1))))
        for changed_part in changes:
            changed = replace(restored, parts=[changed_part])
            with pytest.raises(AIError) as caught:
                await repository.prepare_observation("run", (changed,), first_message_index=0, pending=None, pending_keys=())
            assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        repeated = replace(restored, parts=[replace(restored.parts[0])])
        same_pending = await repository.prepare_observation(
            "run", (), first_message_index=0, pending=repeated, pending_keys=keys,
        )
        assert same_pending.pending == head.pending
        prepared = await repository.prepare_observation("run", (restored,), first_message_index=0, pending=None, pending_keys=())
        await archive.state_store.mutate(lambda tx: repository.commit_observation(tx, prepared))
        assert await repository.verify_observation(prepared)
        assert await repository.verify_observation(same_pending)
        assert await repository.load_messages("run") == (pending,)
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
async def test_pending_missing_clock_factory_does_not_claim_exact_historical_recovery(tmp_path, monkeypatch, backend):
    def omitted_clock(messages):
        values = json.loads(_omitted_payload(messages))
        for value in values:
            for part in value["parts"]:
                part.pop("timestamp", None)
        return canonical_json_bytes(values)

    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="message-defaults-clock", tenant_id="tenant")
    archive = state.run_store.read_store(PersistenceDomain.TRACE)
    repository = archive.transcript_repository
    await repository.create_head("run")
    with monkeypatch.context() as writer:
        writer.setattr(history_module, "encode_model_messages", omitted_clock)
        prepared = await repository.prepare_observation(
            "run", (), first_message_index=0,
            pending=ModelRequest(parts=[ToolReturnPart("tool", "result", tool_call_id="call", timestamp=_NOW)], timestamp=_NOW),
            pending_keys=("tool_result:call",),
        )
        await archive.state_store.mutate(lambda tx: repository.commit_observation(tx, prepared))
    await state.close()
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="message-defaults-clock", tenant_id="tenant")
    try:
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        repository = archive.transcript_repository
        head = await repository.get_head("run")
        chunks, _keys = await archive.state_store.read(lambda tx: repository.capture_pending_in_transaction(tx, head))
        restored = await repository.read_pending_message(head.pending, chunks)
        await asyncio.sleep(0.01)
        with pytest.raises(AIError) as caught:
            await repository.prepare_observation("run", (restored,), first_message_index=0, pending=None, pending_keys=())
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await repository.get_head("run") == head
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("blocks", (1, 2))
async def test_loaded_projection_reuse_digest_describes_original_root_bytes(tmp_path, monkeypatch, backend, blocks):
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="message-defaults-reuse", tenant_id="tenant")
    try:
        repository = state.run_store.read_store(PersistenceDomain.TRACE).transcript_repository
        bundle = PreparedArchive()
        groups = (_messages(),) if blocks == 1 else tuple((message,) for message in _messages())
        projections = []
        with monkeypatch.context() as writer:
            writer.setattr(history_module, "encode_model_messages", _omitted_payload)
            for group in groups:
                projections.append(await repository.prepare_model_projection(group, bundle))
        projection = ContextProjection(tuple(item for part in projections for item in part.items))
        archive = await repository.content_archive()
        await archive.store.mutate(lambda tx: archive.publish(tx, bundle))
        loaded = await repository.load_projected_context("run", projection)
        assert loaded.model_messages() == _messages()
        assert loaded.archive_digest == hashlib.sha256(_omitted_payload(_messages())).hexdigest()
        context = ExecutionInputContext.from_messages(loaded.model_messages())
        service = ExecutionContextArchive(
            state.execution.executions.state_store, state.archive_object_store(PersistenceDomain.EXECUTION),
            namespace="message-defaults-reuse", tenant_id="tenant",
        )
        prepared = await service.prepare(context, history_root=loaded.archive_root, history_digest=loaded.archive_digest)
        assert prepared.archive.scope_id == loaded.archive_root.scope_id
        assert prepared.archive.content_scope == loaded.archive_root.content_scope
        await prepared.archive.store.mutate(lambda tx: prepared.archive.publish(tx, prepared.records))
        assert await service.load(prepared.reference) == context
        assert prepared.reference.payload.decode()["history"] != loaded.archive_root.to_json()
        assert (await repository.load_projected_context("run", projection)).archive_digest == loaded.archive_digest
    finally:
        await state.close()
