#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Original client attempts share the logical interaction's durable lifecycle."""

from linktools.ai.runtime.state import RuntimeState
import gzip
import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus, UsageMetrics
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelClientRequest, ModelClientResponse
from linktools.ai.runtime import RuntimeStorage, RuntimeStoragePlan
from linktools.ai.storage import StorageRoute
from linktools.ai.runtime._model_interaction import StagedModelInteraction
from linktools.ai.runtime.state import RuntimeDomain, SnapshotLimits
from linktools.ai.runtime.state._codec import (
    _CURRENT_CODEC,
    _decode_domain,
    _decode_step_envelope,
    _encode_persisted_domain,
    _encode_step_envelope,
    iter_runtime_object_refs,
)
from linktools.ai.runtime.state._contracts import ModelInteractionRecord, RuntimePayloadRef
from linktools.ai.runtime.state._model_client import resolve_client_requests
from linktools.ai.runtime.state._archive import PreparedArchive
from linktools.ai.runtime.state._step_contracts import AgentRunRecord
from linktools.ai.storage import InMemoryObjectStore, StoredPayload

from .test_evaluation_retention import Offline
from .test_history_projection_conformance import _record

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _storage(backend: str, root: Path) -> RuntimeStorage:
    if backend == "memory":
        return RuntimeStorage.in_memory()
    if backend == "filesystem":
        return RuntimeStorage.filesystem(root / "state")
    return RuntimeStorage.sqlite(root / "state.db")


def _requests(*, large: bool = False) -> tuple[ModelClientRequest, ...]:
    # Unrecognized fields and native role/content shapes must survive unchanged.
    body = (b'{ "messages": [{"role":"developer","content":[{"type":"text","text":"'
            + (b"long context " * 6000 if large else b"native")
            + b'"}]},{"role":"assistant","tool_calls":[{"id":"call","function":{"arguments":"{}"}}]},'
              b'{"role":"tool","tool_call_id":"call","content":"result"}],"extra":{"x":null} }')
    partial_sse = gzip.compress(b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n', mtime=0)
    error_body = b'{"error":{"message":"' + (b"retry " * 6000 if large else b"retry") + b'"}}'
    return (
        ModelClientRequest("openai.chat.completions", body, response=ModelClientResponse(
            error_body, 429, "json", None, True,
        )),
        ModelClientRequest("openai.chat.completions", body, response=ModelClientResponse(
            partial_sse, 200, "sse", "gzip", False,
        ), diagnostics={"started_at": _NOW.isoformat(), "received_bytes": len(partial_sse),
                        "first_chunk_ns": 100, "last_chunk_ns": 200, "read_outcome": "cancelled",
                        "close_started_ns": 300, "close_finished_ns": 400}),
        ModelClientRequest("openai.chat.completions", None, "body_unavailable", ModelClientResponse(
            None, 502, "unknown", None, False, "response_capture_failed",
        )),
        ModelClientRequest("openai.chat.completions", b"", response=ModelClientResponse(
            b"", 204, "unknown", None, True,
        )),
    )


def _staged(requests: tuple[ModelClientRequest, ...] | None) -> StagedModelInteraction:
    return StagedModelInteraction(
        agent_run_id="run", step_index=1, model_request_seq=1,
        purpose="agent", output_retry_index=None, model={"route_id": "default"},
        request_context=None, request_envelope_digest=None, response_context=None,
        status="CANCELLED", error_code=None, duration_ns=42, usage=None,
        started_at=_NOW, finished_at=_NOW, client_requests=requests,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("large", (False, True))
async def test_original_attempts_round_trip_and_relocate_without_sdk_context(
    tmp_path: Path, backend: str, large: bool,
) -> None:
    prepared = PreparedArchive()
    storage = RuntimeState(_storage(backend, tmp_path))
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    await storage.execution.executions.create_with_history_head(_record(ExecutionStatus.STARTED, 1))
    try:
        run = AgentRunRecord("run")
        source = storage.run_store.read_store(RuntimeDomain.TRACE)
        requests = _requests(large=large)
        staged = replace(_staged(requests),
                         model={"route_id": "default", "provider": "openai", "model_name": "gpt-4o"},
                         usage=UsageMetrics(2, 1, 120, 30, 11, 7))
        records = await source.prepare_interactions(run, (staged,), lambda _: b"{}", prepared=prepared)
        record = records[0]
        assert _decode_step_envelope(_encode_step_envelope(record)) == record
        first, retry = record.client_requests[:2]
        assert first.body == retry.body
        assert first.body.codec == ("zlib" if large else "raw")
        assert first.body.raw_size == len(requests[0].body)
        assert first.body.raw_digest == hashlib.sha256(requests[0].body).hexdigest()
        assert first.body.content.payload.kind == ("object" if large else "inline")
        await source.sync_projection(run, events=(), checkpoints=(), interactions=records, execution_id="execution", archive=prepared)
        archived = await source.list_model_interactions(agent_run_id="run")
        resolved = await source.resolve_model_interactions(archived, include_client_requests=True)
        assert resolved == [(None, None, None, requests)]
        assert await source.resolve_model_interaction(archived[0], include_client_requests=True) == resolved[0]
        with pytest.raises(AIError) as raised:
            await source.sync_projection(
                run, events=(), checkpoints=(),
                interactions=(replace(record, client_requests=()),), execution_id="execution",
            archive=prepared)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

        target = storage.run_store.read_store(RuntimeDomain.CHECKPOINT)
        relocated = await target.prepare_relocated_interactions(archived, resolved, prepared=prepared)
        metadata = tuple(replace(
            request, body=None,
            response=replace(request.response, body=None) if request.response is not None else None,
        ) for request in requests)
        expected = [(None, None, None, metadata)]
        assert all(request.body is None for request in relocated[0].client_requests)
        assert relocated[0].model == record.model
        assert relocated[0].usage == record.usage
        await target.sync_projection(run, events=(), checkpoints=(), interactions=relocated, execution_id="execution", archive=prepared)
        assert await target.resolve_model_interactions(relocated, include_client_requests=True) == expected
        restored_to_trace = await source.prepare_relocated_interactions(relocated, expected, prepared=prepared)
        assert await source.resolve_model_interactions(restored_to_trace, include_client_requests=True) == expected
        await source.release_agent_run("run", execution_id="execution")
        assert await target.resolve_model_interactions(relocated, include_client_requests=True) == expected
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("requests", (None, ()))
async def test_missing_capture_and_no_dispatch_remain_distinct_on_wire(requests) -> None:
    prepared = PreparedArchive()
    storage = RuntimeState(RuntimeStorage.in_memory())
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    await storage.execution.executions.create_with_history_head(_record(ExecutionStatus.STARTED, 1))
    try:
        archive = storage.run_store.read_store(RuntimeDomain.TRACE)
        record, = await archive.prepare_interactions(AgentRunRecord("run"), (_staged(requests),), lambda _: b"{}", prepared=prepared)
        assert _decode_step_envelope(_encode_step_envelope(record)).client_requests == requests
        assert await archive.resolve_model_interaction(record, include_client_requests=True) == (None, None, None, requests)
        missing_capture_field = _encode_persisted_domain(record)
        del missing_capture_field["fields"]["client_requests"]
        with pytest.raises(AIError) as missing:
            _decode_domain(missing_capture_field, ModelInteractionRecord, _CURRENT_CODEC, persisted=True)
        assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert missing.value.safe_details["missing_fields"] == ["client_requests"]
        with pytest.raises(ValueError, match="unfinished"):
            replace(record, status="RUNNING", duration_ns=None, finished_at=None, client_requests=())
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ("raw_digest", "raw_size", "compressed_bytes", "domain", "codec"))
async def test_original_body_decoded_integrity_fails_closed(corruption: str) -> None:
    prepared = PreparedArchive()
    storage = RuntimeState(RuntimeStorage.in_memory())
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    await storage.execution.executions.create_with_history_head(_record(ExecutionStatus.STARTED, 1))
    try:
        archive = storage.run_store.read_store(RuntimeDomain.TRACE)
        record, = await archive.prepare_interactions(AgentRunRecord("run"), (_staged(_requests(large=True)),), lambda _: b"{}", prepared=prepared)
        if corruption == "codec":
            encoded = _encode_persisted_domain(record)
            # Unsupported storage compression is not guessed from the bytes.
            encoded["fields"]["client_requests"]["$tuple"][0]["fields"]["body"]["fields"]["codec"] = "unknown"
            with pytest.raises(AIError):
                _decode_domain(encoded, ModelInteractionRecord, _CURRENT_CODEC, persisted=True)
            return
        request = record.client_requests[0]
        body = request.body
        if corruption == "raw_digest":
            body = replace(body, raw_digest="a" * 64)
        elif corruption == "raw_size":
            body = replace(body, raw_size=body.raw_size + 1)
        elif corruption == "compressed_bytes":
            body = replace(body, content=RuntimePayloadRef(
                StoredPayload.inline_bytes(b"not zlib"), RuntimeDomain.TRACE,
            ))
        else:
            body = replace(body, content=replace(body.content, source_domain=RuntimeDomain.CHECKPOINT))
        invalid = replace(record, client_requests=(replace(request, body=body),))
        with pytest.raises(AIError) as raised:
            await archive.resolve_model_interaction(invalid, include_client_requests=True)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
async def test_native_objects_survive_snapshot_and_follow_owner_cleanup(
    tmp_path: Path, backend: str,
) -> None:
    prepared = PreparedArchive()
    storage = RuntimeState(_storage(backend, tmp_path))
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    await storage.execution.executions.create_with_history_head(_record(ExecutionStatus.STARTED, 1))
    requests = _requests(large=True)
    try:
        archive = storage.run_store.read_store(RuntimeDomain.TRACE)
        records = await archive.prepare_interactions(AgentRunRecord("run"), (_staged(requests),), lambda _: b"{}", prepared=prepared)
        await archive.sync_projection(AgentRunRecord("run"), events=(), checkpoints=(), interactions=records, execution_id="execution", archive=prepared)
        references = tuple(set(iter_runtime_object_refs(_encode_persisted_domain(records[0]), default_domain=RuntimeDomain.TRACE)))
        assert len(references) == 2
        retained = await storage.purge_unreferenced_objects(references, exclusive=Offline())
        assert retained.retained == references
    finally:
        await storage.close()

    objects = InMemoryObjectStore("snapshot")
    limits = SnapshotLimits(max_entries=100, max_bytes=1024 * 1024)
    reader = RuntimeState(_storage(backend, tmp_path))
    await reader.initialize(namespace="native-client", tenant_id="tenant", read_only=True)
    try:
        snapshot = await reader.export_snapshot(object_store=objects, limits=limits)
    finally:
        await reader.close()
    restored_root = tmp_path / "restored"
    await RuntimeState.restore_snapshot(snapshot, object_store=objects, root=restored_root, limits=limits)
    restored = RuntimeState(RuntimeStorage.from_root(restored_root))
    await restored.initialize(namespace="native-client", tenant_id="tenant")
    try:
        archive = restored.run_store.read_store(RuntimeDomain.TRACE)
        records = await archive.list_model_interactions(agent_run_id="run")
        assert await archive.resolve_model_interactions(records, include_client_requests=True) == [(None, None, None, requests)]
        references = tuple(set(iter_runtime_object_refs(_encode_persisted_domain(records[0]), default_domain=RuntimeDomain.TRACE)))
        await archive.release_agent_run("run", execution_id="execution")
        result = await restored.purge_unreferenced_objects(references, exclusive=Offline())
        assert result.deleted == references
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_volatile_archive_and_staging_keep_original_attempt_views() -> None:
    prepared = PreparedArchive()
    from linktools.ai.runtime.state._model_interaction_store import ModelInteractionInMemoryStepArchive

    archive = ModelInteractionInMemoryStepArchive(RuntimeDomain.TRACE)
    await archive.initialize()
    try:
        requests = _requests(large=True)
        records = await archive.prepare_interactions(AgentRunRecord("run"), (_staged(requests),), lambda _: b"{}", prepared=prepared)
        assert records[0].client_requests[0].body.content.payload.kind == "inline"
        await archive.sync_projection(AgentRunRecord("run"), events=(), checkpoints=(), interactions=records, execution_id="execution", archive=prepared)
        resolved = await archive.resolve_model_interactions(records, include_client_requests=True)
        assert resolved == [(None, None, None, requests)]
        assert await archive.prepare_relocated_interactions(records, resolved, prepared=prepared) == records
    finally:
        await archive.close()

    storage = RuntimeState(RuntimeStorage(RuntimeStoragePlan(execution=StorageRoute.transient())))
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    try:
        staged = _staged(requests)
        await storage.run_store.register_agent_run(AgentRunRecord("run"))
        storage.run_store.stage_model_interaction(staged)
        assert await storage.run_store.resolve_model_interaction(staged, include_client_requests=True) == (None, None, None, requests)
        from linktools.ai.runtime._transient_history import TransientExecutionHistoryStore
        source = storage.run_store.read_store(RuntimeDomain.TRACE)
        transient = TransientExecutionHistoryStore(source, storage.run_store)
        assert await transient.list_model_interactions(agent_run_id="run") == [staged]
        assert await transient.resolve_model_interactions((staged,)) == [(None, None, None, None)]
        assert await transient.resolve_model_interactions((staged,), include_client_requests=True) == [
            (None, None, None, requests),
        ]
        with pytest.raises(AIError) as raised:
            await source.resolve_model_interaction(staged, include_client_requests=True)
        assert raised.value.code is ErrorCode.STORAGE_DEPENDENCY_NOT_READY

    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_sdk_projection_and_native_bytes_remain_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = PreparedArchive()
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from linktools.ai.runtime._model_interaction import build_context_projection

    storage = RuntimeState(RuntimeStorage.filesystem(tmp_path / "state"))
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    await storage.execution.executions.create_with_history_head(_record(ExecutionStatus.STARTED, 1))
    payloads: dict[str, bytes] = {}

    def intern(raw: bytes) -> tuple[str, int]:
        digest = hashlib.sha256(raw).hexdigest()
        payloads[digest] = raw
        return digest, len(raw)

    try:
        sdk_request = (ModelRequest(parts=[UserPromptPart("SDK input")]),)
        sdk_response = (ModelResponse(parts=[TextPart("SDK projection")]),)
        requests = _requests(large=True)
        staged = replace(
            _staged(requests), status="SUCCEEDED",
            request_context=build_context_projection((), sdk_request, intern),
            request_envelope_digest=intern(b'{"streaming":true}')[0],
            response_context=build_context_projection((), sdk_response, intern),
        )
        archive = storage.run_store.read_store(RuntimeDomain.TRACE)
        records = await archive.prepare_interactions(AgentRunRecord("run"), (staged,), payloads.__getitem__, prepared=prepared)
        await archive.sync_projection(AgentRunRecord("run"), events=(), checkpoints=(), interactions=records, execution_id="execution", archive=prepared)
        objects = storage.object_store(RuntimeDomain.TRACE)
        original_open = objects.open
        raw_keys = {ref.key for _, ref in iter_runtime_object_refs(
            _encode_persisted_domain(records[0]), default_domain=RuntimeDomain.TRACE,
        )}

        async def reject_raw_open(key):
            if key in raw_keys:
                raise AssertionError("unrequested native content read")
            async for chunk in original_open(key):
                yield chunk

        with monkeypatch.context() as patch:
            patch.setattr(objects, "open", reject_raw_open)
            assert await archive.list_model_interactions(agent_run_id="run") == list(records)
            assert await archive.resolve_model_interactions(records) == [
                (sdk_request, sdk_response, b'{"streaming":true}', None),
            ]
            assert (await archive.resolve_model_interaction(records[0]))[3] is None
            metadata = await resolve_client_requests(
                records[0].client_requests, RuntimeDomain.CHECKPOINT, include_bodies=False,
            )
            checkpoint_archive = storage.run_store.read_store(RuntimeDomain.CHECKPOINT)
            relocated = await checkpoint_archive.prepare_relocated_interactions(
                records, [(sdk_request, sdk_response, b'{"streaming":true}', metadata)],
            prepared=prepared)
            assert relocated[0].client_requests[0].body is None
            assert relocated[0].client_requests[0].response.status_code == 429
            assert relocated[0].client_requests[0].record_http_bodies is True
            with pytest.raises(AssertionError, match="unrequested native"):
                await archive.resolve_model_interactions(records, include_client_requests=True)
            with pytest.raises(AIError):
                await archive.prepare_relocated_interactions(
                    records, await archive.resolve_model_interactions(records),
                prepared=prepared)
        assert await archive.resolve_model_interactions(records, include_client_requests=True) == [
            (sdk_request, sdk_response, b'{"streaming":true}', requests),
        ]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_cancelled_preparation_leaves_no_owner_and_body_is_reclaimable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = PreparedArchive()
    import asyncio

    storage = RuntimeState(RuntimeStorage.filesystem(tmp_path / "state"))
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    await storage.execution.executions.create_with_history_head(_record(ExecutionStatus.STARTED, 1))
    try:
        objects = storage.object_store(RuntimeDomain.TRACE)
        original_put = objects.put
        references = []

        async def put_then_cancel(key, chunks, **kwargs):
            from linktools.ai.storage import ObjectRef
            stat = await original_put(key, chunks, **kwargs)
            references.append((RuntimeDomain.TRACE, ObjectRef("runtime", key, stat.digest, stat.size)))
            raise asyncio.CancelledError

        monkeypatch.setattr(objects, "put", put_then_cancel)
        archive = storage.run_store.read_store(RuntimeDomain.TRACE)
        with pytest.raises(asyncio.CancelledError):
            await archive.prepare_interactions(AgentRunRecord("run"), (_staged(_requests(large=True)),), lambda _: b"{}", prepared=prepared)
        assert await archive.list_model_interactions(agent_run_id="run") == []
        assert len(references) == 1
        result = await storage.purge_unreferenced_objects(tuple(references), exclusive=Offline())
        assert result.deleted == tuple(references)
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_interrupted_record_preserves_observed_incomplete_response() -> None:
    prepared = PreparedArchive()
    storage = RuntimeState(RuntimeStorage.in_memory())
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    await storage.execution.executions.create_with_history_head(_record(ExecutionStatus.STARTED, 1))
    try:
        requests = (_requests()[1],)
        staged = replace(_staged(requests), status="INTERRUPTED", duration_ns=None, finished_at=None)
        archive = storage.run_store.read_store(RuntimeDomain.TRACE)
        records = await archive.prepare_interactions(AgentRunRecord("run"), (staged,), lambda _: b"{}", prepared=prepared)
        await archive.sync_projection(AgentRunRecord("run"), events=(), checkpoints=(), interactions=records, execution_id="execution", archive=prepared)
        loaded = await archive.list_model_interactions(agent_run_id="run")
        assert loaded[0].finished_at is None and loaded[0].usage is None
        assert await archive.resolve_model_interactions(loaded, include_client_requests=True) == [
            (None, None, None, requests),
        ]
        assert loaded[0].client_requests[0].response.complete is False
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_recovery_preserves_existing_trace_bodies_without_reading_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = PreparedArchive()
    from linktools.ai.runtime.state._step_contracts import AgentRunCheckpoint

    storage = RuntimeState(RuntimeStorage.filesystem(tmp_path / "state"))
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    await storage.execution.executions.create_with_history_head(_record(ExecutionStatus.STARTED, 1))
    try:
        run = AgentRunRecord("run")
        trace = storage.run_store.read_store(RuntimeDomain.TRACE)
        checkpoint = storage.run_store.read_store(RuntimeDomain.CHECKPOINT)
        requests = _requests(large=True)
        traced = await trace.prepare_interactions(run, (_staged(requests),), lambda _: b"{}", prepared=prepared)
        await trace.sync_projection(run, events=(), checkpoints=(), interactions=traced, execution_id="execution", archive=prepared)
        recovered = await checkpoint.prepare_interactions(run, (_staged(requests),), lambda _: b"{}", prepared=prepared)
        await checkpoint.sync_projection(run, events=(), checkpoints=(), interactions=recovered, archive=prepared)
        await checkpoint.materialize_checkpoint(run, AgentRunCheckpoint("run", 1, [], transcript_message_count_before=0))
        objects = storage.object_store(RuntimeDomain.TRACE)
        original_open = objects.open
        raw_keys = {ref.key for _, ref in iter_runtime_object_refs(
            _encode_persisted_domain(traced[0]), default_domain=RuntimeDomain.TRACE,
        )}

        async def reject_raw_open(key):
            if key in raw_keys:
                raise AssertionError("recovery must not read native bodies")
            async for chunk in original_open(key):
                yield chunk

        with monkeypatch.context() as patch:
            patch.setattr(objects, "open", reject_raw_open)
            await storage.run_store.materialize_from_recovery(
                target=RuntimeDomain.TRACE, agent_run_id="run", execution_id="execution",
            )
        loaded = await trace.list_model_interactions(agent_run_id="run")
        assert loaded[0].client_requests == traced[0].client_requests
        assert await trace.resolve_model_interactions(loaded, include_client_requests=True) == [
            (None, None, None, requests),
        ]
    finally:
        await storage.close()
