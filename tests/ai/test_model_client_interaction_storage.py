#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Original client attempts share the logical interaction's durable lifecycle."""

import gzip
import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelClientRequest, ModelClientResponse
from linktools.ai.runtime import RuntimeStorage, RuntimeStoragePlan, RuntimeStorageRoute
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
        )),
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
    storage = _storage(backend, tmp_path)
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    try:
        run = AgentRunRecord("run")
        source = storage.run_store.read_store(RuntimeDomain.CHECKPOINT)
        requests = _requests(large=large)
        records = await source.prepare_interactions(run, (_staged(requests),), lambda _: b"{}")
        record = records[0]
        assert _decode_step_envelope(_encode_step_envelope(record)) == record
        first, retry = record.client_requests[:2]
        assert first.body == retry.body
        assert first.body.codec == ("zlib" if large else "raw")
        assert first.body.raw_size == len(requests[0].body)
        assert first.body.raw_digest == hashlib.sha256(requests[0].body).hexdigest()
        assert first.body.content.payload.kind == ("object" if large else "inline")
        await source.sync_projection(run, events=(), checkpoints=(), interactions=records)
        archived = await source.list_model_interactions(agent_run_id="run")
        resolved = await source.resolve_model_interactions(archived, include_client_requests=True)
        assert resolved == [(None, None, None, requests)]
        assert await source.resolve_model_interaction(archived[0], include_client_requests=True) == resolved[0]
        with pytest.raises(AIError) as raised:
            await source.sync_projection(
                run, events=(), checkpoints=(),
                interactions=(replace(record, client_requests=()),),
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

        await storage.execution.executions.create_with_history_head(_record(ExecutionStatus.STARTED, 1))
        target = storage.run_store.read_store(RuntimeDomain.TRACE)
        relocated = await target.prepare_relocated_interactions(archived, resolved)
        assert relocated[0].client_requests[0].body.content.source_domain is RuntimeDomain.TRACE
        await target.sync_projection(run, events=(), checkpoints=(), interactions=relocated, execution_id="execution")
        assert await target.resolve_model_interactions(relocated, include_client_requests=True) == resolved
        await source.release_agent_run("run")
        assert await target.resolve_model_interactions(relocated, include_client_requests=True) == resolved
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("requests", (None, ()))
async def test_missing_capture_and_no_dispatch_remain_distinct_on_wire(requests) -> None:
    storage = RuntimeStorage.in_memory()
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    try:
        archive = storage.run_store.read_store(RuntimeDomain.CHECKPOINT)
        record, = await archive.prepare_interactions(AgentRunRecord("run"), (_staged(requests),), lambda _: b"{}")
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
    storage = RuntimeStorage.in_memory()
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    try:
        archive = storage.run_store.read_store(RuntimeDomain.CHECKPOINT)
        record, = await archive.prepare_interactions(AgentRunRecord("run"), (_staged(_requests(large=True)),), lambda _: b"{}")
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
                StoredPayload.inline_bytes(b"not zlib"), RuntimeDomain.CHECKPOINT,
            ))
        else:
            body = replace(body, content=replace(body.content, source_domain=RuntimeDomain.TRACE))
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
    storage = _storage(backend, tmp_path)
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    requests = _requests(large=True)
    try:
        archive = storage.run_store.read_store(RuntimeDomain.CHECKPOINT)
        records = await archive.prepare_interactions(AgentRunRecord("run"), (_staged(requests),), lambda _: b"{}")
        await archive.sync_projection(AgentRunRecord("run"), events=(), checkpoints=(), interactions=records)
        references = tuple(set(iter_runtime_object_refs(_encode_persisted_domain(records[0]), default_domain=RuntimeDomain.CHECKPOINT)))
        assert len(references) == 2
        retained = await storage.purge_unreferenced_objects(references, exclusive=Offline())
        assert retained.retained == references
    finally:
        await storage.close()

    objects = InMemoryObjectStore("snapshot")
    limits = SnapshotLimits(max_entries=100, max_bytes=1024 * 1024)
    reader = _storage(backend, tmp_path)
    await reader.initialize(namespace="native-client", tenant_id="tenant", read_only=True)
    try:
        snapshot = await reader.export_snapshot(object_store=objects, limits=limits)
    finally:
        await reader.close()
    restored_root = tmp_path / "restored"
    await RuntimeStorage.restore_snapshot(snapshot, object_store=objects, root=restored_root, limits=limits)
    restored = RuntimeStorage.from_root(restored_root)
    await restored.initialize(namespace="native-client", tenant_id="tenant")
    try:
        archive = restored.run_store.read_store(RuntimeDomain.CHECKPOINT)
        records = await archive.list_model_interactions(agent_run_id="run")
        assert await archive.resolve_model_interactions(records, include_client_requests=True) == [(None, None, None, requests)]
        references = tuple(set(iter_runtime_object_refs(_encode_persisted_domain(records[0]), default_domain=RuntimeDomain.CHECKPOINT)))
        await archive.release_agent_run("run")
        result = await restored.purge_unreferenced_objects(references, exclusive=Offline())
        assert result.deleted == references
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_volatile_archive_and_staging_keep_original_attempt_views() -> None:
    from linktools.ai.runtime.state._model_interaction_store import ModelInteractionInMemoryStepArchive

    archive = ModelInteractionInMemoryStepArchive(RuntimeDomain.CHECKPOINT)
    await archive.initialize()
    try:
        requests = _requests(large=True)
        records = await archive.prepare_interactions(AgentRunRecord("run"), (_staged(requests),), lambda _: b"{}")
        assert records[0].client_requests[0].body.content.payload.kind == "inline"
        await archive.sync_projection(AgentRunRecord("run"), events=(), checkpoints=(), interactions=records)
        resolved = await archive.resolve_model_interactions(records, include_client_requests=True)
        assert resolved == [(None, None, None, requests)]
        assert await archive.prepare_relocated_interactions(records, resolved) == records
    finally:
        await archive.close()

    storage = RuntimeStorage(RuntimeStoragePlan(execution=RuntimeStorageRoute.transient()))
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
async def test_sdk_recovery_projection_and_native_bytes_remain_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from linktools.ai.runtime._model_interaction import build_context_projection

    storage = RuntimeStorage.filesystem(tmp_path / "state")
    await storage.initialize(namespace="native-client", tenant_id="tenant")
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
        archive = storage.run_store.read_store(RuntimeDomain.CHECKPOINT)
        records = await archive.prepare_interactions(AgentRunRecord("run"), (staged,), payloads.__getitem__)
        await archive.sync_projection(AgentRunRecord("run"), events=(), checkpoints=(), interactions=records)
        objects = storage.object_store(RuntimeDomain.CHECKPOINT)
        original_open = objects.open
        raw_keys = {ref.key for _, ref in iter_runtime_object_refs(
            _encode_persisted_domain(records[0]), default_domain=RuntimeDomain.CHECKPOINT,
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
            with pytest.raises(AssertionError, match="unrequested native"):
                await archive.resolve_model_interactions(records, include_client_requests=True)
            with pytest.raises(AIError):
                await archive.prepare_relocated_interactions(
                    records, await archive.resolve_model_interactions(records),
                )
        assert await archive.resolve_model_interactions(records, include_client_requests=True) == [
            (sdk_request, sdk_response, b'{"streaming":true}', requests),
        ]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_cancelled_preparation_leaves_no_owner_and_body_is_reclaimable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    storage = RuntimeStorage.filesystem(tmp_path / "state")
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    try:
        objects = storage.object_store(RuntimeDomain.CHECKPOINT)
        original_put = objects.put
        references = []

        async def put_then_cancel(key, chunks, **kwargs):
            from linktools.ai.storage import ObjectRef
            stat = await original_put(key, chunks, **kwargs)
            references.append((RuntimeDomain.CHECKPOINT, ObjectRef("runtime", key, stat.digest, stat.size)))
            raise asyncio.CancelledError

        monkeypatch.setattr(objects, "put", put_then_cancel)
        archive = storage.run_store.read_store(RuntimeDomain.CHECKPOINT)
        with pytest.raises(asyncio.CancelledError):
            await archive.prepare_interactions(AgentRunRecord("run"), (_staged(_requests(large=True)),), lambda _: b"{}")
        assert await archive.list_model_interactions(agent_run_id="run") == []
        assert len(references) == 1
        result = await storage.purge_unreferenced_objects(tuple(references), exclusive=Offline())
        assert result.deleted == tuple(references)
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_interrupted_record_preserves_observed_incomplete_response() -> None:
    storage = RuntimeStorage.in_memory()
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    try:
        requests = (_requests()[1],)
        staged = replace(_staged(requests), status="INTERRUPTED", duration_ns=None, finished_at=None)
        archive = storage.run_store.read_store(RuntimeDomain.CHECKPOINT)
        records = await archive.prepare_interactions(AgentRunRecord("run"), (staged,), lambda _: b"{}")
        await archive.sync_projection(AgentRunRecord("run"), events=(), checkpoints=(), interactions=records)
        loaded = await archive.list_model_interactions(agent_run_id="run")
        assert loaded[0].finished_at is None and loaded[0].usage is None
        assert await archive.resolve_model_interactions(loaded, include_client_requests=True) == [
            (None, None, None, requests),
        ]
        assert loaded[0].client_requests[0].response.complete is False
    finally:
        await storage.close()
