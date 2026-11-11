#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Original client attempts share the logical interaction's durable lifecycle."""

from linktools.ai.persistence import Persistence
import gzip
import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus, UsageMetrics, canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelClientRequest, ModelClientResponse
from linktools.ai.persistence import PersistenceStorage, PersistencePlan
from linktools.ai.storage import StorageRoute
from linktools.ai.persistence._model_interaction import StagedModelInteraction
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.persistence.codec import (
    _CURRENT_CODEC,
    _decode_domain,
    _decode_step_envelope,
    _encode_persisted_domain,
    _encode_step_envelope,
    iter_runtime_object_refs,
)
from linktools.ai.persistence.contracts import ModelInteractionRecord, RuntimePayloadRef
from linktools.ai.persistence._model_client import resolve_client_requests
from linktools.ai.persistence._archive import archive_content_scope, ArchiveRoot, ContentRecord, PreparedArchive
from linktools.ai.persistence._step_contracts import AgentRunRecord
from linktools.ai.storage import InMemoryObjectStore, StoredPayload

from .test_evaluation_retention import Offline
from .test_history_projection_conformance import _record
from ._runtime_test_helpers import native_checkpoint, seed_native_snapshot

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _storage(backend: str, root: Path) -> PersistenceStorage:
    if backend == "memory":
        return PersistenceStorage.in_memory()
    if backend == "exclusive_sqlite":
        return PersistenceStorage.from_root(root / "state", exclusive_writer=True)
    return PersistenceStorage.sqlite(root / "state.db")


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
@pytest.mark.parametrize("backend", ("memory", "exclusive_sqlite", "sqlite"))
@pytest.mark.parametrize("large", (False, True))
async def test_original_attempts_round_trip_and_relocate_without_sdk_context(
    tmp_path: Path, backend: str, large: bool,
) -> None:
    prepared = PreparedArchive()
    storage = Persistence(_storage(backend, tmp_path))
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    await storage.execution.executions.create_execution(_record(ExecutionStatus.STARTED, 1))
    target_root = tmp_path / "relocated"
    target_root.mkdir()
    target_storage = Persistence(_storage(backend, target_root))
    await target_storage.initialize(namespace="native-client", tenant_id="tenant")
    await target_storage.execution.executions.create_execution(_record(ExecutionStatus.STARTED, 1))
    try:
        run = AgentRunRecord("run")
        source = storage.run_store.read_store(PersistenceDomain.TRACE)
        requests = _requests(large=large)
        staged = replace(_staged(requests),
                         model={"route_id": "default", "provider": "openai", "model_name": "gpt-4o"},
                         usage=UsageMetrics(2, 1, 120, 30, 11, 7))
        records = await source.prepare_interactions(run, (staged,), lambda _: b"{}", prepared=prepared)
        record = records[0]
        assert _decode_step_envelope(_encode_step_envelope(record)) == record
        first, retry = record.client_requests[:2]
        assert first.body == retry.body
        assert first.body.codec == "raw"
        assert first.body.raw_size == len(requests[0].body)
        assert first.body.raw_digest == hashlib.sha256(requests[0].body).hexdigest()
        assert isinstance(first.body.content, ArchiveRoot)
        content_archive = await source.transcript_repository.content_archive()
        content_ids = await content_archive.content_ids(first.body.content, prepared=prepared)
        content = next(ContentRecord.from_record(item) for item in prepared.records.values()
                       if item.kind == "archive_content" and item.sort_key.decode("ascii") == content_ids[0])
        assert content.compression == ("zlib" if large else "raw")
        assert content.payload is None and isinstance(content.body, bytes)
        await source.sync_projection(run, events=(), checkpoints=(), interactions=records, execution_id="execution", archive=prepared)
        archived = await source.list_model_interactions(agent_run_id="run")
        resolved = await source.resolve_model_interactions(archived, include_client_requests=True)
        assert resolved == [(None, None, None, requests, None)]
        assert await source.resolve_model_interaction(archived[0], include_client_requests=True) == resolved[0]
        with pytest.raises(AIError) as raised:
            await source.sync_projection(
                run, events=(), checkpoints=(),
                interactions=(replace(record, client_requests=()),), execution_id="execution",
            archive=prepared)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

        target = target_storage.run_store.read_store(PersistenceDomain.TRACE)
        target_prepared = PreparedArchive()
        relocated = await target.prepare_relocated_interactions(archived, resolved, prepared=target_prepared)
        expected = [(None, None, None, requests, None)]
        assert relocated[0].client_requests[0].body.raw_digest == record.client_requests[0].body.raw_digest
        assert relocated[0].model == record.model
        assert relocated[0].usage == record.usage
        await target.sync_projection(run, events=(), checkpoints=(), interactions=relocated, execution_id="execution", archive=target_prepared)
        assert await target.resolve_model_interactions(relocated, include_client_requests=True) == expected
        restored_to_trace = await source.prepare_relocated_interactions(relocated, expected, prepared=PreparedArchive())
        assert await source.resolve_model_interactions(restored_to_trace, include_client_requests=True) == expected
        await source.release_agent_run("run", execution_id="execution")
        assert await target.resolve_model_interactions(relocated, include_client_requests=True) == expected
    finally:
        await target_storage.close()
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("requests", (None, ()))
async def test_missing_capture_and_no_dispatch_remain_distinct_on_wire(requests) -> None:
    prepared = PreparedArchive()
    storage = Persistence(PersistenceStorage.in_memory())
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    await storage.execution.executions.create_execution(_record(ExecutionStatus.STARTED, 1))
    try:
        archive = storage.run_store.read_store(PersistenceDomain.TRACE)
        record, = await archive.prepare_interactions(AgentRunRecord("run"), (_staged(requests),), lambda _: b"{}", prepared=prepared)
        assert _decode_step_envelope(_encode_step_envelope(record)).client_requests == requests
        assert await archive.resolve_model_interaction(record, include_client_requests=True) == (None, None, None, requests, None)
        missing_capture_field = _encode_persisted_domain(record)
        del missing_capture_field["fields"]["client_requests"]
        restored = _decode_domain(missing_capture_field, ModelInteractionRecord, _CURRENT_CODEC, persisted=True)
        assert restored == replace(record, client_requests=None)
        assert await archive.resolve_model_interaction(restored, include_client_requests=True) == (None, None, None, None, None)
        with pytest.raises(ValueError, match="unfinished"):
            replace(record, status="RUNNING", duration_ns=None, finished_at=None, client_requests=())
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ("raw_digest", "raw_size", "compressed_bytes", "missing", "domain", "codec"))
async def test_original_body_decoded_integrity_fails_closed(corruption: str) -> None:
    prepared = PreparedArchive()
    storage = Persistence(PersistenceStorage.in_memory())
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    await storage.execution.executions.create_execution(_record(ExecutionStatus.STARTED, 1))
    try:
        archive = storage.run_store.read_store(PersistenceDomain.TRACE)
        record, = await archive.prepare_interactions(AgentRunRecord("run"), (_staged(_requests(large=True)),), lambda _: b"{}", prepared=prepared)
        await archive.sync_projection(AgentRunRecord("run"), events=(), checkpoints=(), interactions=(record,),
                                      execution_id="execution", archive=prepared)
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
        elif corruption in {"compressed_bytes", "missing"}:
            contents = await archive.transcript_repository.content_archive()
            identity, = await contents.content_ids(body.content)
            key = contents._key("archive_content", identity)
            async def damage(tx):
                if corruption == "missing":
                    assert await tx.delete_record(key)
                    return
                record = await tx.get_record(key)
                assert await tx.replace_record(replace(record, payload_bytes=b"not zlib",
                                                       storage_revision=record.storage_revision + 1),
                                               expected_storage_revision=record.storage_revision)
            await contents.store.mutate(damage)
        else:
            body = replace(body, content=replace(body.content, content_scope=archive_content_scope("other", "tenant")))
        invalid = replace(record, client_requests=(replace(request, body=body),))
        with pytest.raises(AIError) as raised:
            await archive.resolve_model_interaction(invalid, include_client_requests=True)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("exclusive_sqlite", "sqlite"))
async def test_client_body_database_content_survives_snapshot_and_follows_owner_cleanup(
    tmp_path: Path, backend: str,
) -> None:
    prepared = PreparedArchive()
    storage = Persistence(_storage(backend, tmp_path))
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    await storage.execution.executions.create_execution(_record(ExecutionStatus.STARTED, 1))
    requests = _requests(large=True)
    try:
        archive = storage.run_store.read_store(PersistenceDomain.TRACE)
        records = await archive.prepare_interactions(AgentRunRecord("run"), (_staged(requests),), lambda _: b"{}", prepared=prepared)
        await archive.sync_projection(AgentRunRecord("run"), events=(), checkpoints=(), interactions=records, execution_id="execution", archive=prepared)
        assert not tuple(iter_runtime_object_refs(_encode_persisted_domain(records[0]), default_domain=PersistenceDomain.TRACE))
        contents = await archive.transcript_repository.content_archive()
        root = records[0].client_requests[0].body.content
        identities = await contents.content_ids(root)
        await storage.purge_unreferenced_objects((), exclusive=Offline())
        assert await contents.read_contents(identities, codec="model-client-body") == (requests[0].body,)
    finally:
        await storage.close()

    objects = InMemoryObjectStore("snapshot")
    limits = SnapshotLimits(max_entries=100, max_bytes=1024 * 1024)
    reader = Persistence(_storage(backend, tmp_path))
    await reader.initialize(namespace="native-client", tenant_id="tenant", read_only=True)
    try:
        snapshot = await reader.export_snapshot(object_store=objects, limits=limits)
    finally:
        await reader.close()
    restored_root = tmp_path / "restored"
    await Persistence.restore_snapshot(snapshot, object_store=objects, root=restored_root, limits=limits)
    restored = Persistence(PersistenceStorage.from_root(restored_root))
    await restored.initialize(namespace="native-client", tenant_id="tenant")
    try:
        archive = restored.run_store.read_store(PersistenceDomain.TRACE)
        records = await archive.list_model_interactions(agent_run_id="run")
        assert await archive.resolve_model_interactions(records, include_client_requests=True) == [(None, None, None, requests, None)]
        contents = await archive.transcript_repository.content_archive()
        root = records[0].client_requests[0].body.content
        identities = await contents.content_ids(root)
        await archive.release_agent_run("run", execution_id="execution")
        await restored.purge_unreferenced_objects((), exclusive=Offline())
        with pytest.raises(AIError) as removed:
            await contents.read_contents(identities, codec="model-client-body")
        assert removed.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_volatile_archive_and_staging_keep_original_attempt_views() -> None:
    prepared = PreparedArchive()
    storage = Persistence(PersistenceStorage.in_memory())
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    await storage.execution.executions.create_execution(_record(ExecutionStatus.STARTED, 1))
    archive = storage.run_store.read_store(PersistenceDomain.TRACE)
    try:
        requests = _requests(large=True)
        records = await archive.prepare_interactions(AgentRunRecord("run"), (_staged(requests),), lambda _: b"{}", prepared=prepared)
        assert records[0].client_requests[0].body.raw_size == len(requests[0].body)
        await archive.sync_projection(AgentRunRecord("run"), events=(), checkpoints=(), interactions=records, execution_id="execution", archive=prepared)
        resolved = await archive.resolve_model_interactions(records, include_client_requests=True)
        assert resolved == [(None, None, None, requests, None)]
        assert await archive.prepare_relocated_interactions(records, resolved, prepared=prepared) == records
    finally:
        await storage.close()

    storage = Persistence(PersistenceStorage(PersistencePlan(execution=StorageRoute.transient())))
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    try:
        staged = _staged(requests)
        await storage.run_store.register_agent_run(AgentRunRecord("run"))
        storage.run_store.stage_model_interaction(staged)
        assert await storage.run_store.resolve_model_interaction(staged, include_client_requests=True) == (None, None, None, requests, None)
        from linktools.ai.history._transient import TransientExecutionHistoryStore
        source = storage.run_store.read_store(PersistenceDomain.TRACE)
        transient = TransientExecutionHistoryStore(source, storage.run_store)
        assert await transient.list_model_interactions(agent_run_id="run") == [staged]
        assert await transient.resolve_model_interactions((staged,)) == [(None, None, None, None, None)]
        assert await transient.resolve_model_interactions((staged,), include_client_requests=True) == [
            (None, None, None, requests, None),
        ]
        with pytest.raises(AIError) as raised:
            await source.resolve_model_interaction(staged, include_client_requests=True)
        assert raised.value.code is ErrorCode.STORAGE_DEPENDENCY_NOT_READY

    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("envelope_size", (16, 80 * 1024))
async def test_sdk_projection_and_native_bytes_remain_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, envelope_size: int,
) -> None:
    prepared = PreparedArchive()
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from linktools.ai.persistence._model_interaction import build_context_projection

    storage = Persistence(PersistenceStorage.from_root(tmp_path / "state", exclusive_writer=True))
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    await storage.execution.executions.create_execution(_record(ExecutionStatus.STARTED, 1))
    payloads: dict[str, bytes] = {}

    def intern(raw: bytes) -> tuple[str, int]:
        digest = hashlib.sha256(raw).hexdigest()
        payloads[digest] = raw
        return digest, len(raw)

    try:
        sdk_request = (ModelRequest(parts=[UserPromptPart("SDK input")]),)
        sdk_response = (ModelResponse(parts=[TextPart("SDK projection")]),)
        envelope = canonical_json_bytes({"streaming": True, "settings": {
            "note": "x" * envelope_size, "opaque": {"kind": "binary", "data": "not base64"},
        }})
        requests = _requests(large=True)
        staged = replace(
            _staged(requests), status="SUCCEEDED",
            request_context=build_context_projection((), sdk_request, intern),
            request_envelope_digest=intern(envelope)[0],
            response_context=build_context_projection((), sdk_response, intern),
        )
        archive = storage.run_store.read_store(PersistenceDomain.TRACE)
        records = await archive.prepare_interactions(AgentRunRecord("run"), (staged,), payloads.__getitem__, prepared=prepared)
        assert records[0].request_envelope.payload.kind == "inline"
        assert records[0].request_envelope.payload.encoding == "json"
        assert not records[0].request_envelope.payload.object_refs()
        await archive.sync_projection(AgentRunRecord("run"), events=(), checkpoints=(), interactions=records, execution_id="execution", archive=prepared)
        contents = await archive.transcript_repository.content_archive()
        original_capture = contents.capture_ranges

        async def reject_raw_capture(roots, *, codec, **kwargs):
            if codec == "model-client-body":
                raise AssertionError("unrequested native content read")
            return await original_capture(roots, codec=codec, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(contents, "capture_ranges", reject_raw_capture)
            assert await archive.list_model_interactions(agent_run_id="run") == list(records)
            assert await archive.resolve_model_interactions(records) == [
                (sdk_request, sdk_response, envelope, None, None),
            ]
            assert (await archive.resolve_model_interaction(records[0]))[3] is None
            metadata = await resolve_client_requests(
                records[0].client_requests, PersistenceDomain.TRACE, include_bodies=False,
            )
            relocated = await archive.prepare_relocated_interactions(
                records, [(sdk_request, sdk_response, envelope, metadata, None)],
            prepared=prepared)
            assert relocated[0].client_requests[0].body is None
            assert relocated[0].request_envelope == records[0].request_envelope
            assert relocated[0].client_requests[0].response.status_code == 429
            assert relocated[0].client_requests[0].record_http_bodies is True
            with pytest.raises(AssertionError, match="unrequested native"):
                await archive.resolve_model_interactions(records, include_client_requests=True)
            with pytest.raises(AIError):
                await archive.prepare_relocated_interactions(
                    records, await archive.resolve_model_interactions(records),
                prepared=prepared)
        assert await archive.resolve_model_interactions(records, include_client_requests=True) == [
            (sdk_request, sdk_response, envelope, requests, None),
        ]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_cancelled_client_preparation_leaves_no_owner_or_binary_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = PreparedArchive()
    import asyncio

    storage = Persistence(PersistenceStorage.from_root(tmp_path / "state", exclusive_writer=True))
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    await storage.execution.executions.create_execution(_record(ExecutionStatus.STARTED, 1))
    try:
        archive = storage.run_store.read_store(PersistenceDomain.TRACE)
        contents = await archive.transcript_repository.content_archive(create=True)
        append = contents.prepare_append

        async def prepare_then_cancel(*args, **kwargs):
            await append(*args, **kwargs)
            raise asyncio.CancelledError

        with monkeypatch.context() as patch:
            patch.setattr(contents, "prepare_append", prepare_then_cancel)
            with pytest.raises(asyncio.CancelledError):
                await archive.prepare_interactions(AgentRunRecord("run"), (_staged(_requests(large=True)),), lambda _: b"{}", prepared=prepared)
        assert await archive.list_model_interactions(agent_run_id="run") == []
        rows = await contents.store.read(lambda tx: tx.scan_records())
        assert not any(record.kind == "archive_content" for record in rows)
        retry = PreparedArchive()
        requests = _requests(large=True)
        records = await archive.prepare_interactions(AgentRunRecord("run"), (_staged(requests),), lambda _: b"{}", prepared=retry)
        await archive.sync_projection(AgentRunRecord("run"), events=(), checkpoints=(), interactions=records,
                                      execution_id="execution", archive=retry)
        assert await archive.resolve_model_interactions(records, include_client_requests=True) == [(None, None, None, requests, None)]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_interrupted_record_preserves_observed_incomplete_response() -> None:
    prepared = PreparedArchive()
    storage = Persistence(PersistenceStorage.in_memory())
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    await storage.execution.executions.create_execution(_record(ExecutionStatus.STARTED, 1))
    try:
        requests = (_requests()[1],)
        staged = replace(_staged(requests), status="INTERRUPTED", duration_ns=None, finished_at=None)
        archive = storage.run_store.read_store(PersistenceDomain.TRACE)
        records = await archive.prepare_interactions(AgentRunRecord("run"), (staged,), lambda _: b"{}", prepared=prepared)
        await archive.sync_projection(AgentRunRecord("run"), events=(), checkpoints=(), interactions=records, execution_id="execution", archive=prepared)
        loaded = await archive.list_model_interactions(agent_run_id="run")
        assert loaded[0].finished_at is None and loaded[0].usage is None
        assert await archive.resolve_model_interactions(loaded, include_client_requests=True) == [
            (None, None, None, requests, None),
        ]
        assert loaded[0].client_requests[0].response.complete is False
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_recovery_preserves_existing_trace_bodies_without_reading_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = PreparedArchive()
    from linktools.ai.persistence._step_contracts import AgentRunCheckpoint

    storage = Persistence(PersistenceStorage.from_root(tmp_path / "state", exclusive_writer=True))
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    await storage.execution.executions.create_execution(_record(ExecutionStatus.STARTED, 1))
    try:
        run = AgentRunRecord("run")
        trace = storage.run_store.read_store(PersistenceDomain.TRACE)
        native = native_checkpoint().store
        storage.run_store.bind_step_store(native)
        await seed_native_snapshot(native, run_id="run", messages=[])
        snapshot = await native.latest_snapshot(run_id="run")
        requests = _requests(large=True)
        traced = await trace.prepare_interactions(run, (_staged(requests),), lambda _: b"{}", prepared=prepared)
        await trace.sync_projection(run, events=(), checkpoints=(), interactions=traced, execution_id="execution", archive=prepared)
        await trace.materialize_checkpoint(
            run, AgentRunCheckpoint("run", 1, [], transcript_message_count_before=0),
            execution_id="execution",
        )
        contents = await trace.transcript_repository.content_archive()
        original_capture = contents.capture_ranges

        async def reject_raw_capture(roots, *, codec, **kwargs):
            if codec == "model-client-body":
                raise AssertionError("recovery must not read native bodies")
            return await original_capture(roots, codec=codec, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(contents, "capture_ranges", reject_raw_capture)
            await storage.run_store.materialize_from_recovery(
                target=PersistenceDomain.TRACE, agent_run_id="run", execution_id="execution",
            )
        assert await native.latest_snapshot(run_id="run") == snapshot
        assert await storage.run_store.read_store(PersistenceDomain.CHECKPOINT).list_model_interactions(
            agent_run_id="run",
        ) == []
        loaded = await trace.list_model_interactions(agent_run_id="run")
        assert loaded[0].client_requests == traced[0].client_requests
        assert await trace.resolve_model_interactions(loaded, include_client_requests=True) == [
            (None, None, None, requests, None),
        ]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_client_body_capture_defers_binary_decode_until_owner_scope_closes(tmp_path: Path, monkeypatch) -> None:
    from linktools.ai.persistence.state import active_state_scope

    storage = Persistence(PersistenceStorage.sqlite(tmp_path / "body-read.db"))
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    await storage.execution.executions.create_execution(_record(ExecutionStatus.STARTED, 1))
    try:
        archive = storage.run_store.read_store(PersistenceDomain.TRACE)
        prepared = PreparedArchive()
        requests = _requests(large=True)
        records = await archive.prepare_interactions(AgentRunRecord("run"), (_staged(requests),), lambda _: b"{}", prepared=prepared)
        await archive.sync_projection(AgentRunRecord("run"), events=(), checkpoints=(), interactions=records,
                                      execution_id="execution", archive=prepared)
        contents = await archive.transcript_repository.content_archive()
        read = contents._read_content
        decoded = []

        async def outside_scope(content):
            assert active_state_scope() is None
            decoded.append(content.raw_digest)
            return await read(content)

        monkeypatch.setattr(contents, "_read_content", outside_scope)
        async def capture():
            assert active_state_scope() is not None
            return await archive.capture_model_interactions(records, include_client_requests=True, include_content=False)

        load = await archive.transcript_repository.read_content(capture)
        assert not decoded
        assert await load() == [(None, None, None, requests, None)]
        assert set(decoded) == {hashlib.sha256(body).hexdigest() for request in requests
                               for body in (request.body, None if request.response is None else request.response.body)
                               if body is not None}
    finally:
        await storage.close()
