#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authorized original client bodies load independently of SDK projections."""

from linktools.ai.persistence import Persistence
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, ModelResponse, TextPart, UserPromptPart

from linktools.ai.core import (
    ExecutionStatus, HmacCursorSigner, Page, Principal, TenantAuthorizationPolicy, agent_conversation_id, agent_run_id,
)
from linktools.ai.persistence._memory import InMemoryStateStore
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.runtime import RuntimeExecutions
from linktools.ai.persistence import PersistenceStorage, PersistencePlan
from linktools.ai.storage import StorageRoute
from linktools.ai.history._service import DefaultExecutionHistoryService
from linktools.ai.persistence._model_interaction import (
    StagedContextInline,
    StagedContextProjection,
    build_context_projection,
)
from linktools.ai.history._transient import TransientExecutionHistoryStore
from linktools.ai.execution import UsageReadCutoff
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence._archive import ArchiveRoot, ContentArchive, PreparedArchive
from linktools.ai.persistence.state import active_state_scope
from linktools.ai.persistence._archive_closure import iter_archive_roots, iter_native_message_roots
from linktools.ai.persistence.codec import _encode_persisted_domain, iter_runtime_object_refs
from linktools.ai.persistence.contracts import ContextProjection, RuntimePayloadRef, TranscriptSpanRef
from linktools.ai.persistence._model_interaction_store import ModelInteractionStateStepArchive
from linktools.ai.persistence._step_contracts import AgentRunRecord
from linktools.ai.storage import StoredPayload

from .test_history_projection_conformance import _message_content_records, _reader, _record
from .test_model_client_interaction_storage import _requests, _staged, _storage
from .test_runtime_history_public import _Executions


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "exclusive_sqlite", "sqlite"))
async def test_public_client_reads_are_independent_authorized_and_cursor_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    storage = Persistence(_storage(backend, tmp_path))
    await storage.initialize(namespace="history", tenant_id="tenant")
    try:
        await storage.execution.executions.create_execution(_record(ExecutionStatus.STARTED, 1))
        run = AgentRunRecord(
            agent_run_id=agent_run_id(
                namespace="history", tenant_id="tenant", execution_id="execution", agent_run_seq=1,
            ),
            agent_conversation_id=agent_conversation_id(
                namespace="history", tenant_id="tenant", execution_id="execution",
            ),
            metadata={"agent_run_seq": "1"},
        )
        payloads = {}

        def intern(raw):
            digest = hashlib.sha256(raw).hexdigest()
            payloads[digest] = raw
            return digest, len(raw)

        sdk_request = (ModelRequest(parts=[UserPromptPart(["SDK input " * 6000, BinaryContent(b"SDK image", media_type="image/png")])]),)
        sdk_response = (ModelResponse(parts=[TextPart("SDK output " * 6000)]),)
        original = _requests(large=True)
        staged = replace(
            _staged(original), agent_run_id=run.agent_run_id,
            request_context=build_context_projection((), sdk_request, intern),
            request_envelope_digest=intern(b'{"streaming":true}')[0],
            response_context=build_context_projection((), sdk_response, intern),
        )
        entries = (staged, replace(staged, model_request_seq=2, client_requests=None),
                   replace(staged, model_request_seq=3, client_requests=()),
                   replace(staged, model_request_seq=4, status="INTERRUPTED", response_context=None,
                           duration_ns=None, finished_at=None, client_requests=(original[1],)))
        archive = storage.run_store.read_store(PersistenceDomain.TRACE)
        bundle = PreparedArchive()
        records = await archive.prepare_interactions(run, entries, payloads.__getitem__, prepared=bundle)
        await archive.sync_projection(run, events=(), checkpoints=(), interactions=records, execution_id="execution", archive=bundle)
        service = DefaultExecutionHistoryService(
            storage.execution.executions, TenantAuthorizationPolicy("tenant"), _reader(storage),
            HmacCursorSigner("client-read", b"client-read-key"),
        )
        history = History(service, tenant_id="tenant")
        principal = Principal("reader", "tenant")
        objects = storage.object_store(PersistenceDomain.TRACE)
        opened = []



        object_stores = {id(objects): objects}
        archive_objects = storage.archive_object_store(PersistenceDomain.TRACE)
        object_stores[id(archive_objects)] = archive_objects
        for object_store in object_stores.values():
            async def track_open(key, original_open=object_store.open):
                opened.append(key)
                async for chunk in original_open(key):
                    yield chunk
            monkeypatch.setattr(object_store, "open", track_open)

        async def object_keys(value):
            encoded = _encode_persisted_domain(value)
            keys = {ref.key for _, ref in iter_runtime_object_refs(
                encoded, default_domain=PersistenceDomain.TRACE,
            )}
            roots = (*iter_archive_roots(encoded), *iter_native_message_roots(encoded))
            if roots:
                contents = await archive.transcript_repository.content_archive()
                values = await _message_content_records(contents, roots)
                keys.update(ref.key for content in values for ref in content.object_refs())
            return keys

        sdk_keys = await object_keys(replace(records[0], client_requests=None))
        raw_roots = {body.content for request in records[0].client_requests
                     for body in (request.body, None if request.response is None else request.response.body)
                     if body is not None}
        assert raw_roots and all(isinstance(root, ArchiveRoot) for root in raw_roots)
        assert not await object_keys(records[0]) - sdk_keys
        if backend != "memory":
            assert sdk_keys
        captured, hydrated, decoded = [], [], []
        original_capture = ContentArchive.capture_ranges
        async def track_client_bodies(owner, roots, *, codec, **kwargs):
            load = await original_capture(owner, roots, codec=codec, **kwargs)
            if codec != "model-client-body":
                return load
            captured.extend(roots)
            async def hydrate():
                assert active_state_scope() is None
                values = await load()
                hydrated.extend(roots)
                decoded.extend(raw for group in values for raw in group)
                return values
            return hydrate
        monkeypatch.setattr(ContentArchive, "capture_ranges", track_client_bodies)

        metadata = await history.model_interactions("execution", principal=principal, limit=1)
        item, = metadata.items
        assert not item.content_included and not item.client_requests_included
        assert item.client_requests is None and item.request == {} and item.response is None
        assert opened == [] and captured == hydrated == decoded == []

        sdk = await history.model_interactions("execution", principal=principal, include_content=True, limit=1)
        assert sdk.items[0].request and sdk.items[0].response
        assert sdk.items[0].client_requests is None and not sdk.items[0].client_requests_included
        assert captured == hydrated == decoded == []
        if backend != "memory":
            assert sdk_keys.issubset(opened)

        opened.clear()
        raw = await history.model_interactions("execution", principal=principal, include_client_requests=True, limit=1)
        item, = raw.items
        assert item.client_requests_included and item.client_requests == original
        assert not item.content_included and item.request == {} and item.response is None
        with pytest.raises(ValueError, match="omitted client"):
            replace(item, client_requests_included=False)
        with pytest.raises(TypeError, match="client request content"):
            replace(item, client_requests=list(original))
        assert opened == []
        assert set(captured) == set(hydrated) == raw_roots
        assert set(decoded) == {body for request in original
            for body in (request.body, None if request.response is None else request.response.body) if body is not None}

        opened.clear()
        captured.clear()
        hydrated.clear()
        decoded.clear()
        for cursor, flags in ((metadata.next_cursor, {"include_client_requests": True}),
                              (raw.next_cursor, {}), (raw.next_cursor, {"include_client_requests": True, "include_content": True})):
            with pytest.raises(AIError) as caught:
                await history.model_interactions("execution", principal=principal, cursor=cursor, **flags)
            assert caught.value.code is ErrorCode.CURSOR_INVALID
            assert opened == [] and captured == hydrated == decoded == []
        with pytest.raises(AIError) as caught:
            await history.model_interactions("execution", principal=Principal("other", "other"), include_content=True, include_client_requests=True)
        assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert opened == [] and captured == hydrated == decoded == []

        class DeniedPolicy:
            async def authorize(self, principal, action, resource):
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)

        denied = DefaultExecutionHistoryService(storage.execution.executions, DeniedPolicy(), _reader(storage))
        with pytest.raises(AIError) as caught:
            await denied.model_interactions("execution", principal=principal, include_content=True, include_client_requests=True)
        assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert opened == [] and captured == hydrated == decoded == []
        with pytest.raises(AIError) as caught:
            await history.model_interactions("execution", principal=principal, include_client_requests=1)
        assert caught.value.code is ErrorCode.REQUEST_FIELD_INVALID
        assert opened == [] and captured == hydrated == decoded == []

        second = await history.model_interactions(
            "execution", principal=principal, cursor=raw.next_cursor, include_client_requests=True, limit=1,
        )
        assert second.items[0].client_requests_included and second.items[0].client_requests is None
        third = await history.model_interactions(
            "execution", principal=principal, cursor=second.next_cursor, include_client_requests=True, limit=1,
        )
        assert third.items[0].client_requests == () and third.items[0].model_request_seq == 3
        assert captured == hydrated == decoded == []
        fourth = await history.model_interactions(
            "execution", principal=principal, cursor=third.next_cursor, include_client_requests=True, limit=1,
        )
        assert fourth.next_cursor is None and fourth.items[0].status == "INTERRUPTED"
        assert fourth.items[0].client_requests[0].response.complete is False
        fourth_roots = {body.content for request in records[3].client_requests
                        for body in (request.body, request.response.body) if body is not None}
        assert set(captured) == set(hydrated) == fourth_roots
        captured.clear()
        hydrated.clear()
        decoded.clear()
        both = await history.model_interactions(
            "execution", principal=principal, include_content=True, include_client_requests=True,
            cutoffs=(UsageReadCutoff("execution", 1, 1),),
        )
        assert len(both.items) == 1
        assert both.items[0].request == sdk.items[0].request
        assert both.items[0].response == sdk.items[0].response
        assert both.items[0].client_requests == original
        assert set(captured) == set(hydrated) == raw_roots
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_existing_custom_readers_and_services_keep_default_call_shape() -> None:
    calls = []

    class ExistingReader:
        async def model_interactions(self, execution_id, *, tenant_id, cursor, limit, cutoffs=None, include_content=True):
            calls.append((execution_id, tenant_id, include_content))
            return Page(())

    class ExistingService:
        async def model_interactions(self, execution_id, *, principal, cursor=None, limit=100, cutoffs=None, include_content=False):
            calls.append((execution_id, principal.tenant_id, include_content))
            return Page(())

    async def unused_get(*args):
        raise AssertionError("content reads do not construct an execution")

    principal = Principal("reader", "tenant")
    readers = (
        History(DefaultExecutionHistoryService(
            _Executions(), TenantAuthorizationPolicy("tenant"), ExistingReader(),
        ), tenant_id="tenant"),
        History(ExistingService(), tenant_id="tenant"),
        RuntimeExecutions(ExistingService(), unused_get, bind_execution=None),
    )
    for reader in readers:
        assert (await reader.model_interactions("execution", principal=principal)).items == ()
        assert (await reader.model_interactions(
            "execution", principal=principal, include_content=True, include_client_requests=False,
        )).items == ()
    assert calls == [("execution", "tenant", False), ("execution", "tenant", True)] * 3


@pytest.mark.asyncio
async def test_raw_only_volatile_reads_do_not_require_sdk_checkpoint_or_payload() -> None:
    original = _requests()
    archive = ModelInteractionStateStepArchive(
        InMemoryStateStore(), object_store=None, namespace="test", tenant_id="tenant",
        runtime_domain=PersistenceDomain.TRACE,
    )
    await archive.state_store.initialize()
    await archive.initialize()
    try:
        bundle = PreparedArchive()
        record, = await archive.prepare_interactions(AgentRunRecord("run"), (_staged(original),), lambda _: b"{}", prepared=bundle)
        content_archive = await archive.transcript_repository.content_archive()
        await content_archive.store.mutate(lambda transaction: content_archive.publish(transaction, bundle))
        pending_sdk = replace(record, request_context=ContextProjection((
            TranscriptSpanRef(PersistenceDomain.TRACE, "run", 0, 1),
        )), request_envelope=RuntimePayloadRef(
            StoredPayload.inline_bytes(b"{}"), PersistenceDomain.TRACE,
        ))
        assert await archive.resolve_model_interactions(
            (pending_sdk,), include_client_requests=True, include_content=False,
        ) == [(None, None, None, original, None)]
        assert await archive.resolve_model_interaction(
            pending_sdk, include_client_requests=True, include_content=False,
        ) == (None, None, None, original, None)
        with pytest.raises(AIError) as caught:
            await archive.resolve_model_interaction(pending_sdk)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await archive.close()
        await archive.state_store.close()

    storage = Persistence(PersistenceStorage(PersistencePlan(execution=StorageRoute.transient())))
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    try:
        staged = replace(
            _staged(original), request_context=StagedContextProjection((StagedContextInline("a" * 64, 2),)),
            request_envelope_digest="b" * 64,
        )
        await storage.run_store.register_agent_run(AgentRunRecord("run"))
        storage.run_store.stage_model_interaction(staged)
        transient = TransientExecutionHistoryStore(
            storage.run_store.read_store(PersistenceDomain.TRACE), storage.run_store,
        )
        assert await transient.resolve_model_interactions(
            (staged,), include_client_requests=True, include_content=False,
        ) == [(None, None, None, original, None)]
        assert await storage.run_store.resolve_model_interaction(
            staged, include_client_requests=True, include_content=False,
        ) == (None, None, None, original, None)
        with pytest.raises(AIError) as caught:
            await transient.resolve_model_interactions((staged,))
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reader", ("page", "identities", "staged_fallback"))
async def test_selected_interaction_content_survives_concurrent_native_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str,
) -> None:
    import asyncio

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from linktools.ai.migrate import provision_database
    from linktools.ai.persistence import SqlStepStore
    from linktools.ai.persistence.state import active_state_scope

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'interaction-read.db'}")
    async with engine.begin() as connection:
        await connection.execute(text("PRAGMA journal_mode=WAL"))
    await provision_database(engine)
    state = Persistence(PersistenceStorage.sql(engine))
    await state.initialize(namespace="history", tenant_id="tenant")
    captured_owner, collected = asyncio.Event(), asyncio.Event()
    task = None
    try:
        await state.execution.executions.create_execution(_record(ExecutionStatus.STARTED, 1))
        run = AgentRunRecord(
            agent_run_id=agent_run_id(namespace="history", tenant_id="tenant", execution_id="execution", agent_run_seq=1),
            agent_conversation_id=agent_conversation_id(namespace="history", tenant_id="tenant", execution_id="execution"),
            metadata={"agent_run_seq": "1"},
        )
        payloads = {}

        def intern(raw):
            digest = hashlib.sha256(raw).hexdigest()
            payloads[digest] = raw
            return digest, len(raw)

        request = (ModelRequest(parts=[UserPromptPart("selected input")]),)
        response = (ModelResponse(parts=[TextPart("selected output")]),)
        staged = replace(
            _staged(None), agent_run_id=run.agent_run_id,
            request_context=build_context_projection((), request, intern),
            request_envelope_digest=intern(b"{}")[0],
            response_context=build_context_projection((), response, intern),
        )
        lookahead = replace(staged, model_request_seq=2,
            response_context=build_context_projection((), (ModelResponse(parts=[TextPart("unselected output")]),), intern))
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        prepared = PreparedArchive()
        records = await archive.prepare_interactions(run, (staged, lookahead), payloads.__getitem__, prepared=prepared)
        await archive.sync_projection(run, events=(), checkpoints=(), interactions=records,
                                      execution_id="execution", archive=prepared)
        service = DefaultExecutionHistoryService(
            state.execution.executions, TenantAuthorizationPolicy("tenant"), _reader(state),
            HmacCursorSigner("concurrent-read", b"concurrent-read-key"),
        )
        original_capture = SqlStepStore.capture_message_roots
        hydrated = []
        paused = False

        async def capture_native(self, roots, **kwargs):
            nonlocal paused
            if roots and not paused:
                paused = True
                assert active_state_scope() is not None
                captured_owner.set()
                await collected.wait()
            load = await original_capture(self, roots, **kwargs)

            async def hydrate():
                assert active_state_scope() is None
                groups = await load()
                hydrated.extend(body for group in groups for body in group)
                return groups

            return hydrate

        monkeypatch.setattr(SqlStepStore, "capture_message_roots", capture_native)

        async def read():
            if reader == "staged_fallback":
                values = await state.run_store.resolve_model_interactions((staged,))
                assert values[0][0] == request and values[0][1] == response
                return
            principal = Principal("reader", "tenant")
            if reader == "identities":
                items = await service.read_model_interactions(
                    "execution", principal=principal, identities=((1, 1),), include_content=True,
                )
            else:
                page = await service.model_interactions("execution", principal=principal, include_content=True, limit=1)
                assert page.next_cursor is not None
                items = page.items
            assert len(items) == 1 and items[0].model_request_seq == 1
            assert items[0].request and items[0].response

        task = asyncio.create_task(read())
        await asyncio.wait_for(captured_owner.wait(), timeout=5)
        async with engine.begin() as connection:
            for table in ("ai_state_facts", "ai_state_records", "ai_step_sequences", "ai_step_messages"):
                await connection.execute(text(f"DELETE FROM {table}"))
        collected.set()
        await asyncio.wait_for(task, timeout=5)
        assert len(hydrated) == 2
        assert not any(b"unselected" in body for body in hydrated)
        assert active_state_scope() is None
    finally:
        collected.set()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await state.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_exact_client_reads_reopen_cold_archive_without_sdk_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PersistenceStorage.sqlite(tmp_path / "cold-client.db")
    original = _requests(large=True)
    storage = Persistence(config)
    await storage.initialize(namespace="history", tenant_id="tenant")
    try:
        await storage.execution.executions.create_execution(_record(ExecutionStatus.STARTED, 1))
        run = AgentRunRecord(
            agent_run_id=agent_run_id(
                namespace="history", tenant_id="tenant", execution_id="execution", agent_run_seq=1,
            ),
            agent_conversation_id=agent_conversation_id(
                namespace="history", tenant_id="tenant", execution_id="execution",
            ),
            metadata={"agent_run_seq": "1"},
        )
        archive = storage.run_store.read_store(PersistenceDomain.TRACE)
        bundle = PreparedArchive()
        records = await archive.prepare_interactions(
            run, (replace(_staged(original), agent_run_id=run.agent_run_id),),
            lambda _: b"{}", prepared=bundle,
        )
        await archive.sync_projection(
            run, events=(), checkpoints=(), interactions=records,
            execution_id="execution", archive=bundle,
        )
    finally:
        await storage.close()

    storage = Persistence(config)
    await storage.initialize(namespace="history", tenant_id="tenant")
    try:
        history = History(DefaultExecutionHistoryService(
            storage.execution.executions, TenantAuthorizationPolicy("tenant"), _reader(storage),
        ), tenant_id="tenant")
        principal = Principal("reader", "tenant")
        group = storage.execution.executions.state_store.storage_group
        assert group.opened_archive is None

        read_content = ContentArchive._read_content
        control_codecs = {"execution-binding", "runtime-policy"}
        metadata_reads = []

        async def read_control_only(owner, content):
            assert content.codec in control_codecs, "metadata read hydrated message or client content"
            metadata_reads.append(content.codec)
            return await read_content(owner, content)

        with monkeypatch.context() as metadata_patch:
            metadata_patch.setattr(ContentArchive, "_read_content", read_control_only)
            metadata, = await history.read_model_interactions(
                "execution", principal=principal, identities=((1, 1),),
            )
        assert not metadata.content_included and not metadata.client_requests_included
        assert metadata.client_requests is None
        assert "runtime-policy" in metadata_reads

        decoded = []

        async def decode_outside_scope(owner, content):
            if content.codec not in control_codecs:
                assert active_state_scope() is None
                decoded.append(content.raw_digest)
            return await read_content(owner, content)

        monkeypatch.setattr(ContentArchive, "_read_content", decode_outside_scope)
        item, = await history.read_model_interactions(
            "execution", principal=principal, identities=((1, 1),),
            include_client_requests=True, include_content=False,
        )
        assert item.client_requests_included and item.client_requests == original
        assert not item.content_included and item.request == {} and item.response is None
        assert set(decoded) == {
            hashlib.sha256(body).hexdigest() for request in original
            for body in (request.body, None if request.response is None else request.response.body)
            if body is not None
        }
    finally:
        await storage.close()
