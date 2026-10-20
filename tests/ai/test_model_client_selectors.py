#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact model-request selectors retain identity, authorization, and lazy bodies."""

from linktools.ai.runtime.state import RuntimeState
import hashlib
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from time import time
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from linktools.ai.core import (
    ExecutionLineageKind, ExecutionStatus, HmacCursorSigner, Page, Principal,
    TenantAuthorizationPolicy, agent_conversation_id, agent_run_id, canonical_sha256,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelClientRequest, ModelClientResponse
from linktools.ai.runtime import Execution, RuntimeExecutions, RuntimeHistory
from linktools.ai.core import _signed_cursor as _cursor
from linktools.ai.runtime._execution import DefaultExecutionService
from linktools.ai.runtime._history_service import DefaultExecutionHistoryService
from linktools.ai.runtime._model_interaction import build_context_projection
from linktools.ai.runtime.service_api import UsageReadCutoff
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._archive import PreparedArchive
from linktools.ai.runtime.state._archive_closure import iter_archive_roots
from linktools.ai.runtime.state._codec import _encode_persisted_domain, iter_runtime_object_refs
from linktools.ai.runtime.state._step_contracts import AgentRunRecord

from .test_history_projection_conformance import _reader, _record
from .test_model_client_interaction_storage import _staged, _storage
from .test_runtime_history_public import _Executions


def _execution_service(history):
    return DefaultExecutionService(
        None, None, TenantAuthorizationPolicy("tenant"), sessions=None, catalog=None,
        compiler=None, runtime_bridge=SimpleNamespace(port=None), live_broker=None,
        history_reader=None, history_service=history,
    )


async def _object_keys(value, archive):
    encoded = _encode_persisted_domain(value)
    keys = {ref.key for _, ref in iter_runtime_object_refs(
        encoded, default_domain=RuntimeDomain.TRACE,
    )}
    roots = tuple(iter_archive_roots(encoded))
    if roots:
        contents = await archive.transcript_repository.content_archive()
        ids = await contents.content_ids_many(roots)
        records = await contents.content_records(tuple(identity for group in ids for identity in group))
        keys.update(content.payload.ref.key for content in records if content.payload.ref is not None)
    return keys


@asynccontextmanager
async def _seed(backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    storage = RuntimeState(_storage(backend, tmp_path))
    await storage.initialize(namespace="history", tenant_id="tenant")
    try:
        root = _record(ExecutionStatus.STARTED, 2)
        child = replace(
            root, execution_id="child", parent_execution_id="execution", parent_invocation_id="child-call",
            lineage_kind=ExecutionLineageKind.SUBAGENT, agent_run_seq=1,
            created_at=root.created_at + timedelta(seconds=1),
        )
        for record in (root, child):
            await storage.execution.executions.create_with_history_head(record)
        archive = storage.run_store.read_store(RuntimeDomain.TRACE)
        records, bodies, sdk_keys, raw_keys = {}, {}, {}, {}
        payloads = {}

        def intern(raw):
            digest = hashlib.sha256(raw).hexdigest()
            payloads[digest] = raw
            return digest, len(raw)

        for execution_id, run_sequence in (("execution", 1), ("execution", 2), ("child", 1)):
            run = AgentRunRecord(
                agent_run_id=agent_run_id(
                    namespace="history", tenant_id="tenant", execution_id=execution_id,
                    agent_run_seq=run_sequence,
                ),
                agent_conversation_id=agent_conversation_id(
                    namespace="history", tenant_id="tenant", execution_id=execution_id,
                ),
                metadata={"agent_run_seq": str(run_sequence)},
            )
            staged = []
            for sequence in (1, 2, 3):
                identity = (execution_id, run_sequence, sequence)
                marker = ":".join(map(str, identity))
                bodies[identity] = (ModelClientRequest(
                    "openai.chat.completions",
                    ('{"identity":"' + marker + '","input":"' + "native input " * 6000 + '"}').encode(),
                    response=ModelClientResponse(
                        ('{"identity":"' + marker + '","output":"' + "native output " * 6000 + '"}').encode(),
                        200, "json", None, True,
                    ),
                ),)
                staged.append(replace(
                    _staged(bodies[identity]), agent_run_id=run.agent_run_id, model_request_seq=sequence,
                    request_context=build_context_projection(
                        (), (ModelRequest(parts=[UserPromptPart(marker + " SDK input " * 6000)]),), intern,
                    ),
                    request_envelope_digest=intern(('{"identity":"' + marker + '"}').encode())[0],
                    response_context=build_context_projection(
                        (), (ModelResponse(parts=[TextPart(marker + " SDK output " * 6000)]),), intern,
                    ),
                ))
            bundle = PreparedArchive()
            prepared = await archive.prepare_interactions(run, tuple(staged), payloads.__getitem__, prepared=bundle)
            await archive.sync_projection(run, events=(), checkpoints=(), interactions=prepared, execution_id=execution_id, archive=bundle)
            for record in prepared:
                identity = (execution_id, run_sequence, record.model_request_seq)
                records[identity] = record
                sdk_keys[identity] = await _object_keys(replace(record, client_requests=None), archive)
                raw_keys[identity] = await _object_keys(record, archive) - sdk_keys[identity]
        reader = _reader(storage)
        signer = HmacCursorSigner("client-selectors", b"client-selectors-key")
        service = DefaultExecutionHistoryService(
            storage.execution.executions, TenantAuthorizationPolicy("tenant"), reader, signer,
        )
        objects = storage.object_store(RuntimeDomain.TRACE)
        original_resolve = archive.resolve_model_interactions
        original_list_children = storage.execution.executions.list_children
        opened, resolved, children_reads = [], [], []



        async def track_resolve(interactions, **kwargs):
            resolved.extend((value.agent_run_id, value.model_request_seq) for value in interactions)
            return await original_resolve(interactions, **kwargs)

        async def track_children(execution_id, *, tenant_id):
            children_reads.append(execution_id)
            return await original_list_children(execution_id, tenant_id=tenant_id)

        monkeypatch.setattr(storage.execution.executions, "list_children", track_children)
        object_stores = {id(objects): objects}
        archive_objects = storage.archive_object_store(RuntimeDomain.TRACE)
        object_stores[id(archive_objects)] = archive_objects
        for object_store in object_stores.values():
            async def track_open(key, original_open=object_store.open):
                opened.append(key)
                async for chunk in original_open(key):
                    yield chunk
            monkeypatch.setattr(object_store, "open", track_open)
        monkeypatch.setattr(archive, "resolve_model_interactions", track_resolve)
        yield SimpleNamespace(
            storage=storage, reader=reader, service=service, signer=signer,
            history=RuntimeHistory(service, tenant_id="tenant"), principal=Principal("reader", "tenant"),
            records=records, bodies=bodies, sdk_keys=sdk_keys, raw_keys=raw_keys,
            opened=opened, resolved=resolved, children_reads=children_reads,
        )
    finally:
        await storage.close()


def _identity(item):
    return item.execution_id, item.agent_run_seq, item.model_request_seq


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_model_client_selectors_resolve_only_exact_identity_bodies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    async with _seed(backend, tmp_path, monkeypatch) as data:
        identity = ("execution", 1, 2)
        options = {"principal": data.principal, "agent_run_seq": 1, "model_request_seq": 2}
        metadata = await data.history.model_interactions("execution", **options)
        assert [_identity(item) for item in metadata.items] == [identity]
        assert metadata.next_cursor is None
        assert metadata.items[0].client_requests is None and metadata.items[0].request == {}
        assert data.opened == [] and data.resolved == []

        raw = await data.history.model_interactions("execution", include_client_requests=True, **options)
        assert [_identity(item) for item in raw.items] == [identity]
        assert raw.items[0].client_requests == data.bodies[identity]
        assert raw.items[0].client_requests_included and not raw.items[0].content_included
        assert raw.items[0].request == {} and raw.items[0].response is None
        assert data.resolved == [(data.records[identity].agent_run_id, 2)]
        assert not set(data.opened) - data.raw_keys[identity]
        if backend == "sqlite":
            assert data.raw_keys[identity] and set(data.opened) == data.raw_keys[identity]

        data.opened.clear()
        data.resolved.clear()
        sdk = await data.history.model_interactions("execution", include_content=True, **options)
        assert sdk.items[0].request and sdk.items[0].response
        assert sdk.items[0].client_requests is None and not sdk.items[0].client_requests_included
        assert data.resolved == [(data.records[identity].agent_run_id, 2)]
        assert not set(data.opened) - data.sdk_keys[identity]
        if backend == "sqlite":
            assert data.sdk_keys[identity] and set(data.opened) == data.sdk_keys[identity]

        data.opened.clear()
        data.resolved.clear()
        child = await data.history.model_interactions(
            "child", principal=data.principal, agent_run_seq=1, model_request_seq=2, include_client_requests=True,
        )
        child_identity = ("child", 1, 2)
        assert [_identity(item) for item in child.items] == [child_identity]
        assert child.items[0].client_requests == data.bodies[child_identity]
        assert data.resolved == [(data.records[child_identity].agent_run_id, 2)]
        assert not set(data.opened) - data.raw_keys[child_identity]
        assert data.children_reads == []

        data.opened.clear()
        data.resolved.clear()
        for selectors in ({"agent_run_seq": 99}, {"model_request_seq": 99},
                          {"agent_run_seq": 1, "model_request_seq": 2,
                           "cutoffs": (UsageReadCutoff("execution", 1, 1),)}):
            empty = await data.history.model_interactions(
                "execution", principal=data.principal, include_client_requests=True, **selectors,
            )
            assert empty.items == () and empty.next_cursor is None
        assert data.opened == [] and data.resolved == []


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_model_client_selectors_paging_and_cursor_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    async with _seed(backend, tmp_path, monkeypatch) as data:
        for public in (True, False):
            read = data.history.model_interactions if public else data.reader.model_interactions
            options = {"principal": data.principal} if public else {"tenant_id": "tenant", "include_content": False}
            pages = []
            cursor = None
            while True:
                page = await read("execution", cursor=cursor, limit=1, agent_run_seq=1, **options)
                pages.extend(_identity(item) for item in page.items)
                cursor = page.next_cursor
                if cursor is None:
                    break
            assert pages == [("execution", 1, sequence) for sequence in (1, 2, 3)]

            first = await read("execution", cursor=None, limit=1, model_request_seq=2, **options)
            assert first.next_cursor is not None
            second = await read("execution", cursor=first.next_cursor, limit=1, model_request_seq=2, **options)
            third = await read("execution", cursor=second.next_cursor, limit=1, model_request_seq=2, **options)
            assert [_identity(page.items[0]) for page in (first, second, third)] == [
                ("execution", 1, 2), ("execution", 2, 2), ("child", 1, 2),
            ]
            assert third.next_cursor is None
            for selectors in ({}, {"model_request_seq": 1}, {"agent_run_seq": 1, "model_request_seq": 2}):
                with pytest.raises(AIError) as caught:
                    await read("execution", cursor=first.next_cursor, limit=1, **selectors, **options)
                assert caught.value.code is ErrorCode.CURSOR_INVALID

            unfiltered = await read("execution", cursor=None, limit=1, **options)
            legacy_cursor = unfiltered.next_cursor
            if public:
                envelope = data.signer.decode(legacy_cursor)
                assert envelope.filter_digest == canonical_sha256({
                    "execution_id": "execution", "query_kind": "model_interactions",
                    "include_content": False, "filters": None,
                })
                legacy_cursor = envelope.position
            envelope = HmacCursorSigner("history", b"history-key").decode(legacy_cursor)
            assert envelope.filter_digest == canonical_sha256({
                "execution_id": "execution", "projection_version": 1,
            })
            cursor_second = [int(time())]
            with monkeypatch.context() as cursor_clock:
                cursor_clock.setattr(_cursor, "time", SimpleNamespace(time=lambda: cursor_second[0]))
                omitted = await read("execution", cursor=unfiltered.next_cursor, limit=1, **options)
                cursor_second[0] += 1
                defaults = await read(
                    "execution", cursor=unfiltered.next_cursor, limit=1,
                    agent_run_seq=None, model_request_seq=None, **options,
                )
                assert omitted.items == defaults.items
                signer = data.signer if public else HmacCursorSigner("history", b"history-key")
                assert signer.decode(defaults.next_cursor).expires_at == signer.decode(omitted.next_cursor).expires_at + 1
                after_omitted = await read("execution", cursor=omitted.next_cursor, limit=1, **options)
                after_defaults = await read("execution", cursor=defaults.next_cursor, limit=1, **options)
                assert after_omitted.items == after_defaults.items
                assert [_identity(item) for item in after_omitted.items] == [("execution", 1, 3)]
            with pytest.raises(AIError) as caught:
                await read("execution", cursor=unfiltered.next_cursor, limit=1, agent_run_seq=1, **options)
            assert caught.value.code is ErrorCode.CURSOR_INVALID
        assert data.opened == [] and data.resolved == []


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_model_client_selectors_require_real_execution_and_tenant_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    async with _seed(backend, tmp_path, monkeypatch) as data:
        authorized = []

        class Policy:
            async def authorize(self, principal, action, resource):
                authorized.append(resource.id)
                if resource.id == "child":
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)

        history = RuntimeHistory(DefaultExecutionHistoryService(
            data.storage.execution.executions, Policy(), data.reader, data.signer,
        ), tenant_id="tenant")
        await history.model_interactions("execution", principal=data.principal, agent_run_seq=1, model_request_seq=2)
        assert authorized == ["execution"]
        for execution_id, principal in (("child", data.principal), ("missing", data.principal),
                                        ("execution", Principal("other", "other"))):
            with pytest.raises(AIError) as caught:
                await history.model_interactions(
                    execution_id, principal=principal, agent_run_seq=1, model_request_seq=2,
                    include_content=True, include_client_requests=True,
                )
            assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert authorized == ["execution", "child"]
        assert data.opened == [] and data.resolved == []

        for read, options in (
            (data.history.model_interactions, {"principal": data.principal}),
            (data.reader.model_interactions, {"tenant_id": "tenant", "cursor": None, "limit": 1}),
        ):
            for field in ("agent_run_seq", "model_request_seq"):
                for invalid in (0, True, "1", 1.5):
                    with pytest.raises(AIError) as caught:
                        await read("execution", include_client_requests=True, **options, **{field: invalid})
                    assert caught.value.code is ErrorCode.REQUEST_FIELD_INVALID
        assert data.opened == [] and data.resolved == []


@pytest.mark.asyncio
async def test_model_client_selectors_keep_default_custom_reader_and_service_calls() -> None:
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
    executions = RuntimeExecutions(ExistingService(), unused_get)
    readers = (
        RuntimeHistory(DefaultExecutionHistoryService(
            _Executions(), TenantAuthorizationPolicy("tenant"), ExistingReader(),
        ), tenant_id="tenant"),
        RuntimeHistory(ExistingService(), tenant_id="tenant"),
        executions,
        _execution_service(ExistingService()),
    )
    for reader in readers:
        assert (await reader.model_interactions("execution", principal=principal)).items == ()
        assert (await reader.model_interactions(
            "execution", principal=principal, agent_run_seq=None, model_request_seq=None,
            include_content=True, include_client_requests=False,
        )).items == ()
    execution = Execution(SimpleNamespace(executions=executions), "execution", principal, None)
    assert (await execution.model_interactions()).items == ()
    assert (await execution.model_interactions(
        agent_run_seq=None, model_request_seq=None, include_content=True, include_client_requests=False,
    )).items == ()
    assert calls == [("execution", "tenant", False), ("execution", "tenant", True)] * 5


@pytest.mark.asyncio
async def test_model_client_selectors_propagate_through_execution_facades() -> None:
    calls = []

    class Reader:
        async def model_interactions(
            self, execution_id, *, tenant_id, cursor, limit, cutoffs=None,
            include_content=True, include_client_requests=False,
            agent_run_seq=None, model_request_seq=None,
        ):
            calls.append((execution_id, tenant_id, agent_run_seq, model_request_seq,
                          include_content, include_client_requests))
            return Page(())

    principal = Principal("reader", "tenant")
    history = DefaultExecutionHistoryService(_Executions(), TenantAuthorizationPolicy("tenant"), Reader())
    service = _execution_service(history)
    executions = RuntimeExecutions(service, None)
    for target in (service, executions, RuntimeHistory(history, tenant_id="tenant")):
        await target.model_interactions(
            "execution", principal=principal, agent_run_seq=2, model_request_seq=3,
            include_client_requests=True,
        )
    execution = Execution(SimpleNamespace(executions=executions), "execution", principal, None)
    await execution.model_interactions(agent_run_seq=2, model_request_seq=3, include_client_requests=True)
    assert calls == [("execution", "tenant", 2, 3, False, True)] * 4
