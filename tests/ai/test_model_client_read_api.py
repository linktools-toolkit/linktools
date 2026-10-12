#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authorized original client bodies load independently of SDK projections."""

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from linktools.ai.core import (
    ExecutionStatus, HmacCursorSigner, Page, Principal, TenantAuthorizationPolicy, agent_conversation_id, agent_run_id,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeExecutions, RuntimeHistory, RuntimeStorage, RuntimeStoragePlan, RuntimeStorageRoute
from linktools.ai.runtime._history_service import DefaultExecutionHistoryService
from linktools.ai.runtime._model_interaction import StagedContextInline, StagedContextProjection, build_context_projection
from linktools.ai.runtime._transient_history import TransientExecutionHistoryStore
from linktools.ai.runtime.service_api import UsageReadCutoff
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._codec import _encode_persisted_domain, iter_runtime_object_refs
from linktools.ai.runtime.state._contracts import ContextProjection, RuntimePayloadRef
from linktools.ai.runtime.state._model_interaction_store import ModelInteractionInMemoryStepArchive
from linktools.ai.runtime.state._step_contracts import AgentRunRecord
from linktools.ai.storage import StoredPayload

from .test_history_projection_conformance import _reader, _record
from .test_model_client_interaction_storage import _requests, _staged, _storage
from .test_runtime_history_public import _Executions


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_public_client_reads_are_independent_authorized_and_cursor_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    storage = _storage(backend, tmp_path)
    await storage.initialize(namespace="history", tenant_id="tenant")
    try:
        await storage.execution.executions.create_with_history_head(_record(ExecutionStatus.STARTED, 1))
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

        sdk_request = (ModelRequest(parts=[UserPromptPart("SDK input " * 6000)]),)
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
        archive = storage.run_store.read_store(RuntimeDomain.EXECUTION)
        records = await archive.prepare_interactions(run, entries, payloads.__getitem__)
        await archive.sync_projection(run, events=(), checkpoints=(), interactions=records, execution_id="execution")
        service = DefaultExecutionHistoryService(
            storage.execution.executions, TenantAuthorizationPolicy("tenant"), _reader(storage),
            HmacCursorSigner("client-read", b"client-read-key"),
        )
        history = RuntimeHistory(service, tenant_id="tenant")
        principal = Principal("reader", "tenant")
        objects = storage.object_store(RuntimeDomain.EXECUTION)
        original_open = objects.open
        opened = []

        async def track_open(key):
            opened.append(key)
            async for chunk in original_open(key):
                yield chunk

        monkeypatch.setattr(objects, "open", track_open)

        def object_keys(value):
            return {ref.key for _, ref in iter_runtime_object_refs(
                _encode_persisted_domain(value), default_domain=RuntimeDomain.EXECUTION,
            )}

        sdk_keys = object_keys(replace(records[0], client_requests=None))
        raw_keys = object_keys(records[0]) - sdk_keys
        if backend != "memory":
            assert sdk_keys and raw_keys and sdk_keys.isdisjoint(raw_keys)

        metadata = await history.model_interactions("execution", principal=principal, limit=1)
        item, = metadata.items
        assert not item.content_included and not item.client_requests_included
        assert item.client_requests is None and item.request == {} and item.response is None
        assert opened == []

        sdk = await history.model_interactions("execution", principal=principal, include_content=True, limit=1)
        assert sdk.items[0].request and sdk.items[0].response
        assert sdk.items[0].client_requests is None and not sdk.items[0].client_requests_included
        assert not raw_keys.intersection(opened)
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
        assert not sdk_keys.intersection(opened)
        if backend != "memory":
            assert raw_keys.issubset(opened)

        opened.clear()
        for cursor, flags in ((metadata.next_cursor, {"include_client_requests": True}),
                              (raw.next_cursor, {}), (raw.next_cursor, {"include_client_requests": True, "include_content": True})):
            with pytest.raises(AIError) as caught:
                await history.model_interactions("execution", principal=principal, cursor=cursor, **flags)
            assert caught.value.code is ErrorCode.CURSOR_INVALID
            assert opened == []
        with pytest.raises(AIError) as caught:
            await history.model_interactions("execution", principal=Principal("other", "other"), include_content=True, include_client_requests=True)
        assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert opened == []

        class DeniedPolicy:
            async def authorize(self, principal, action, resource):
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)

        denied = DefaultExecutionHistoryService(storage.execution.executions, DeniedPolicy(), _reader(storage))
        with pytest.raises(AIError) as caught:
            await denied.model_interactions("execution", principal=principal, include_content=True, include_client_requests=True)
        assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert opened == []
        with pytest.raises(AIError) as caught:
            await history.model_interactions("execution", principal=principal, include_client_requests=1)
        assert caught.value.code is ErrorCode.REQUEST_FIELD_INVALID
        assert opened == []

        second = await history.model_interactions(
            "execution", principal=principal, cursor=raw.next_cursor, include_client_requests=True, limit=1,
        )
        assert second.items[0].client_requests_included and second.items[0].client_requests is None
        third = await history.model_interactions(
            "execution", principal=principal, cursor=second.next_cursor, include_client_requests=True, limit=1,
        )
        assert third.items[0].client_requests == () and third.items[0].model_request_seq == 3
        fourth = await history.model_interactions(
            "execution", principal=principal, cursor=third.next_cursor, include_client_requests=True, limit=1,
        )
        assert fourth.next_cursor is None and fourth.items[0].status == "INTERRUPTED"
        assert fourth.items[0].client_requests[0].response.complete is False
        both = await history.model_interactions(
            "execution", principal=principal, include_content=True, include_client_requests=True,
            cutoffs=(UsageReadCutoff("execution", 1, 1),),
        )
        assert len(both.items) == 1
        assert both.items[0].request == sdk.items[0].request
        assert both.items[0].response == sdk.items[0].response
        assert both.items[0].client_requests == original
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
        RuntimeHistory(DefaultExecutionHistoryService(
            _Executions(), TenantAuthorizationPolicy("tenant"), ExistingReader(),
        ), tenant_id="tenant"),
        RuntimeHistory(ExistingService(), tenant_id="tenant"),
        RuntimeExecutions(ExistingService(), unused_get),
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
    archive = ModelInteractionInMemoryStepArchive(RuntimeDomain.RECOVERY)
    await archive.initialize()
    try:
        record, = await archive.prepare_interactions(AgentRunRecord("run"), (_staged(original),), lambda _: b"{}")
        pending_sdk = replace(record, request_context=ContextProjection(()), request_envelope=RuntimePayloadRef(
            StoredPayload.inline_bytes(b"{}"), RuntimeDomain.RECOVERY,
        ))
        assert await archive.resolve_model_interactions(
            (pending_sdk,), include_client_requests=True, include_content=False,
        ) == [(None, None, None, original)]
        assert await archive.resolve_model_interaction(
            pending_sdk, include_client_requests=True, include_content=False,
        ) == (None, None, None, original)
        with pytest.raises(AIError) as caught:
            await archive.resolve_model_interaction(pending_sdk)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await archive.close()

    storage = RuntimeStorage(RuntimeStoragePlan(execution=RuntimeStorageRoute.transient()))
    await storage.initialize(namespace="native-client", tenant_id="tenant")
    try:
        staged = replace(
            _staged(original), request_context=StagedContextProjection((StagedContextInline("a" * 64, 2),)),
            request_envelope_digest="b" * 64,
        )
        await storage.run_store.register_agent_run(AgentRunRecord("run"))
        storage.run_store.stage_model_interaction(staged)
        transient = TransientExecutionHistoryStore(
            storage.run_store.read_store(RuntimeDomain.EXECUTION), storage.run_store,
        )
        assert await transient.resolve_model_interactions(
            (staged,), include_client_requests=True, include_content=False,
        ) == [(None, None, None, original)]
        assert await storage.run_store.resolve_model_interaction(
            staged, include_client_requests=True, include_content=False,
        ) == (None, None, None, original)
        with pytest.raises(AIError) as caught:
            await transient.resolve_model_interactions((staged,))
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await storage.close()
