#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""History content reads share Runtime caching without changing domain ownership."""

import hashlib
import zlib
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from linktools.ai.persistence import Persistence
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.core import Principal, TenantAuthorizationPolicy, canonical_json_bytes, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.model.messages import encode_model_messages
from linktools.ai.runtime._execution import DefaultExecutionService, _ExecutionRuntimeBridge
from linktools.ai.execution.events import LiveExecutionEventBroker
from linktools.ai.runtime._factory import _borrowed_runtime_history
from linktools.ai.persistence import PersistenceDomain, PersistencePlan
from linktools.ai.storage import StorageRoute
from linktools.ai.persistence._step_archive import StateStepArchive
from linktools.ai.persistence._archive import PreparedArchive
from linktools.ai.storage import InMemoryCacheBackend, InMemoryObjectStore, ObjectRef, StoredPayload, StorageReadCache

from .test_runtime_history_public import _ResultExecutions, _TaskResults
from .test_execution_result_read_io import _seed


class _CountingObjects(InMemoryObjectStore):
    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    async def open(self, key: str) -> AsyncIterator[bytes]:
        self.reads += 1
        async for chunk in super().open(key):
            yield chunk


class _RecordingCache(InMemoryCacheBackend):
    def __init__(self) -> None:
        super().__init__(max_bytes=1024 * 1024)
        self.writes: list[tuple[str, bytes]] = []

    async def put(self, key: str, content: bytes) -> None:
        self.writes.append((key, content))
        await super().put(key, content)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_storage_reader", (False, True), ids=("direct", "cached"))
async def test_history_and_task_results_keep_execution_object_ownership(
    tmp_path: Path, use_storage_reader: bool,
) -> None:
    objects = _CountingObjects()
    value = {"result": "persisted output"}
    content = canonical_json_bytes(value)
    digest = hashlib.sha256(content).hexdigest()

    async def chunks() -> AsyncIterator[bytes]:
        yield content

    await objects.put("result", chunks(), expected_size=len(content), expected_digest=digest)
    reference = ObjectRef("runtime", "result", digest, len(content))
    storage = Persistence(PersistenceStorage.sqlite(
        tmp_path / "runtime.db", object_store=objects,
        cache=InMemoryCacheBackend(max_bytes=1024),
    ))
    await storage.initialize(namespace="history", tenant_id="tenant")
    try:
        executions = _ResultExecutions()
        executions.result.output = StoredPayload.object(reference)
        tasks = _TaskResults()
        tasks.record = replace(tasks.record, result_digest=canonical_sha256(value))
        tasks._graph_state = replace(
            tasks._graph_state,
            node_states=(replace(tasks._graph_state.node_states[0], result_digest=canonical_sha256(value)),),
        )
        domains: list[PersistenceDomain] = []

        async def read(domain: PersistenceDomain, ref: ObjectRef) -> bytes:
            domains.append(domain)
            return await storage.read_object(domain, ref)

        history = History(
            SimpleNamespace(), tenant_id="tenant", executions=executions, tasks=tasks,
            authorization=TenantAuthorizationPolicy("tenant"),
            execution_objects=objects,
            task_objects=InMemoryObjectStore(),
            object_reader=read if use_storage_reader else None,
        )
        principal = Principal("reader", "tenant", "service")
        assert (await history.result("execution", principal=principal)).output == value
        assert await history.task_result("graph", "node", principal=principal) == value
        assert objects.reads == (1 if use_storage_reader else 2)
        assert domains == ([PersistenceDomain.EXECUTION] * 2 if use_storage_reader else [])

        with pytest.raises(AIError) as error:
            await history.result("execution", principal=Principal("reader", "other", "service"))
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert objects.reads == (1 if use_storage_reader else 2)
        assert len(domains) == (2 if use_storage_reader else 0)

        executions.result.output = StoredPayload.inline_json(value)
        assert (await history.result("execution", principal=principal)).output == value
        assert len(domains) == (2 if use_storage_reader else 0)
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("domain", (PersistenceDomain.CONVERSATION, PersistenceDomain.TRACE, PersistenceDomain.CHECKPOINT))
async def test_composed_transcript_archives_cache_verified_compressed_bytes(
    tmp_path: Path, domain: PersistenceDomain,
) -> None:
    objects = _CountingObjects()
    cache = _RecordingCache()
    route = StorageRoute.sqlite(tmp_path / "runtime.db", cache=cache)
    storage = Persistence(PersistenceStorage.from_plan(
        PersistencePlan(conversation=route, execution=route, trace=route, checkpoint=route.with_cache(None)),
        object_store=objects,
    ))
    await storage.initialize(
        namespace="history", tenant_id="tenant",
    )
    try:
        archive = storage.run_store.read_store(domain)
        assert isinstance(archive, StateStepArchive)
        repository = archive.transcript_repository
        messages = (ModelResponse(parts=[TextPart("compressed body " * 2048)]),)
        await repository.create_head("run")
        prepared = PreparedArchive()
        chunks = await repository.prepare_chunks(
            "run", messages, first_message_index=0, prepared=prepared,
        )
        assert len(chunks) == 1
        chunk = chunks[0]
        async def publish(transaction):
            await repository.publish_archive(prepared)
            await repository.append_chunks(transaction, "run", chunks)

        await archive.state_store.mutate(publish)
        contents = await repository.content_archive()
        identities = await contents.content_ids(chunk.content)
        records = await contents.content_records(identities)
        assert len(records) == 1
        content = records[0]
        assert content.compression == "zlib" and content.payload.kind == "object"

        assert await repository.load_messages("run") == messages
        assert await repository.load_messages("run") == messages
        if domain is PersistenceDomain.CHECKPOINT:
            assert objects.reads == 2
            assert cache.writes == []
            assert await repository.read_payload(StoredPayload.inline_bytes(b"inline body")) == b"inline body"
            assert objects.reads == 2
            return
        assert objects.reads == 1
        assert len(cache.writes) == 1
        key, _envelope = cache.writes[0]

        async def unexpected_origin():
            pytest.fail("fresh verified compressed body must stay hot")

        compressed = await StorageReadCache(cache).get(key, unexpected_origin)
        assert hashlib.sha256(compressed).hexdigest() == content.payload.digest
        assert len(compressed) == content.payload.size
        assert zlib.decompress(compressed) == encode_model_messages(messages)
        assert chunk.raw_digest != content.payload.digest

        await cache.put(key, b"corrupt cached body")
        assert await repository.load_messages("run") == messages
        assert objects.reads == 2

        inline = StoredPayload.inline_bytes(b"inline body")
        assert await repository.read_payload(inline) == b"inline body"
        assert objects.reads == 2
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_storage_reader", (False, True), ids=("direct", "cached"))
async def test_live_result_reads_preserve_authorization_and_direct_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_storage_reader: bool,
) -> None:
    objects = _CountingObjects()
    content = b'{"result":"live output"}'
    digest = hashlib.sha256(content).hexdigest()

    async def chunks() -> AsyncIterator[bytes]:
        yield content

    await objects.put("result", chunks(), expected_size=len(content), expected_digest=digest)
    reference = ObjectRef("runtime", "result", digest, len(content))
    storage = Persistence(PersistenceStorage.sqlite(
        tmp_path / "runtime.db", object_store=objects,
        cache=InMemoryCacheBackend(max_bytes=1024),
    ))
    await storage.initialize(namespace="history", tenant_id="tenant")
    try:
        record = await _seed(storage, output=StoredPayload.object(reference))
        domains: list[PersistenceDomain] = []

        async def read(domain: PersistenceDomain, ref: ObjectRef) -> bytes:
            domains.append(domain)
            return await storage.read_object(domain, ref)

        service = DefaultExecutionService(
            storage.execution, objects, TenantAuthorizationPolicy("tenant"),
            sessions=storage.conversation.sessions, catalog=None, compiler=None,
            runtime_bridge=_ExecutionRuntimeBridge(),
            live_broker=LiveExecutionEventBroker(), history_reader=None,
            object_reader=read if use_storage_reader else None,
        )
        principal = Principal("principal", "tenant", "service")
        assert (await service.result("execution", principal=principal)).output == {"result": "live output"}
        assert (await service.wait("execution", principal=principal)).output == {"result": "live output"}
        assert objects.reads == (1 if use_storage_reader else 2)
        assert domains == ([PersistenceDomain.EXECUTION] * 2 if use_storage_reader else [])

        with pytest.raises(AIError) as error:
            await service.result("execution", principal=Principal("principal", "other", "service"))
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert objects.reads == (1 if use_storage_reader else 2)

        foreign = replace(reference, store_id="foreign")
        async def foreign_result(execution_id: str, *, tenant_id: str):
            assert execution_id == "execution" and tenant_id == "tenant"
            return replace(record, result=replace(record.result, output=StoredPayload.object(foreign)))

        monkeypatch.setattr(storage.execution.executions, "get_fresh", foreign_result)
        with pytest.raises(AIError) as error:
            await service.result("execution", principal=principal)
        assert error.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
        assert objects.reads == (1 if use_storage_reader else 2)
        assert len(domains) == (2 if use_storage_reader else 0)
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_borrowed_and_read_only_history_use_shared_result_cache(tmp_path: Path) -> None:
    objects = _CountingObjects()
    cache = InMemoryCacheBackend(max_bytes=1024)
    content = b'{"result":"persisted output"}'
    digest = hashlib.sha256(content).hexdigest()

    async def chunks() -> AsyncIterator[bytes]:
        yield content

    await objects.put("result", chunks(), expected_size=len(content), expected_digest=digest)
    reference = ObjectRef("runtime", "result", digest, len(content))
    storage = Persistence(PersistenceStorage.sqlite(tmp_path / "runtime.db", object_store=objects, cache=cache))
    await storage.initialize(namespace="history", tenant_id="tenant")
    principal = Principal("principal", "tenant", "service")
    try:
        await _seed(storage, output=StoredPayload.object(reference))
        history = _borrowed_runtime_history(
            SimpleNamespace(), tenant_id="tenant", state=storage, run_store=storage.run_store,
            authorization=TenantAuthorizationPolicy("tenant"), artifact=SimpleNamespace(),
        )
        assert (await history.result("execution", principal=principal)).output == {"result": "persisted output"}
        assert (await history.result("execution", principal=principal)).output == {"result": "persisted output"}
        assert objects.reads == 1
    finally:
        await storage.close()

    reopened = PersistenceStorage.sqlite(tmp_path / "runtime.db", object_store=objects, cache=cache)
    async with Persistence.open('history', storage=reopened, tenant_id='tenant', read_only=True) as history_persistence_277, History.open(persistence=history_persistence_277) as history:
        assert (await history.result("execution", principal=principal)).output == {"result": "persisted output"}
        assert objects.reads == 1


@pytest.mark.asyncio
async def test_history_result_confirms_source_membership_inside_body_cache_window(tmp_path: Path, monkeypatch) -> None:
    from linktools.ai.storage import _write_cache

    now = [1000.0]
    monkeypatch.setattr(_write_cache, "time", lambda: now[0])
    objects = _CountingObjects()
    content = b'{"text":"retained result"}'
    digest = hashlib.sha256(content).hexdigest()

    async def chunks():
        yield content

    await objects.put("result", chunks(), expected_size=len(content), expected_digest=digest)
    reference = ObjectRef("runtime", "result", digest, len(content))
    storage = Persistence(PersistenceStorage.sqlite(tmp_path / "runtime.db", object_store=objects,
                                    cache=InMemoryCacheBackend(max_bytes=100000)))
    await storage.initialize(namespace="history", tenant_id="tenant")
    principal = Principal("principal", "tenant", "service")
    try:
        await _seed(storage, output=StoredPayload.object(reference))
        history = _borrowed_runtime_history(SimpleNamespace(), tenant_id="tenant", state=storage, run_store=storage.run_store,
                                           authorization=TenantAuthorizationPolicy("tenant"), artifact=SimpleNamespace())
        assert (await history.result("execution", principal=principal)).output == {"text": "retained result"}
        await objects.delete_object(reference.key, expected_digest=reference.digest)
        now[0] = 1002.0
        assert await storage.read_object(PersistenceDomain.EXECUTION, reference) == content
        with pytest.raises(AIError) as raised:
            await history.result("execution", principal=principal)
        assert raised.value.code is ErrorCode.STORAGE_NOT_FOUND
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer", ("checkpoint", "verify", "relocate", "deferred"))
@pytest.mark.parametrize("domain", (PersistenceDomain.CHECKPOINT, PersistenceDomain.CONVERSATION))
async def test_checkpoint_consumers_confirm_source_inside_body_cache_window(
    tmp_path: Path, monkeypatch, consumer: str, domain: PersistenceDomain,
) -> None:
    from linktools.ai.runtime._local import LocalExecutionBackend
    from linktools.ai.persistence._step_contracts import AgentRunCheckpoint, AgentRunRecord
    from linktools.ai.storage import _write_cache

    now = [1000.0]
    monkeypatch.setattr(_write_cache, "time", lambda: now[0])
    objects = _CountingObjects()
    storage = Persistence(PersistenceStorage.sqlite(tmp_path / "checkpoint.db", object_store=objects,
                                    cache=InMemoryCacheBackend(max_bytes=100000)))
    await storage.initialize(
        namespace="checkpoint", tenant_id="tenant",
    )
    try:
        archive = storage.run_store.read_store(domain)
        run = AgentRunRecord("run")
        messages = [ModelResponse(parts=[TextPart("retained checkpoint " * 2048)])]
        checkpoint = AgentRunCheckpoint("run", 1, messages, state="interrupted", context_messages=messages,
                                        transcript_message_count_before=0)
        await archive.materialize_checkpoint(run, checkpoint)
        assert await archive.latest_checkpoint(agent_run_id="run", include_interrupted=True) is not None
        references = [value async for value in objects.list_objects()]
        assert references
        for reference in references:
            await objects.delete_object(reference.key, expected_digest=reference.digest)
        now[0] = 1002.0
        # The ordinary observer still has its permitted hot bytes. The consumers
        # below must instead validate the current recovery source.
        if domain is not PersistenceDomain.CHECKPOINT:
            assert await archive.transcript_repository.load_messages("run") == tuple(messages)
        with pytest.raises(AIError) as raised:
            if consumer == "checkpoint":
                await archive.latest_checkpoint(agent_run_id="run", include_interrupted=True)
            elif consumer == "verify":
                await archive.verify_checkpoint_projection(agent_run_id="run", checkpoint=checkpoint)
            elif consumer == "relocate":
                if domain is PersistenceDomain.CONVERSATION:
                    await archive.relocate_conversation_checkpoint(run, checkpoint)
                else:
                    await archive.relocate_run_checkpoint(run, checkpoint)
            else:
                backend = object.__new__(LocalExecutionBackend)
                backend._run_stores = {PersistenceDomain.CHECKPOINT: archive}
                await backend.load_interrupted_messages("run")
        assert raised.value.code is ErrorCode.STORAGE_NOT_FOUND
    finally:
        await storage.close()
