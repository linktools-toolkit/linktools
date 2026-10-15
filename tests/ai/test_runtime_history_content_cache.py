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
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.core import Principal, TenantAuthorizationPolicy, canonical_json_bytes, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeHistory, RuntimeStorage
from linktools.ai.runtime._message import encode_model_messages
from linktools.ai.runtime._execution import DefaultExecutionService
from linktools.ai.runtime._event import LiveExecutionEventBroker
from linktools.ai.runtime._factory import _borrowed_runtime_history
from linktools.ai.runtime.state import RuntimeDomain, RuntimeStoragePlan, RuntimeStorageRoute
from linktools.ai.runtime.state._step_archive import StateStepArchive
from linktools.ai.storage import InMemoryCacheBackend, InMemoryObjectStore, ObjectRef, StoredPayload

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
    storage = RuntimeStorage.sqlite(
        tmp_path / "runtime.db", object_store=objects,
        cache_backend=InMemoryCacheBackend(max_bytes=1024),
    )
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
        domains: list[RuntimeDomain] = []

        async def read(domain: RuntimeDomain, ref: ObjectRef) -> bytes:
            domains.append(domain)
            return await storage.read_object(domain, ref)

        history = RuntimeHistory(
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
        assert domains == ([RuntimeDomain.EXECUTION] * 2 if use_storage_reader else [])

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
@pytest.mark.parametrize("domain", (RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY))
async def test_composed_transcript_archives_cache_verified_compressed_bytes(
    tmp_path: Path, domain: RuntimeDomain,
) -> None:
    objects = _CountingObjects()
    cache = _RecordingCache()
    route = RuntimeStorageRoute.sqlite(tmp_path / "runtime.db")
    storage = RuntimeStorage.from_plan(
        RuntimeStoragePlan(conversation=route, execution=route),
        object_store=objects, cache_backend=cache,
    )
    await storage.initialize(namespace="history", tenant_id="tenant")
    try:
        archive = storage.run_store.read_store(domain)
        assert isinstance(archive, StateStepArchive)
        repository = archive.transcript_repository
        messages = (ModelResponse(parts=[TextPart("compressed body " * 2048)]),)
        await repository.create_head("run")
        chunks = await repository.prepare_chunks("run", messages, first_message_index=0)
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.codec == "zlib" and chunk.content.payload.kind == "object"
        await archive.state_store.mutate(lambda transaction: repository.append_chunks(transaction, "run", chunks))

        assert await repository.load_messages("run") == messages
        assert await repository.load_messages("run") == messages
        assert objects.reads == 1
        assert len(cache.writes) == 1
        key, compressed = cache.writes[0]
        assert hashlib.sha256(compressed).hexdigest() == chunk.content.payload.digest
        assert len(compressed) == chunk.content.payload.size
        assert zlib.decompress(compressed) == encode_model_messages(messages)
        assert chunk.raw_digest != chunk.content.payload.digest

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
    storage = RuntimeStorage.sqlite(
        tmp_path / "runtime.db", object_store=objects,
        cache_backend=InMemoryCacheBackend(max_bytes=1024),
    )
    await storage.initialize(namespace="history", tenant_id="tenant")
    try:
        record = await _seed(storage, output=StoredPayload.object(reference))
        domains: list[RuntimeDomain] = []

        async def read(domain: RuntimeDomain, ref: ObjectRef) -> bytes:
            domains.append(domain)
            return await storage.read_object(domain, ref)

        service = DefaultExecutionService(
            storage.execution, objects, TenantAuthorizationPolicy("tenant"),
            sessions=storage.conversation.sessions, catalog=None, compiler=None,
            runtime_bridge=SimpleNamespace(port=None),
            live_broker=LiveExecutionEventBroker(), history_reader=None,
            object_reader=read if use_storage_reader else None,
        )
        principal = Principal("principal", "tenant", "service")
        assert (await service.result("execution", principal=principal)).output == {"result": "live output"}
        assert (await service.wait("execution", principal=principal)).output == {"result": "live output"}
        assert objects.reads == (1 if use_storage_reader else 2)
        assert domains == ([RuntimeDomain.EXECUTION] * 2 if use_storage_reader else [])

        with pytest.raises(AIError) as error:
            await service.result("execution", principal=Principal("principal", "other", "service"))
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert objects.reads == (1 if use_storage_reader else 2)

        foreign = replace(reference, store_id="foreign")
        async def foreign_result(execution_id: str, *, tenant_id: str):
            assert execution_id == "execution" and tenant_id == "tenant"
            return replace(record, result=replace(record.result, output=StoredPayload.object(foreign)))

        monkeypatch.setattr(storage.execution.executions, "get", foreign_result)
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
    storage = RuntimeStorage.sqlite(tmp_path / "runtime.db", object_store=objects, cache_backend=cache)
    await storage.initialize(namespace="history", tenant_id="tenant")
    principal = Principal("principal", "tenant", "service")
    try:
        await _seed(storage, output=StoredPayload.object(reference))
        history = _borrowed_runtime_history(
            SimpleNamespace(), tenant_id="tenant", storage=storage,
            authorization=TenantAuthorizationPolicy("tenant"), artifact=SimpleNamespace(),
        )
        assert (await history.result("execution", principal=principal)).output == {"result": "persisted output"}
        assert (await history.result("execution", principal=principal)).output == {"result": "persisted output"}
        assert objects.reads == 1
    finally:
        await storage.close()

    reopened = RuntimeStorage.sqlite(tmp_path / "runtime.db", object_store=objects, cache_backend=cache)
    async with RuntimeHistory.open("history", storage=reopened, tenant_id="tenant") as history:
        assert (await history.result("execution", principal=principal)).output == {"result": "persisted output"}
        assert objects.reads == 1
