#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline cleanup scans physical sources without losing domain references."""

from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.runtime.state import RuntimeState
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.agent import AgentInputCaptureRef
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.runtime.state import ArtifactRecord, RuntimeDomain, input_capture_key
from linktools.ai.runtime.state._codec import encode_domain
from linktools.ai.runtime.state._contracts import RuntimePayloadRef
from linktools.ai.runtime.state._object_cleanup import purge_unreferenced_objects
from linktools.ai.storage import FilesystemObjectStore, InMemoryObjectStore, ObjectStat, ObjectStoreMaintenance, SqlObjectStore, StoredPayload

from .test_evaluation_retention import NOW, Offline, put


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_cleanup_shared_source_io_does_not_scale_with_domain_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'state.sqlite'}")
    await provision_database(engine)
    objects = (InMemoryObjectStore() if backend == "memory" else
               FilesystemObjectStore(tmp_path / "objects") if backend == "filesystem" else SqlObjectStore(engine))
    storage = RuntimeState(RuntimeStorage.sql(engine, object_store=objects))
    await storage.initialize(namespace="evaluation-retention", tenant_id="tenant")
    try:
        shared = await put(objects, "shared", {"content": "another owner"})
        captured = await put(objects, "captured", {"content": "independent capture"})
        orphan = await put(objects, "orphan", {"content": "unreferenced"})
        await storage.artifact.records.put_metadata(ArtifactRecord(
            "artifact", "execution", "producer", "application/json", shared, NOW))
        foreign = await put(objects, input_capture_key("other", "other", "agent", "independent"), {
            "kind": "agent", "version": 1, "namespace": "other", "tenant_id": "other",
            "payload": encode_domain(captured),
        })
        await put(objects, "v1/input-capture/declaration/foreign", {
            "capture": encode_domain(AgentInputCaptureRef("other", "other", "independent", foreign.digest, None)),
        })
        reads: list[str] = []
        listings: list[None] = []
        statements: list[str] = []
        original_open, original_list = objects.open, objects.list_objects

        async def opened(key: str) -> AsyncIterator[bytes]:
            reads.append(key)
            stream = original_open(key)
            try:
                async for value in stream:
                    yield value
            finally:
                await stream.aclose()

        async def listed() -> AsyncIterator[ObjectStat]:
            listings.append(None)
            async for value in original_list():
                yield value

        def capture(_connection: object, _cursor: object, sql: str, _parameters: object,
                    _context: object, _many: bool) -> None:
            if sql.startswith("SELECT"):
                statements.append(sql)

        monkeypatch.setattr(objects, "open", opened)
        monkeypatch.setattr(objects, "list_objects", listed)
        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        candidates = tuple((RuntimeDomain.EVALUATION, ref) for ref in (shared, captured, orphan))
        result = await storage.purge_unreferenced_objects(candidates, exclusive=Offline(), limit=1)
        assert result.retained == candidates[:2]
        if isinstance(objects, ObjectStoreMaintenance):
            assert result.deleted == candidates[2:]
        else:
            assert result.blocked == candidates[2:]
        assert len(listings) <= 1
        assert len(reads) <= 2
        assert len(statements) <= 35
        assert await objects.stat(shared.key) is not None
        assert await objects.stat(captured.key) is not None
    finally:
        await storage.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("malformed", "unknown_version", "transport"))
async def test_cleanup_does_not_merge_distinct_sources_with_the_same_store_id(
    monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    first, second = InMemoryObjectStore(), InMemoryObjectStore()
    assert first.store_id == second.store_id
    key = "v1/asset-snapshot/root"
    await put(first, key, {"kind": "asset-snapshot", "format_version": 1, "entries": []})
    value = ([] if fault == "malformed" else
             {"kind": "asset-snapshot", "format_version": 2 if fault == "unknown_version" else 1, "entries": []})
    await put(second, key, value)
    orphan = await put(first, "orphan", "unreferenced")
    failure = RuntimeError("second source unavailable")

    async def failed(key: str) -> AsyncIterator[bytes]:
        raise failure
        yield b""  # pragma: no cover

    if fault == "transport":
        monkeypatch.setattr(second, "open", failed)
    sources = {domain: first for domain in RuntimeDomain}
    sources[RuntimeDomain.TASK] = second
    with pytest.raises(RuntimeError if fault == "transport" else AIError) as raised:
        await purge_unreferenced_objects(((RuntimeDomain.EVALUATION, orphan),),
            namespace="cleanup", tenant_id="tenant", stores={}, object_stores=sources, exclusive=Offline())
    if fault == "transport":
        assert raised.value is failure
    else:
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert await first.stat(orphan.key) is not None


@pytest.mark.asyncio
async def test_cleanup_preserves_implicit_and_explicit_domain_dependencies() -> None:
    primary, task, execution = (InMemoryObjectStore() for _ in range(3))
    implicit = await put(primary, "implicit", "kept by physical source")
    explicit = await put(execution, "explicit", "kept by explicit source")
    await put(primary, "v1/asset-snapshot/root", {"kind": "asset-snapshot", "format_version": 1,
        "entries": [{"content": {"key": implicit.key, "digest": implicit.digest, "size": implicit.size}}]})
    await put(task, "v1/input-capture/declaration/root", {
        "payload": encode_domain(RuntimePayloadRef(StoredPayload.object(explicit), RuntimeDomain.EXECUTION)),
    })
    sources = {domain: primary for domain in RuntimeDomain}
    sources.update({RuntimeDomain.TASK: task, RuntimeDomain.EXECUTION: execution})
    candidates = ((RuntimeDomain.ARTIFACT, implicit), (RuntimeDomain.EXECUTION, explicit))
    result = await purge_unreferenced_objects(candidates, namespace="cleanup", tenant_id="tenant",
        stores={}, object_stores=sources, exclusive=Offline())
    assert result.retained == candidates
    assert not result.deleted


@pytest.mark.asyncio
async def test_cleanup_shared_manifest_still_validates_each_reference_size() -> None:
    objects = InMemoryObjectStore()
    reference = await put(objects, "v1/asset-snapshot/root", {
        "kind": "asset-snapshot", "format_version": 1, "entries": [],
    })
    orphan = await put(objects, "orphan", "unreferenced")
    first, second = RuntimeState(RuntimeStorage.in_memory()), RuntimeState(RuntimeStorage.in_memory())
    await first.initialize(namespace="cleanup", tenant_id="tenant")
    await second.initialize(namespace="cleanup", tenant_id="tenant")
    try:
        for storage, ref in ((first, reference), (second, replace(reference, size=reference.size + 1))):
            await storage.artifact.records.put_metadata(ArtifactRecord(
                "artifact", "execution", "producer", "application/json", ref, NOW))
        stores = {RuntimeDomain.ARTIFACT: first.artifact.records.state_store,
                  RuntimeDomain.EXECUTION: second.artifact.records.state_store}
        with pytest.raises(AIError) as raised:
            await purge_unreferenced_objects(((RuntimeDomain.EVALUATION, orphan),),
                namespace="cleanup", tenant_id="tenant", stores=stores,
                object_stores={domain: objects for domain in RuntimeDomain}, exclusive=Offline())
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await objects.stat(orphan.key) is not None
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_cleanup_reuses_capture_metadata_only_within_exclusive_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    from collections import Counter
    from linktools.ai.runtime.state import input_capture_expiry_key
    from .test_state_batch_delete_backends import _record

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'captures.sqlite'}")
    await provision_database(engine)
    objects = (InMemoryObjectStore() if backend == "memory" else
               FilesystemObjectStore(tmp_path / "objects") if backend == "filesystem" else SqlObjectStore(engine))
    storage = RuntimeState(RuntimeStorage.sql(engine, object_store=objects))
    await storage.initialize(namespace="cleanup", tenant_id="tenant")
    try:
        target = await put(objects, "target", {"content": "retained"})
        capture = await put(objects, input_capture_key("cleanup", "tenant", "agent", "captured"), {
            "kind": "agent", "version": 1, "namespace": "cleanup", "tenant_id": "tenant",
            "payload": encode_domain(target),
        })
        ref = AgentInputCaptureRef("cleanup", "tenant", "captured", capture.digest, None)
        await storage.artifact.records.state_store.mutate(lambda tx: tx.insert_record(
            replace(_record(1), data={"captures": [encode_domain(ref)] * 16})))
        original = objects.stat
        reads: Counter[str] = Counter()
        failed = False
        failure = RuntimeError("metadata transport failure")
        async def stat(key):
            nonlocal failed
            reads[key] += 1
            if not failed:
                failed = True
                raise failure
            return await original(key)
        monkeypatch.setattr(objects, "stat", stat)
        candidate = ((RuntimeDomain.EVALUATION, target),)
        with pytest.raises(RuntimeError) as raised:
            await storage.purge_unreferenced_objects(candidate, exclusive=Offline())
        assert raised.value is failure
        reads.clear()
        assert (await storage.purge_unreferenced_objects(candidate, exclusive=Offline())).retained == candidate
        assert reads[input_capture_expiry_key(capture.key)] <= 1
        assert reads[capture.key] <= 1
        changed = replace(_record(1), storage_revision=1, data={"captures": [encode_domain(ref)] * 15
                          + [encode_domain(replace(ref, digest="0" * 64))]})
        assert await storage.artifact.records.state_store.mutate(lambda tx: tx.replace_record(
            changed, expected_storage_revision=0))
        with pytest.raises(AIError) as mismatch:
            await storage.purge_unreferenced_objects(candidate, exclusive=Offline())
        assert mismatch.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        await put(objects, input_capture_expiry_key(capture.key), {
            "namespace": "cleanup", "tenant_id": "tenant", "capture_key": capture.key,
            "expired_at": NOW.isoformat(),
        })
        reads.clear()
        assert not (await storage.purge_unreferenced_objects(candidate, exclusive=Offline())).retained
        assert reads[input_capture_expiry_key(capture.key)] <= 1
    finally:
        await storage.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_cleanup_expiry_metadata_does_not_cross_object_instances() -> None:
    from linktools.ai.runtime.state import input_capture_expiry_key

    first, second = InMemoryObjectStore(), InMemoryObjectStore()
    assert first.store_id == second.store_id
    target = await put(second, "target", {"content": "retained"})
    key = input_capture_key("cleanup", "tenant", "agent", "capture")
    payload = {"kind": "agent", "version": 1, "namespace": "cleanup", "tenant_id": "tenant",
               "payload": encode_domain(RuntimePayloadRef(StoredPayload.object(target), RuntimeDomain.ARTIFACT))}
    await put(first, key, payload)
    await put(second, key, payload)
    await put(first, input_capture_expiry_key(key), {
        "namespace": "cleanup", "tenant_id": "tenant", "capture_key": key, "expired_at": NOW.isoformat(),
    })
    sources = {domain: first for domain in RuntimeDomain}
    sources[RuntimeDomain.ARTIFACT] = second
    candidate = ((RuntimeDomain.ARTIFACT, target),)
    result = await purge_unreferenced_objects(candidate, namespace="cleanup", tenant_id="tenant", stores={},
                                              object_stores=sources, exclusive=Offline())
    assert result.retained == candidate
