#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline cleanup scans physical sources without losing domain references."""

from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import TypeVar

import pytest

from linktools.ai.persistence import Persistence
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.agent import AgentInputCaptureRef
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence.contracts import ArtifactRecord
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence.captures import capture_record_digest, input_capture_key, publish_capture_record
from linktools.ai.persistence._memory import InMemoryStateStore
from linktools.ai.persistence.state import StateCallback
from linktools.ai.persistence.codec import encode_domain
from linktools.ai.persistence.contracts import RuntimePayloadRef
from linktools.ai.persistence._object_cleanup import purge_unreferenced_objects
from linktools.ai.storage import (
    FilesystemObjectStore,
    InMemoryObjectStore,
    ObjectStat,
    ObjectStoreDeletion,
    SqlObjectStore,
    StoredPayload,
)

from .test_evaluation_retention import NOW, Offline, put


ValueT = TypeVar("ValueT")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_cleanup_shared_source_io_does_not_scale_with_domain_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'state.sqlite'}")
    await provision_database(engine)
    objects = (InMemoryObjectStore() if backend == "memory" else
               FilesystemObjectStore(tmp_path / "objects") if backend == "filesystem" else SqlObjectStore(engine))
    storage = Persistence(PersistenceStorage.sql(engine, object_store=objects))
    await storage.initialize(namespace="evaluation-retention", tenant_id="tenant")
    try:
        shared = await put(objects, "shared", {"content": "another owner"})
        captured = await put(objects, "captured", {"content": "independent capture"})
        orphan = await put(objects, "orphan", {"content": "unreferenced"})
        await storage.artifact.records.put_metadata(ArtifactRecord(
            "artifact", "execution", "producer", "application/json", shared, NOW))
        await publish_capture_record(storage.task.tasks.state_store, namespace="other", tenant_id="other",
            key=input_capture_key("other", "other", "agent", "independent"), value={
                "kind": "agent", "version": 1, "namespace": "other", "tenant_id": "other",
                "payload": encode_domain(captured),
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
        candidates = tuple((PersistenceDomain.EVALUATION, ref) for ref in (shared, captured, orphan))
        result = await storage.purge_unreferenced_objects(candidates, exclusive=Offline(), limit=1)
        assert result.retained == candidates[:2]
        if isinstance(objects, ObjectStoreDeletion):
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
    sources = {domain: first for domain in PersistenceDomain}
    sources[PersistenceDomain.TASK] = second
    with pytest.raises(RuntimeError if fault == "transport" else AIError) as raised:
        await purge_unreferenced_objects(((PersistenceDomain.EVALUATION, orphan),),
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
    state = InMemoryStateStore()
    await state.initialize()
    try:
        await publish_capture_record(state, namespace="cleanup", tenant_id="tenant",
            key=input_capture_key("cleanup", "tenant", "agent", "root"), value={
                "kind": "agent", "version": 1, "namespace": "cleanup", "tenant_id": "tenant",
                "payload": encode_domain(RuntimePayloadRef(StoredPayload.object(explicit), PersistenceDomain.EXECUTION)),
            })
        sources = {domain: primary for domain in PersistenceDomain}
        sources.update({PersistenceDomain.TASK: task, PersistenceDomain.EXECUTION: execution})
        candidates = ((PersistenceDomain.ARTIFACT, implicit), (PersistenceDomain.EXECUTION, explicit))
        result = await purge_unreferenced_objects(candidates, namespace="cleanup", tenant_id="tenant",
            stores={PersistenceDomain.TASK: state}, object_stores=sources, exclusive=Offline())
        assert result.retained == candidates
        assert not result.deleted
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_cleanup_shared_manifest_still_validates_each_reference_size() -> None:
    objects = InMemoryObjectStore()
    reference = await put(objects, "v1/asset-snapshot/root", {
        "kind": "asset-snapshot", "format_version": 1, "entries": [],
    })
    orphan = await put(objects, "orphan", "unreferenced")
    first, second = Persistence(PersistenceStorage.in_memory()), Persistence(PersistenceStorage.in_memory())
    await first.initialize(namespace="cleanup", tenant_id="tenant")
    await second.initialize(namespace="cleanup", tenant_id="tenant")
    try:
        for storage, ref in ((first, reference), (second, replace(reference, size=reference.size + 1))):
            await storage.artifact.records.put_metadata(ArtifactRecord(
                "artifact", "execution", "producer", "application/json", ref, NOW))
        stores = {PersistenceDomain.ARTIFACT: first.artifact.records.state_store,
                  PersistenceDomain.EXECUTION: second.artifact.records.state_store}
        with pytest.raises(AIError) as raised:
            await purge_unreferenced_objects(((PersistenceDomain.EVALUATION, orphan),),
                namespace="cleanup", tenant_id="tenant", stores=stores,
                object_stores={domain: objects for domain in PersistenceDomain}, exclusive=Offline())
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
    from linktools.ai.persistence.captures import input_capture_expiry_key
    from .test_state_batch_delete_backends import _record

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'captures.sqlite'}")
    await provision_database(engine)
    objects = (InMemoryObjectStore() if backend == "memory" else
               FilesystemObjectStore(tmp_path / "objects") if backend == "filesystem" else SqlObjectStore(engine))
    storage = Persistence(PersistenceStorage.sql(engine, object_store=objects))
    await storage.initialize(namespace="cleanup", tenant_id="tenant")
    try:
        target = await put(objects, "target", {"content": "retained"})
        state = storage.task.tasks.state_store
        capture_key = input_capture_key("cleanup", "tenant", "agent", "captured")
        capture = await publish_capture_record(state, namespace="cleanup", tenant_id="tenant", key=capture_key, value={
            "kind": "agent", "version": 1, "namespace": "cleanup", "tenant_id": "tenant",
            "payload": encode_domain(target),
        })
        ref = AgentInputCaptureRef("cleanup", "tenant", "captured", capture_record_digest(capture), None)
        await storage.artifact.records.state_store.mutate(lambda tx: tx.insert_record(
            replace(_record(1), data={"captures": [encode_domain(ref)] * 16})))
        original_read = state.read
        reads: list[None] = []
        failed = False
        failure = RuntimeError("metadata transport failure")

        async def read(callback: StateCallback[ValueT]) -> ValueT:
            nonlocal failed
            reads.append(None)
            if not failed:
                failed = True
                raise failure
            return await original_read(callback)

        monkeypatch.setattr(state, "read", read)
        candidate = ((PersistenceDomain.EVALUATION, target),)
        with pytest.raises(RuntimeError) as raised:
            await storage.purge_unreferenced_objects(candidate, exclusive=Offline())
        assert raised.value is failure
        reads.clear()
        assert (await storage.purge_unreferenced_objects(candidate, exclusive=Offline())).retained == candidate
        repeated_reference_reads = len(reads)
        single = replace(_record(1), storage_revision=1, data={"captures": [encode_domain(ref)]})
        assert await storage.artifact.records.state_store.mutate(lambda tx: tx.replace_record(
            single, expected_storage_revision=0))
        reads.clear()
        assert (await storage.purge_unreferenced_objects(candidate, exclusive=Offline())).retained == candidate
        assert 0 < len(reads) == repeated_reference_reads
        changed = replace(_record(1), storage_revision=2, data={"captures": [encode_domain(ref)] * 15
                          + [encode_domain(replace(ref, digest="0" * 64))]})
        assert await storage.artifact.records.state_store.mutate(lambda tx: tx.replace_record(
            changed, expected_storage_revision=1))
        with pytest.raises(AIError) as mismatch:
            await storage.purge_unreferenced_objects(candidate, exclusive=Offline())
        assert mismatch.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        await publish_capture_record(state, namespace="cleanup", tenant_id="tenant",
            key=input_capture_expiry_key(capture_key), value={
                "namespace": "cleanup", "tenant_id": "tenant", "capture_key": capture_key,
                "expired_at": NOW.isoformat(),
            })
        reads.clear()
        assert not (await storage.purge_unreferenced_objects(candidate, exclusive=Offline())).retained
        assert reads
    finally:
        await storage.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_cleanup_expiry_metadata_does_not_cross_state_instances() -> None:
    from linktools.ai.persistence.captures import input_capture_expiry_key

    first, second = InMemoryStateStore(), InMemoryStateStore()
    await first.initialize()
    await second.initialize()
    objects = InMemoryObjectStore()
    target = await put(objects, "target", {"content": "retained"})
    key = input_capture_key("cleanup", "tenant", "agent", "capture")
    payload = {"kind": "agent", "version": 1, "namespace": "cleanup", "tenant_id": "tenant",
               "payload": encode_domain(RuntimePayloadRef(StoredPayload.object(target), PersistenceDomain.ARTIFACT))}
    try:
        for state in (first, second):
            await publish_capture_record(state, namespace="cleanup", tenant_id="tenant", key=key, value=payload)
        await publish_capture_record(first, namespace="cleanup", tenant_id="tenant",
            key=input_capture_expiry_key(key), value={
                "namespace": "cleanup", "tenant_id": "tenant", "capture_key": key, "expired_at": NOW.isoformat(),
            })
        candidate = ((PersistenceDomain.ARTIFACT, target),)
        result = await purge_unreferenced_objects(candidate, namespace="cleanup", tenant_id="tenant",
            stores={PersistenceDomain.TASK: first, PersistenceDomain.ARTIFACT: second},
            object_stores={domain: objects for domain in PersistenceDomain}, exclusive=Offline())
        assert result.retained == candidate
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_cleanup_without_deletion_capability_stays_blocked() -> None:
    objects = InMemoryObjectStore()
    orphan = await put(objects, "orphan", "unreferenced")

    class ReadOnlyObjects:
        store_id = objects.store_id
        put = objects.put
        stat = objects.stat
        open = objects.open
        validate_integrity = objects.validate_integrity
        list_objects = objects.list_objects

    sources = {domain: ReadOnlyObjects() for domain in PersistenceDomain}
    candidate = ((PersistenceDomain.EVALUATION, orphan),)
    result = await purge_unreferenced_objects(candidate, namespace="cleanup", tenant_id="tenant",
        stores={}, object_stores=sources, exclusive=Offline())
    assert result.blocked == candidate
    assert result.deleted == result.missing == result.retained == ()
    assert await objects.stat(orphan.key) is not None
