#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Memory scope ownership survives full-length paths and durable round trips."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai_harness.memory import MemoryOperation

from linktools.ai.core import canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.memory import PersistentMemoryStore
from linktools.ai.memory import _store as memory_store
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence._memory import InMemoryStateStorageGroup, InMemoryStateStore
from linktools.ai.persistence._repository_common import domain_data
from linktools.ai.persistence._snapshot_validation import validate_snapshot_domain
from linktools.ai.persistence.codec import _decode_enveloped_domain, decode_envelope, encode_envelope
from linktools.ai.persistence.contracts import MemoryRecord
from linktools.ai.persistence.repositories import MemoryRepositoryImpl
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.persistence.state import SORT_KEY_MAX_BYTES, scope_key
from linktools.ai.storage import InMemoryObjectStore, StoredPayload


def _store(state: Persistence, scope: str, execution: str | None = None) -> PersistentMemoryStore:
    return PersistentMemoryStore(
        state.memory, object_store=state.object_store(PersistenceDomain.MEMORY),
        namespace=state.namespace, tenant_id=state.tenant_id,
        execution_id=execution or "execution", memory_scope=scope, transient=execution is not None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_memory_maximum_scope_and_path_preserve_capture_replay(
    tmp_path: Path, backend: str,
) -> None:
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "memory.db")
    namespace, tenant, scope, execution = "n" * 256, "t" * 128, "é" * 128, "e" * 128
    path = ("x" * 199 + "/") * 5 + "x" * (SORT_KEY_MAX_BYTES - 1000)
    assert len(path.encode()) == 1152 and len(("capture:" + path).encode()) == 1160
    async with Persistence.open(namespace, tenant_id=tenant, storage=storage) as state:
        persistent, transient, other = _store(state, scope), _store(state, scope, execution), _store(state, scope, "other")
        operation = MemoryOperation("create", canonical_sha256({"content": "note"}))
        created = await persistent.write(path, "note", expected_version=None, operation=operation)
        capture = await persistent.capture()
        assert await transient.read(path, max_chars=100) is None
        assert await other.capture() == {}
        await transient.seed_capture(capture)
        await transient.seed_capture(capture)
        restored = await transient.read(path, max_chars=100)
        assert restored is not None and restored.version == created.version
        assert await transient.list_paths(path[:200], limit=1) == [path]
        records = await state.memory.records.capture_scope(
            tenant_id=tenant, memory_scope=scope, transient_execution_id=execution,
        )
        assert len(records) == 1
        assert (records[0].memory_scope, records[0].transient_execution_id) == (scope, execution)
        updated = await transient.write(path, "changed", expected_version=restored.version, operation=operation)
        assert updated.version != created.version
        assert updated.version == "m1:" + canonical_sha256({
            "namespace": namespace, "tenant_id": tenant, "scope": scope,
            "transient_execution_id": execution, "path": path, "identity": operation.id,
        })
        await transient.seed_capture(capture)
        assert (await transient.read(path, max_chars=100)).content == "changed"
        await transient.delete(path, expected_version=updated.version)
        await transient.seed_capture(capture)
        assert await transient.read(path, max_chars=100) is None
        assert (await persistent.read(path, max_chars=100)).content == "note"
        assert await other.get_operation(operation) is None
        assert await other.list_paths(limit=10) == []


@pytest.mark.asyncio
async def test_memory_global_scope_isolates_shared_bare_state_store() -> None:
    state = InMemoryStateStore(InMemoryStateStorageGroup())
    await state.initialize()
    now = datetime.now(timezone.utc)
    try:
        owners = (("namespace", "tenant", "é", None), ("other", "tenant", "é", None),
                  ("namespace", "other", "é", None), ("namespace", "tenant", "e\u0301", None),
                  ("namespace", "tenant", "é", "execution"))
        for namespace, tenant, scope, execution in owners:
            repository = MemoryRepositoryImpl(state, namespace=namespace, tenant_id=tenant)
            value = MemoryRecord(
                memory_store._memory_id(scope, "memory/note", execution), scope,
                StoredPayload.inline_text(namespace + tenant + scope), {"path": "memory/note"},
                1, now, now, transient_execution_id=execution,
            )
            await state.mutate(lambda tx: tx.insert_record(repository._stored("memory", (value.memory_id,), value)))
        for namespace, tenant, scope, execution in owners:
            repository = MemoryRepositoryImpl(state, namespace=namespace, tenant_id=tenant)
            values = await repository.capture_scope(tenant_id=tenant, memory_scope=scope, transient_execution_id=execution)
            assert len(values) == 1
            assert values[0].content.decode() == namespace + tenant + scope
            assert values[0].transient_execution_id == execution
    finally:
        await state.close()


def _record() -> MemoryRecord:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return MemoryRecord("memory", "private-scope", StoredPayload.inline_text("note"),
                        {"path": "memory/note"}, 1, now, now, transient_execution_id="private-execution")


@pytest.mark.parametrize("field,value", (("memory_scope", "x" * 257),
                                        ("transient_execution_id", "x" * 129),
                                        ("memory_scope", ""), ("transient_execution_id", "")))
def test_memory_scope_codec_and_snapshot_classify_invalid_owner_as_integrity(
    field: str, value: str,
) -> None:
    record = _record()
    with pytest.raises(ValueError):
        replace(record, **{field: value})
    encoded = domain_data(record)
    assert _decode_enveloped_domain(encoded, MemoryRecord) == record
    payload = deepcopy(decode_envelope(encoded).value)
    payload["payload"]["fields"][field] = value
    corrupt = encode_envelope(payload)
    with pytest.raises(AIError) as decoded:
        _decode_enveloped_domain(corrupt, MemoryRecord)
    assert decoded.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    repository = MemoryRepositoryImpl(InMemoryStateStore(InMemoryStateStorageGroup()), namespace="namespace", tenant_id="tenant")
    stored = replace(repository._stored("memory", (record.memory_id,), record), data=corrupt)
    with pytest.raises(AIError) as restored:
        validate_snapshot_domain(namespace="namespace", tenant_id="tenant", domain=PersistenceDomain.MEMORY,
                                 records=(stored,), aliases=(), facts=(), operations=(), sequences={})
    assert restored.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ("scope", "execution", "index"))
async def test_memory_repository_rejects_mismatched_scope_projection(change: str) -> None:
    repository = MemoryRepositoryImpl(InMemoryStateStore(InMemoryStateStorageGroup()), namespace="namespace", tenant_id="tenant")
    record = _record()
    stored = repository._stored("memory", (record.memory_id,), record)
    if change == "index":
        stored = replace(stored, scope_key=scope_key("other", "tenant", "memory", "memory", "memory_scope", ("persistent", "scope")))
    else:
        changed = replace(record, **({"memory_scope": "other"} if change == "scope" else {"transient_execution_id": "other"}))
        stored = replace(stored, data=domain_data(changed))
    with pytest.raises(AIError) as raised:
        await repository._decode(stored, MemoryRecord)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_memory_scope_snapshot_restores_current_records_and_receipts(tmp_path: Path) -> None:
    storage = PersistenceStorage.from_root(tmp_path / "source")
    operation = MemoryOperation("write", "a" * 64)
    async with Persistence.open("memory-snapshot", tenant_id="tenant", storage=storage) as state:
        source = _store(state, "é", "execution")
        await source.write("memory/note", "note", expected_version=None, operation=operation)
        expected = await source.read("memory/note", max_chars=100)
    archive = InMemoryObjectStore()
    limits = SnapshotLimits(max_entries=1000, max_bytes=1024 * 1024)
    async with Persistence.open("memory-snapshot", tenant_id="tenant", storage=storage, read_only=True) as state:
        snapshot = await state.export_snapshot(object_store=archive, limits=limits)
    root = tmp_path / "restored"
    await Persistence.restore_snapshot(snapshot, object_store=archive, root=root, limits=limits)
    async with Persistence.open("memory-snapshot", tenant_id="tenant", storage=PersistenceStorage.from_root(root)) as state:
        restored = _store(state, "é", "execution")
        assert await restored.read("memory/note", max_chars=100) == expected
        assert (await restored.get_operation(operation)).replayed
        assert await _store(state, "é").capture() == {}
        assert await _store(state, "é", "other").capture() == {}


@pytest.mark.asyncio
async def test_memory_scope_is_private_in_record_repr_and_mutation_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    record = _record()
    assert record.memory_scope not in repr(record) and record.transient_execution_id not in repr(record)
    logs: list[str] = []
    monkeypatch.setattr(memory_store._logger, "debug", lambda message, *args: logs.append(message % args))
    async with Persistence.open("memory-privacy", storage=PersistenceStorage.in_memory()) as state:
        store = _store(state, record.memory_scope, record.transient_execution_id)
        written = await store.write("memory/note", "note", expected_version=None)
        await store.delete("memory/note", expected_version=written.version)
    assert logs
    assert all(record.memory_scope not in line and record.transient_execution_id not in line for line in logs)


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ({"memory_scope": "x" * 257},
                                   {"memory_scope": "scope", "transient_execution_id": "x" * 129}))
async def test_memory_scope_queries_validate_raw_owner_before_storage(owner: dict[str, str]) -> None:
    repository = MemoryRepositoryImpl(InMemoryStateStore(InMemoryStateStorageGroup()), namespace="namespace", tenant_id="tenant")
    with pytest.raises(ValueError):
        await repository.capture_scope(tenant_id="tenant", **owner)
    with pytest.raises(ValueError):
        await repository.list(tenant_id="tenant", prefix="", cursor=None, limit=1, **owner)


def test_memory_scope_codec_uses_only_declared_optional_execution_default() -> None:
    record = replace(_record(), transient_execution_id=None)
    payload = deepcopy(decode_envelope(domain_data(record)).value)
    del payload["payload"]["fields"]["transient_execution_id"]
    assert _decode_enveloped_domain(encode_envelope(payload), MemoryRecord) == record
    del payload["payload"]["fields"]["memory_scope"]
    with pytest.raises(AIError) as raised:
        _decode_enveloped_domain(encode_envelope(payload), MemoryRecord)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
