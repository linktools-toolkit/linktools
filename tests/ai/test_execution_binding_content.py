#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared bindings preserve complete recovery contracts and archive ownership."""

from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import canonical_json_bytes, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain
from linktools.ai.persistence._archive import ContentRecord
from linktools.ai.persistence._archive_closure import iter_archive_content_refs, rekey_archive_roots
from linktools.ai.persistence._execution_binding import expand_snapshot_binding
from linktools.ai.persistence._payload import encode_payload_bytes
from linktools.ai.persistence._repository_common import project_record
from linktools.ai.persistence.codec import _decode_enveloped_domain, iter_runtime_object_refs
from linktools.ai.persistence.contracts import ExecutionRecord, RuntimePayloadRef
from linktools.ai.persistence.state import RecordQuery
from linktools.ai.storage import StoredPayload

from .test_incremental_projection_boundaries import _storage
from .test_runtime_persistence_evidence import _execution


def _large_execution():
    execution = _execution()
    binding = replace(execution.binding, output_schema={"type": "string", "description": "schema " * 20000})
    return replace(execution, binding=binding)


def _shared_record():
    execution = _large_execution()
    record = project_record(namespace="bindings", tenant_id="tenant", domain=PersistenceDomain.EXECUTION,
                            kind="execution", identity=execution.execution_id, value=execution)
    data = record.data
    binding = data["value"]["payload"]["fields"]["binding"]
    raw = canonical_json_bytes(binding)
    compression, digest, size, encoded = encode_payload_bytes(raw)
    content = ContentRecord("execution-binding", digest, size, compression, StoredPayload.inline_bytes(encoded))
    scope = canonical_sha256(["bindings", "tenant"])
    reference = {"$execution_binding": content.identity, "archive_scope": "source", "content_scope": scope}
    compact = replace(record, data={**data, "value": {**data["value"], "payload": {
        **data["value"]["payload"], "fields": {**data["value"]["payload"]["fields"], "binding": reference},
    }}})
    return execution, record, compact, content, ("source", scope, content.identity)


def test_complete_binding_snapshot_round_trip_and_scope_remap() -> None:
    execution, full, compact, content, key = _shared_record()
    assert len(canonical_json_bytes(compact.data)) < len(canonical_json_bytes(full.data)) // 10
    assert tuple(iter_archive_content_refs(compact.data)) == (key,)
    restored = expand_snapshot_binding(compact, {key: content}, namespace="bindings", tenant_id="tenant")
    assert restored == full
    assert _decode_enveloped_domain(restored.data, ExecutionRecord).binding.to_payload() == execution.binding.to_payload()
    remapped = replace(compact, data=rekey_archive_roots(dict(compact.data), {"source": "target"}))
    target_key = ("target", *key[1:])
    assert tuple(iter_archive_content_refs(remapped.data)) == (target_key,)
    assert expand_snapshot_binding(remapped, {target_key: content}, namespace="bindings", tenant_id="tenant") == full


def test_binding_archive_reference_does_not_hide_runtime_object_references() -> None:
    from linktools.ai.persistence.codec import encode_domain
    from linktools.ai.storage import ObjectRef

    _, full, compact, _, _ = _shared_record()
    reference = ObjectRef("payloads", "execution/result", "a" * 64, 32)
    for record in (full, compact):
        data = dict(record.data)
        data["value"]["payload"]["fields"]["repository_instructions"] = encode_domain(
            RuntimePayloadRef(StoredPayload.object(reference), PersistenceDomain.TASK),
        )
        assert tuple(iter_runtime_object_refs(data, default_domain=PersistenceDomain.EXECUTION)) == (
            (PersistenceDomain.TASK, reference),
        )


@pytest.mark.parametrize("field,value", (("$execution_binding", "invalid"), ("archive_scope", ""),
                                         ("content_scope", "invalid"), ("unexpected", True)))
def test_runtime_object_scan_rejects_malformed_binding_reference(field: str, value: object) -> None:
    _, _, compact, _, _ = _shared_record()
    data = dict(compact.data)
    data["value"]["payload"]["fields"]["binding"][field] = value
    with pytest.raises(AIError) as error:
        tuple(iter_runtime_object_refs(data, default_domain=PersistenceDomain.EXECUTION))
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("corruption", ("missing", "body", "codec", "tenant"))
def test_binding_snapshot_rejects_missing_or_corrupt_content(corruption: str) -> None:
    _, _, compact, content, key = _shared_record()
    contents = {key: content}
    if corruption == "missing":
        contents = {}
    elif corruption == "body":
        contents[key] = replace(content, payload=StoredPayload.inline_bytes(b"corrupt"))
    elif corruption == "codec":
        contents[key] = replace(content, codec="different")
    with pytest.raises(AIError) as error:
        expand_snapshot_binding(compact, contents, namespace="bindings",
                                tenant_id="other" if corruption == "tenant" else "tenant")
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_execution_bindings_share_content_and_survive_cold_reads(tmp_path: Path, backend: str) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="bindings", tenant_id="tenant")
    try:
        repository = state.execution.executions
        execution = _large_execution()
        other = replace(execution, execution_id="other", root_execution_id="other")
        await repository.create_execution(execution)
        await repository.create_execution(other)
        records = await repository.state_store.read(lambda tx: tx.list_records(RecordQuery(kind="execution")))
        refs = [tuple(iter_archive_content_refs(record.data)) for record in records]
        assert len(records) == 2 and refs[0] == refs[1] and len(refs[0]) == 1
        _, archive = await repository.state_store.storage_group.open_archive()
        contents = await archive.read(lambda tx: tx.list_records(RecordQuery(kind="archive_content")))
        assert len([record for record in contents if record.data["codec"] == "execution-binding"]) == 1
        from linktools.ai.persistence._execution_repositories import ExecutionRepositoryImpl
        cold = ExecutionRepositoryImpl(repository.state_store, namespace="bindings", tenant_id="tenant")
        await cold.initialize_content(create=False)
        assert (await repository.state_store.read(lambda tx: cold.get_in_transaction(
            tx, "execution", tenant_id="tenant"))).binding.to_payload() == execution.binding.to_payload()
        bare = ExecutionRepositoryImpl(repository.state_store, namespace="bindings", tenant_id="tenant")
        assert await repository.state_store.read(
            lambda tx: bare.get_in_transaction(tx, "execution", tenant_id="tenant"),
        ) == execution
        assert await bare.get("execution", tenant_id="tenant") == execution
        inline = replace(execution, execution_id="inline", root_execution_id="inline")
        await bare.create_execution(inline)
        rows = await repository.state_store.read(lambda tx: tx.list_records(RecordQuery(kind="execution")))
        inline_row = next(row for row in rows if row.key_digest == repository.execution_record_key("inline"))
        assert not tuple(iter_archive_content_refs(inline_row.data))
        changed = replace(execution, revision=execution.revision + 1, error_code="diagnostic")
        await repository.compare_and_swap("execution", tenant_id="tenant",
                                          expected_revision=execution.revision, next_record=changed)
        assert await repository.get_fresh("execution", tenant_id="tenant") == changed
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("external_transaction", (False, True))
async def test_uninitialized_repository_keeps_complete_binding_inline(external_transaction: bool) -> None:
    from linktools.ai.persistence._execution_repositories import ExecutionRepositoryImpl
    from linktools.ai.persistence._memory import InMemoryStateStorageGroup, InMemoryStateStore

    group = InMemoryStateStorageGroup()
    store = InMemoryStateStore(group)
    await store.initialize()
    repository = ExecutionRepositoryImpl(store, namespace="bindings", tenant_id="tenant")
    try:
        execution = _large_execution()
        if external_transaction:
            await store.mutate(lambda tx: repository.create_execution_in_transaction(tx, execution))
        else:
            await repository.create_execution(execution)
        assert await repository.get("execution", tenant_id="tenant") == execution
        cold = ExecutionRepositoryImpl(store, namespace="bindings", tenant_id="tenant")
        assert await store.read(lambda tx: cold.get_in_transaction(tx, "execution", tenant_id="tenant")) == execution
        assert await group.open_archive() is None
        records = await store.read(lambda tx: tx.list_records(RecordQuery(kind="execution")))
        assert not tuple(iter_archive_content_refs(records[0].data))
    finally:
        await store.close()
        await group.close()


@pytest.mark.asyncio
async def test_binding_and_execution_publish_in_one_transaction(tmp_path: Path) -> None:
    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="bindings", tenant_id="tenant")
    try:
        repository = state.execution.executions
        execution = _large_execution()
        await repository.initialize_content(create=True)
        async def abort(transaction):
            await repository.create_execution_in_transaction(transaction, execution)
            raise ValueError("abort owner publication")
        with pytest.raises(ValueError, match="abort owner"):
            await repository.state_store.mutate(abort)
        assert await repository.get("execution", tenant_id="tenant") is None
        _, archive = await repository.state_store.storage_group.open_archive()
        contents = await archive.read(lambda tx: tx.list_records(RecordQuery(kind="archive_content")))
        assert not contents
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_snapshot_capture_reads_shared_bindings_in_batches(tmp_path: Path, monkeypatch) -> None:
    from linktools.ai.persistence import snapshots
    from linktools.ai.persistence._archive import ContentArchive
    from linktools.ai.storage import InMemoryObjectStore, ObjectRef

    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="bindings", tenant_id="tenant")
    try:
        repository = state.execution.executions
        execution = _large_execution()
        for identity in ("first", "second", "third"):
            await repository.create_execution(replace(execution, execution_id=identity, root_execution_id=identity))
        batches = []
        read_contents = ContentArchive.read_contents

        async def read_batch(self, identities, *, codec, reserve=None):
            if codec == "execution-binding":
                batches.append(tuple(identities))
            return await read_contents(self, identities, codec=codec, reserve=reserve)

        expected = ObjectRef("snapshot", "captured", "a" * 64, 0)

        async def accept_capture(source, **kwargs):
            records = source.readers[PersistenceDomain.EXECUTION].records
            assert len(records) == 3
            assert all(tuple(iter_archive_content_refs(record.data)) for record in records)
            return expected

        monkeypatch.setattr(ContentArchive, "read_contents", read_batch)
        monkeypatch.setattr(snapshots, "_export_snapshot", accept_capture)
        objects = InMemoryObjectStore()
        resources = snapshots.SnapshotResources(
            namespace="bindings", tenant_id="tenant", read_only=True,
            durable_domains=frozenset({PersistenceDomain.EXECUTION}),
            stores={PersistenceDomain.EXECUTION: repository.state_store},
            object_stores={PersistenceDomain.EXECUTION: objects},
        )
        result = await snapshots.RuntimeSnapshotOperations(resources, require_ready=lambda: None).export_snapshot(
            object_store=objects, limits=snapshots.SnapshotLimits(100, 2_000_000),
        )
        assert result == expected
        # One shared body is hydrated once for the capture batch, not per execution.
        assert len(batches) == 1 and len(batches[0]) == 1
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_archive_handle_without_transaction_enlistment_keeps_binding_inline(monkeypatch) -> None:
    from types import SimpleNamespace
    from linktools.ai.persistence import _execution_binding as binding_module
    from linktools.ai.persistence._execution_binding import ExecutionBindingContent
    from linktools.ai.persistence._memory import InMemoryStateStorageGroup, InMemoryStateStore

    group = InMemoryStateStorageGroup()
    store = InMemoryStateStore(group)
    await store.initialize()
    content = ExecutionBindingContent(store, namespace="bindings", tenant_id="tenant")
    await content.initialize(create=True)
    try:
        _, record, _, _, _ = _shared_record()
        # The group handle exists, but this earlier captured callback has no
        # transaction for it. No archive read or write may be attempted.
        monkeypatch.setattr(binding_module, "active_state_scope",
                            lambda: SimpleNamespace(group=group, transactions={store: object()}))
        assert await content.compact(record) == record
        unbound = ExecutionBindingContent(store, namespace="bindings", tenant_id="tenant")
        unbound.enable_sharing()
        assert await unbound.compact(record) == record
        _, _, compact, _, _ = _shared_record()
        with pytest.raises(AIError) as error:
            await unbound.expand(compact)
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        monkeypatch.undo()
        await store.close()
        await group.close()


def _binding_manifest(copies: int):
    from linktools.ai.persistence.codec import encode_record
    from linktools.ai.persistence.state import record_key_digest

    _, _, compact, content, key = _shared_record()
    archive_record = replace(compact, kind="archive_content", sort_key=key[2],
        scope_digest=bytes.fromhex(key[1]), data=content.to_json(),
        key_digest=record_key_digest("", "", "archive", "archive_content", [key[1], key[2]]))
    manifest = {
        "kind": "runtime-storage-snapshot", "format_version": 1,
        "namespace": "bindings", "tenant_id": "tenant",
        "domains": {"execution": {"records": [encode_record(compact) for _ in range(copies)],
                                  "facts": [], "operations": [], "aliases": [], "sequences": []}},
        "archives": [{"scope_id": key[0], "domain": "execution", "records": [encode_record(archive_record)]}],
        "objects": [],
    }
    return manifest, compact, content, key


@pytest.mark.parametrize("copies", (1, 3))
def test_snapshot_binding_budget_counts_each_retained_execution(copies: int) -> None:
    from linktools.ai.persistence.snapshots import snapshot_manifest_usage
    from linktools.ai.snapshot._snapshot import _storage_snapshot_usage

    manifest, compact, content, key = _binding_manifest(copies)
    expected = copies + 1, copies * content.raw_size
    assert snapshot_manifest_usage(manifest) == expected
    assert _storage_snapshot_usage(manifest) == expected
    # Identical content is persisted once but each execution retains its decoded binding.
    assert content.compression == "zlib"
    assert len(canonical_json_bytes(manifest)) < copies * content.raw_size


def test_snapshot_binding_reserves_before_decompression(monkeypatch) -> None:
    from linktools.ai.persistence import _execution_binding as bindings
    from linktools.ai.persistence.snapshots import SnapshotLimits, _SnapshotCaptureBudget

    _, compact, content, key = _binding_manifest(1)
    decode = bindings.decode_payload_bytes
    calls = []

    def decode_content(*args, **kwargs):
        calls.append(True)
        return decode(*args, **kwargs)

    monkeypatch.setattr(bindings, "decode_payload_bytes", decode_content)
    budget = _SnapshotCaptureBudget(SnapshotLimits(100, content.raw_size - 1))
    with pytest.raises(AIError) as error:
        expand_snapshot_binding(compact, {key: content}, namespace="bindings", tenant_id="tenant",
                                reserve=budget.reserve_bytes)
    assert error.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED
    assert not calls
    budget = _SnapshotCaptureBudget(SnapshotLimits(100, content.raw_size))
    restored = expand_snapshot_binding(compact, {key: content}, namespace="bindings", tenant_id="tenant",
                                       reserve=budget.reserve_bytes)
    assert restored == _shared_record()[1]
    assert budget.byte_size == content.raw_size
    assert calls == [True]


@pytest.mark.asyncio
async def test_snapshot_restore_rejects_binding_budget_before_hydration(tmp_path: Path, monkeypatch) -> None:
    from linktools.ai.persistence import snapshots
    from linktools.ai.storage import InMemoryObjectStore, ObjectRef

    manifest, _, content, _ = _binding_manifest(3)
    raw = canonical_json_bytes(manifest)
    objects = InMemoryObjectStore()

    async def chunks():
        yield raw

    stat = await objects.put("snapshot", chunks())

    def reject_hydration(*args, **kwargs):
        raise AssertionError("binding hydration must follow the aggregate budget check")

    def reject_target(*args, **kwargs):
        raise AssertionError("an over-budget snapshot must not open a target")

    monkeypatch.setattr(snapshots, "expand_snapshot_binding", reject_hydration)
    with pytest.raises(AIError) as error:
        await snapshots.restore_snapshot(ObjectRef(objects.store_id, stat.key, stat.digest, stat.size),
            object_store=objects, root=tmp_path / "restore", open_target=reject_target,
            limits=snapshots.SnapshotLimits(100, len(raw) + 3 * content.raw_size - 1))
    assert error.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED


@pytest.mark.asyncio
async def test_snapshot_capture_rejects_binding_budget_before_hydration(tmp_path: Path, monkeypatch) -> None:
    from linktools.ai.persistence import snapshots
    from linktools.ai.persistence._archive import ContentArchive
    from linktools.ai.storage import InMemoryObjectStore

    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="bindings", tenant_id="tenant")
    try:
        repository = state.execution.executions
        await repository.create_execution(_large_execution())
        _, _, content, _ = _binding_manifest(1)

        async def reject_hydration(*args, **kwargs):
            raise AssertionError("binding hydration must follow the capture budget check")

        monkeypatch.setattr(ContentArchive, "_read_content", reject_hydration)
        objects = InMemoryObjectStore()
        resources = snapshots.SnapshotResources(
            namespace="bindings", tenant_id="tenant", read_only=True,
            durable_domains=frozenset({PersistenceDomain.EXECUTION}),
            stores={PersistenceDomain.EXECUTION: repository.state_store},
            object_stores={PersistenceDomain.EXECUTION: objects},
        )
        with pytest.raises(AIError) as error:
            await snapshots.RuntimeSnapshotOperations(resources, require_ready=lambda: None).export_snapshot(
                object_store=objects, limits=snapshots.SnapshotLimits(100, content.raw_size - 1))
        assert error.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_small_bindings_stay_inline_without_archive_hydration(tmp_path: Path, monkeypatch, backend: str) -> None:
    from linktools.ai.persistence._archive import ContentArchive
    from linktools.ai.storage import PayloadPolicy

    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="bindings", tenant_id="tenant")
    try:
        repository = state.execution.executions
        assert repository.state_store.storage_group.opened_archive is None
        execution = _execution()
        stored = project_record(namespace="bindings", tenant_id="tenant", domain=PersistenceDomain.EXECUTION,
                                kind="execution", identity=execution.execution_id, value=execution)
        assert len(canonical_json_bytes(stored.data["value"]["payload"]["fields"]["binding"])) <= PayloadPolicy().inline_limit_bytes

        async def reject_archive(*args, **kwargs):
            raise AssertionError("small execution metadata must not hydrate shared bindings")

        monkeypatch.setattr(ContentArchive, "read_contents", reject_archive)
        assert await repository.create_execution(execution) == execution
        records = await repository.state_store.read(lambda tx: tx.list_records(RecordQuery(kind="execution")))
        assert len(records) == 1 and not tuple(iter_archive_content_refs(records[0].data))
        assert await repository.get_fresh(execution.execution_id, tenant_id="tenant") == execution
        changed = replace(execution, revision=execution.revision + 1)
        assert await repository.compare_and_swap(execution.execution_id, tenant_id="tenant",
                                                expected_revision=execution.revision, next_record=changed) == changed
        assert await repository.get(execution.execution_id, tenant_id="tenant") == changed
    finally:
        await state.close()
