#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared bindings preserve complete recovery contracts and archive ownership."""

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain
from linktools.ai.persistence._archive import ContentRecord, archive_content_scope
from linktools.ai.persistence._archive_closure import iter_archive_content_refs, rekey_archive_roots
from linktools.ai.persistence._execution_content import expand_snapshot_execution_content
from linktools.ai.persistence._payload import encode_payload_bytes
from linktools.ai.persistence._repository_common import project_record
from linktools.ai.persistence.codec import _decode_enveloped_domain, _execution_binding_summary, iter_runtime_object_refs
from linktools.ai.persistence.contracts import ExecutionRecord, RuntimePayloadRef
from linktools.ai.persistence.state import RecordQuery, StoredRecord, active_state_scope
from linktools.ai.storage import StoredPayload

from .test_incremental_projection_boundaries import _storage
from .test_runtime_persistence_evidence import _execution


def _large_execution() -> ExecutionRecord:
    execution = _execution()
    binding = replace(execution.binding, output_schema={"type": "string", "description": "schema " * 20000})
    return replace(execution, binding=binding)


def _shared_record() -> tuple[ExecutionRecord, StoredRecord, StoredRecord, ContentRecord, tuple[str, str, str]]:
    execution = _large_execution()
    record = project_record(namespace="bindings", tenant_id="tenant", domain=PersistenceDomain.EXECUTION,
                            kind="execution", identity=(execution.execution_id,), value=execution)
    data = record.data
    binding = data["value"]["payload"]["fields"]["binding"]
    raw = canonical_json_bytes(binding)
    compression, encoded = encode_payload_bytes(raw)
    digest, size = hashlib.sha256(raw).hexdigest(), len(raw)
    content = ContentRecord("execution-binding", digest, size, compression, body=encoded)
    scope = archive_content_scope("bindings", "tenant")
    reference = {"$execution_binding": content.identity, "archive_scope": "source", "content_scope": scope}
    compact = replace(record, data={**data, "value": {**data["value"], "payload": {
        **data["value"]["payload"], "fields": {**data["value"]["payload"]["fields"], "binding": reference},
    }}})
    return execution, record, compact, content, ("source", scope, content.identity)


def test_complete_binding_snapshot_round_trip_and_scope_remap() -> None:
    execution, full, compact, content, key = _shared_record()
    assert len(canonical_json_bytes(compact.data)) < len(canonical_json_bytes(full.data)) // 10
    assert tuple(iter_archive_content_refs(compact.data)) == (key,)
    restored = expand_snapshot_execution_content(compact, {key: content}, namespace="bindings", tenant_id="tenant")
    assert restored == full
    assert _decode_enveloped_domain(restored.data, ExecutionRecord).binding.to_payload() == execution.binding.to_payload()
    remapped = replace(compact, data=rekey_archive_roots(dict(compact.data), {"source": "target"}))
    target_key = ("target", *key[1:])
    assert tuple(iter_archive_content_refs(remapped.data)) == (target_key,)
    assert expand_snapshot_execution_content(remapped, {target_key: content}, namespace="bindings", tenant_id="tenant") == full


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
        contents[key] = replace(content, body=b"corrupt")
    elif corruption == "codec":
        contents[key] = replace(content, codec="different")
    with pytest.raises(AIError) as error:
        expand_snapshot_execution_content(compact, contents, namespace="bindings",
                                tenant_id="other" if corruption == "tenant" else "tenant")
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_execution_contents_share_content_and_survive_cold_reads(tmp_path: Path, backend: str) -> None:
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
        assert len(records) == 2 and refs[0] == refs[1]
        assert len([ref for ref in refs[0] if ref[2].startswith("execution-binding:")]) == 1
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
        inline_row = next(row for row in rows if row.key == repository.execution_record_key("inline"))
        assert any(ref[2].startswith("execution-binding:") for ref in iter_archive_content_refs(inline_row.data))
        assert "summary" not in inline_row.data["value"]["payload"]["fields"]["binding"]
        changed = replace(execution, revision=execution.revision + 1, error_code="diagnostic")
        await repository.compare_and_swap("execution", tenant_id="tenant",
                                          expected_revision=execution.revision, next_record=changed)
        assert await repository.get_fresh("execution", tenant_id="tenant") == changed
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("external_transaction", (False, True))
async def test_uninitialized_repository_prepares_fixed_binding_before_publication(external_transaction: bool) -> None:
    from linktools.ai.persistence._execution_repositories import ExecutionRepositoryImpl
    from linktools.ai.persistence._memory import InMemoryStateStorageGroup, InMemoryStateStore

    group = InMemoryStateStorageGroup()
    store = InMemoryStateStore(group)
    await store.initialize()
    repository = ExecutionRepositoryImpl(store, namespace="bindings", tenant_id="tenant")
    try:
        execution = _large_execution()
        if external_transaction:
            with pytest.raises(AIError) as unprepared:
                await store.mutate(lambda tx: repository.create_execution_in_transaction(tx, execution))
            assert unprepared.value.code is ErrorCode.STORAGE_DEPENDENCY_NOT_READY
            assert await repository.get("execution", tenant_id="tenant") is None
            await repository.prepare_content(execution)
            await store.mutate(lambda tx: repository.create_execution_in_transaction(tx, execution))
        else:
            await repository.create_execution(execution)
        assert await repository.get("execution", tenant_id="tenant") == execution
        cold = ExecutionRepositoryImpl(store, namespace="bindings", tenant_id="tenant")
        assert await store.read(lambda tx: cold.get_in_transaction(tx, "execution", tenant_id="tenant")) == execution
        assert await group.open_archive() is not None
        records = await store.read(lambda tx: tx.list_records(RecordQuery(kind="execution")))
        assert any(ref[2].startswith("execution-binding:") for ref in iter_archive_content_refs(records[0].data))
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
    from linktools.ai.persistence import _execution_content as binding_module
    from linktools.ai.persistence._execution_content import ExecutionContent
    from linktools.ai.persistence._memory import InMemoryStateStorageGroup, InMemoryStateStore

    group = InMemoryStateStorageGroup()
    store = InMemoryStateStore(group)
    await store.initialize()
    content = ExecutionContent(store, namespace="bindings", tenant_id="tenant")
    await content.initialize(create=True)
    try:
        execution, record, _, _, _ = _shared_record()
        # The group handle exists, but this earlier captured callback has no
        # transaction for it. No archive read or write may be attempted.
        monkeypatch.setattr(binding_module, "active_state_scope",
                            lambda: SimpleNamespace(group=group, transactions={store: object()}, writable=False))
        assert await content.encode(execution) == record.data["value"]["payload"]
        unbound = ExecutionContent(store, namespace="bindings", tenant_id="tenant")
        unbound.enable_sharing()
        assert await unbound.encode(execution) == record.data["value"]["payload"]
        _, _, compact, _, _ = _shared_record()
        with pytest.raises(AIError) as error:
            await unbound.expand(compact)
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        monkeypatch.undo()
        await store.close()
        await group.close()


def _binding_manifest(copies: int):
    from linktools.ai.persistence._archive import archive_scope_parts, decode_archive_scope
    from linktools.ai.persistence.codec import encode_record
    from linktools.ai.persistence.state import record_key

    _, _, compact, content, key = _shared_record()
    scope = decode_archive_scope(key[1])
    namespace, tenant_id = archive_scope_parts(scope)
    archive_record = replace(compact, kind="archive_content", sort_key=key[2].encode("utf-8"),
        scope_key=scope, data=content.to_json(), payload_bytes=content.body,
        key=record_key(namespace, tenant_id, "archive", "archive_content", (key[2],)))
    assert ContentRecord.from_record(archive_record) == content
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
    from linktools.ai.persistence import _execution_content as bindings
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
        expand_snapshot_execution_content(compact, {key: content}, namespace="bindings", tenant_id="tenant",
                                reserve=budget.reserve_bytes)
    assert error.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED
    assert not calls
    budget = _SnapshotCaptureBudget(SnapshotLimits(100, content.raw_size))
    restored = expand_snapshot_execution_content(compact, {key: content}, namespace="bindings", tenant_id="tenant",
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

    stat = await objects.put("snapshot", chunks(), expected_size=len(raw),
                             expected_digest=hashlib.sha256(raw).hexdigest())

    def reject_hydration(*args, **kwargs):
        raise AssertionError("binding hydration must follow the aggregate budget check")

    def reject_target(*args, **kwargs):
        raise AssertionError("an over-budget snapshot must not open a target")

    monkeypatch.setattr(snapshots, "expand_snapshot_execution_content", reject_hydration)
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
@pytest.mark.parametrize("kind", ("agent", "task"))
async def test_small_bindings_share_complete_content_across_executions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, kind: str,
) -> None:
    from linktools.ai.persistence._archive import ContentArchive
    from linktools.ai.persistence._execution_repositories import ExecutionRepositoryImpl
    from linktools.ai.storage import PayloadPolicy
    from linktools.ai.task import TaskBindingContract

    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="bindings", tenant_id="tenant")
    try:
        repository = state.execution.executions
        execution = _execution()
        schema = {"type": "string", "description": "output contract " * 64}
        if kind == "agent":
            execution = replace(execution, binding=replace(execution.binding, output_schema=schema))
        else:
            execution = replace(execution, mode=None, planning=None, thinking=None, agent_run_seq=0,
                binding=TaskBindingContract("task", 1, "none", schema, None, 3, 0))
        executions = tuple(replace(execution, execution_id=f"execution-{index}", root_execution_id=f"execution-{index}")
                           for index in range(20))
        inline = tuple(project_record(namespace="bindings", tenant_id="tenant", domain=PersistenceDomain.EXECUTION,
            kind="execution", identity=(value.execution_id,), value=value) for value in executions)
        assert len(canonical_json_bytes(inline[0].data["value"]["payload"]["fields"]["binding"])) < PayloadPolicy().inline_limit_bytes
        for value in executions:
            await repository.create_execution(value)
        records = await repository.state_store.read(lambda tx: tx.list_records(RecordQuery(kind="execution")))
        references = [tuple(iter_archive_content_refs(record.data)) for record in records]
        assert len(records) == len(executions) and all(reference == references[0] for reference in references)
        assert len([ref for ref in references[0] if ref[2].startswith("execution-binding:")]) == 1
        _, archive = repository.state_store.storage_group.opened_archive
        archive_records = await archive.read(lambda tx: tx.scan_records())
        contents = [record for record in archive_records if record.kind == "archive_content"
                    and record.data["codec"] == "execution-binding"]
        assert len(contents) == 1 and contents[0].data["codec"] == "execution-binding"

        def record_bytes(values: tuple[StoredRecord, ...]) -> int:
            return sum(len(canonical_json_bytes(record.data)) + len(record.payload_bytes or b"") for record in values)

        # Include archive metadata and binary bodies, not only the compact owner JSON.
        assert record_bytes((*records, *archive_records)) < record_bytes(inline)
        cold = ExecutionRepositoryImpl(repository.state_store, namespace="bindings", tenant_id="tenant")

        body_reads = []
        original_read = ContentArchive._read_content
        async def track_archive(self: ContentArchive, content: ContentRecord) -> bytes:
            body_reads.append(content.codec)
            return await original_read(self, content)

        with monkeypatch.context() as patch:
            patch.setattr(ContentArchive, "_read_content", track_archive)
            assert (await cold.get_view(executions[0].execution_id, tenant_id="tenant"))[1].binding_kind == kind
        assert "execution-binding" in body_reads
        assert set(body_reads) <= {"execution-binding", "runtime-policy"}
        assert await cold.get_fresh(executions[0].execution_id, tenant_id="tenant") == executions[0]
        changed = replace(executions[0], revision=executions[0].revision + 1)
        assert await repository.compare_and_swap(changed.execution_id, tenant_id="tenant",
            expected_revision=executions[0].revision, next_record=changed) == changed
        assert await cold.get_fresh(changed.execution_id, tenant_id="tenant") == changed
        assert len([row for row in await archive.read(lambda tx: tx.list_records(RecordQuery(kind="archive_content")))
                    if row.data["codec"] == "execution-binding"]) == 1

        different = replace(execution.binding, **({"output_schema": {"type": "integer"}} if kind == "agent"
                            else {"output_contract": {"type": "integer"}}))
        variant = replace(execution, execution_id="variant", root_execution_id="variant", binding=different)
        await repository.create_execution(variant)
        assert len([row for row in await archive.read(lambda tx: tx.list_records(RecordQuery(kind="archive_content")))
                    if row.data["codec"] == "execution-binding"]) == 2
        assert await cold.get_fresh("variant", tenant_id="tenant") == variant
    finally:
        await state.close()


def test_binding_capture_detaches_live_declarations_without_freezing_sources() -> None:
    from linktools.ai.spec import CapabilityRef
    source = {"type": "string", "description": "old"}
    selected = [CapabilityRef("capability", "first", 1)]
    old = replace(_large_execution().binding, output_schema=source, selected=selected)
    source["description"] = "new"
    selected.append(CapabilityRef("capability", "second", 1))
    current = replace(old, output_schema=source, selected=selected)
    assert old.output_schema["description"] == "old"
    assert tuple(ref.id for ref in old.selected) == ("first",)
    assert current.output_schema["description"] == "new"
    assert tuple(ref.id for ref in current.selected) == ("first", "second")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("kind", ("agent", "task"))
async def test_archived_binding_views_read_the_fixed_contract_in_the_owner_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, kind: str,
) -> None:
    from linktools.ai.execution import project_execution_view
    from linktools.ai.core import ExecutionLineageKind
    from linktools.ai.persistence._archive import ContentArchive
    from linktools.ai.persistence._execution_repositories import ExecutionRepositoryImpl
    from linktools.ai.task import TaskBindingContract

    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="bindings", tenant_id="tenant")
    try:
        repository = state.execution.executions
        execution = _large_execution()
        if kind == "task":
            execution = replace(execution, mode=None, planning=None, thinking=None, agent_run_seq=0,
                binding=TaskBindingContract("task", 1, "none", execution.binding.output_schema,
                    None, 3, 0), task_attempt=2)
        await repository.create_execution(execution)
        child = None
        if kind == "agent":
            child = replace(execution, execution_id="child", parent_execution_id="execution",
                parent_invocation_id="invocation", lineage_kind=ExecutionLineageKind.SUBAGENT)
            await repository.create_execution(child)
        cold = ExecutionRepositoryImpl(repository.state_store, namespace="bindings", tenant_id="tenant")

        reads = []
        original_read = ContentArchive._read_content
        async def track_contract_read(self: ContentArchive, content: ContentRecord) -> bytes:
            scope = active_state_scope()
            assert scope is not None and not scope.writable
            assert repository.state_store in scope.transactions and self.store in scope.transactions
            reads.append(content.codec)
            return await original_read(self, content)

        with monkeypatch.context() as patch:
            patch.setattr(ContentArchive, "_read_content", track_contract_read)
            expected = project_execution_view(execution)
            assert (await cold.get_view("execution", tenant_id="tenant"))[1] == expected
            assert (await cold.get_view_fresh("execution", tenant_id="tenant"))[1] == expected
            expected_children = () if child is None else (project_execution_view(child),)
            assert await cold.list_child_views("execution", tenant_id="tenant") == expected_children
            parent, children = await cold.get_view_with_children("execution", tenant_id="tenant")
            assert parent[1] == expected and children == expected_children
        assert reads.count("execution-binding") >= 3
        assert set(reads) <= {"execution-binding", "runtime-policy"}
        assert await cold.get("execution", tenant_id="tenant") == execution
        _, archive = repository.state_store.storage_group.opened_archive
        rows = await archive.read(lambda tx: tx.list_records(RecordQuery(kind="archive_content")))
        owner = next(row for row in rows if row.data["codec"] == "execution-binding")
        await archive.mutate(lambda tx: tx.delete_record(owner.key))
        with pytest.raises(AIError) as missing:
            await cold.get_view_fresh("execution", tenant_id="tenant")
        assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


@pytest.mark.parametrize("field,value", (("id", "other"), ("revision", 2),
    ("kind", "task"), ("max_attempts", 3), ("binding_digest", "f" * 64),
    ("capability_refs", [{"invalid": True}])))
def test_binding_reference_rejects_a_redundant_or_inconsistent_summary(field: str, value: object) -> None:
    import json

    execution, _, compact, content, key = _shared_record()
    data = json.loads(canonical_json_bytes(compact.data))
    data["value"]["payload"]["fields"]["binding"]["summary"] = {
        **_execution_binding_summary(execution.binding), field: value}
    compact = replace(compact, data=data)
    with pytest.raises(AIError) as error:
        expand_snapshot_execution_content(compact, {key: content}, namespace="bindings", tenant_id="tenant")
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("kind,identity,revision", (
    ("agent", "bad*id", 1), ("task", "bad/id", 1),
    ("agent", "agent", True), ("task", "task", 0),
))
def test_archived_binding_view_rejects_invalid_declared_identity(kind: str, identity: str, revision: object) -> None:
    import json
    from linktools.ai.persistence.codec import _encode_persisted_domain
    from linktools.ai.task import TaskBindingContract

    execution, _, compact, _, key = _shared_record()
    binding = (execution.binding if kind == "agent" else
               TaskBindingContract("task", 1, "none", {"type": "string"}, None, 3, 0))
    wire = json.loads(canonical_json_bytes(_encode_persisted_domain(binding)))
    declaration = wire["agent_spec"] if kind == "agent" else wire["fields"]
    declaration.update({"id": identity, "revision": revision})
    raw = canonical_json_bytes(wire)
    compression, encoded = encode_payload_bytes(raw)
    content = ContentRecord("execution-binding", hashlib.sha256(raw).hexdigest(), len(raw), compression, body=encoded)
    data = json.loads(canonical_json_bytes(compact.data))
    data["value"]["payload"]["fields"]["binding"]["$execution_binding"] = content.identity
    malformed = replace(compact, data=data)
    reference = (*key[:2], content.identity)
    with pytest.raises(AIError) as error:
        expand_snapshot_execution_content(malformed, {reference: content}, namespace="bindings", tenant_id="tenant")
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_binding_body_is_decoded_once_per_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    from linktools.ai.persistence._archive import ContentArchive

    async with Persistence.open("bindings", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        repository = state.execution.executions
        execution = _large_execution()
        await repository.create_execution(execution)
        calls = []
        original = ContentArchive._read_content

        async def read(self, content):
            if content.codec == "execution-binding":
                calls.append(content.identity)
            return await original(self, content)

        monkeypatch.setattr(ContentArchive, "_read_content", read)

        async def read_twice(transaction):
            first = await repository.get_in_transaction(transaction, "execution", tenant_id="tenant")
            second = await repository.get_in_transaction(transaction, "execution", tenant_id="tenant")
            assert first.binding == second.binding == execution.binding

        await repository.state_store.mutate(read_twice)
        assert len(calls) == 1
        await repository.state_store.mutate(read_twice)
        assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("corruption", ("missing", "body"))
@pytest.mark.parametrize("codec", ("execution-binding", "runtime-policy"))
async def test_configuration_body_reuse_rechecks_current_source(
    tmp_path: Path, backend: str, corruption: str, codec: str,
) -> None:
    async with Persistence.open("bindings", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        repository = state.execution.executions
        await repository.create_execution(_large_execution())
        group = repository.state_store.storage_group
        _, archive = await group.open_archive()

        async def corrupt_after_read(transactions):
            owner = transactions.transaction(repository.state_store)
            await repository.get_in_transaction(owner, "execution", tenant_id="tenant")
            contents = transactions.transaction(archive)
            records = await contents.list_records(RecordQuery(kind="archive_content"))
            content = next(record for record in records if record.data["codec"] == codec)
            if corruption == "missing":
                await contents.delete_record(content.key)
            else:
                await contents.replace_record(replace(content, payload_bytes=b"corrupt",
                    storage_revision=content.storage_revision + 1), expected_storage_revision=content.storage_revision)
            with pytest.raises(AIError) as invalid:
                await repository.get_in_transaction(owner, "execution", tenant_id="tenant")
            assert invalid.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

        await group.mutate((repository.state_store, archive), corrupt_after_read)
