#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operation CAS preserves durable identity across storage backends."""

from linktools.ai.persistence import Persistence
import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from linktools.ai.core import (
    CommitPolicy,
    JsonValue,
    OperationKind,
    canonical_json_bytes,
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationStatus,
    ResourceKind,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.persistence.codec import decode_operation, encode_operation
from linktools.ai.persistence.state import (
    OperationScanCursor, StateTransaction, StoredOperation, operation_key,
)
from linktools.ai.persistence import PersistenceDomain, PersistenceStorage
from linktools.ai.persistence.repositories import OperationLedgerRepositoryImpl, build_repository_bundle
from linktools.ai.storage import InMemoryObjectStore

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _storage(backend: str, root: Path) -> PersistenceStorage:
    if backend == "memory":
        return PersistenceStorage.in_memory()
    if backend == "local_sqlite":
        return PersistenceStorage.from_root(root, exclusive_writer=True)
    return PersistenceStorage.sqlite(root / "runtime.db")


@pytest_asyncio.fixture(params=("memory", "local_sqlite", "sqlite"))
async def storage(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> AsyncIterator[Persistence]:
    state = _storage(request.param, tmp_path / "runtime")
    state = Persistence(state)
    await state.initialize(namespace="operations", tenant_id="tenant")
    try:
        yield state
    finally:
        await state.close()


def _operation() -> OperationLedgerInput:
    return OperationLedgerInput(
        operation_id="operation",
        tenant_id="tenant",
        resource_kind=ResourceKind.SESSION,
        resource_id="session",
        execution_id=None,
        operation_kind=OperationKind.SESSION_UPDATE,
        status=OperationStatus.PENDING,
        request_digest="a" * 64,
        result_ref=None,
        result_digest=None,
        error_code=None,
        compactable=True,
        created_at=_NOW,
        updated_at=_NOW,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("operation_id", "other-operation"),
        ("tenant_id", "other-tenant"),
        ("resource_kind", ResourceKind.EXECUTION),
        ("resource_id", "other-session"),
        ("execution_id", "execution"),
        ("operation_kind", OperationKind.SESSION_CLOSE),
        ("request_digest", "b" * 64),
        ("compactable", False),
        ("sequence", 99),
        ("created_at", _NOW + timedelta(seconds=1)),
    ),
)
async def test_operation_cas_rejects_changed_immutable_identity(
    storage: Persistence,
    field: str,
    value: object,
) -> None:
    operations = storage.conversation.operations
    current = await operations.append(_operation())
    candidate = replace(current, status=OperationStatus.RUNNING, **{field: value})
    with pytest.raises(AIError) as raised:
        await operations.compare_and_swap(
            current.operation_id,
            tenant_id="tenant",
            expected_status=current.status,
            next_record=candidate,
        )
    assert raised.value.code is (
        ErrorCode.STORAGE_OWNER_MISMATCH
        if field == "tenant_id"
        else ErrorCode.STORAGE_CONFLICT
    )
    assert await operations.get("operation", tenant_id="tenant") == current
    assert await operations.get("other-operation", tenant_id="tenant") is None
    assert await operations.list_pending(
        ResourceKind.SESSION, "session", tenant_id="tenant", limit=10
    ) == (current,)
    assert await operations.list_pending(
        ResourceKind.SESSION, "other-session", tenant_id="tenant", limit=10
    ) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal", (OperationStatus.SUCCEEDED, OperationStatus.FAILED, OperationStatus.CANCELLED)
)
async def test_operation_cas_roundtrips_mutable_fields_and_keeps_terminal_status(
    storage: Persistence,
    terminal: OperationStatus,
) -> None:
    operations = storage.conversation.operations
    current = await operations.append(_operation())
    for status in (OperationStatus.RUNNING, terminal, terminal):
        candidate = replace(
            current,
            status=status,
            result_ref="result",
            result_digest="c" * 64,
            error_code="example-error" if terminal is OperationStatus.FAILED else None,
            updated_at=current.updated_at + timedelta(seconds=1),
        )
        assert await operations.compare_and_swap(
            current.operation_id,
            tenant_id="tenant",
            expected_status=current.status,
            next_record=candidate,
        ) == candidate
        assert await operations.get("operation", tenant_id="tenant") == candidate
        current = candidate
    assert await operations.list_pending(
        ResourceKind.SESSION, "session", tenant_id="tenant", limit=10
    ) == ()
    with pytest.raises(AIError) as raised:
        await operations.compare_and_swap(
            current.operation_id,
            tenant_id="tenant",
            expected_status=terminal,
            next_record=replace(current, status=OperationStatus.RUNNING),
        )
    assert raised.value.code is ErrorCode.STORAGE_CONFLICT
    assert await operations.get("operation", tenant_id="tenant") == current


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation_id", "tenant_id", "candidate_tenant", "expected_status", "code"),
    (
        ("missing", "other", "tenant", OperationStatus.RUNNING, ErrorCode.STORAGE_OWNER_MISMATCH),
        ("missing", "tenant", "other", OperationStatus.RUNNING, ErrorCode.STORAGE_OWNER_MISMATCH),
        ("missing", "tenant", "tenant", OperationStatus.PENDING, ErrorCode.STORAGE_CONFLICT),
        ("operation", "tenant", "tenant", OperationStatus.RUNNING, ErrorCode.STORAGE_CONFLICT),
    ),
)
async def test_operation_cas_preserves_owner_and_status_conflicts(
    storage: Persistence,
    operation_id: str,
    tenant_id: str,
    candidate_tenant: str,
    expected_status: OperationStatus,
    code: ErrorCode,
) -> None:
    operations = storage.conversation.operations
    current = await operations.append(_operation())
    with pytest.raises(AIError) as raised:
        await operations.compare_and_swap(
            operation_id,
            tenant_id=tenant_id,
            expected_status=expected_status,
            next_record=replace(current, tenant_id=candidate_tenant, sequence=99),
        )
    assert raised.value.code is code
    assert await operations.get("operation", tenant_id="tenant") == current


@pytest.mark.asyncio
async def test_operation_cas_competing_transitions_have_one_durable_winner(
    storage: Persistence,
) -> None:
    operations = storage.conversation.operations
    current = await operations.append(_operation())
    candidates = (
        replace(current, status=OperationStatus.RUNNING),
        replace(current, status=OperationStatus.CANCELLED),
    )
    results = await asyncio.gather(
        *(
            operations.compare_and_swap(
                current.operation_id,
                tenant_id="tenant",
                expected_status=OperationStatus.PENDING,
                next_record=candidate,
            )
            for candidate in candidates
        ),
        return_exceptions=True,
    )
    successes = [value for value in results if isinstance(value, OperationLedgerRecord)]
    failures = [value for value in results if isinstance(value, AIError)]
    assert len(successes) == len(failures) == 1
    assert failures[0].code is ErrorCode.STORAGE_CONFLICT
    assert await operations.get("operation", tenant_id="tenant") == successes[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("local_sqlite", "sqlite"))
async def test_operation_cas_output_survives_snapshot_restore(
    backend: str,
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime"
    state = _storage(backend, root)
    state = Persistence(state)
    await state.initialize(namespace="operations", tenant_id="tenant")
    try:
        current = await state.conversation.operations.append(replace(_operation(), operation_id="界" * 42 + "ab"))
        candidate = replace(current, status=OperationStatus.RUNNING, updated_at=_NOW + timedelta(seconds=1))
        await state.conversation.operations.compare_and_swap(
            current.operation_id,
            tenant_id="tenant",
            expected_status=current.status,
            next_record=candidate,
        )
    finally:
        await state.close()

    objects = InMemoryObjectStore("snapshot")
    limits = SnapshotLimits(max_entries=100, max_bytes=1024 * 1024)
    reader = _storage(backend, root)
    reader = Persistence(reader)
    await reader.initialize(namespace="operations", tenant_id="tenant", read_only=True)
    try:
        reference = await reader.export_snapshot(object_store=objects, limits=limits)
    finally:
        await reader.close()
    restored_root = tmp_path / "restored"
    await Persistence.restore_snapshot(
        reference, object_store=objects, root=restored_root, limits=limits
    )
    restored = PersistenceStorage.from_root(restored_root)
    restored = Persistence(restored)
    await restored.initialize(namespace="operations", tenant_id="tenant")
    try:
        assert await restored.conversation.operations.get(candidate.operation_id, tenant_id="tenant") == candidate
        assert await restored.conversation.operations.list_pending(
            ResourceKind.SESSION, "session", tenant_id="tenant", limit=10
        ) == (candidate,)
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_operation_ids_keep_exact_utf8_addresses_replay_and_scan_order(storage: Persistence) -> None:
    operations = storage.conversation.operations
    ids = ("A", "a", "é", "e\u0301", "a" * 64, "x" * 128, "界" * 42 + "ab")
    values = tuple([await operations.append(replace(_operation(), operation_id=value)) for value in ids])
    assert tuple(value.sequence for value in values) == tuple(range(1, len(ids) + 1))
    assert await operations.append(replace(_operation(), operation_id=ids[-1])) == values[-1]
    with pytest.raises(AIError) as conflict:
        await operations.append(replace(_operation(), operation_id=ids[-1], request_digest="b" * 64))
    assert conflict.value.code is ErrorCode.STORAGE_CONFLICT
    assert await operations.list_pending(ResourceKind.SESSION, "session", tenant_id="tenant", limit=10) == values

    seen = []
    cursor = None
    while True:
        page = await operations.state_store.read(lambda tx: tx.scan_operations_page(after=cursor, limit=2))
        if not page:
            break
        seen.extend(page)
        cursor = OperationScanCursor(page[-1].key)
    assert [value.key for value in seen] == sorted(operation_key("operations", "tenant", "conversation", value) for value in ids)
    for value in seen:
        assert decode_operation(encode_operation(value)) == value
        assert await operations.state_store.read(lambda tx: tx.get_operation(value.key)) == value
    stream = seen[0].stream_key
    assert await operations.state_store.read(lambda tx: tx.get_sequence(stream)) == len(ids)


@pytest.mark.parametrize("identity", ("", "x" * 129, "界" * 43, " leading", "trailing ", "a\nb", "\ud800"))
def test_operation_address_rejects_invalid_idempotency_ids(identity: str) -> None:
    with pytest.raises(AIError) as error:
        operation_key("operations", "tenant", "conversation", identity)
    assert error.value.code is ErrorCode.IDEMPOTENCY_KEY_INVALID


@pytest.mark.asyncio
@pytest.mark.parametrize("custom", (False, True))
async def test_public_operation_repositories_isolate_scopes_on_one_store(
    storage: Persistence, monkeypatch: pytest.MonkeyPatch, custom: bool,
) -> None:
    from .test_state_fact_reader import _LegacyTransaction, _custom_transactions

    store = storage.conversation.operations.state_store
    if custom:
        from linktools.ai.persistence._memory import InMemoryStateStore
        _custom_transactions(monkeypatch, "memory" if isinstance(store, InMemoryStateStore) else "sql", _LegacyTransaction)
    scopes = (("operations", "tenant", PersistenceDomain.CONVERSATION),
              ("other", "tenant", PersistenceDomain.CONVERSATION),
              ("operations", "other", PersistenceDomain.CONVERSATION),
              ("operations", "tenant", PersistenceDomain.EXECUTION))
    repositories = [build_repository_bundle(store, namespace=namespace, tenant_id=tenant, domain=domain)["operations"]
                    for namespace, tenant, domain in scopes]
    values = []
    for index, (repository, (_, tenant, _)) in enumerate(zip(repositories, scopes, strict=True)):
        assert isinstance(repository, OperationLedgerRepositoryImpl)
        assert await repository.get("same-ID", tenant_id=tenant) is None
        value = replace(_operation(), operation_id="same-ID", tenant_id=tenant, resource_id=f"resource-{index}")
        values.append(await repository.append(value))
        assert await repository.append(value) == values[-1]
    for repository, (_, tenant, _), value in zip(repositories, scopes, values, strict=True):
        assert await repository.get("same-ID", tenant_id=tenant) == value
        assert await repository.list_pending(ResourceKind.SESSION, value.resource_id, tenant_id=tenant, limit=10) == (value,)
        updated = replace(value, status=OperationStatus.RUNNING)
        assert await repository.compare_and_swap("same-ID", tenant_id=tenant, expected_status=value.status,
                                                 next_record=updated) == updated
        assert await repository.get("same-ID", tenant_id=tenant) == updated


@pytest.mark.asyncio
async def test_capability_receipts_and_markers_keep_separate_operation_families(storage: Persistence) -> None:
    from copy import copy
    from hashlib import sha256
    from linktools.ai.persistence.capabilities import CapabilityWriter, RuntimeCapabilityWrites
    from linktools.ai.persistence._capability_marker import has_capability_partition_marker
    from linktools.ai.persistence.state import capability_receipt_key

    persisted = []

    class Capability:
        commit = CommitPolicy.sync()
        writer: CapabilityWriter | None = None

        def bind_persistence(self, writer: CapabilityWriter) -> "Capability":
            bound = copy(self)
            bound.writer = writer
            return bound

        async def persist_in_transaction(self, transaction: StateTransaction, payload: Mapping[str, JsonValue]) -> None:
            persisted.append(dict(payload))

    store = storage.conversation.operations.state_store
    name = "capability-partitions"
    ordinary = StoredOperation(operation_key("operations", "tenant", "conversation", name), b"o" * 32, 0, "ordinary", False, {})
    await store.mutate(lambda tx: tx.insert_operation(ordinary))
    write_key = "界" * 42 + "ab"
    scopes = (("one", "tenant"), ("two", "tenant"), ("one", "other"))
    for index, (namespace, tenant) in enumerate(scopes):
        address = capability_receipt_key(namespace, tenant, "probe", 1, "run", write_key)
        # Same tuple bytes with the ordinary family remain a different identity.
        value = StoredOperation(b"\x01" + address[1:], b"ordinary" + bytes((index,)), index + 1, "ordinary", False, {})
        await store.mutate(lambda tx: tx.insert_operation(value))
        writes = RuntimeCapabilityWrites(
            {"probe": store}, namespace=namespace, tenant_id=tenant,
            durable_capability_ids=frozenset({"probe"}), require_active_run=lambda *_: None,
            has_active_execution=lambda _: True, marker_store=store,
        )
        try:
            capability = writes.bind("probe", "run", "execution", Capability())
            payload = {"scope": index}
            await capability.writer.submit(key=write_key, payload=payload)
            await capability.writer.submit(key=write_key, payload=payload)
            receipt = await store.read(lambda tx: tx.get_operation(address))
            assert receipt.stream_key == address
            assert receipt.data == {"request_digest": sha256(canonical_json_bytes(payload)).hexdigest()}
        finally:
            await writes.close()
    assert persisted == [{"scope": index} for index in range(len(scopes))]
    assert await store.read(has_capability_partition_marker)
    marker = await store.read(lambda tx: tx.get_operation(b"\x00capability-partitions"))
    assert marker.stream_key == marker.key == b"\x00capability-partitions"
    assert await store.read(lambda tx: tx.get_operation(ordinary.key)) == ordinary


@pytest.mark.asyncio
@pytest.mark.parametrize(("field", "identity"), (("operation_id", "other"), ("operation_id", "x" * 129),
                                                 ("operation_id", "a\nb"), ("tenant_id", "other")))
async def test_operation_readers_reject_corrupt_persisted_identity(storage: Persistence, field: str, identity: str) -> None:
    from linktools.ai.persistence import PersistenceDomain
    from linktools.ai.persistence._repository_common import domain_data
    from linktools.ai.persistence._snapshot_validation import validate_snapshot_domain

    operations = storage.conversation.operations
    await operations.append(_operation())
    key = operation_key("operations", "tenant", "conversation", "operation")
    stored = await operations.state_store.read(lambda tx: tx.get_operation(key))
    corrupted = replace(stored, data=domain_data(replace(_operation(), **{field: identity})))
    assert await operations.state_store.mutate(lambda tx: tx.replace_operation(corrupted, expected_state="PENDING"))
    with pytest.raises(AIError) as direct:
        await operations.get("operation", tenant_id="tenant")
    assert direct.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    with pytest.raises(AIError) as listed:
        await operations.list_pending(ResourceKind.SESSION, "session", tenant_id="tenant", limit=10)
    assert listed.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    with pytest.raises(AIError) as snapshot:
        validate_snapshot_domain(namespace="operations", tenant_id="tenant", domain=PersistenceDomain.CONVERSATION,
                                 records=(), aliases=(), facts=(), operations=(corrupted,),
                                 sequences={corrupted.stream_key: corrupted.sequence})
    assert snapshot.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("physical_field", ("key", "stream_key"))
@pytest.mark.parametrize("scope", (("other", "tenant", "conversation"),
                                   ("operations", "other", "conversation"),
                                   ("operations", "tenant", "execution")))
async def test_operation_readers_reject_foreign_physical_scope(storage: Persistence, scope: tuple[str, str, str], physical_field: str) -> None:
    from linktools.ai.persistence.repositories import decode_operation as decode_ledger_operation
    from linktools.ai.persistence._snapshot_validation import validate_snapshot_domain

    operations = storage.conversation.operations
    await operations.append(_operation())
    stored = await operations.state_store.read(lambda tx: tx.get_operation(
        operation_key("operations", "tenant", "conversation", "operation")))
    from linktools.ai.persistence.state import stream_key
    physical = (operation_key(*scope, "operation") if physical_field == "key"
                else stream_key(*scope, "operation", ["SESSION", "session"]))
    corrupted = replace(stored, **{physical_field: physical})
    with pytest.raises(AIError, match="STORAGE_INTEGRITY_ERROR"):
        decode_ledger_operation(corrupted, namespace="operations", tenant_id="tenant", domain=PersistenceDomain.CONVERSATION)
    with pytest.raises(AIError, match="STORAGE_INTEGRITY_ERROR"):
        validate_snapshot_domain(namespace="operations", tenant_id="tenant", domain=PersistenceDomain.CONVERSATION,
                                 records=(), aliases=(), facts=(), operations=(corrupted,),
                                 sequences={corrupted.stream_key: corrupted.sequence})
