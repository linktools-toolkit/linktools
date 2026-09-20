#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable Runtime snapshot persisted-field validation."""

import hashlib
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest

from linktools.ai.core import (
    OperationKind,
    OperationLedgerInput,
    OperationStatus,
    ResourceKind,
    SessionStatus,
    ToolOperationStatus,
    canonical_json_bytes,
    canonical_sha256,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeSnapshot, RuntimeState
from linktools.ai.runtime.state import RuntimeDomain, SnapshotLimits
from linktools.ai.runtime.state._codec import (
    _encode_persisted_domain,
    encode_envelope,
    encode_record,
    wire_type_id,
)
from linktools.ai.runtime.state._contracts import (
    ExecutionRecord,
    SessionRecord,
    ToolOperationRecord,
)
from linktools.ai.runtime.state._repository_common import domain_data, project_record
from linktools.ai.runtime.state import _snapshot_validation as snapshot_validation
from linktools.ai.runtime.state._snapshot_validation import (
    canonical_snapshot_indexes,
    validate_snapshot_domain,
)
from linktools.ai.runtime.state._store import (
    StoredAlias,
    StoredFact,
    StoredOperation,
    StoredRecord,
    alias_digest,
    operation_key,
    scope_digest,
    sequence_key,
    sortable_identity,
    stream_digest,
)
from linktools.ai.storage import InMemoryObjectStore, ObjectRef
from linktools.ai.workspace import Workspace


async def _put(store: InMemoryObjectStore, key: str, payload: bytes) -> ObjectRef:
    digest = hashlib.sha256(payload).hexdigest()

    async def chunks():
        yield payload

    await store.put(
        key,
        chunks(),
        expected_size=len(payload),
        expected_digest=digest,
    )
    return ObjectRef(store.store_id, key, digest, len(payload))


class _SnapshotGuard:
    @asynccontextmanager
    async def offline_exclusivity(self):
        yield


@pytest.mark.asyncio
async def test_runtime_state_snapshot_identity_ignores_target_store(
    tmp_path,
) -> None:
    root = tmp_path / "runtime"
    writable = RuntimeState.from_root(root)
    await writable.initialize(namespace="runtime", tenant_id="tenant")
    await writable.close()

    state = RuntimeState.from_root(root)
    await state.initialize(namespace="runtime", tenant_id="tenant", read_only=True)
    limits = SnapshotLimits(max_entries=1000, max_bytes=1024 * 1024)
    first = InMemoryObjectStore("snapshot-a")
    second = InMemoryObjectStore("snapshot-b")
    try:
        first_ref = await state.export_snapshot(object_store=first, limits=limits)
        second_ref = await state.export_snapshot(object_store=second, limits=limits)
    finally:
        await state.close()

    assert first_ref.key == second_ref.key
    assert first_ref.digest == second_ref.digest
    assert first_ref.size == second_ref.size


@pytest.mark.asyncio
async def test_runtime_snapshot_identity_ignores_target_store(tmp_path) -> None:
    root = tmp_path / "runtime"
    writable = RuntimeState.from_root(root)
    await writable.initialize(namespace="runtime", tenant_id="tenant")
    await writable.close()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "README.md").write_text("workspace", encoding="utf-8")
    workspace = Workspace.load(workspace_root)

    limits = SnapshotLimits(max_entries=1000, max_bytes=1024 * 1024)
    first = InMemoryObjectStore("snapshot-a")
    second = InMemoryObjectStore("snapshot-b")
    first_ref = await RuntimeSnapshot.create(
        "runtime",
        tenant_id="tenant",
        state=RuntimeState.from_root(root),
        object_store=first,
        workspace=workspace,
        exclusive=_SnapshotGuard(),
        limits=limits,
    )
    second_ref = await RuntimeSnapshot.create(
        "runtime",
        tenant_id="tenant",
        state=RuntimeState.from_root(root),
        object_store=second,
        workspace=workspace,
        exclusive=_SnapshotGuard(),
        limits=limits,
    )

    assert first_ref.key == second_ref.key
    assert first_ref.digest == second_ref.digest
    assert first_ref.size == second_ref.size


@pytest.mark.asyncio
async def test_runtime_snapshot_rejects_coerced_object_ref_fields() -> None:
    store = InMemoryObjectStore("snapshot")
    manifest = {
        "kind": "runtime-snapshot",
        "format_version": 1,
        "namespace": "runtime",
        "tenant_id": "tenant",
        "state": {
            "store_id": "snapshot",
            "key": "state",
            "digest": "a" * 64,
            "size": "1",
        },
        "workspace": {
            "present": False,
            "entries": [],
        },
        "metadata": {},
    }
    ref = await _put(
        store,
        "runtime-snapshot",
        canonical_json_bytes(manifest),
    )

    with pytest.raises(AIError) as raised:
        await RuntimeSnapshot.verify(
            ref,
            object_store=store,
            limits=SnapshotLimits(max_entries=10, max_bytes=4096),
        )

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_runtime_snapshot_rejects_coerced_format_version() -> None:
    store = InMemoryObjectStore("snapshot")
    manifest = {
        "kind": "runtime-snapshot",
        "format_version": 1.0,
        "namespace": "runtime",
        "tenant_id": "tenant",
        "state": {
            "store_id": "snapshot",
            "key": "state",
            "digest": "a" * 64,
            "size": 1,
        },
        "workspace": {
            "present": False,
            "entries": [],
        },
        "metadata": {},
    }
    ref = await _put(
        store,
        "runtime-snapshot-version",
        canonical_json_bytes(manifest),
    )

    with pytest.raises(AIError) as raised:
        await RuntimeSnapshot.verify(
            ref,
            object_store=store,
            limits=SnapshotLimits(max_entries=10, max_bytes=4096),
        )

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_runtime_state_restore_rejects_mismatched_logical_record_key_before_target_write(
    tmp_path,
) -> None:
    namespace = "runtime"
    tenant_id = "tenant"
    session = SessionRecord(
        session_id="session",
        owner_principal_id="principal",
        status=SessionStatus.OPEN,
        revision=1,
        cwd=None,
        metadata={},
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        closed_at=None,
        active_execution_id=None,
        agent_id="agent",
    )
    record = StoredRecord(
        b"x" * 32,
        scope_digest(
            namespace,
            tenant_id,
            "conversation",
            "session",
            "owner",
            session.owner_principal_id,
        ),
        None,
        "session",
        sortable_identity(session.session_id),
        session.status.value,
        0,
        None,
        0,
        None,
        encode_envelope(
            {
                "type": wire_type_id(session),
                "payload": _encode_persisted_domain(session),
            }
        ),
    )
    manifest = {
        "kind": "runtime-state-snapshot",
        "format_version": 1,
        "namespace": namespace,
        "tenant_id": tenant_id,
        "domains": {
            "conversation": {
                "records": [encode_record(record)],
                "aliases": [],
                "facts": [],
                "operations": [],
                "sequences": [],
            }
        },
        "objects": [],
    }
    store = InMemoryObjectStore("snapshot")
    ref = await _put(
        store,
        "invalid-state-snapshot",
        canonical_json_bytes(manifest),
    )
    target = tmp_path / "restored"

    with pytest.raises(AIError) as raised:
        await RuntimeState.restore_snapshot(
            ref,
            object_store=store,
            root=target,
            limits=SnapshotLimits(max_entries=20, max_bytes=16 * 1024),
        )

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert not target.exists()



def test_snapshot_aliases_are_rebuilt_from_tool_operation_identity() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    operation = ToolOperationRecord(
        tool_operation_id=canonical_sha256({"tool_operation": 1}),
        execution_id="execution",
        step_run_id="run",
        tool_call_id="call",
        idempotency_key_digest=canonical_sha256({"idempotency": 1}),
        tool_name="tool",
        arguments_digest=canonical_sha256({"arguments": 1}),
        binding_digest=canonical_sha256({"binding": 1}),
        replay_safe=True,
        status=ToolOperationStatus.PENDING,
        owner=None,
        fence=0,
        lease_expires_at=None,
        error_code=None,
        created_at=now,
        updated_at=now,
    )
    record = project_record(
        namespace="runtime",
        tenant_id="tenant",
        domain=RuntimeDomain.RECOVERY,
        kind="tool_operation",
        identity=operation.tool_operation_id,
        value=operation,
        state=operation.status.value,
    )

    aliases, sequences = canonical_snapshot_indexes(
        namespace="runtime",
        tenant_id="tenant",
        domain=RuntimeDomain.RECOVERY,
        records=(record,),
        facts=(),
        operations=(),
    )

    expected_alias = alias_digest(
        "runtime",
        "tenant",
        RuntimeDomain.RECOVERY.value,
        "tool_call",
        [operation.step_run_id, operation.tool_call_id],
    )
    assert aliases == (StoredAlias(expected_alias, record.key_digest),)
    assert sequences == {}

    validate_snapshot_domain(
        namespace="runtime",
        tenant_id="tenant",
        domain=RuntimeDomain.RECOVERY,
        records=(record,),
        aliases=aliases,
        facts=(),
        operations=(),
        sequences=sequences,
    )
    with pytest.raises(AIError) as raised:
        validate_snapshot_domain(
            namespace="runtime",
            tenant_id="tenant",
            domain=RuntimeDomain.RECOVERY,
            records=(record,),
            aliases=(StoredAlias(b"x" * 32, record.key_digest),),
            facts=(),
            operations=(),
            sequences=sequences,
        )
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_snapshot_operation_sequence_is_rebuilt_from_ledger_anchor() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    operation = OperationLedgerInput(
        operation_id=canonical_sha256({"operation": 1}),
        tenant_id="tenant",
        resource_kind=ResourceKind.SESSION,
        resource_id="session",
        execution_id=None,
        operation_kind=OperationKind.SESSION_UPDATE,
        status=OperationStatus.SUCCEEDED,
        request_digest=canonical_sha256({"request": 1}),
        result_ref=None,
        result_digest=None,
        error_code=None,
        compactable=True,
        created_at=now,
        updated_at=now,
    )
    stored = StoredOperation(
        operation_key(
            "runtime",
            "tenant",
            RuntimeDomain.CONVERSATION.value,
            operation.operation_id,
        ),
        stream_digest(
            "runtime",
            "tenant",
            RuntimeDomain.CONVERSATION.value,
            "operation",
            [operation.resource_kind.value, operation.resource_id],
        ),
        7,
        operation.status.value,
        operation.compactable,
        domain_data(operation),
    )

    aliases, sequences = canonical_snapshot_indexes(
        namespace="runtime",
        tenant_id="tenant",
        domain=RuntimeDomain.CONVERSATION,
        records=(),
        facts=(),
        operations=(stored,),
    )

    assert aliases == ()
    assert sequences == {
        sequence_key(
            "runtime",
            "tenant",
            RuntimeDomain.CONVERSATION.value,
            "operation",
            [operation.resource_kind.value, operation.resource_id],
        ): 7
    }



def _execution_stub(*, event_sequence: int) -> ExecutionRecord:
    execution = object.__new__(ExecutionRecord)
    object.__setattr__(execution, "execution_id", "execution")
    object.__setattr__(execution, "event_sequence", event_sequence)
    return execution


def _execution_fact(owner_key: bytes, sequence: int = 1) -> StoredFact:
    return StoredFact(
        stream_digest(
            "runtime",
            "tenant",
            RuntimeDomain.EXECUTION.value,
            "execution",
            "execution",
        ),
        sequence,
        owner_key,
        "EVENT",
        None,
        None,
        {},
    )


def test_snapshot_execution_facts_do_not_create_sequence_rows() -> None:
    owner_key = b"e" * 32
    execution = _execution_stub(event_sequence=1)
    sequences = snapshot_validation._canonical_sequences(
        "runtime",
        "tenant",
        RuntimeDomain.EXECUTION,
        (_execution_fact(owner_key),),
        (),
        {owner_key: execution},
    )

    assert sequences == {}


def test_snapshot_rejects_truncated_execution_event_stream() -> None:
    owner_key = b"e" * 32
    execution = _execution_stub(event_sequence=2)
    owner_record = StoredRecord(
        owner_key,
        None,
        None,
        "execution",
        "execution",
        None,
        0,
        None,
        0,
        None,
        {},
    )

    with pytest.raises(AIError) as raised:
        snapshot_validation._validate_facts(
            "runtime",
            "tenant",
            RuntimeDomain.EXECUTION,
            (_execution_fact(owner_key),),
            {owner_key: owner_record},
            {owner_key: execution},
        )

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
