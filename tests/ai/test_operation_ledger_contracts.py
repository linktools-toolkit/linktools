#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operation CAS preserves durable identity across storage backends."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from linktools.ai.core import (
    OperationKind,
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationStatus,
    ResourceKind,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeStorage, SnapshotLimits
from linktools.ai.storage import InMemoryObjectStore

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _storage(backend: str, root: Path) -> RuntimeStorage:
    if backend == "memory":
        return RuntimeStorage.in_memory()
    if backend == "filesystem":
        return RuntimeStorage.filesystem(root)
    return RuntimeStorage.sqlite(root / "runtime.db")


@pytest_asyncio.fixture(params=("memory", "filesystem", "sqlite"))
async def storage(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> AsyncIterator[RuntimeStorage]:
    state = _storage(request.param, tmp_path / "runtime")
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
    storage: RuntimeStorage,
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
    storage: RuntimeStorage,
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
    storage: RuntimeStorage,
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
    storage: RuntimeStorage,
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
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
async def test_operation_cas_output_survives_snapshot_restore(
    backend: str,
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime"
    state = _storage(backend, root)
    await state.initialize(namespace="operations", tenant_id="tenant")
    try:
        current = await state.conversation.operations.append(_operation())
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
    await reader.initialize(namespace="operations", tenant_id="tenant", read_only=True)
    try:
        reference = await reader.export_snapshot(object_store=objects, limits=limits)
    finally:
        await reader.close()
    restored_root = tmp_path / "restored"
    await RuntimeStorage.restore_snapshot(
        reference, object_store=objects, root=restored_root, limits=limits
    )
    restored = RuntimeStorage.from_root(restored_root)
    await restored.initialize(namespace="operations", tenant_id="tenant")
    try:
        assert await restored.conversation.operations.get("operation", tenant_id="tenant") == candidate
        assert await restored.conversation.operations.list_pending(
            ResourceKind.SESSION, "session", tenant_id="tenant", limit=10
        ) == (candidate,)
    finally:
        await restored.close()
