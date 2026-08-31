#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for ToolOperation optimistic CAS convergence."""

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from linktools.ai.core import ToolOperationStatus, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.runtime._tool import RuntimeToolOperationBridge, ToolOperationRecord
from linktools.ai.runtime.state import RuntimeState, RuntimeStateCommands, ToolOperationAdmission
from linktools.ai.runtime.state._repositories import ToolRepositoryImpl
from linktools.ai.storage import StoredPayload
from sqlalchemy.ext.asyncio import create_async_engine


def _record(
    *,
    status: ToolOperationStatus = ToolOperationStatus.CLAIMED,
    owner: str = "tool-owner",
    fence: int = 1,
    result_payload: StoredPayload | None = None,
    error_code: str | None = None,
    error_payload: StoredPayload | None = None,
) -> ToolOperationRecord:
    now = datetime.now(timezone.utc)
    return ToolOperationRecord(
        tool_operation_id="tool-operation",
        tenant_id="tenant",
        step_run_id="step-run",
        tool_call_id="tool-call",
        idempotency_key_digest=canonical_sha256({"call": "tool-call"}),
        tool_name="tool",
        arguments_digest=canonical_sha256({"args": True}),
        binding_digest=canonical_sha256({"binding": True}),
        replay_safe=True,
        status=status,
        owner=owner,
        fence=fence,
        lease_expires_at=(
            now + timedelta(seconds=60)
            if status is ToolOperationStatus.CLAIMED
            else None
        ),
        error_code=error_code,
        created_at=now,
        updated_at=now,
        result_payload=result_payload,
        error_payload=error_payload,
    )


def _admission(*, owner: str = "tool-owner") -> ToolOperationAdmission:
    return ToolOperationAdmission(
        tenant_id="tenant",
        tool_operation_id="tool-operation",
        step_run_id="step-run",
        recovery_step_run_id=None,
        tool_call_id="tool-call",
        idempotency_key_digest=canonical_sha256({"call": "tool-call"}),
        tool_name="tool",
        arguments_digest=canonical_sha256({"args": True}),
        binding_digest=canonical_sha256({"binding": True}),
        replay_safe=True,
        owner=owner,
        lease_seconds=60,
    )


@pytest.mark.asyncio
async def test_sqlite_materializes_convergent_tool_repository(tmp_path) -> None:
    database = tmp_path / "state.sqlite"
    engine = create_async_engine(f"sqlite+aiosqlite:///{database}")
    await provision_runtime_database(engine)
    await engine.dispose()

    state = RuntimeState.sqlite(database)
    await state.initialize(namespace="tool-cas", tenant_id="tenant")
    try:
        repository = state.recovery.tools
        assert type(repository) is ToolRepositoryImpl

        request = _admission()
        first, second = await asyncio.gather(
            repository.admit(request),
            repository.admit(request),
        )
        assert first.tool_operation_id == second.tool_operation_id
        assert first.owner == second.owner == request.owner
        assert first.fence == second.fence == 1

        renewed = await repository.renew(
            request.tool_operation_id,
            tenant_id=request.tenant_id,
            owner=request.owner,
            fence=first.fence,
            lease_seconds=60,
        )
        payload = StoredPayload.inline_bytes(b"result")
        terminal = await repository.complete_payload(
            request.tool_operation_id,
            tenant_id=request.tenant_id,
            owner=request.owner,
            fence=renewed.fence,
            result_payload=payload,
        )
        assert terminal.status is ToolOperationStatus.COMPLETED
        assert terminal.result_payload == payload
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_tool_repository_retries_raw_storage_conflict() -> None:
    repository = object.__new__(ToolRepositoryImpl)
    committed = _record(status=ToolOperationStatus.COMPLETED)
    attempts = 0

    async def operation() -> ToolOperationRecord:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return committed

    result = await repository._retry_storage_conflict(operation)

    assert result == committed
    assert attempts == 2


@pytest.mark.asyncio
async def test_tool_repository_does_not_retry_semantic_conflict() -> None:
    repository = object.__new__(ToolRepositoryImpl)
    attempts = 0

    async def operation() -> ToolOperationRecord:
        nonlocal attempts
        attempts += 1
        raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)

    with pytest.raises(AIError) as raised:
        await repository._retry_storage_conflict(operation)

    assert raised.value.code is ErrorCode.TOOL_OPERATION_CONFLICT
    assert attempts == 1




class _GroupTransaction:
    def __init__(self, transaction: object) -> None:
        self._transaction = transaction

    def transaction(self, store: object) -> object:
        del store
        return self._transaction


class _StorageGroup:
    def __init__(self) -> None:
        self.attempts = 0
        self.transaction = object()
        self.conflict_once = True

    async def mutate(self, stores, callback):
        del stores
        self.attempts += 1
        if self.conflict_once:
            self.conflict_once = False
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return await callback(_GroupTransaction(self.transaction))


class _StateStore:
    def __init__(self, group: _StorageGroup) -> None:
        self.storage_group = group


class _CommandTools:
    def __init__(self, current: ToolOperationRecord) -> None:
        self.current = current
        self.group = _StorageGroup()
        self.state_store = _StateStore(self.group)
        self.complete_calls = 0
        self.fail_calls = 0

    async def get_operation(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
    ) -> ToolOperationRecord:
        del tool_operation_id, tenant_id
        return self.current

    async def complete_in_transaction(
        self,
        transaction: object,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        result_payload: StoredPayload,
    ) -> ToolOperationRecord:
        del transaction, tool_operation_id, tenant_id
        self.complete_calls += 1
        self.current = replace(
            self.current,
            status=ToolOperationStatus.COMPLETED,
            owner=owner,
            fence=fence,
            lease_expires_at=None,
            result_payload=result_payload,
        )
        return self.current

    async def fail_in_transaction(
        self,
        transaction: object,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        error_code: str,
        error_payload: StoredPayload | None,
    ) -> ToolOperationRecord:
        del transaction, tool_operation_id, tenant_id
        self.fail_calls += 1
        self.current = replace(
            self.current,
            status=ToolOperationStatus.FAILED,
            owner=owner,
            fence=fence,
            lease_expires_at=None,
            error_code=error_code,
            error_payload=error_payload,
        )
        return self.current


@pytest.mark.asyncio
async def test_tool_terminal_command_retries_after_outer_transaction_exits() -> None:
    tools = _CommandTools(_record())
    commands = object.__new__(RuntimeStateCommands)
    commands._tools = tools
    commands._background_tasks = set()
    payload = StoredPayload.inline_bytes(b"result")

    result = await commands.commit_tool_terminal(
        "tool-operation",
        tenant_id="tenant",
        owner="tool-owner",
        fence=1,
        result_payload=payload,
    )

    assert tools.group.attempts == 2
    assert tools.complete_calls == 1
    assert result.status is ToolOperationStatus.COMPLETED
    assert result.result_payload == payload


@pytest.mark.asyncio
async def test_tool_terminal_command_preserves_new_owner_after_conflict() -> None:
    tools = _CommandTools(_record(owner="new-owner", fence=2))
    commands = object.__new__(RuntimeStateCommands)
    commands._tools = tools
    commands._background_tasks = set()

    with pytest.raises(AIError) as raised:
        await commands.commit_tool_terminal(
            "tool-operation",
            tenant_id="tenant",
            owner="tool-owner",
            fence=1,
            result_payload=StoredPayload.inline_bytes(b"result"),
        )

    assert raised.value.code is ErrorCode.TOOL_OPERATION_CONFLICT
    assert tools.group.attempts == 1
    assert tools.complete_calls == 0


class _ReadbackRepository:
    def __init__(self, record: ToolOperationRecord) -> None:
        self.record = record

    async def get_operation(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
    ) -> ToolOperationRecord:
        del tool_operation_id, tenant_id
        return self.record


class _BlockingUnknownRepository(_ReadbackRepository):
    def __init__(self) -> None:
        super().__init__(_record())
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def mark_effect_unknown(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        error_code: str | None,
    ) -> ToolOperationRecord:
        del tool_operation_id, tenant_id
        self.calls += 1
        self.started.set()
        await self.release.wait()
        self.record = replace(
            self.record,
            status=ToolOperationStatus.EFFECT_UNKNOWN,
            owner=owner,
            fence=fence,
            lease_expires_at=None,
            error_code=error_code,
        )
        return self.record


@pytest.mark.asyncio
async def test_tool_unknown_resolves_durable_truth_before_propagating_cancellation() -> None:
    repository = _BlockingUnknownRepository()
    bridge = object.__new__(RuntimeToolOperationBridge)
    bridge._repository = repository
    bridge._tenant_id = "tenant"
    bridge._execution_id = "execution"
    bridge._owner = "tool-owner"
    bridge._background_tasks = set()
    decision = SimpleNamespace(
        operation_id="tool-operation",
        owner="tool-owner",
        fence=1,
    )

    task = asyncio.create_task(bridge.unknown(decision, RuntimeError("boom")))
    await repository.started.wait()
    task.cancel()
    repository.release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert repository.calls == 1
    assert repository.record.status is ToolOperationStatus.EFFECT_UNKNOWN
    assert repository.record.owner == decision.owner
    assert repository.record.fence == decision.fence
    assert repository.record.error_code == ErrorCode.TOOL_EFFECT_UNKNOWN.value


@pytest.mark.asyncio
async def test_tool_bridge_preserves_result_conflict_instead_of_integrity_error() -> None:
    expected = StoredPayload.inline_bytes(b"expected")
    observed = _record(
        status=ToolOperationStatus.COMPLETED,
        result_payload=StoredPayload.inline_bytes(b"different"),
    )
    bridge = object.__new__(RuntimeToolOperationBridge)
    bridge._repository = _ReadbackRepository(observed)
    bridge._tenant_id = "tenant"
    bridge._background_tasks = set()
    decision = SimpleNamespace(
        operation_id="tool-operation",
        owner="tool-owner",
        fence=1,
    )

    async def failed_commit() -> ToolOperationRecord:
        raise AIError(ErrorCode.TOOL_RESULT_CONFLICT)

    with pytest.raises(AIError) as raised:
        await bridge._finish_with_readback(
            failed_commit,
            decision,
            expected_status=ToolOperationStatus.COMPLETED,
            expected_payload=expected,
        )

    assert raised.value.code is ErrorCode.TOOL_RESULT_CONFLICT


@pytest.mark.asyncio
async def test_tool_bridge_requires_terminal_owner_and_fence_identity() -> None:
    payload = StoredPayload.inline_bytes(b"result")
    observed = _record(
        status=ToolOperationStatus.COMPLETED,
        owner="new-owner",
        fence=2,
        result_payload=payload,
    )
    bridge = object.__new__(RuntimeToolOperationBridge)
    bridge._repository = _ReadbackRepository(observed)
    bridge._tenant_id = "tenant"
    bridge._background_tasks = set()
    decision = SimpleNamespace(
        operation_id="tool-operation",
        owner="tool-owner",
        fence=1,
    )

    async def failed_commit() -> ToolOperationRecord:
        raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)

    with pytest.raises(AIError) as raised:
        await bridge._finish_with_readback(
            failed_commit,
            decision,
            expected_status=ToolOperationStatus.COMPLETED,
            expected_payload=payload,
        )

    assert raised.value.code is ErrorCode.TOOL_OPERATION_CONFLICT
