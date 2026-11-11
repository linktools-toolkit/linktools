#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for ToolOperation optimistic CAS convergence."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from linktools.ai.persistence import Persistence
from linktools.ai.core import ExecutionStatus, ToolOperationStatus, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.runtime._tool import RuntimeToolOperationBridge, ToolOperationDecision, ToolOperationRecord
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence.commands import StateCommands
from linktools.ai.persistence.contracts import ToolOperationAdmission
from linktools.ai.storage import FilesystemObjectStore, InMemoryObjectStore, StoredPayload
from linktools.ai.persistence.state import Observed
from sqlalchemy.ext.asyncio import create_async_engine


class _CommandExecution:
    def __init__(self, state_store):
        self.state_store = state_store

    async def get(self, execution_id, *, tenant_id):
        return SimpleNamespace(status=ExecutionStatus.STARTED)

    async def get_in_transaction(self, transaction, execution_id, *, tenant_id):
        return await self.get(execution_id, tenant_id=tenant_id)


def _record(
    *,
    status: ToolOperationStatus = ToolOperationStatus.CLAIMED,
    owner: str = "tool-owner",
    fence: int = 1,
    replay_safe: bool = True,
    result_payload: StoredPayload | None = None,
    error_code: str | None = None,
    error_payload: StoredPayload | None = None,
) -> ToolOperationRecord:
    now = datetime.now(timezone.utc)
    return ToolOperationRecord(
        tool_operation_id="tool-operation",
        execution_id="execution",
        agent_run_id="agent-run",
        tool_call_id="tool-call",
        tool_name="tool",
        arguments_digest=canonical_sha256({"args": True}),
        binding_digest=canonical_sha256({"binding": True}),
        replay_safe=replay_safe,
        status=status,
        owner=owner,
        fence=fence,
        error_code=error_code,
        created_at=now,
        updated_at=now,
        result_payload=result_payload,
        error_payload=error_payload,
    )


def _admission(*, owner: str = "tool-owner") -> ToolOperationAdmission:
    return ToolOperationAdmission(
        execution_id="execution",
        tool_operation_id="tool-operation",
        agent_run_id="agent-run",
        recovery_agent_run_id=None,
        tool_call_id="tool-call",
        tool_name="tool",
        arguments_digest=canonical_sha256({"args": True}),
        binding_digest=canonical_sha256({"binding": True}),
        replay_safe=True,
        owner=owner,
    )


@pytest.mark.asyncio
async def test_sqlite_parallel_admissions_share_one_effect_record(tmp_path) -> None:
    database = tmp_path / "state.sqlite"
    engine = create_async_engine(f"sqlite+aiosqlite:///{database}")
    await provision_runtime_database(engine)
    await engine.dispose()

    state = Persistence(PersistenceStorage.sqlite(
        database,
        object_store=FilesystemObjectStore(tmp_path / "objects"),
    ))
    await state.initialize(namespace="tool-cas", tenant_id="tenant")
    try:
        repository = state.checkpoint.tools

        request = _admission()
        first, second = await asyncio.gather(
            repository.admit(request),
            repository.admit(request),
        )
        assert first.tool_operation_id == second.tool_operation_id
        assert first.execution_id == second.execution_id == request.execution_id
        assert first.owner == second.owner == request.owner
        assert first.fence == second.fence == 1
        assert await repository.list_by_execution(
            request.execution_id,
            tenant_id="tenant",
        ) == (first,)
        assert await repository.list_by_execution(
            "other-execution",
            tenant_id="tenant",
        ) == ()

        for change in (
            {"arguments_digest": canonical_sha256({"args": False})},
            {"binding_digest": canonical_sha256({"binding": False})},
            {"tool_name": "other-tool"},
            {"tool_call_id": "other-call"},
            {"agent_run_id": "other-run"},
        ):
            with pytest.raises(AIError) as error:
                await repository.admit(replace(request, **change))
            assert error.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        assert await repository.list_by_execution(
            request.execution_id, tenant_id="tenant",
        ) == (first,)
        effect = replace(first, effect_idempotency_key="provider-effect-key")
        assert effect.to_tool_effect(request.agent_run_id).idempotency_key == "provider-effect-key"
        assert effect.fence == first.fence

        payload = StoredPayload.inline_bytes(b"result")
        terminal = await repository.complete_payload(
            request.tool_operation_id,
            tenant_id="tenant",
            owner=request.owner,
            fence=first.fence,
            result_payload=payload,
        )
        assert terminal.status is ToolOperationStatus.COMPLETED
        assert terminal.result_payload == payload
    finally:
        await state.close()


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

    async def observe_operation(self, tool_operation_id: str, *, tenant_id: str) -> Observed[ToolOperationRecord]:
        return Observed(await self.get_operation(tool_operation_id, tenant_id=tenant_id), 0)

    async def complete_in_transaction(
        self,
        transaction: object,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        result_payload: StoredPayload,
        observed: Observed[ToolOperationRecord] | None = None,
    ) -> Observed[ToolOperationRecord]:
        del transaction, tool_operation_id, tenant_id
        self.complete_calls += 1
        self.current = replace(
            self.current,
            status=ToolOperationStatus.COMPLETED,
            owner=owner,
            fence=fence,
            result_payload=result_payload,
        )
        return Observed(self.current, 1)

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
        observed: Observed[ToolOperationRecord] | None = None,
    ) -> Observed[ToolOperationRecord]:
        del transaction, tool_operation_id, tenant_id
        self.fail_calls += 1
        self.current = replace(
            self.current,
            status=ToolOperationStatus.FAILED,
            owner=owner,
            fence=fence,
            error_code=error_code,
            error_payload=error_payload,
        )
        return Observed(self.current, 1)


@pytest.mark.asyncio
async def test_tool_terminal_command_retries_after_outer_transaction_exits() -> None:
    tools = _CommandTools(_record())
    commands = object.__new__(StateCommands)
    commands._tools = tools
    commands._observe_cancellation = True
    commands._execution = _CommandExecution(tools.state_store)
    commands._background_tasks = set()
    payload = StoredPayload.inline_bytes(b"result")

    result, cancellation_requested = await commands.commit_tool_terminal(
        "tool-operation",
        tenant_id="tenant",
        owner="tool-owner",
        fence=1,
        result_payload=payload,
    )

    assert tools.group.attempts == 2
    assert tools.complete_calls == 1
    assert result.value.status is ToolOperationStatus.COMPLETED
    assert result.value.result_payload == payload
    assert not cancellation_requested


@pytest.mark.asyncio
async def test_tool_terminal_command_preserves_new_owner_after_conflict() -> None:
    tools = _CommandTools(_record(owner="new-owner", fence=2))
    commands = object.__new__(StateCommands)
    commands._tools = tools
    commands._observe_cancellation = True
    commands._execution = _CommandExecution(tools.state_store)
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


    async def observe_operation(self, tool_operation_id: str, *, tenant_id: str) -> Observed[ToolOperationRecord]:
        return Observed(await self.get_operation(tool_operation_id, tenant_id=tenant_id), 0)


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
            error_code=error_code,
        )
        return self.record


def _readback_bridge(repository: _ReadbackRepository) -> RuntimeToolOperationBridge:
    return RuntimeToolOperationBridge(
        repository, InMemoryObjectStore(), namespace="tool-cas", tenant_id="tenant",
        execution_id="execution", agent_run_id="agent-run", owner="tool-owner",
        binding_digest=repository.record.binding_digest, background_tasks=set(),
    )


@pytest.mark.asyncio
async def test_tool_unknown_resolves_durable_truth_before_propagating_cancellation() -> None:
    repository = _BlockingUnknownRepository()
    bridge = _readback_bridge(repository)
    decision = ToolOperationDecision(
        operation_id="tool-operation",
        owner="tool-owner",
        fence=1,
        replay_safe=True,
    )

    task = asyncio.create_task(bridge.unknown(decision, RuntimeError("boom")))
    await asyncio.wait_for(repository.started.wait(), 5)
    task.cancel()
    repository.release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert repository.calls == 1
    assert repository.record.status is ToolOperationStatus.EFFECT_UNKNOWN
    assert repository.record.owner == decision.owner
    assert repository.record.fence == decision.fence
    assert repository.record.error_code == ErrorCode.TOOL_EFFECT_UNKNOWN.value
    assert await bridge._repository.get_operation("tool-operation", tenant_id="tenant") == repository.record


@pytest.mark.asyncio
async def test_tool_bridge_preserves_result_conflict_instead_of_integrity_error() -> None:
    expected = StoredPayload.inline_bytes(b"expected")
    observed = _record(
        status=ToolOperationStatus.COMPLETED,
        result_payload=StoredPayload.inline_bytes(b"different"),
    )
    bridge = _readback_bridge(_ReadbackRepository(observed))
    decision = ToolOperationDecision(
        operation_id="tool-operation",
        owner="tool-owner",
        fence=1,
        replay_safe=True,
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
    bridge = _readback_bridge(_ReadbackRepository(observed))
    decision = ToolOperationDecision(
        operation_id="tool-operation",
        owner="tool-owner",
        fence=1,
        replay_safe=True,
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
