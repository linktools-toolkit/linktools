#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused regression coverage for the v5 Runtime convergence contract."""

import asyncio
import ast
import contextlib
import inspect
import json
import sqlite3
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.adapter import StepExecutionHistoryReader, build_filesystem_runtime, build_in_memory_runtime
from linktools.ai.app import RuntimePersistenceConfig, build_runtime_access, build_runtime_services, open_runtime_services
from linktools.ai.app import open_workspace_runtime
from linktools.ai.core import Page, Principal, TenantAuthorizationPolicy
from linktools.ai.errors import ErrorCode, AIError
from linktools.ai.core import canonical_sha256, idempotency_key_hash
from linktools.ai.core import ApprovalDecision, ApprovalStatus, ExecutionEventType, ExecutionLineageKind, ExecutionStatus, IdempotencyStatus, OperationKind, OperationStatus, ResourceKind, SessionStatus, StopReason, TaskStatus
from linktools.ai.runtime import ApprovalRecord, ExecutionCancelRequestCommit, ExecutionRecord, ExecutionTerminalCommit, IdempotencyRecord, IdempotencyTerminalUpdate, OperationLedgerInput, ResultRecord, RuntimePersistence, SessionHeadAdvance, SessionRecord
from linktools.ai.runtime import ApprovalDecisionRequest, CancelExecutionRequest, CreateSessionRequest, ExecutionHandle, ExecutionRequest, ExecutionView, ForkExecutionRequest, RetryExecutionRequest, RuntimeServiceIdentity, WorkflowQueryResult, WorkflowUpdateResult
from linktools.ai.runtime import CancelEffectOutcome
from linktools.ai.task import CancelGraphRequest, TaskGraph, TaskGraphHandle, TaskGraphRequest, TaskNode
from linktools.ai.agent import WorkspaceAgentResult, WorkspaceAgentRunner
from linktools.ai.workspace import Workspace, trusted_workspace_principal
from tests.ai.persistence.helper import _open_sql_workspace, open_sql_resources


class _History:
    async def trace(self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int) -> Page[object]:
        return Page((), None)

    async def transcript(self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int) -> Page[object]:
        return Page((), None)


class _Launcher:
    def __init__(self) -> None:
        self.started: list[str] = []

    async def start(self, request: ExecutionRequest, execution: ExecutionRecord) -> None:
        self.started.append(execution.execution_id)
        return None

    async def cancel(self, execution: ExecutionRecord) -> object:
        return None


class _Gateway:
    def __init__(self) -> None:
        self.execution_starts: list[str] = []
        self.task_starts: list[str] = []
        self.updates: list[str] = []

    async def start_execution(self, workflow_id: str, request: ExecutionRequest) -> ExecutionHandle:
        self.execution_starts.append(workflow_id)
        return ExecutionHandle(workflow_id)

    async def update_execution(self, workflow_id: str, operation: str, payload: dict[str, object]) -> WorkflowUpdateResult:
        self.updates.append(operation)
        return WorkflowUpdateResult(workflow_id, "UPDATED")

    async def query_execution(self, workflow_id: str, query: str) -> WorkflowQueryResult:
        return WorkflowQueryResult(workflow_id, "RUNNING", {})

    async def cancel_execution(self, workflow_id: str) -> object:
        return None

    async def start_task_graph(self, workflow_id: str, request: TaskGraphRequest) -> TaskGraphHandle:
        self.task_starts.append(workflow_id)
        return TaskGraphHandle(workflow_id, workflow_id)

    async def cancel_task_graph(self, workflow_id: str, cancel_request_id: str) -> object:
        return None


class _TaskGateway:
    def __init__(self, *, start_error: "Exception | None" = None, cancel_error: "Exception | None" = None) -> None:
        self.start_error = start_error
        self.cancel_error = cancel_error
        self.task_starts: list[str] = []
        self.task_cancels: list[str] = []
        self.start_entered = asyncio.Event()
        self.start_release = asyncio.Event()
        self.cancel_entered = asyncio.Event()
        self.cancel_release = asyncio.Event()
        self.block_start = False
        self.block_cancel = False

    async def start_task_graph(self, workflow_id: str, request: TaskGraphRequest) -> TaskGraphHandle:
        self.task_starts.append(workflow_id)
        self.start_entered.set()
        if self.block_start:
            await self.start_release.wait()
        if self.start_error is not None:
            raise self.start_error
        return TaskGraphHandle(workflow_id, workflow_id)

    async def cancel_task_graph(self, workflow_id: str, cancel_request_id: str) -> object:
        self.task_cancels.append(workflow_id)
        self.cancel_entered.set()
        if self.block_cancel:
            await self.cancel_release.wait()
        if self.cancel_error is not None:
            raise self.cancel_error
        return None


class _CancelRaceLauncher:
    def __init__(self) -> None:
        self.calls = 0
        self.first_cancel_entered = asyncio.Event()
        self.first_cancel_release = asyncio.Event()

    async def start(self, request: ExecutionRequest, execution: ExecutionRecord) -> None:
        return None

    async def cancel(self, execution: ExecutionRecord) -> CancelEffectOutcome:
        self.calls += 1
        if self.calls == 1:
            self.first_cancel_entered.set()
            await self.first_cancel_release.wait()
            return CancelEffectOutcome.UNKNOWN
        return CancelEffectOutcome.CONFIRMED


class _BarrierRunner:
    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self._entered = 0
        self._release = asyncio.Event()

    async def binding_digest(self, agent_id: str | None) -> str:
        return "b" * 64

    async def run(
        self,
        agent_id: str | None,
        prompt: str,
        history: list[object],
        conversation_id: str,
        *,
        step_store: object,
        step_run_id: str,
        segment_sequence: int,
        parent_step_run_id: str | None = None,
        memory_namespace: str | None = None,
        memory_store: object | None = None,
        on_event: object | None = None,
    ) -> WorkspaceAgentResult:
        async with self._guard:
            self._entered += 1
            if self._entered == 2:
                self._release.set()
        await self._release.wait()
        return WorkspaceAgentResult(step_run_id, prompt, [])


class _BlockingRunner:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self._wait_forever = asyncio.Event()

    async def binding_digest(self, agent_id: str | None) -> str:
        return "c" * 64

    async def run(
        self,
        agent_id: str | None,
        prompt: str,
        history: list[object],
        conversation_id: str,
        *,
        step_store: object,
        step_run_id: str,
        segment_sequence: int,
        parent_step_run_id: str | None = None,
        memory_namespace: str | None = None,
        memory_store: object | None = None,
        on_event: object | None = None,
    ) -> WorkspaceAgentResult:
        self.started.set()
        await self._wait_forever.wait()
        return WorkspaceAgentResult(step_run_id, prompt, [])


class _ReturnBarrierRunner:
    def __init__(self) -> None:
        self.ready = asyncio.Event()
        self.release = asyncio.Event()

    async def binding_digest(self, agent_id: str | None) -> str:
        return "d" * 64

    async def run(
        self,
        agent_id: str | None,
        prompt: str,
        history: list[object],
        conversation_id: str,
        *,
        step_store: object,
        step_run_id: str,
        segment_sequence: int,
        parent_step_run_id: str | None = None,
        memory_namespace: str | None = None,
        memory_store: object | None = None,
        on_event: object | None = None,
    ) -> WorkspaceAgentResult:
        self.ready.set()
        await self.release.wait()
        return WorkspaceAgentResult(step_run_id, prompt, [])


def _cancel_request(principal: Principal, request_id: str) -> CancelExecutionRequest:
    return CancelExecutionRequest(principal, request_id, True)


def _cancel_operation(operation_id: str, execution_id: str, tenant_id: str, status: OperationStatus = OperationStatus.PENDING) -> OperationLedgerInput:
    now = datetime.now(timezone.utc)
    return OperationLedgerInput(
        operation_id,
        tenant_id,
        ResourceKind.EXECUTION,
        execution_id,
        execution_id,
        OperationKind.EXECUTION_CANCEL,
        status,
        f"digest-{operation_id}",
        None,
        None,
        None,
        True,
        now,
        now,
    )


def _terminal_fixture(now: datetime, *, tenant_id: str, session_id: str, execution_id: str) -> tuple[SessionRecord, ExecutionRecord, ExecutionRecord, IdempotencyRecord, ResultRecord]:
    session = SessionRecord(session_id, tenant_id, "owner", "binding", SessionStatus.OPEN, 0, 0, None, {}, now, now, None, "head-a")
    execution = ExecutionRecord(execution_id, tenant_id, session_id, "binding", None, execution_id, None, "head-a", ExecutionLineageKind.SESSION_RESUME, ExecutionStatus.STARTED, 1, 0, 1, None, None, None, {}, now, now)
    terminal = replace(execution, status=ExecutionStatus.SUCCEEDED, revision=2, event_sequence=1, result_ref="digest", result_digest="digest", updated_at=now)
    identity = IdempotencyRecord(tenant_id, "session.resume", idempotency_key_hash("terminal-key"), "request", execution_id, IdempotencyStatus.STARTED, None, None, now, now)
    result = ResultRecord(execution_id, tenant_id, ExecutionStatus.SUCCEEDED, "none", 1, "none", "digest", "digest", StopReason.END_TURN, 0, 0, 0, now)
    return session, execution, terminal, identity, result


async def _seed_approval(persistence: RuntimePersistence, execution_id: str, approval_id: str, now: datetime) -> None:
    await persistence.executions.create(
        ExecutionRecord(execution_id, "tenant", None, "b" * 64, None, execution_id, None, None, ExecutionLineageKind.RUN, ExecutionStatus.STARTED, 1, 0, 0, None, None, None, {}, now, now)
    )
    await persistence.approvals.create(
        ApprovalRecord(approval_id, execution_id, "tenant", "operation", ApprovalStatus.PENDING, None, None, None, None, now, None)
    )


async def _assert_stale_terminal_is_atomic(persistence: RuntimePersistence) -> None:
    now = datetime.now(timezone.utc)
    session, execution, terminal, identity, result = _terminal_fixture(now, tenant_id="tenant", session_id="session", execution_id="execution")
    await persistence.sessions.create(session)
    await persistence.executions.create(execution)
    await persistence.idempotency.reserve(identity)
    commit = ExecutionTerminalCommit(
        1,
        0,
        terminal,
        result,
        ExecutionEventType.EXECUTION_SUCCEEDED,
        {"run_id": "run"},
        IdempotencyTerminalUpdate(identity.scope, identity.key_hash, identity.status, IdempotencyStatus.COMPLETED, identity.request_digest, "digest", None),
        None,
        SessionHeadAdvance("session", "head-b", "execution"),
    )
    with pytest.raises(AIError) as error:
        await persistence.results.commit_terminal(commit)
    assert error.value.code is ErrorCode.STORAGE_CONFLICT
    assert await persistence.executions.get("execution", tenant_id="tenant") == execution
    assert await persistence.results.get("execution", tenant_id="tenant") is None
    assert (await persistence.events.list("execution", tenant_id="tenant", after_sequence=0, limit=10)).items == ()
    assert await persistence.idempotency.get(identity.scope, identity.key_hash, tenant_id="tenant") == identity
    assert await persistence.sessions.get("session", tenant_id="tenant") == session


@pytest.mark.asyncio
async def test_terminal_session_head_stale_cas_is_atomic_for_memory_and_sqlite(tmp_path: Path) -> None:
    memory = build_in_memory_runtime(namespace="v5-memory")
    await memory.initialize()
    try:
        await _assert_stale_terminal_is_atomic(memory.persistence)
    finally:
        await memory.close()

    path = tmp_path / "runtime.db"
    config = RuntimePersistenceConfig.sqlite(str(path), namespace="v5-sql", deployment_id="test")
    async with open_sql_resources(config) as resources:
        await _assert_stale_terminal_is_atomic(resources.domain)
    with sqlite3.connect(path) as connection:
        assert connection.execute("select profile from ai_runtime_sessions where session_id = ?", ("session",)).fetchone() == ("",)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_workspace_resume_advances_head_from_the_previous_session_head(tmp_path: Path, backend: str) -> None:
    workspace = Workspace.load(tmp_path / backend)
    runner = WorkspaceAgentRunner(workspace.root, workspace.config, model=TestModel())
    config = RuntimePersistenceConfig.in_memory(namespace=workspace.workspace_id) if backend == "memory" else RuntimePersistenceConfig.sqlite(str(tmp_path / "runtime.db"), namespace=workspace.workspace_id, deployment_id="test")
    runtime_context = open_workspace_runtime(workspace, config=config, runner=runner) if backend == "memory" else _open_sql_workspace(workspace, config, runner=runner)
    async with runtime_context as runtime:
        results = [
            await runtime.run("main", prompt, idempotency_key=key, memory_namespace="test")
            for prompt, key in (("one", "k1"), ("two", "k2"), ("three", "k3"))
        ]
        records = [await runtime._resources.domain.executions.get(item.execution_id, tenant_id=workspace.workspace_id) for item in results]
        session = await runtime._resources.domain.sessions.get("main", tenant_id=workspace.workspace_id)
    assert len({item.execution_id for item in results}) == 3
    assert all(item is not None and item.status is ExecutionStatus.SUCCEEDED for item in records)
    assert records[0] is not None and records[0].base_execution_id is None
    assert records[1] is not None and records[1].base_execution_id == results[0].execution_id
    assert records[2] is not None and records[2].base_execution_id == results[1].execution_id
    assert session is not None and session.head_execution_id == results[2].execution_id


@pytest.mark.asyncio
async def test_concurrent_resume_has_one_session_head_winner(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    runner = _BarrierRunner()
    async with open_workspace_runtime(workspace, config=RuntimePersistenceConfig.in_memory(namespace=workspace.workspace_id), runner=runner) as runtime:
        outcomes = await asyncio.gather(
            runtime.run("main", "one", idempotency_key="first", memory_namespace="test"),
            runtime.run("main", "two", idempotency_key="second", memory_namespace="test"),
            return_exceptions=True,
        )
        successful = tuple(item for item in outcomes if not isinstance(item, BaseException))
        failures = tuple(item for item in outcomes if isinstance(item, BaseException))
        session = await runtime._resources.domain.sessions.get("main", tenant_id=workspace.workspace_id)
        executions = await runtime._resources.domain.executions.list_by_session("main", tenant_id=workspace.workspace_id)
    assert len(successful) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], AIError)
    assert failures[0].code is ErrorCode.STORAGE_CONFLICT
    assert session is not None and session.head_execution_id == successful[0].execution_id
    assert sum(item.status is ExecutionStatus.SUCCEEDED for item in executions) == 1
    assert sum(item.status is ExecutionStatus.FAILED for item in executions) == 1


@pytest.mark.asyncio
async def test_cancel_natural_terminal_wins_without_effect_unknown_or_storage_failure(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    runner = _ReturnBarrierRunner()
    principal = trusted_workspace_principal(workspace.workspace_id)
    async with open_workspace_runtime(workspace, config=RuntimePersistenceConfig.in_memory(namespace=workspace.workspace_id), runner=runner) as runtime:
        run_task = asyncio.create_task(runtime.run("main", "natural", idempotency_key="natural", memory_namespace="test"))
        await runner.ready.wait()
        execution = (await runtime._resources.domain.executions.list_by_session("main", tenant_id=workspace.workspace_id))[0]
        cancel_entered = asyncio.Event()
        allow_cancel = asyncio.Event()

        async def observe_cancel(record: ExecutionRecord) -> CancelEffectOutcome:
            cancel_entered.set()
            await allow_cancel.wait()
            assert not runtime._launcher._tasks[record.execution_id].cancelled()
            return CancelEffectOutcome.UNKNOWN

        runtime._launcher.cancel = observe_cancel
        cancel_task = asyncio.create_task(
            runtime._services.execution.cancel(execution.execution_id, _cancel_request(principal, "natural-cancel"))
        )
        await cancel_entered.wait()
        runner.release.set()
        completed = await run_task
        current = await runtime._resources.domain.executions.get(completed.execution_id, tenant_id=workspace.workspace_id)
        assert current is not None and current.status is ExecutionStatus.SUCCEEDED
        allow_cancel.set()
        cancel_result = await cancel_task
        operation = await runtime._resources.domain.operations.get(idempotency_key_hash("natural-cancel"), tenant_id=workspace.workspace_id)
        assert not cancel_result.cancelled
        assert operation is not None and operation.status is OperationStatus.SUCCEEDED
        assert operation.error_code is None


@pytest.mark.asyncio
async def test_workspace_launcher_cancel_uses_final_task_state(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    async with open_workspace_runtime(workspace, config=RuntimePersistenceConfig.in_memory(namespace=workspace.workspace_id), runner=_BlockingRunner()) as runtime:
        now = datetime.now(timezone.utc)
        execution = ExecutionRecord("launcher-state", workspace.workspace_id, None, "b" * 64, None, "launcher-state", None, None, ExecutionLineageKind.RUN, ExecutionStatus.STARTED, 1, 0, 0, None, None, None, {}, now, now)

        async def natural() -> WorkspaceAgentResult:
            return WorkspaceAgentResult("run", "", [])

        natural_task = asyncio.create_task(natural())
        await natural_task
        runtime._launcher._tasks[execution.execution_id] = natural_task
        assert await runtime._launcher.cancel(execution) is CancelEffectOutcome.UNKNOWN

        async def failed() -> WorkspaceAgentResult:
            raise RuntimeError("runner failure")

        failed_task = asyncio.create_task(failed())
        await asyncio.gather(failed_task, return_exceptions=True)
        runtime._launcher._tasks[execution.execution_id] = failed_task
        assert await runtime._launcher.cancel(execution) is CancelEffectOutcome.UNKNOWN
        runtime._launcher._tasks.pop(execution.execution_id, None)


@pytest.mark.asyncio
async def test_confirmed_cancel_requires_task_cancelled_and_closes_operation(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    runner = _BlockingRunner()
    principal = trusted_workspace_principal(workspace.workspace_id)
    async with open_workspace_runtime(workspace, config=RuntimePersistenceConfig.in_memory(namespace=workspace.workspace_id), runner=runner) as runtime:
        run_task = asyncio.create_task(runtime.run("main", "cancel", idempotency_key="cancel", memory_namespace="test"))
        await runner.started.wait()
        execution = (await runtime._resources.domain.executions.list_by_session("main", tenant_id=workspace.workspace_id))[0]
        cancel_result = await runtime._services.execution.cancel(execution.execution_id, _cancel_request(principal, "confirmed-cancel"))
        with contextlib.suppress(asyncio.CancelledError):
            await run_task
        current = await runtime._resources.domain.executions.get(execution.execution_id, tenant_id=workspace.workspace_id)
        operation = await runtime._resources.domain.operations.get(idempotency_key_hash("confirmed-cancel"), tenant_id=workspace.workspace_id)
        assert cancel_result.cancelled
        assert current is not None and current.status is ExecutionStatus.CANCELLED
        assert operation is not None and operation.status is OperationStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_cancel_backend_failure_keeps_stable_error_in_operation_ledger(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    runner = _ReturnBarrierRunner()
    principal = trusted_workspace_principal(workspace.workspace_id)
    async with open_workspace_runtime(workspace, config=RuntimePersistenceConfig.in_memory(namespace=workspace.workspace_id), runner=runner) as runtime:
        run_task = asyncio.create_task(runtime.run("main", "backend", idempotency_key="backend", memory_namespace="test"))
        await runner.ready.wait()
        execution = (await runtime._resources.domain.executions.list_by_session("main", tenant_id=workspace.workspace_id))[0]

        async def fail_cancel(record: ExecutionRecord) -> CancelEffectOutcome:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        runtime._launcher.cancel = fail_cancel
        with pytest.raises(AIError) as error:
            await runtime._services.execution.cancel(execution.execution_id, _cancel_request(principal, "backend-cancel"))
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        operation = await runtime._resources.domain.operations.get(idempotency_key_hash("backend-cancel"), tenant_id=workspace.workspace_id)
        assert operation is not None and operation.status is OperationStatus.FAILED
        assert operation.error_code == ErrorCode.STORAGE_INTEGRITY_ERROR.value
        runner.release.set()
        await run_task


@pytest.mark.asyncio
async def test_cancel_terminal_wins_before_request_cancel_cas(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    runner = _ReturnBarrierRunner()
    principal = trusted_workspace_principal(workspace.workspace_id)
    async with open_workspace_runtime(workspace, config=RuntimePersistenceConfig.in_memory(namespace=workspace.workspace_id), runner=runner) as runtime:
        run_task = asyncio.create_task(runtime.run("main", "cas", idempotency_key="cas", memory_namespace="test"))
        await runner.ready.wait()
        execution = (await runtime._resources.domain.executions.list_by_session("main", tenant_id=workspace.workspace_id))[0]
        cas_entered = asyncio.Event()
        continue_cas = asyncio.Event()
        request_cancel = runtime._resources.domain.executions.request_cancel

        async def delayed_request_cancel(commit: ExecutionCancelRequestCommit) -> ExecutionRecord:
            cas_entered.set()
            await continue_cas.wait()
            return await request_cancel(commit)

        runtime._resources.domain.executions.request_cancel = delayed_request_cancel
        cancel_task = asyncio.create_task(
            runtime._services.execution.cancel(execution.execution_id, _cancel_request(principal, "cas-cancel"))
        )
        await cas_entered.wait()
        runner.release.set()
        await run_task
        continue_cas.set()
        cancel_result = await cancel_task
        operation = await runtime._resources.domain.operations.get(idempotency_key_hash("cas-cancel"), tenant_id=workspace.workspace_id)
        current = await runtime._resources.domain.executions.get(execution.execution_id, tenant_id=workspace.workspace_id)
        assert not cancel_result.cancelled
        assert current is not None and current.status is ExecutionStatus.SUCCEEDED
        assert operation is not None and operation.status is OperationStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_concurrent_cancel_request_ids_share_terminal_but_keep_ledgers(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    runner = _BlockingRunner()
    principal = trusted_workspace_principal(workspace.workspace_id)
    async with open_workspace_runtime(workspace, config=RuntimePersistenceConfig.in_memory(namespace=workspace.workspace_id), runner=runner) as runtime:
        run_task = asyncio.create_task(runtime.run("main", "concurrent-cancel", idempotency_key="concurrent-cancel", memory_namespace="test"))
        await runner.started.wait()
        execution = (await runtime._resources.domain.executions.list_by_session("main", tenant_id=workspace.workspace_id))[0]
        results = await asyncio.gather(
            runtime._services.execution.cancel(execution.execution_id, _cancel_request(principal, "cancel-a")),
            runtime._services.execution.cancel(execution.execution_id, _cancel_request(principal, "cancel-b")),
        )
        with contextlib.suppress(asyncio.CancelledError):
            await run_task
        current = await runtime._resources.domain.executions.get(execution.execution_id, tenant_id=workspace.workspace_id)
        operations = [
            await runtime._resources.domain.operations.get(idempotency_key_hash(request_id), tenant_id=workspace.workspace_id)
            for request_id in ("cancel-a", "cancel-b")
        ]
        assert [item.cancelled for item in results] == [True, True]
        assert current is not None and current.status is ExecutionStatus.CANCELLED
        assert all(item is not None and item.status is OperationStatus.SUCCEEDED and item.error_code is None for item in operations)


async def _assert_request_cancel_operation_parity(persistence: RuntimePersistence) -> None:
    now = datetime.now(timezone.utc)
    execution_id = "cancelling-execution"
    tenant_id = "tenant"
    execution = ExecutionRecord(execution_id, tenant_id, None, "b" * 64, None, execution_id, None, None, ExecutionLineageKind.RUN, ExecutionStatus.STARTED, 1, 0, 0, None, None, None, {}, now, now)
    await persistence.executions.create(execution)
    first = await persistence.operations.append(_cancel_operation("cancel-first", execution_id, tenant_id))
    current = await persistence.executions.request_cancel(ExecutionCancelRequestCommit(execution_id, tenant_id, 1, 0, first.operation_id, now))
    assert current.status is ExecutionStatus.CANCELLING
    with pytest.raises(AIError) as missing:
        await persistence.executions.request_cancel(ExecutionCancelRequestCommit(execution_id, tenant_id, 1, 0, "missing", now))
    assert missing.value.code is ErrorCode.STORAGE_CONFLICT
    completed = await persistence.operations.append(_cancel_operation("cancel-completed", execution_id, tenant_id, OperationStatus.SUCCEEDED))
    with pytest.raises(AIError) as non_pending:
        await persistence.executions.request_cancel(ExecutionCancelRequestCommit(execution_id, tenant_id, 1, 0, completed.operation_id, now))
    assert non_pending.value.code is ErrorCode.STORAGE_CONFLICT
    mismatch = await persistence.operations.append(_cancel_operation("cancel-mismatch", "other-execution", tenant_id))
    with pytest.raises(AIError) as wrong_execution:
        await persistence.executions.request_cancel(ExecutionCancelRequestCommit(execution_id, tenant_id, 1, 0, mismatch.operation_id, now))
    assert wrong_execution.value.code is ErrorCode.STORAGE_CONFLICT
    second = await persistence.operations.append(_cancel_operation("cancel-second", execution_id, tenant_id))
    assert await persistence.executions.request_cancel(ExecutionCancelRequestCommit(execution_id, tenant_id, 1, 0, second.operation_id, now)) == current


@pytest.mark.asyncio
async def test_memory_and_sqlite_request_cancel_operation_preconditions_match(tmp_path: Path) -> None:
    memory = build_in_memory_runtime(namespace="cancel-parity-memory")
    await memory.initialize()
    try:
        await _assert_request_cancel_operation_parity(memory.persistence)
    finally:
        await memory.close()
    config = RuntimePersistenceConfig.sqlite(str(tmp_path / "cancel-parity.db"), namespace="cancel-parity-sql", deployment_id="test")
    async with open_sql_resources(config) as resources:
        await _assert_request_cancel_operation_parity(resources.domain)


@pytest.mark.asyncio
async def test_runtime_access_exposes_query_only_wrappers() -> None:
    runtime = build_in_memory_runtime(namespace="v5-access")
    await runtime.initialize()
    try:
        services = build_runtime_services(
            runtime.persistence,
            TenantAuthorizationPolicy(),
            grant_key=b"v5-access-key",
            history_reader=_History(),
            schema_digest=runtime.persistence.atomic_domain_id,
            execution_launcher=_Launcher(),
        )
        access = build_runtime_access(services)
        assert hasattr(access.execution, "inspect")
        assert hasattr(access.session, "list")
        assert hasattr(access.task, "inspect_graph")
        assert hasattr(access.approval, "list")
        assert not hasattr(access.approval, "decide")
        assert not hasattr(access.task, "run_graph")
        assert not hasattr(access.task, "cancel_graph")
        assert not hasattr(access.session, "create")
        assert not hasattr(access.session, "resume")
        assert not hasattr(access.session, "fork")
        assert not hasattr(access.session, "update")
        assert not hasattr(access.session, "close")
        assert hasattr(services.approval, "decide")
        assert hasattr(services.task, "run_graph")
        assert hasattr(services.session, "create")
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_runtime_composition_is_driven_by_injected_dependencies() -> None:
    runtime = build_in_memory_runtime(namespace="v5-composition")
    await runtime.initialize()
    try:
        principal = Principal("owner", "tenant")
        explicit = _Launcher()
        gateway = _Gateway()
        explicit_services = build_runtime_services(
            runtime.persistence,
            TenantAuthorizationPolicy(),
            grant_key=b"v5-composition-key",
            history_reader=_History(),
            schema_digest=runtime.persistence.atomic_domain_id,
            execution_launcher=explicit,
        )
        explicit_handle = await explicit_services.execution.run("b" * 64, ExecutionRequest("explicit", principal, "explicit-key", memory_namespace="test"))
        assert explicit.started == [explicit_handle.execution_id]

        gateway_services = build_runtime_services(
            runtime.persistence,
            TenantAuthorizationPolicy(),
            grant_key=b"v5-composition-key",
            history_reader=_History(),
            schema_digest=runtime.persistence.atomic_domain_id,
            workflow_gateway=gateway,
        )
        gateway_handle = await gateway_services.execution.run("b" * 64, ExecutionRequest("gateway", principal, "gateway-key", memory_namespace="test"))
        assert gateway.execution_starts == [gateway_handle.execution_id]

        combined_launcher = _Launcher()
        combined_gateway = _Gateway()
        combined_services = build_runtime_services(
            runtime.persistence,
            TenantAuthorizationPolicy(),
            grant_key=b"v5-composition-key",
            history_reader=_History(),
            schema_digest=runtime.persistence.atomic_domain_id,
            execution_launcher=combined_launcher,
            workflow_gateway=combined_gateway,
        )
        combined_handle = await combined_services.execution.run("b" * 64, ExecutionRequest("combined", principal, "combined-key", memory_namespace="test"))
        graph = TaskGraph("combined-graph", (TaskNode("node"),))
        await combined_services.task.run_graph("b" * 64, TaskGraphRequest(graph, principal, "graph-key"))
        assert combined_launcher.started == [combined_handle.execution_id]
        assert combined_gateway.execution_starts == []
        assert combined_gateway.task_starts == [graph.graph_id]
        await _seed_approval(runtime.persistence, "approval-combined", "approval-combined", datetime.now(timezone.utc))
        await combined_services.approval.decide(
            "approval-combined",
            ApprovalDecisionRequest(principal, "approval-combined", "decision-combined", ApprovalDecision.APPROVE),
        )
        assert combined_gateway.updates == []

        gateway_approval = _Gateway()
        gateway_only = build_runtime_services(
            runtime.persistence,
            TenantAuthorizationPolicy(),
            grant_key=b"v5-composition-key",
            history_reader=_History(),
            schema_digest=runtime.persistence.atomic_domain_id,
            workflow_gateway=gateway_approval,
        )
        await _seed_approval(runtime.persistence, "approval-gateway", "approval-gateway", datetime.now(timezone.utc))
        await gateway_only.approval.decide(
            "approval-gateway",
            ApprovalDecisionRequest(principal, "approval-gateway", "decision-gateway", ApprovalDecision.APPROVE),
        )
        assert gateway_approval.updates == ["approve"]
        with pytest.raises(AIError) as error:
            build_runtime_services(
                runtime.persistence,
                TenantAuthorizationPolicy(),
                grant_key=b"v5-composition-key",
                history_reader=_History(),
                schema_digest=runtime.persistence.atomic_domain_id,
            )
        assert error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
    finally:
        await runtime.close()


def test_public_runtime_contract_has_no_profile_or_replacement_category() -> None:
    import linktools.ai.core as core

    from linktools.ai.core import ExecutionEventType
    from linktools.ai.runtime import ExecutionRecord
    from linktools.ai.task import TaskGraphRequest

    assert not hasattr(core, "ExecutionProfile")
    assert not hasattr(ErrorCode, "PROFILE_NOT_ALLOWED")
    assert not hasattr(ErrorCode, "CAPABILITY_DISABLED_FOR_PROFILE")
    for contract in (ExecutionRequest, TaskGraphRequest, RuntimeServiceIdentity, SessionRecord, ExecutionRecord, ExecutionHandle, ExecutionView):
        assert "profile" not in {item.name for item in fields(contract)}
    assert "profile" not in inspect.signature(build_runtime_services).parameters
    assert "temporal_enabled" not in inspect.signature(build_runtime_services).parameters
    assert "temporal_enabled" not in inspect.signature(open_runtime_services).parameters
    assert "temporal_enabled" not in inspect.signature(open_workspace_runtime).parameters
    assert ExecutionEventType.EXECUTION_SUCCEEDED.value == "EXECUTION_SUCCEEDED"
    source_root = Path("linktools-ai/src/linktools/ai")
    forbidden = ("ExecutionProfile", "local-coding", "production-service", "production-sandboxed", "ExecutionMode", "RuntimeMode", "DeploymentMode")
    for path in source_root.rglob("*.py"):
        if path.parts[-2:] == ("adapter", "schema.py"):
            source = path.read_text(encoding="utf-8").replace('Column("profile"', "").replace('owner="adapter.sql"', "")
        else:
            source = path.read_text(encoding="utf-8")
        assert not any(value in source for value in forbidden), path


@pytest.mark.asyncio
async def test_file_decoder_ignores_legacy_profile_fields(tmp_path: Path) -> None:
    workspace_id = "legacy-workspace"
    runtime = build_filesystem_runtime(str(tmp_path), workspace_id=workspace_id)
    await runtime.initialize()
    now = datetime.now(timezone.utc)
    await runtime.persistence.sessions.create(SessionRecord("legacy", workspace_id, "owner", "binding", SessionStatus.OPEN, 0, 0, None, {}, now, now, None))
    await runtime.persistence.executions.create(ExecutionRecord("legacy-execution", workspace_id, "legacy", "binding", None, "legacy-execution", None, None, ExecutionLineageKind.RUN, ExecutionStatus.PENDING_START, 0, 0, 0, None, None, None, {}, now, now))
    await runtime.close()
    state_path = next(tmp_path.rglob("state.json"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["records"]["sessions"][0]["profile"] = "local-coding"
    state["records"]["executions"][0]["profile"] = "production-service"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    reopened = build_filesystem_runtime(str(tmp_path), workspace_id=workspace_id)
    await reopened.initialize()
    try:
        session = await reopened.persistence.sessions.get("legacy", tenant_id=workspace_id)
        execution = await reopened.persistence.executions.get("legacy-execution", tenant_id=workspace_id)
        assert session is not None and execution is not None
        assert not hasattr(session, "profile")
        assert not hasattr(execution, "profile")
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_sql_legacy_profile_tombstone_is_not_domain_semantic(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    config = RuntimePersistenceConfig.sqlite(str(path), namespace="legacy-sql", deployment_id="test")
    now = datetime.now(timezone.utc)
    session = SessionRecord("legacy-sql-session", "legacy-sql", "owner", "binding", SessionStatus.OPEN, 0, 0, None, {}, now, now, None, None)
    async with open_sql_resources(config) as resources:
        await resources.domain.sessions.create(session)
        with sqlite3.connect(path) as connection:
            connection.execute("update ai_runtime_sessions set profile = ? where session_id = ?", ("local-coding", session.session_id))
            connection.commit()
    async with open_sql_resources(config) as resources:
        loaded = await resources.domain.sessions.get(session.session_id, tenant_id=session.tenant_id)
        assert loaded is not None and not hasattr(loaded, "profile")
        updated = replace(loaded, revision=loaded.revision + 1, updated_at=datetime.now(timezone.utc))
        await resources.domain.sessions.compare_and_swap(session.session_id, tenant_id=session.tenant_id, expected_revision=loaded.revision, next_record=updated)
    with sqlite3.connect(path) as connection:
        assert connection.execute("select profile from ai_runtime_sessions where session_id = ?", (session.session_id,)).fetchone() == ("",)
    source = Path("linktools-ai/src/linktools/ai/adapter/_sql.py").read_text(encoding="utf-8")
    assert source.count('profile=""') == 2
    assert "profile" not in source.replace('profile=""', "")
    ast.parse(source)


@pytest.mark.asyncio
async def test_profile_removal_preserves_idempotency_conflicts_and_digest_scope() -> None:
    runtime = build_in_memory_runtime(namespace="profile-boundary")
    await runtime.initialize()
    try:
        principal = Principal("owner", "tenant")
        binding_digest = "b" * 64
        launcher = _Launcher()
        services = build_runtime_services(
            runtime.persistence,
            TenantAuthorizationPolicy(),
            grant_key=b"profile-boundary-key",
            history_reader=_History(),
            schema_digest=runtime.persistence.atomic_domain_id,
            execution_launcher=launcher,
        )
        now = datetime.now(timezone.utc)
        await runtime.persistence.sessions.create(SessionRecord("resume", "tenant", "owner", binding_digest, SessionStatus.OPEN, 0, 0, None, {}, now, now, None, None))
        await runtime.persistence.executions.create(ExecutionRecord("source", "tenant", None, binding_digest, None, "source", None, None, ExecutionLineageKind.RUN, ExecutionStatus.STARTED, 1, 0, 0, None, None, None, {}, now, now))

        async def reserve_old(scope: str, key: str, execution_id: str) -> None:
            await runtime.persistence.idempotency.reserve(
                IdempotencyRecord("tenant", scope, idempotency_key_hash(key), canonical_sha256({"scope": scope, "profile": "local-coding", "legacy": True}), execution_id, IdempotencyStatus.STARTED, None, None, now, now)
            )

        for scope, key, execution_id in (
            ("execution.run", "old-run", "old-run-execution"),
            ("execution.retry", "old-retry", "old-retry-execution"),
            ("execution.fork", "old-fork", "old-fork-execution"),
            ("session.resume", "old-resume", "old-resume-execution"),
        ):
            await reserve_old(scope, key, execution_id)
        old_create = OperationLedgerInput(
            idempotency_key_hash("old-create"),
            "tenant",
            ResourceKind.SESSION,
            "old-session",
            None,
            OperationKind.SESSION_CREATE,
            OperationStatus.PENDING,
            canonical_sha256({"scope": "session.create", "profile": "local-coding", "legacy": True}),
            None,
            None,
            None,
            True,
            now,
            now,
        )
        await runtime.persistence.operations.append(old_create)
        pre_sessions = await runtime.persistence.sessions.list(tenant_id="tenant")
        pre_executions = await runtime.persistence.executions.list_by_session("resume", tenant_id="tenant")
        pre_execution_ids = {"source", *(item.execution_id for item in pre_executions)}
        pre_session_ids = {item.session_id for item in pre_sessions}
        pre_resume = await runtime.persistence.sessions.get("resume", tenant_id="tenant")
        pre_source = await runtime.persistence.executions.get("source", tenant_id="tenant")
        pre_legacy_idempotency = {
            scope: await runtime.persistence.idempotency.get(scope, idempotency_key_hash(key), tenant_id="tenant")
            for scope, key in (
                ("execution.run", "old-run"),
                ("execution.retry", "old-retry"),
                ("execution.fork", "old-fork"),
                ("session.resume", "old-resume"),
            )
        }
        pre_legacy_operation = await runtime.persistence.operations.get(old_create.operation_id, tenant_id="tenant")
        pre_started = tuple(launcher.started)
        old_calls = (
            services.execution.run(binding_digest, ExecutionRequest("run", principal, "old-run", memory_namespace="test")),
            services.execution.retry(binding_digest, "source", RetryExecutionRequest("retry", principal, "old-retry")),
            services.execution.fork(binding_digest, "source", ForkExecutionRequest("fork", principal, "old-fork")),
            services.execution.run_for_session(binding_digest, "resume", ExecutionRequest("resume", principal, "old-resume", memory_namespace="test")),
            services.session.create(binding_digest, CreateSessionRequest(principal, "old-session", "old-create")),
        )
        for call in old_calls:
            with pytest.raises(AIError) as error:
                await call
            assert error.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        post_sessions = await runtime.persistence.sessions.list(tenant_id="tenant")
        post_executions = await runtime.persistence.executions.list_by_session("resume", tenant_id="tenant")
        post_execution_ids = {"source", *(item.execution_id for item in post_executions)}
        post_session_ids = {item.session_id for item in post_sessions}
        post_resume = await runtime.persistence.sessions.get("resume", tenant_id="tenant")
        post_source = await runtime.persistence.executions.get("source", tenant_id="tenant")
        post_legacy_idempotency = {
            scope: await runtime.persistence.idempotency.get(scope, idempotency_key_hash(key), tenant_id="tenant")
            for scope, key in (
                ("execution.run", "old-run"),
                ("execution.retry", "old-retry"),
                ("execution.fork", "old-fork"),
                ("session.resume", "old-resume"),
            )
        }
        post_legacy_operation = await runtime.persistence.operations.get(old_create.operation_id, tenant_id="tenant")
        assert post_execution_ids == pre_execution_ids
        assert post_session_ids == pre_session_ids
        assert post_resume == pre_resume
        assert post_source == pre_source
        assert tuple(launcher.started) == pre_started
        assert post_legacy_idempotency == pre_legacy_idempotency
        assert post_legacy_operation == pre_legacy_operation
        assert await runtime.persistence.sessions.get("old-session", tenant_id="tenant") is None
        assert await runtime.persistence.executions.get("old-run-execution", tenant_id="tenant") is None

        fresh = await services.execution.run(binding_digest, ExecutionRequest("fresh", principal, "fresh-run", memory_namespace="test"))
        identity = await runtime.persistence.idempotency.get("execution.run", idempotency_key_hash("fresh-run"), tenant_id="tenant")
        assert identity is not None
        assert identity.request_digest == canonical_sha256(
            {
                "prompt": "fresh",
                "binding_digest": binding_digest,
                "scope": "execution",
                "principal_id": "owner",
                "tenant_id": "tenant",
                "session_id": None,
                "source_execution_id": None,
                "base_execution_id": None,
                    "parent_execution_id": None,
                    "root_identity": "$self",
                    "lineage_kind": ExecutionLineageKind.RUN.value,
                    "memory_namespace_digest": canonical_sha256("test"),
                }
        )
        assert fresh.execution_id != "source"
        created = await services.session.create(binding_digest, CreateSessionRequest(principal, "fresh-session", "fresh-create"))
        create_operation = await runtime.persistence.operations.get(idempotency_key_hash("fresh-create"), tenant_id="tenant")
        assert created.session_id == "fresh-session" and create_operation is not None
        assert create_operation.request_digest == canonical_sha256({"action": "session.create", "tenant_id": "tenant", "principal_id": "owner", "session_id": "fresh-session", "binding": binding_digest})

        graph_request = TaskGraphRequest(TaskGraph("profile-graph", (TaskNode("node"),)), principal, "fresh-graph")
        await services.task.run_graph(binding_digest, graph_request)
        graph_operation = await runtime.persistence.operations.get(idempotency_key_hash("fresh-graph"), tenant_id="tenant")
        assert graph_operation is not None
        expected_graph_digest = canonical_sha256({"graph_id": "profile-graph", "nodes": ["node"], "binding": binding_digest, "limits": {"max_nodes": graph_request.limits.max_nodes, "max_depth": graph_request.limits.max_depth, "max_budget": graph_request.limits.max_budget, "max_concurrency": graph_request.limits.max_concurrency}})
        assert graph_operation.request_digest == expected_graph_digest
        assert graph_operation.request_digest != canonical_sha256({"graph_id": "profile-graph", "nodes": ["node"], "binding": binding_digest, "profile": "local-coding", "limits": {"max_nodes": graph_request.limits.max_nodes, "max_depth": graph_request.limits.max_depth, "max_budget": graph_request.limits.max_budget, "max_concurrency": graph_request.limits.max_concurrency}})
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_task_run_same_key_has_single_effect_owner() -> None:
    runtime = build_in_memory_runtime(namespace="task-run-owner")
    await runtime.initialize()
    try:
        principal = Principal("owner", "tenant")
        gateway = _TaskGateway()
        gateway.block_start = True
        services = build_runtime_services(
            runtime.persistence,
            TenantAuthorizationPolicy(),
            grant_key=b"task-run-owner-key",
            history_reader=_History(),
            schema_digest=runtime.persistence.atomic_domain_id,
            workflow_gateway=gateway,
            execution_launcher=_Launcher(),
        )
        create_plan = runtime.persistence.tasks.create_plan
        create_count = 0

        async def counted_create_plan(graph: TaskGraph, *, tenant_id: str) -> object:
            nonlocal create_count
            create_count += 1
            return await create_plan(graph, tenant_id=tenant_id)

        runtime.persistence.tasks.create_plan = counted_create_plan
        request = TaskGraphRequest(TaskGraph("owner-graph", (TaskNode("node"),)), principal, "owner-key")
        winner = asyncio.create_task(services.task.run_graph("b" * 64, request))
        await gateway.start_entered.wait()
        loser = asyncio.create_task(services.task.run_graph("b" * 64, request))
        with pytest.raises(AIError) as conflict:
            await loser
        assert conflict.value.code is ErrorCode.STORAGE_CONFLICT
        gateway.start_release.set()
        result = await winner
        operation = await runtime.persistence.operations.get(idempotency_key_hash("owner-key"), tenant_id="tenant")
        plan = await runtime.persistence.tasks.get_plan("owner-graph", tenant_id="tenant")
        assert result.graph_id == "owner-graph"
        assert operation is not None and operation.status is OperationStatus.SUCCEEDED
        assert plan is not None
        assert create_count == 1
        replay = await services.task.run_graph("b" * 64, request)
        assert replay == result
        assert gateway.task_starts == ["owner-graph"]
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_task_failed_operation_replays_stable_error() -> None:
    runtime = build_in_memory_runtime(namespace="task-failed-replay")
    await runtime.initialize()
    try:
        principal = Principal("owner", "tenant")
        gateway = _TaskGateway(start_error=AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY))
        services = build_runtime_services(
            runtime.persistence,
            TenantAuthorizationPolicy(),
            grant_key=b"task-failed-replay-key",
            history_reader=_History(),
            schema_digest=runtime.persistence.atomic_domain_id,
            workflow_gateway=gateway,
            execution_launcher=_Launcher(),
        )
        request = TaskGraphRequest(TaskGraph("failed-graph", (TaskNode("node"),)), principal, "failed-key")
        with pytest.raises(AIError) as first_error:
            await services.task.run_graph("b" * 64, request)
        with pytest.raises(AIError) as replay_error:
            await services.task.run_graph("b" * 64, request)
        operation = await runtime.persistence.operations.get(idempotency_key_hash("failed-key"), tenant_id="tenant")
        assert first_error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
        assert replay_error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
        assert operation is not None and operation.status is OperationStatus.FAILED
        assert operation.error_code == ErrorCode.RUNTIME_DEPENDENCY_NOT_READY.value
        assert gateway.task_starts == ["failed-graph"]

        gateway.start_error = RuntimeError("gateway unavailable")
        ordinary = TaskGraphRequest(TaskGraph("ordinary-failed-graph", (TaskNode("node"),)), principal, "ordinary-failed-key")
        with pytest.raises(RuntimeError):
            await services.task.run_graph("b" * 64, ordinary)
        with pytest.raises(AIError) as ordinary_replay:
            await services.task.run_graph("b" * 64, ordinary)
        ordinary_operation = await runtime.persistence.operations.get(idempotency_key_hash("ordinary-failed-key"), tenant_id="tenant")
        assert ordinary_replay.value.code is ErrorCode.STORAGE_UNAVAILABLE
        assert ordinary_operation is not None and ordinary_operation.error_code == ErrorCode.STORAGE_UNAVAILABLE.value
        assert gateway.task_starts == ["failed-graph", "ordinary-failed-graph"]

        missing_plan = TaskGraphRequest(TaskGraph("missing-plan", (TaskNode("node"),)), principal, "missing-plan-key")
        missing_digest = canonical_sha256({"graph_id": "missing-plan", "nodes": ["node"], "binding": "b" * 64, "limits": {"max_nodes": missing_plan.limits.max_nodes, "max_depth": missing_plan.limits.max_depth, "max_budget": missing_plan.limits.max_budget, "max_concurrency": missing_plan.limits.max_concurrency}})
        now = datetime.now(timezone.utc)
        await runtime.persistence.operations.append(OperationLedgerInput(idempotency_key_hash("missing-plan-key"), "tenant", ResourceKind.TASK_GRAPH, "missing-plan", None, OperationKind.TASK_NODE, OperationStatus.SUCCEEDED, missing_digest, "missing-plan", "missing-digest", None, True, now, now))
        with pytest.raises(AIError) as missing_error:
            await services.task.run_graph("b" * 64, missing_plan)
        assert missing_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

        corrupt_plan = TaskGraphRequest(TaskGraph("corrupt-plan", (TaskNode("node"),)), principal, "corrupt-plan-key")
        corrupt_digest = canonical_sha256({"graph_id": "corrupt-plan", "nodes": ["node"], "binding": "b" * 64, "limits": {"max_nodes": corrupt_plan.limits.max_nodes, "max_depth": corrupt_plan.limits.max_depth, "max_budget": corrupt_plan.limits.max_budget, "max_concurrency": corrupt_plan.limits.max_concurrency}})
        await runtime.persistence.operations.append(OperationLedgerInput(idempotency_key_hash("corrupt-plan-key"), "tenant", ResourceKind.TASK_GRAPH, "corrupt-plan", None, OperationKind.TASK_NODE, OperationStatus.FAILED, corrupt_digest, None, None, "UNKNOWN_PERSISTED_ERROR", True, now, now))
        with pytest.raises(AIError) as corrupt_error:
            await services.task.run_graph("b" * 64, corrupt_plan)
        assert corrupt_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_task_cancel_same_key_has_single_effect_owner() -> None:
    runtime = build_in_memory_runtime(namespace="task-cancel-owner")
    await runtime.initialize()
    try:
        principal = Principal("owner", "tenant")
        gateway = _TaskGateway()
        services = build_runtime_services(
            runtime.persistence,
            TenantAuthorizationPolicy(),
            grant_key=b"task-cancel-owner-key",
            history_reader=_History(),
            schema_digest=runtime.persistence.atomic_domain_id,
            workflow_gateway=gateway,
            execution_launcher=_Launcher(),
        )
        graph_request = TaskGraphRequest(TaskGraph("cancel-owner-graph", (TaskNode("node"),)), principal, "cancel-owner-run")
        await services.task.run_graph("b" * 64, graph_request)
        cancel_plan = runtime.persistence.tasks.cancel_plan
        cancel_count = 0

        async def counted_cancel_plan(graph_id: str, *, tenant_id: str) -> object:
            nonlocal cancel_count
            cancel_count += 1
            return await cancel_plan(graph_id, tenant_id=tenant_id)

        runtime.persistence.tasks.cancel_plan = counted_cancel_plan
        gateway.block_cancel = True
        request = CancelGraphRequest(principal, "cancel-owner-key")
        winner = asyncio.create_task(services.task.cancel_graph("cancel-owner-graph", request))
        await gateway.cancel_entered.wait()
        loser = asyncio.create_task(services.task.cancel_graph("cancel-owner-graph", request))
        with pytest.raises(AIError) as conflict:
            await loser
        assert conflict.value.code is ErrorCode.STORAGE_CONFLICT
        gateway.cancel_release.set()
        result = await winner
        replay = await services.task.cancel_graph("cancel-owner-graph", request)
        operation = await runtime.persistence.operations.get(idempotency_key_hash("cancel-owner-key"), tenant_id="tenant")
        assert result == replay
        assert result.status is TaskStatus.CANCELLED
        assert cancel_count == 1
        assert gateway.task_cancels == ["cancel-owner-graph"]
        assert operation is not None and operation.status is OperationStatus.SUCCEEDED
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_execution_cancel_same_id_unknown_then_confirmed_converges() -> None:
    runtime = build_in_memory_runtime(namespace="cancel-unknown-confirmed")
    await runtime.initialize()
    try:
        tenant_id = "tenant"
        principal = Principal("owner", tenant_id)
        now = datetime.now(timezone.utc)
        execution_id = "cancel-race-execution"
        await runtime.persistence.executions.create(
            ExecutionRecord(execution_id, tenant_id, None, "b" * 64, None, execution_id, None, None, ExecutionLineageKind.RUN, ExecutionStatus.STARTED, 1, 0, 1, None, None, None, {}, now, now)
        )
        launcher = _CancelRaceLauncher()
        services = build_runtime_services(
            runtime.persistence,
            TenantAuthorizationPolicy(),
            grant_key=b"cancel-race-key",
            history_reader=_History(),
            schema_digest=runtime.persistence.atomic_domain_id,
            execution_launcher=launcher,
        )
        request = _cancel_request(principal, "cancel-race-request")
        unknown = asyncio.create_task(services.execution.cancel(execution_id, request))
        await launcher.first_cancel_entered.wait()
        confirmed = await services.execution.cancel(execution_id, request)
        launcher.first_cancel_release.set()
        unknown_result = await unknown
        operation = await runtime.persistence.operations.get(idempotency_key_hash("cancel-race-request"), tenant_id=tenant_id)
        execution = await runtime.persistence.executions.get(execution_id, tenant_id=tenant_id)
        assert confirmed.cancelled
        assert unknown_result.cancelled
        assert execution is not None and execution.status is ExecutionStatus.CANCELLED
        assert operation is not None and operation.status is OperationStatus.SUCCEEDED
        assert (await services.execution.cancel(execution_id, request)).cancelled
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("execution_status", "expected_cancelled"),
    ((ExecutionStatus.SUCCEEDED, False), (ExecutionStatus.CANCELLED, True)),
)
async def test_execution_cancel_effect_unknown_resolves_actual_terminal(execution_status: ExecutionStatus, expected_cancelled: bool) -> None:
    runtime = build_in_memory_runtime(namespace=f"cancel-unknown-{execution_status.value.lower()}")
    await runtime.initialize()
    try:
        tenant_id = "tenant"
        principal = Principal("owner", tenant_id)
        now = datetime.now(timezone.utc)
        execution_id = f"terminal-{execution_status.value.lower()}"
        execution = ExecutionRecord(execution_id, tenant_id, None, "b" * 64, None, execution_id, None, None, ExecutionLineageKind.RUN, execution_status, 1, 0, 0, None, "terminal-digest", None, {}, now, now)
        await runtime.persistence.executions.create(execution)
        operation_id = idempotency_key_hash(f"unknown-{execution_id}")
        digest = canonical_sha256({"action": "execution.cancel", "tenant_id": tenant_id, "principal_id": principal.principal_id, "execution_id": execution_id, "force": True})
        await runtime.persistence.operations.append(OperationLedgerInput(operation_id, tenant_id, ResourceKind.EXECUTION, execution_id, execution_id, OperationKind.EXECUTION_CANCEL, OperationStatus.EFFECT_UNKNOWN, digest, None, None, None, True, now, now))
        launcher = _Launcher()
        services = build_runtime_services(
            runtime.persistence,
            TenantAuthorizationPolicy(),
            grant_key=b"cancel-unknown-key",
            history_reader=_History(),
            schema_digest=runtime.persistence.atomic_domain_id,
            execution_launcher=launcher,
        )
        result = await services.execution.cancel(execution_id, _cancel_request(principal, f"unknown-{execution_id}"))
        current_operation = await runtime.persistence.operations.get(operation_id, tenant_id=tenant_id)
        assert result.cancelled is expected_cancelled
        assert current_operation is not None and current_operation.status is OperationStatus.SUCCEEDED
        assert launcher.started == []

        failed_operation_id = idempotency_key_hash(f"failed-{execution_id}")
        failed_digest = canonical_sha256({"action": "execution.cancel", "tenant_id": tenant_id, "principal_id": principal.principal_id, "execution_id": execution_id, "force": True})
        await runtime.persistence.operations.append(OperationLedgerInput(failed_operation_id, tenant_id, ResourceKind.EXECUTION, execution_id, execution_id, OperationKind.EXECUTION_CANCEL, OperationStatus.FAILED, failed_digest, None, None, ErrorCode.RUNTIME_DEPENDENCY_NOT_READY.value, True, now, now))
        with pytest.raises(AIError) as failed_error:
            await services.execution.cancel(execution_id, _cancel_request(principal, f"failed-{execution_id}"))
        failed_operation = await runtime.persistence.operations.get(failed_operation_id, tenant_id=tenant_id)
        assert failed_error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
        assert failed_operation is not None and failed_operation.status is OperationStatus.FAILED
    finally:
        await runtime.close()


def test_temporal_contract_fields_and_registered_class_names_are_stable() -> None:
    from linktools.ai.temporal import EvaluationActivity, ExecuteActivity, SessionActivity, TaskActivity
    from linktools.ai.temporal.workflow import EvaluationWorkflow, ExecutionWorkflow, ExecutionWorkflowInput, ExecutionWorkflowResult, ExecutionWorkflowState, SessionWorkflow, TaskWorkflow, TaskWorkflowInput

    assert "profile" not in {item.name for item in fields(ExecutionWorkflowInput)}
    assert "profile" not in {item.name for item in fields(ExecutionWorkflowState)}
    assert "profile" not in {item.name for item in fields(ExecutionWorkflowResult)}
    assert "profile" not in {item.name for item in fields(TaskWorkflowInput)}
    assert tuple(item.__name__ for item in (ExecutionWorkflow, EvaluationWorkflow, SessionWorkflow, TaskWorkflow)) == ("ExecutionWorkflow", "EvaluationWorkflow", "SessionWorkflow", "TaskWorkflow")
    assert tuple(item.__name__ for item in (ExecuteActivity, EvaluationActivity, SessionActivity, TaskActivity)) == ("ExecuteActivity", "EvaluationActivity", "SessionActivity", "TaskActivity")
    activity_source = Path("linktools-ai/src/linktools/ai/temporal/_activity.py").read_text(encoding="utf-8")
    assert all(f'_temporal_activity.defn(name="{name}")' in activity_source for name in (
        "execute", "load_input", "fix_bundle_route", "fix_binding", "load_prompt", "reserve_budget", "run_agent", "process_deferred", "commit_result", "settle_budget", "evaluation", "session_mutation", "task_graph",
    ))
    workflow_sources = tuple(
        Path("linktools-ai/src/linktools/ai/temporal/workflow", name).read_text(encoding="utf-8")
        for name in ("_execution.py", "_evaluation.py", "_session.py", "_graph.py")
    )
    assert all(f'_temporal_workflow.defn(name="{name}")' in source for source, name in zip(workflow_sources, ("ExecutionWorkflow", "EvaluationWorkflow", "SessionWorkflow", "TaskWorkflow"), strict=True))


@pytest.mark.asyncio
async def test_workbench_owns_sqlite_lock_and_generic_services_do_not(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    config = RuntimePersistenceConfig.sqlite(str(tmp_path / "lock.db"), namespace=workspace.workspace_id, deployment_id="test")
    owner_context = _open_sql_workspace(workspace, config, runner=WorkspaceAgentRunner(workspace.root, workspace.config, model=TestModel()))
    owner = await owner_context.__aenter__()
    contender = _open_sql_workspace(workspace, config, runner=WorkspaceAgentRunner(workspace.root, workspace.config, model=TestModel()))
    with pytest.raises(AIError) as error:
        await contender.__aenter__()
    assert error.value.code is ErrorCode.STORAGE_CONFLICT
    service_engine = create_async_engine(f"sqlite+aiosqlite:///{config.location}")
    service_factory = async_sessionmaker(service_engine, expire_on_commit=False)
    services_context = open_runtime_services(config, TenantAuthorizationPolicy(), grant_key=b"v5-lock-key", execution_launcher=_Launcher(), engine=service_engine, session_factory=service_factory)
    await services_context.__aenter__()
    await services_context.__aexit__(None, None, None)
    await service_engine.dispose()
    memory_root = tmp_path / "memory"
    memory_workspace = Workspace.load(memory_root)
    async with open_workspace_runtime(memory_workspace, config=RuntimePersistenceConfig.in_memory(namespace=memory_workspace.workspace_id), runner=WorkspaceAgentRunner(memory_workspace.root, memory_workspace.config, model=TestModel())):
        pass
    assert not tuple(memory_root.rglob("*.local.lock"))

    async def broken_shutdown() -> None:
        raise RuntimeError("shutdown failure")

    owner.shutdown = broken_shutdown
    with pytest.raises(RuntimeError):
        await owner_context.__aexit__(None, None, None)
    async with _open_sql_workspace(workspace, config, runner=WorkspaceAgentRunner(workspace.root, workspace.config, model=TestModel())):
        pass


@pytest.mark.asyncio
async def test_workspace_session_listing_reads_past_the_first_page(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    runner = WorkspaceAgentRunner(workspace.root, workspace.config, model=TestModel())
    async with open_workspace_runtime(workspace, config=RuntimePersistenceConfig.in_memory(namespace=workspace.workspace_id), runner=runner) as runtime:
        for index in range(201):
            await runtime.open_session(f"session-{index:03d}")
        sessions = await runtime.list_sessions()
    assert len(sessions) == 201


@pytest.mark.asyncio
async def test_shutdown_cleans_active_execution_on_the_second_session_page(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    config = RuntimePersistenceConfig.sqlite(str(tmp_path / "runtime.db"), namespace=workspace.workspace_id, deployment_id="test")
    runner = _BlockingRunner()
    target_id = "session-200"
    execution_id = ""
    async with _open_sql_workspace(workspace, config, runner=runner) as runtime:
        for index in range(201):
            await runtime.open_session(f"session-{index:03d}")
        run_task = asyncio.create_task(runtime.run(target_id, "block", idempotency_key="blocking", memory_namespace="test"))
        await runner.started.wait()
        execution = await runtime._resources.domain.executions.list_by_session(target_id, tenant_id=workspace.workspace_id)
        execution_id = execution[0].execution_id
        await runtime.shutdown()
        with contextlib.suppress(asyncio.CancelledError):
            await run_task
        assert runtime._launcher.active_execution_ids() == ()
    async with _open_sql_workspace(workspace, config, runner=WorkspaceAgentRunner(workspace.root, workspace.config, model=TestModel())) as reopened:
        record = await reopened._resources.domain.executions.get(execution_id, tenant_id=workspace.workspace_id)
        assert record is not None and record.status is ExecutionStatus.CANCELLED
        assert reopened._launcher.active_execution_ids() == ()


@pytest.mark.asyncio
async def test_shutdown_rejects_new_mutations_after_admission_closes(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    runner = _BlockingRunner()
    async with open_workspace_runtime(workspace, config=RuntimePersistenceConfig.in_memory(namespace=workspace.workspace_id), runner=runner) as runtime:
        await runtime.shutdown()
        with pytest.raises(AIError) as error:
            await runtime.open_session("closed")
        assert error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY


@pytest.mark.asyncio
async def test_history_trace_and_transcript_use_the_stable_cursor_error(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    runner = WorkspaceAgentRunner(workspace.root, workspace.config, model=TestModel())
    async with open_workspace_runtime(workspace, config=RuntimePersistenceConfig.in_memory(namespace=workspace.workspace_id), runner=runner) as runtime:
        result = await runtime.run("main", "hello", idempotency_key="history", memory_namespace="test")
        principal = trusted_workspace_principal(workspace.workspace_id)
        access = build_runtime_access(runtime._services)
        trace = await access.execution.trace(result.execution_id, principal=principal)
        transcript = await access.execution.transcript(result.execution_id, principal=principal)
        for reader, page in ((access.execution.trace, trace), (access.execution.transcript, transcript)):
            for cursor in ("invalid", "-1", str(len(page.items) + 1)):
                with pytest.raises(AIError) as error:
                    await reader(result.execution_id, principal=principal, cursor=cursor)
                assert error.value.code is ErrorCode.CURSOR_INVALID
            await reader(result.execution_id, principal=principal, cursor=None)
            await reader(result.execution_id, principal=principal, cursor="0")
            await reader(result.execution_id, principal=principal, cursor=str(len(page.items)))


@pytest.mark.asyncio
async def test_history_projection_has_no_app_dependency() -> None:
    assert StepExecutionHistoryReader.__module__ == "linktools.ai.adapter._history"
