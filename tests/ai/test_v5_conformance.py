#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused regression coverage for the v5 Runtime convergence contract."""

import asyncio
import contextlib
import inspect
import json
import sqlite3
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from linktools.ai.adapter.history import StepExecutionHistoryReader
from linktools.ai.adapter.memory import build_file_runtime, build_memory_runtime
from linktools.ai.app.assembly import RuntimeStoreConfig, build_runtime_access, build_runtime_services, open_runtime_services, open_runtime_store
from linktools.ai.app.workbench import open_workspace_runtime
from linktools.ai.core import Page, Principal, TenantAuthorizationPolicy
from linktools.ai.core.errors import ErrorCode, AIError
from linktools.ai.core.ids import idempotency_key_hash
from linktools.ai.core.value import ApprovalDecision, ApprovalStatus, ExecutionEventType, ExecutionLineageKind, ExecutionStatus, IdempotencyStatus, SessionStatus, StopReason
from linktools.ai.runtime.persistence import ApprovalRecord, ExecutionRecord, ExecutionTerminalCommit, IdempotencyRecord, IdempotencyTerminalUpdate, ResultRecord, RuntimePersistence, SessionHeadAdvance, SessionRecord
from linktools.ai.runtime.services import ApprovalDecisionRequest, ExecutionHandle, ExecutionRequest, ExecutionView, RuntimeServiceIdentity, TaskGraphHandle, WorkflowQueryResult, WorkflowUpdateResult
from linktools.ai.task import TaskGraph, TaskGraphRequest, TaskNode
from linktools.ai.agent.runner import WorkspaceAgentResult, WorkspaceAgentRunner
from linktools.ai.workspace import Workspace, trusted_workspace_principal


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
        on_event: object | None = None,
    ) -> WorkspaceAgentResult:
        self.started.set()
        await self._wait_forever.wait()
        return WorkspaceAgentResult(step_run_id, prompt, [])


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
    memory = build_memory_runtime(namespace="v5-memory")
    await memory.initialize()
    try:
        await _assert_stale_terminal_is_atomic(memory.persistence)
    finally:
        await memory.close()

    path = tmp_path / "runtime.db"
    config = RuntimeStoreConfig.sqlite(str(path), namespace="v5-sql", deployment_id="test")
    async with open_runtime_store(config) as stores:
        await _assert_stale_terminal_is_atomic(stores.domain)
    with sqlite3.connect(path) as connection:
        assert connection.execute("select profile from ai_runtime_sessions where session_id = ?", ("session",)).fetchone() == ("",)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_workspace_resume_advances_head_from_the_previous_session_head(tmp_path: Path, backend: str) -> None:
    workspace = Workspace.load(tmp_path / backend)
    runner = WorkspaceAgentRunner(workspace.root, workspace.config, model=TestModel())
    config = RuntimeStoreConfig.memory(namespace=workspace.workspace_id) if backend == "memory" else RuntimeStoreConfig.sqlite(str(tmp_path / "runtime.db"), namespace=workspace.workspace_id, deployment_id="test")
    async with open_workspace_runtime(workspace, config=config, runner=runner) as runtime:
        results = [
            await runtime.run("main", prompt, idempotency_key=key)
            for prompt, key in (("one", "k1"), ("two", "k2"), ("three", "k3"))
        ]
        records = [await runtime._stores.domain.executions.get(item.execution_id, tenant_id=workspace.workspace_id) for item in results]
        session = await runtime._stores.domain.sessions.get("main", tenant_id=workspace.workspace_id)
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
    async with open_workspace_runtime(workspace, config=RuntimeStoreConfig.memory(namespace=workspace.workspace_id), runner=runner) as runtime:
        outcomes = await asyncio.gather(
            runtime.run("main", "one", idempotency_key="first"),
            runtime.run("main", "two", idempotency_key="second"),
            return_exceptions=True,
        )
        successful = tuple(item for item in outcomes if not isinstance(item, BaseException))
        failures = tuple(item for item in outcomes if isinstance(item, BaseException))
        session = await runtime._stores.domain.sessions.get("main", tenant_id=workspace.workspace_id)
        executions = await runtime._stores.domain.executions.list_by_session("main", tenant_id=workspace.workspace_id)
    assert len(successful) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], AIError)
    assert failures[0].code is ErrorCode.STORAGE_CONFLICT
    assert session is not None and session.head_execution_id == successful[0].execution_id
    assert sum(item.status is ExecutionStatus.SUCCEEDED for item in executions) == 1
    assert sum(item.status is ExecutionStatus.FAILED for item in executions) == 1


@pytest.mark.asyncio
async def test_runtime_access_exposes_query_only_wrappers() -> None:
    runtime = build_memory_runtime(namespace="v5-access")
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
    runtime = build_memory_runtime(namespace="v5-composition")
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
        explicit_handle = await explicit_services.execution.run("b" * 64, ExecutionRequest("explicit", principal, "explicit-key"))
        assert explicit.started == [explicit_handle.execution_id]

        gateway_services = build_runtime_services(
            runtime.persistence,
            TenantAuthorizationPolicy(),
            grant_key=b"v5-composition-key",
            history_reader=_History(),
            schema_digest=runtime.persistence.atomic_domain_id,
            workflow_gateway=gateway,
        )
        gateway_handle = await gateway_services.execution.run("b" * 64, ExecutionRequest("gateway", principal, "gateway-key"))
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
        combined_handle = await combined_services.execution.run("b" * 64, ExecutionRequest("combined", principal, "combined-key"))
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

    from linktools.ai.core.value import ExecutionEventType
    from linktools.ai.runtime.persistence import ExecutionRecord
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
    runtime = build_file_runtime(str(tmp_path), workspace_id=workspace_id)
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
    reopened = build_file_runtime(str(tmp_path), workspace_id=workspace_id)
    await reopened.initialize()
    try:
        session = await reopened.persistence.sessions.get("legacy", tenant_id=workspace_id)
        execution = await reopened.persistence.executions.get("legacy-execution", tenant_id=workspace_id)
        assert session is not None and execution is not None
        assert not hasattr(session, "profile")
        assert not hasattr(execution, "profile")
    finally:
        await reopened.close()


def test_temporal_contract_fields_and_registered_class_names_are_stable() -> None:
    from linktools.ai.temporal.activity import ExecuteActivity, EvaluationActivity, SessionActivity, TaskActivity
    from linktools.ai.temporal.workflow.dag import TaskWorkflowInput
    from linktools.ai.temporal.workflow.dag import TaskWorkflow
    from linktools.ai.temporal.workflow.mutation import SessionWorkflow
    from linktools.ai.temporal.workflow.run import ExecutionWorkflowInput, ExecutionWorkflowResult, ExecutionWorkflowState
    from linktools.ai.temporal.workflow.run import ExecutionWorkflow
    from linktools.ai.temporal.workflow.suite import EvaluationWorkflow

    assert "profile" not in {item.name for item in fields(ExecutionWorkflowInput)}
    assert "profile" not in {item.name for item in fields(ExecutionWorkflowState)}
    assert "profile" not in {item.name for item in fields(ExecutionWorkflowResult)}
    assert "profile" not in {item.name for item in fields(TaskWorkflowInput)}
    assert tuple(item.__name__ for item in (ExecutionWorkflow, EvaluationWorkflow, SessionWorkflow, TaskWorkflow)) == ("ExecutionWorkflow", "EvaluationWorkflow", "SessionWorkflow", "TaskWorkflow")
    assert tuple(item.__name__ for item in (ExecuteActivity, EvaluationActivity, SessionActivity, TaskActivity)) == ("ExecuteActivity", "EvaluationActivity", "SessionActivity", "TaskActivity")


@pytest.mark.asyncio
async def test_workbench_owns_sqlite_lock_and_generic_services_do_not(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    config = RuntimeStoreConfig.sqlite(str(tmp_path / "lock.db"), namespace=workspace.workspace_id, deployment_id="test")
    owner_context = open_workspace_runtime(workspace, config=config, runner=WorkspaceAgentRunner(workspace.root, workspace.config, model=TestModel()))
    owner = await owner_context.__aenter__()
    contender = open_workspace_runtime(workspace, config=config, runner=WorkspaceAgentRunner(workspace.root, workspace.config, model=TestModel()))
    with pytest.raises(AIError) as error:
        await contender.__aenter__()
    assert error.value.code is ErrorCode.STORAGE_CONFLICT
    services_context = open_runtime_services(config, TenantAuthorizationPolicy(), grant_key=b"v5-lock-key", execution_launcher=_Launcher())
    await services_context.__aenter__()
    await services_context.__aexit__(None, None, None)
    memory_root = tmp_path / "memory"
    memory_workspace = Workspace.load(memory_root)
    async with open_workspace_runtime(memory_workspace, config=RuntimeStoreConfig.memory(namespace=memory_workspace.workspace_id), runner=WorkspaceAgentRunner(memory_workspace.root, memory_workspace.config, model=TestModel())):
        pass
    assert not tuple(memory_root.rglob("*.local.lock"))

    async def broken_shutdown() -> None:
        raise RuntimeError("shutdown failure")

    owner.shutdown = broken_shutdown
    with pytest.raises(RuntimeError):
        await owner_context.__aexit__(None, None, None)
    async with open_workspace_runtime(workspace, config=config, runner=WorkspaceAgentRunner(workspace.root, workspace.config, model=TestModel())):
        pass


@pytest.mark.asyncio
async def test_workspace_session_listing_reads_past_the_first_page(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    runner = WorkspaceAgentRunner(workspace.root, workspace.config, model=TestModel())
    async with open_workspace_runtime(workspace, config=RuntimeStoreConfig.memory(namespace=workspace.workspace_id), runner=runner) as runtime:
        for index in range(201):
            await runtime.open_session(f"session-{index:03d}")
        sessions = await runtime.list_sessions()
    assert len(sessions) == 201


@pytest.mark.asyncio
async def test_shutdown_cleans_active_execution_on_the_second_session_page(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    config = RuntimeStoreConfig.sqlite(str(tmp_path / "runtime.db"), namespace=workspace.workspace_id, deployment_id="test")
    runner = _BlockingRunner()
    target_id = "session-200"
    execution_id = ""
    async with open_workspace_runtime(workspace, config=config, runner=runner) as runtime:
        for index in range(201):
            await runtime.open_session(f"session-{index:03d}")
        run_task = asyncio.create_task(runtime.run(target_id, "block", idempotency_key="blocking"))
        await runner.started.wait()
        execution = await runtime._stores.domain.executions.list_by_session(target_id, tenant_id=workspace.workspace_id)
        execution_id = execution[0].execution_id
        await runtime.shutdown()
        with contextlib.suppress(asyncio.CancelledError):
            await run_task
        assert runtime._launcher.active_execution_ids() == ()
    async with open_workspace_runtime(workspace, config=config, runner=WorkspaceAgentRunner(workspace.root, workspace.config, model=TestModel())) as reopened:
        record = await reopened._stores.domain.executions.get(execution_id, tenant_id=workspace.workspace_id)
        assert record is not None and record.status is ExecutionStatus.CANCELLED
        assert reopened._launcher.active_execution_ids() == ()


@pytest.mark.asyncio
async def test_shutdown_rejects_new_mutations_after_admission_closes(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    runner = _BlockingRunner()
    async with open_workspace_runtime(workspace, config=RuntimeStoreConfig.memory(namespace=workspace.workspace_id), runner=runner) as runtime:
        await runtime.shutdown()
        with pytest.raises(AIError) as error:
            await runtime.open_session("closed")
        assert error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY


@pytest.mark.asyncio
async def test_history_trace_and_transcript_use_the_stable_cursor_error(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    runner = WorkspaceAgentRunner(workspace.root, workspace.config, model=TestModel())
    async with open_workspace_runtime(workspace, config=RuntimeStoreConfig.memory(namespace=workspace.workspace_id), runner=runner) as runtime:
        result = await runtime.run("main", "hello", idempotency_key="history")
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
    assert StepExecutionHistoryReader.__module__ == "linktools.ai.adapter.history"
