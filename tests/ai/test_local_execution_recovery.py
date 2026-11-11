#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local execution recovery and worker-supervision contracts."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import TypeVar

import pytest
from linktools.ai.agent import AgentBindingContract
from linktools.ai.agent._output import bind_output
from linktools.ai.core import (
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    IdempotencyStatus,
    Principal,
    StopReason,
    ToolOperationStatus,
    UsageMetrics,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionRecoveryDetail, ExecutionRecoveryState, ExecutionRequest
from linktools.ai.runtime._execution import CancelEffectOutcome, ExecutionStartIdentity
from linktools.ai.runtime._local import LocalExecutionBackend, _is_infrastructure_error
from linktools.ai.runtime._pending_turn import _PendingExecutionTurns
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.commands import StateCommands
from linktools.ai.persistence.contracts import (
    ToolOperationAdmission,
    ExecutionCancelRequestCommit,
    ExecutionRecord,
    StoredUserInput,
)
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import InMemoryObjectStore, StoredPayload


def _binding_contract() -> AgentBindingContract:
    output = bind_output()
    return AgentBindingContract(
        agent_spec=AgentSpec("default", tool_retries=3, output_retries=3),
        model_contract={"route_id": "default", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
    )


def _binding() -> object:
    binding_contract = _binding_contract()
    compiled_agent = SimpleNamespace(
        digest="b" * 64,
        spec=SimpleNamespace(id="default"),
        selected_tools=(),
    )
    return SimpleNamespace(
        binding_digest=binding_contract.binding_digest,
        binding_contract=binding_contract,
        compiled_agent=compiled_agent,
    )


def _request() -> ExecutionRequest:
    return ExecutionRequest(
        user_prompt="prompt",
        principal=Principal("owner", "tenant"),
        idempotency_key="idempotency",
        memory_scope=None,
        mode="run",
        planning=False,
        thinking=False,
    )


def _record() -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    binding_contract = _binding_contract()
    return ExecutionRecord(
        restart_checkpoints=True,
        durable_trace=True,
        execution_id="execution",
        session_id=None,
        parent_execution_id=None,
        root_execution_id="execution",
        previous_execution_id=None,
        fork_base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.STARTED,
        revision=0,
        event_seq=0,
        agent_run_seq=0,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        mode="run",
        planning=False,
        thinking=False,
        binding=binding_contract,
        principal_id="principal",
        principal_kind="service",
        stored_user_input=StoredUserInput(
            "text",
            StoredPayload.inline_text("prompt"),
        ),
    )


_ValueT = TypeVar("_ValueT")


class _ReadStateStore:
    async def read(self, callback: Callable[[object], Awaitable[_ValueT]]) -> _ValueT:
        return await callback(None)


class _Executions:
    def __init__(self, record: ExecutionRecord) -> None:
        self.record = record
        self.state_store = _ReadStateStore()

    async def get(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord:
        del execution_id, tenant_id
        return self.record


class _ExecutionState:
    def __init__(self, record: ExecutionRecord) -> None:
        self.executions = _Executions(record)


def _backend() -> LocalExecutionBackend:
    record = _record()
    backend = object.__new__(LocalExecutionBackend)
    backend._execution_buffer = None
    backend._pending_turns = _PendingExecutionTurns()
    backend._turn_monitors = {}
    backend._turn_owners = {}

    async def no_resolution_operations(
        *_args: object, **_kwargs: object
    ) -> tuple[object, ...]:
        return ()

    backend._execution = _ExecutionState(record)
    backend._checkpoint = SimpleNamespace(
        operations=SimpleNamespace(list_pending=no_resolution_operations)
    )
    binding = _binding()
    backend._catalog = SimpleNamespace(binding=lambda digest: binding)
    backend._restore_binding = None
    backend._accepting = True
    backend._recovery_enabled = False
    backend._tenant_id = "tenant"
    backend._namespace = "test"
    backend._tasks = {}
    backend._captured_usage = {}
    backend._worker_failures = {}
    backend._worker_cancel_requests = set()
    backend._handoff_scopes = {}
    backend._worker_shutdown_requests = set()
    backend._recovery_reconcile_tasks = {}
    backend._terminal_events = {}
    backend._pending_audit_events = {}
    backend._pending_audit_locks = {}
    backend._approval_pause_segments = {}
    backend._agent_run_only_worker_exits = set()
    backend._repository_instruction_provenance = {}
    backend._checkpoint_tasks = set()
    backend._execution_durable_tasks = {}
    backend._metric_recorder = None
    backend._worker_completed = None
    backend._tool_operations = None
    backend._live_broker = SimpleNamespace(complete=lambda _execution_id: None)
    return backend


@pytest.mark.asyncio
async def test_text_output_keeps_terminal_authority_without_object_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _backend()
    current = backend._execution.executions.record
    objects = InMemoryObjectStore()
    backend._execution_objects = objects

    async def reject_object_write(*args, **kwargs):
        pytest.fail("ordinary execution output must remain with its state owner")

    async def observe_terminal_authority(
        execution: ExecutionRecord, *, producer_generation: int | None = None,
        tool_operations: object | None = None,
    ) -> ExecutionRecord:
        return execution

    monkeypatch.setattr(objects, "put", reject_object_write)
    monkeypatch.setattr(backend, "_check_reconcile_tool_effects", observe_terminal_authority)
    observed = await backend._commit_success(
        current, _binding(), "answer" * 20000, UsageMetrics(), "agent-run",
    )
    assert observed == current


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_backend", ("memory", "sqlite"))
async def test_prepare_start_persists_exact_binding_and_execution_policy(
    tmp_path: Path, storage_backend: str,
) -> None:
    backend = _backend()
    execution = replace(
        _record(), status=ExecutionStatus.PENDING_START,
        mode="plan", planning=True, thinking=True,
    )
    request = replace(_request(), mode="plan", planning=True, thinking=True)
    identity = ExecutionStartIdentity("scope", "a" * 64, "b" * 64)
    storage = (PersistenceStorage.in_memory() if storage_backend == "memory"
               else PersistenceStorage.sqlite(tmp_path / "prepare-start.db"))
    async with Persistence.open("prepare-start", tenant_id="tenant", storage=storage) as state:
        await state.execution.executions.create_execution(execution)
        backend._execution = state.execution
        backend._recovery_enabled = True
        backend._runtime_commands = StateCommands(
            state.execution.executions, namespace="prepare-start",
            events=state.execution.events, background_tasks=set(),
        )

        started = await backend.prepare_start(request, execution, identity)

        persisted = await state.execution.executions.get(execution.execution_id, tenant_id="tenant")
        assert persisted == started
        assert started.status is ExecutionStatus.STARTED
        assert started.execution_id == execution.execution_id
        assert started.binding == execution.binding
        assert (started.mode, started.planning, started.thinking) == ("plan", True, True)
        assert (started.restart_checkpoints, started.durable_trace, started.conversation) == (
            execution.restart_checkpoints, execution.durable_trace, execution.conversation,
        )
        assert started.stored_user_input == execution.stored_user_input
        assert started.recovery_state is ExecutionRecoveryState.ADMITTED
        assert started.recovery == ExecutionRecoveryDetail()
        assert (started.producer_generation, started.producer_claim_id) == (0, None)
        assert started.revision == execution.revision + 1
        assert started.event_seq == execution.event_seq + 1
        claimed = await state.execution.idempotency.get(
            identity.scope, identity.idempotency_key, tenant_id="tenant",
        )
        assert claimed is not None
        assert claimed.resource_id == execution.execution_id
        assert claimed.request_digest == identity.request_digest
        assert claimed.status is IdempotencyStatus.STARTED
        events = await state.execution.events.list(
            execution.execution_id, tenant_id="tenant", after_event_seq=0, limit=10,
        )
        assert tuple(event.event_type for event in events.items) == (ExecutionEventType.EXECUTION_STARTED,)


@pytest.mark.parametrize(
    ("error", "expected"),
    (
        (ValueError("business"), False),
        (AIError(ErrorCode.OUTPUT_VALIDATION_FAILED), False),
        (AIError(ErrorCode.STORAGE_INTEGRITY_ERROR), True),
        (AIError(ErrorCode.AGENT_BINDING_UNAVAILABLE), True),
        (AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE), True),
        (AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY), True),
        (AIError(ErrorCode.SERVICE_NOT_READY), True),
    ),
)
def test_local_infrastructure_failure_classification(
    error: Exception,
    expected: bool,
) -> None:
    assert _is_infrastructure_error(error) is expected


@pytest.mark.asyncio
async def test_local_worker_failure_is_consumed_and_observable() -> None:
    backend = _backend()
    completed: list[str] = []

    async def worker_completed(execution_id: str) -> None:
        completed.append(execution_id)

    backend._worker_completed = worker_completed

    async def fail() -> None:
        raise RuntimeError("worker failed")

    task = asyncio.create_task(fail(), name="ai-execution-execution")
    backend._tasks["execution"] = task
    task.add_done_callback(
        lambda completed: backend._task_done("execution", completed)
    )
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)

    failure = backend.worker_failure("execution", tenant_id="tenant")
    assert failure is not None
    assert failure.code is ErrorCode.INTERNAL_ERROR
    assert failure.safe_details == {
        "phase": "local_execution_worker",
        "execution_id": "execution",
    }
    assert completed == []


@pytest.mark.asyncio
async def test_local_old_worker_callback_cannot_clear_new_owner() -> None:
    backend = _backend()

    async def fail() -> None:
        raise RuntimeError("old worker failed")

    async def wait() -> None:
        await asyncio.Event().wait()

    old_task = asyncio.create_task(fail())
    new_task = asyncio.create_task(wait())
    try:
        await asyncio.sleep(0)
        captured = object()
        backend._tasks["execution"] = new_task
        backend._captured_usage["execution"] = captured

        backend._task_done("execution", old_task)

        assert backend._tasks["execution"] is new_task
        assert backend._captured_usage["execution"] is captured
        assert "execution" not in backend._worker_failures
    finally:
        new_task.cancel()
        await asyncio.gather(old_task, new_task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    (
        ExecutionStatus.SUCCEEDED,
        ExecutionStatus.FAILED,
        ExecutionStatus.CANCELLED,
        ExecutionStatus.CANCELLING,
        ExecutionStatus.FINALIZING,
    ),
)
async def test_local_cancel_without_worker_confirms_terminal_or_finalizing_state(
    status: ExecutionStatus,
) -> None:
    backend = _backend()
    current = replace(_record(), status=status)
    backend._execution.executions.record = current

    assert await backend.cancel(current) is CancelEffectOutcome.CONFIRMED


@pytest.mark.asyncio
async def test_local_cancel_without_worker_is_unknown_for_active_execution() -> None:
    backend = _backend()
    current = _record()
    backend._execution.executions.record = current

    assert await backend.cancel(current) is CancelEffectOutcome.UNKNOWN


class _ToolOperations:
    def __init__(self, records: tuple[object, ...]) -> None:
        self.records = records

    async def list_by_execution(
        self, execution_id: str, *, tenant_id: str,
    ) -> tuple[object, ...]:
        del execution_id, tenant_id
        return self.records


class _FailingToolOperations:
    async def list_by_execution(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> tuple[object, ...]:
        del execution_id, tenant_id
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)


def _tool_operation(
    status: ToolOperationStatus,
    *,
    replay_safe: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        tool_operation_id="tool-operation",
        execution_id="execution",
        agent_run_id="agent-run",
        tool_call_id="tool-call",
        tool_name="tool",
        replay_safe=replay_safe,
        status=status,
        fence=1,
        error_code=(
            ErrorCode.TOOL_EFFECT_UNKNOWN.value
            if status is ToolOperationStatus.EFFECT_UNKNOWN
            else None
        ),
    )


@pytest.mark.asyncio
async def test_recovery_effect_query_does_not_reconcile_claims() -> None:
    backend = _backend()
    operations = _ToolOperations(
        (_tool_operation(ToolOperationStatus.CLAIMED),),
    )
    backend._tool_operations = operations

    effects = await backend._recovery_failure_effects(
        "execution",
        tenant_id="tenant",
    )

    assert effects == ()
    assert operations.records[0].status is ToolOperationStatus.CLAIMED


@pytest.mark.asyncio
async def test_local_cancel_without_worker_requires_recovery_for_unknown_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _backend()
    current = replace(_record(), status=ExecutionStatus.CANCELLING)
    backend._execution.executions.record = current
    backend._tool_operations = _ToolOperations(
        (_tool_operation(ToolOperationStatus.EFFECT_UNKNOWN),)
    )
    observed: list[tuple[ExecutionRecord, AIError, tuple[object, ...]]] = []

    async def commit_recovery(
        execution: ExecutionRecord,
        error: AIError,
        effects: tuple[object, ...],
        *,
        producer_generation: int | None = None,
    ) -> ExecutionRecord:
        observed.append((execution, error, effects))
        return replace(execution, status=ExecutionStatus.RECOVERY_REQUIRED)

    monkeypatch.setattr(backend, "_commit_recovery_required", commit_recovery)

    outcome = await backend.cancel(current)

    assert outcome is CancelEffectOutcome.UNKNOWN
    assert len(observed) == 1
    assert observed[0][1].code is ErrorCode.TOOL_EFFECT_UNKNOWN
    assert observed[0][2][0].operation_id == "tool-operation"


@pytest.mark.asyncio
async def test_local_failure_uses_tool_ledger_even_for_an_unrelated_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _backend()
    current = _record()
    backend._execution.executions.record = current
    backend._tool_operations = _ToolOperations(
        (_tool_operation(ToolOperationStatus.EFFECT_UNKNOWN),)
    )
    observed: list[tuple[ExecutionRecord, AIError, tuple[object, ...]]] = []

    async def commit_recovery(
        execution: ExecutionRecord,
        error: AIError,
        effects: tuple[object, ...],
        *,
        producer_generation: int | None = None,
    ) -> ExecutionRecord:
        observed.append((execution, error, effects))
        return replace(
            execution,
            status=ExecutionStatus.RECOVERY_REQUIRED,
            error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value,
        )

    monkeypatch.setattr(backend, "_commit_recovery_required", commit_recovery)

    failed = await backend._commit_failure(current, ValueError("unrelated"))

    assert failed.status is ExecutionStatus.RECOVERY_REQUIRED
    assert failed.error_code == ErrorCode.TOOL_EFFECT_UNKNOWN.value
    assert observed[0][1].code is ErrorCode.TOOL_EFFECT_UNKNOWN


@pytest.mark.asyncio
async def test_local_cancel_without_worker_allows_deferred_pending_tool_call() -> None:
    backend = _backend()
    current = replace(_record(), status=ExecutionStatus.CANCELLING)
    backend._execution.executions.record = current
    backend._tool_operations = _ToolOperations(
        (
            _tool_operation(
                ToolOperationStatus.PENDING,
                replay_safe=True,
            ),
        )
    )

    assert await backend.cancel(current) is CancelEffectOutcome.CONFIRMED


@asynccontextmanager
async def _tool_cancel_state(
    tmp_path: Path, backend: str, replay_safe: bool,
) -> AsyncIterator[tuple[Persistence, StateCommands, ExecutionRecord]]:
    storage = (PersistenceStorage.in_memory() if backend == "memory"
               else PersistenceStorage.sqlite(tmp_path / "tool-cancel.db"))
    async with Persistence.open("tool-cancel", tenant_id="tenant", storage=storage) as state:
        current = replace(_record(), revision=1, agent_run_seq=1,
                          producer_generation=1, producer_claim_id="owner")
        await state.execution.executions.create_execution(current)
        commands = StateCommands(
            state.execution.executions, namespace="tool-cancel", events=state.execution.events,
            tools=state.checkpoint.tools, background_tasks=set(),
        )
        await commands.commit_tool_admission(ToolOperationAdmission(
            execution_id="execution", tool_operation_id="tool-operation",
            agent_run_id="agent-run", recovery_agent_run_id=None,
            tool_call_id="tool-call",
            tool_name="tool", arguments_digest="b" * 64, binding_digest="c" * 64,
            replay_safe=replay_safe, owner="owner", producer_generation=1,
        ))
        yield state, commands, current


async def _request_tool_cancel(state: Persistence, current: ExecutionRecord) -> ExecutionRecord:
    return await state.execution.executions.request_cancel(ExecutionCancelRequestCommit(
        execution_id=current.execution_id, expected_revision=current.revision,
        expected_event_seq=current.event_seq, operation_id="cancel",
        requested_at=datetime.now(timezone.utc),
    ))


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_backend", ("memory", "sqlite"))
@pytest.mark.parametrize("replay_safe", (False, True))
@pytest.mark.parametrize("force", (False, True))
async def test_local_cancel_without_worker_preserves_live_claim_even_when_forced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage_backend: str, replay_safe: bool, force: bool,
) -> None:
    async with _tool_cancel_state(tmp_path, storage_backend, replay_safe) as (state, commands, current):
        current = await _request_tool_cancel(state, current)
        backend = _backend()
        backend._execution = state.execution
        backend._tool_operations = state.checkpoint.tools
        backend._runtime_commands = commands
        observed: list[tuple[ExecutionRecord, tuple[object, ...]]] = []

        async def commit_recovery(
            execution: ExecutionRecord, error: AIError, effects: tuple[object, ...],
            *, producer_generation: int | None = None,
        ) -> ExecutionRecord:
            assert error.code is ErrorCode.TOOL_EFFECT_UNKNOWN
            observed.append((execution, effects))
            return replace(execution, status=ExecutionStatus.RECOVERY_REQUIRED)

        monkeypatch.setattr(backend, "_commit_recovery_required", commit_recovery)
        assert await backend.cancel(current, force=force) is CancelEffectOutcome.UNKNOWN
        assert observed == []
        persisted = await state.execution.executions.get("execution", tenant_id="tenant")
        assert persisted.status is ExecutionStatus.CANCELLING
        operation = await state.checkpoint.tools.get_operation("tool-operation", tenant_id="tenant")
        assert operation.status is ToolOperationStatus.CLAIMED
        assert (operation.owner, operation.fence, operation.producer_generation) == ("owner", 1, 1)
        # A cancel intent, even forced, does not prove that the remote effect stopped.
        result, _ = await commands.commit_tool_terminal(
            "tool-operation", tenant_id="tenant", owner="owner", fence=1,
            result_payload=StoredPayload.inline_json("late known result"),
        )
        assert result.value.status is ToolOperationStatus.COMPLETED
        assert result.value.result_payload.decode() == "late known result"


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_authority", ("started", "wrong_claim", "wrong_generation"))
async def test_tool_cancellation_requires_exact_cancelling_turn(
    tmp_path: Path, invalid_authority: str,
) -> None:
    async with _tool_cancel_state(tmp_path, "memory", False) as (state, commands, current):
        if invalid_authority != "started":
            await _request_tool_cancel(state, current)
        with pytest.raises(AIError) as raised:
            await commands.reconcile_tool_turn_cancellation(
                "tool-operation", execution_id="execution",
                producer_generation=2 if invalid_authority == "wrong_generation" else 1,
                producer_claim_id="other" if invalid_authority == "wrong_claim" else "owner",
            )
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        operation = await state.checkpoint.tools.get_operation("tool-operation", tenant_id="tenant")
        assert operation.status is ToolOperationStatus.CLAIMED
        assert operation.owner == "owner"


@pytest.mark.asyncio
async def test_local_cancel_with_tool_ledger_read_failure_is_not_confirmed() -> None:
    backend = _backend()
    current = replace(_record(), status=ExecutionStatus.CANCELLING)
    backend._execution.executions.record = current
    backend._tool_operations = _FailingToolOperations()

    with pytest.raises(AIError) as raised:
        await backend.cancel(current)

    assert raised.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    assert backend._execution.executions.record.status is ExecutionStatus.CANCELLING


@pytest.mark.asyncio
async def test_local_failure_does_not_terminalize_when_tool_ledger_read_fails() -> None:
    backend = _backend()
    current = _record()
    backend._execution.executions.record = current
    backend._tool_operations = _FailingToolOperations()

    with pytest.raises(AIError) as raised:
        await backend._commit_failure(current, ValueError("unrelated"))

    assert raised.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    assert backend._execution.executions.record is current


class _Sessions:
    def __init__(self) -> None:
        self.releases: list[tuple[str, str, str]] = []
        self.history_unavailable: list[bool] = []
        self.require_turn_commit: list[bool] = []

    async def release_execution(
        self,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
        history_unavailable: bool = False,
        require_turn_commit: bool = False,
    ) -> object:
        self.releases.append((session_id, tenant_id, execution_id))
        self.history_unavailable.append(history_unavailable)
        self.require_turn_commit.append(require_turn_commit)
        return SimpleNamespace(active_execution_id=None)


@pytest.mark.asyncio
async def test_recovered_started_cancel_commits_cancel_event_before_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _backend()
    current = replace(_record(), recovery=ExecutionRecoveryDetail(agent_run_id="agent-run"))
    backend._execution.executions.record = current
    operation = SimpleNamespace(operation_id="cancel-operation")
    events: list[ExecutionEventType] = []
    settled: list[tuple[object, ExecutionRecord]] = []

    async def commit_cancel_checkpoint(
        commit: ExecutionCancelRequestCommit,
        *,
        expected_status: ExecutionStatus,
    ) -> ExecutionRecord:
        assert commit.operation_id == operation.operation_id
        assert expected_status is ExecutionStatus.STARTED
        events.append(ExecutionEventType.CANCEL_REQUESTED)
        return replace(
            current,
            status=ExecutionStatus.CANCELLING,
            revision=current.revision + 1,
            event_seq=current.event_seq + 1,
        )

    async def commit_terminal(
        execution: ExecutionRecord,
        status: ExecutionStatus,
        output: object,
        error_code: str | None,
        stop_reason: object,
        **kwargs: object,
    ) -> ExecutionRecord:
        assert execution.status is ExecutionStatus.CANCELLING
        assert execution.recovery == current.recovery
        assert kwargs["agent_run_id"] == current.recovery.agent_run_id
        assert output is None
        assert error_code == ErrorCode.EXECUTION_CANCELLED.value
        assert stop_reason is StopReason.CANCELLED
        events.append(ExecutionEventType.EXECUTION_CANCELLED)
        return replace(execution, status=status)

    async def settle_cancel_operation(
        candidate: object,
        execution: ExecutionRecord,
    ) -> None:
        settled.append((candidate, execution))

    monkeypatch.setattr(backend, "commit_cancel_checkpoint", commit_cancel_checkpoint)
    monkeypatch.setattr(backend, "_commit_terminal", commit_terminal)
    monkeypatch.setattr(backend, "_settle_cancel_operation", settle_cancel_operation)

    terminal = await backend._complete_recovered_cancel(
        current,
        (operation,),
    )

    assert terminal.status is ExecutionStatus.CANCELLED
    assert events == [
        ExecutionEventType.CANCEL_REQUESTED,
        ExecutionEventType.EXECUTION_CANCELLED,
    ]
    assert settled == [(operation, terminal)]


@pytest.mark.asyncio
async def test_abort_start_releases_its_session_when_admission_checkpoint_is_absent() -> None:
    backend = _backend()
    current = replace(
        _record(),
        session_id="session",
        status=ExecutionStatus.FAILED,
    )
    backend._execution.executions.record = current
    sessions = _Sessions()
    backend._conversation = SimpleNamespace(sessions=sessions)

    stale = replace(current, recovery=ExecutionRecoveryDetail(agent_run_id="stale-run"))
    await backend.abort_start(stale)

    assert current.recovery is None
    assert backend._execution.executions.record is current
    assert sessions.releases == [("session", "tenant", "execution")]
    assert sessions.history_unavailable == [False]
    assert sessions.require_turn_commit == [current.conversation]


@pytest.mark.asyncio
async def test_abort_start_rejects_an_execution_that_already_started() -> None:
    backend = _backend()
    current = replace(
        _record(),
        session_id="session",
        status=ExecutionStatus.CANCELLED,
        started_at=datetime.now(timezone.utc),
    )
    backend._execution.executions.record = current
    sessions = _Sessions()
    backend._conversation = SimpleNamespace(sessions=sessions)

    with pytest.raises(AIError) as raised:
        await backend.abort_start(replace(current, started_at=None))

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert backend._execution.executions.record is current
    assert sessions.releases == []


@pytest.mark.asyncio
async def test_local_cancel_before_worker_coroutine_starts_confirms_cancelling_state() -> None:
    backend = _backend()
    current = replace(_record(), status=ExecutionStatus.CANCELLING)
    backend._execution.executions.record = current

    async def wait() -> None:
        await asyncio.Event().wait()

    task = asyncio.create_task(wait())
    backend._tasks["execution"] = task
    outcome = await backend.cancel(current)

    assert outcome is CancelEffectOutcome.CONFIRMED
    assert backend._tasks == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("new_worker", (False, True))
async def test_completion_cleanup_waits_for_control_and_preserves_new_worker(new_worker: bool) -> None:
    backend = _backend()
    release = asyncio.Event()
    completed = []
    backend._terminal_events["execution"] = asyncio.Event()
    backend._terminal_events["execution"].set()
    backend._pending_audit_locks["execution"] = asyncio.Lock()

    async def worker_completed(execution_id: str) -> None:
        completed.append(execution_id)

    async def control() -> None:
        await release.wait()
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    backend._worker_completed = worker_completed
    control_task = asyncio.create_task(control())
    backend._execution_durable_tasks["execution"] = {control_task}
    cleanup = asyncio.create_task(backend._complete_worker_handoff("execution"))
    installed = None
    try:
        await asyncio.sleep(0)
        assert not cleanup.done()
        if new_worker:
            installed = asyncio.create_task(asyncio.Event().wait())
            backend._tasks["execution"] = installed
        release.set()
        outcomes = await asyncio.gather(control_task, return_exceptions=True)
        assert isinstance(outcomes[0], AIError)
        await cleanup
        assert backend._worker_failures == {}
        if new_worker:
            assert backend._tasks["execution"] is installed
            assert "execution" in backend._terminal_events
            assert "execution" in backend._pending_audit_locks
            assert completed == []
        else:
            assert backend._terminal_events == {}
            assert backend._pending_audit_locks == {}
            assert backend._execution_durable_tasks == {}
            assert completed == ["execution"]
            await asyncio.wait_for(backend.wait_terminal("execution", tenant_id="tenant"), 1)
            assert backend._terminal_events == {}
    finally:
        release.set()
        if installed is not None:
            installed.cancel()
            await asyncio.gather(installed, return_exceptions=True)
        await asyncio.gather(control_task, cleanup, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("unknown", "expected_revision", "unrelated_transition"))
async def test_cancel_terminal_does_not_rebuild_without_known_cancel_transition(
    monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    backend = _backend()
    current = replace(_record(), recovery=ExecutionRecoveryDetail(agent_run_id="agent-run"))
    backend._execution.executions.record = current
    failure = AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN if boundary == "unknown"
                      else ErrorCode.EXECUTION_RESULT_CONFLICT)
    dispatched: list[ExecutionRecord] = []

    async def no_effects(*args: object, **kwargs: object) -> None:
        return None

    async def terminal(
        execution: ExecutionRecord, recovery_checkpoint: ExecutionRecord, **kwargs: object,
    ) -> ExecutionRecord:
        assert recovery_checkpoint == execution
        dispatched.append(execution)
        backend._execution.executions.record = replace(current,
            status=ExecutionStatus.STARTED if boundary == "unrelated_transition" else ExecutionStatus.CANCELLING,
            revision=current.revision + 1, event_seq=current.event_seq + 1)
        raise failure

    monkeypatch.setattr(backend, "_check_reconcile_tool_effects", no_effects)
    monkeypatch.setattr(backend, "_same_terminal_storage_group", lambda *args, **kwargs: True)
    monkeypatch.setattr(backend, "_commit_same_group_recovery_terminal", terminal)
    with pytest.raises(AIError) as raised:
        await backend._commit_terminal(current, ExecutionStatus.CANCELLED, None,
            ErrorCode.EXECUTION_CANCELLED.value, StopReason.CANCELLED,
            expected_revision=current.revision if boundary == "expected_revision" else None)
    assert raised.value is failure
    assert dispatched == [current]
