#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local execution recovery and worker-supervision contracts."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from linktools.ai.agent import AgentBindingContract
from linktools.ai.agent._output import bind_output
from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    Principal,
    ToolOperationStatus,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ExecutionRequest
from linktools.ai.runtime._execution import CancelEffectOutcome, ExecutionStartIdentity
from linktools.ai.runtime._local import LocalExecutionBackend, _is_infrastructure_error
from linktools.ai.runtime.state._contracts import ExecutionRecord, StoredUserInput
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import StoredPayload


def _binding_contract() -> AgentBindingContract:
    output = bind_output()
    return AgentBindingContract(
        agent_spec=AgentSpec("default"),
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
        execution_id="execution",
        session_id=None,
        parent_execution_id=None,
        root_execution_id="execution",
        previous_execution_id=None,
        fork_base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.STARTED,
        revision=0,
        event_sequence=0,
        agent_run_sequence=0,
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


class _Executions:
    def __init__(self, record: ExecutionRecord) -> None:
        self.record = record

    async def get(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord:
        del execution_id, tenant_id
        return self.record


class _ExecutionState:
    def __init__(self, record: ExecutionRecord) -> None:
        self.executions = _Executions(record)


class _StartCommands:
    def __init__(self, execution: ExecutionRecord) -> None:
        self.execution = execution
        self.recovery_checkpoint = None

    async def commit_start_attempt_checkpoint(
        self,
        claim: object,
        *,
        recovery_checkpoint: object,
        session_id: str | None,
        expected_cursor: object,
    ) -> ExecutionRecord:
        del claim, session_id, expected_cursor
        self.recovery_checkpoint = recovery_checkpoint
        return replace(self.execution, status=ExecutionStatus.STARTED)


def _backend() -> LocalExecutionBackend:
    record = _record()
    backend = object.__new__(LocalExecutionBackend)

    async def no_resolution_operations(
        *_args: object, **_kwargs: object
    ) -> tuple[object, ...]:
        return ()

    backend._execution = _ExecutionState(record)
    backend._recovery = SimpleNamespace(
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
    backend._worker_shutdown_requests = set()
    backend._terminal_events = {}
    backend._pending_audit_events = {}
    backend._pending_audit_locks = {}
    backend._approval_pause_segments = {}
    backend._agent_run_only_worker_exits = set()
    backend._repository_instruction_provenance = {}
    backend._checkpoint_tasks = set()
    backend._execution_durable_tasks = {}
    backend._metric_recorder = None
    backend._tool_operations = None
    backend._live_broker = SimpleNamespace(complete=lambda _execution_id: None)
    return backend


@pytest.mark.asyncio
async def test_prepare_start_persists_exact_binding_and_execution_policy() -> None:
    backend = _backend()
    execution = replace(_record(), status=ExecutionStatus.PENDING_START)
    commands = _StartCommands(execution)
    backend._recovery_enabled = True
    backend._runtime_commands = commands

    started = await backend.prepare_start(
        _request(),
        execution,
        ExecutionStartIdentity("scope", "key", "request"),
    )

    assert started.status is ExecutionStatus.STARTED
    checkpoint = commands.recovery_checkpoint
    assert checkpoint is not None
    assert checkpoint.execution_id == execution.execution_id
    assert checkpoint.state.value == "admitted"
    assert checkpoint.pending_tools is None


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
    def __init__(
        self,
        records: tuple[object, ...],
        *,
        expired_claims: bool = False,
    ) -> None:
        self.records = records
        self.expired_claims = expired_claims
        self.reconcile_calls = 0

    async def list_by_execution(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> tuple[object, ...]:
        del execution_id, tenant_id
        return self.records

    async def reconcile_expired_claim(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
    ) -> object:
        del tenant_id
        self.reconcile_calls += 1
        record = next(
            value
            for value in self.records
            if value.tool_operation_id == tool_operation_id
        )
        if not self.expired_claims:
            return record
        return SimpleNamespace(
            **{
                **vars(record),
                "status": ToolOperationStatus.EFFECT_UNKNOWN,
                "error_code": ErrorCode.TOOL_EFFECT_UNKNOWN.value,
            }
        )


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
        idempotency_key_digest="key-digest",
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
        expired_claims=True,
    )
    backend._tool_operations = operations

    effects = await backend._recovery_failure_effects(
        "execution",
        tenant_id="tenant",
    )

    assert effects == ()
    assert operations.reconcile_calls == 0


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


@pytest.mark.asyncio
async def test_local_cancel_without_worker_does_not_confirm_live_tool_claim() -> None:
    backend = _backend()
    current = replace(_record(), status=ExecutionStatus.CANCELLING)
    backend._execution.executions.record = current
    backend._tool_operations = _ToolOperations(
        (_tool_operation(ToolOperationStatus.CLAIMED),)
    )

    assert await backend.cancel(current) is CancelEffectOutcome.UNKNOWN


@pytest.mark.asyncio
async def test_local_cancel_without_worker_reconciles_expired_claim_to_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _backend()
    current = replace(_record(), status=ExecutionStatus.CANCELLING)
    backend._execution.executions.record = current
    backend._tool_operations = _ToolOperations(
        (_tool_operation(ToolOperationStatus.CLAIMED),),
        expired_claims=True,
    )
    observed: list[tuple[ExecutionRecord, tuple[object, ...]]] = []

    async def commit_recovery(
        execution: ExecutionRecord,
        error: AIError,
        effects: tuple[object, ...],
    ) -> ExecutionRecord:
        assert error.code is ErrorCode.TOOL_EFFECT_UNKNOWN
        observed.append((execution, effects))
        return replace(execution, status=ExecutionStatus.RECOVERY_REQUIRED)

    monkeypatch.setattr(backend, "_commit_recovery_required", commit_recovery)

    assert await backend.cancel(current) is CancelEffectOutcome.UNKNOWN
    assert len(observed) == 1
    assert observed[0][1][0].operation_id == "tool-operation"


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


class _Checkpoints:
    def __init__(self, checkpoint: object | None) -> None:
        self.checkpoint = checkpoint
        self.reads = 0

    async def get(self, execution_id: str, *, tenant_id: str) -> object | None:
        del execution_id, tenant_id
        self.reads += 1
        return self.checkpoint


class _Sessions:
    def __init__(self) -> None:
        self.releases: list[tuple[str, str, str]] = []

    async def release_execution(
        self,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
    ) -> object:
        self.releases.append((session_id, tenant_id, execution_id))
        return SimpleNamespace(active_execution_id=None)


@pytest.mark.asyncio
async def test_abort_start_releases_its_session_when_admission_checkpoint_is_absent() -> None:
    backend = _backend()
    current = replace(
        _record(),
        session_id="session",
        status=ExecutionStatus.FAILED,
    )
    backend._execution.executions.record = current
    checkpoints = _Checkpoints(None)
    sessions = _Sessions()
    backend._recovery = SimpleNamespace(checkpoints=checkpoints)
    backend._conversation = SimpleNamespace(sessions=sessions)

    await backend.abort_start(current)

    assert checkpoints.reads == 1
    assert sessions.releases == [("session", "tenant", "execution")]


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
    checkpoints = _Checkpoints(None)
    sessions = _Sessions()
    backend._recovery = SimpleNamespace(checkpoints=checkpoints)
    backend._conversation = SimpleNamespace(sessions=sessions)

    with pytest.raises(AIError) as raised:
        await backend.abort_start(current)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert checkpoints.reads == 0
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
