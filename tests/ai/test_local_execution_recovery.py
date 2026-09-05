#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local execution recovery and worker-supervision contracts."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._output import bind_output
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ExecutionRequest
from linktools.ai.runtime._execution import CancelEffectOutcome, ExecutionStartIdentity
from linktools.ai.runtime._local import LocalExecutionBackend, _is_infrastructure_error
from linktools.ai.runtime.state import ExecutionRecord
from linktools.ai.spec import AgentSpec


def _binding_snapshot() -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("default"),
        model={"route_id": "default", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
        binding_digest="a" * 64,
    )


def _binding() -> object:
    snapshot = _binding_snapshot()
    definition = SimpleNamespace(
        digest="b" * 64,
        spec=SimpleNamespace(id="default"),
    )
    return SimpleNamespace(
        digest=snapshot.binding_digest,
        snapshot=snapshot,
        definition=definition,
    )


def _request() -> ExecutionRequest:
    return ExecutionRequest(
        user_prompt="prompt",
        user_prompt_codec="text",
        principal=Principal("owner", "tenant"),
        idempotency_key="idempotency",
        memory_scope=None,
        mode="run",
        planning=False,
        thinking=False,
    )


def _record() -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    return ExecutionRecord(
        execution_id="execution",
        tenant_id="tenant",
        session_id=None,
        binding_digest="a" * 64,
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
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
        binding=_binding_snapshot(),
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
    backend._execution = _ExecutionState(record)
    binding = _binding()
    backend._catalog = SimpleNamespace(binding=lambda digest: binding)
    backend._accepting = True
    backend._recovery_enabled = False
    backend._tenant_id = "tenant"
    backend._tasks = {}
    backend._captured_usage = {}
    backend._worker_failures = {}
    backend._worker_cancel_requests = set()
    backend._terminal_events = {}
    backend._pending_audit_events = {}
    backend._pending_audit_locks = {}
    backend._approval_pause_segments = {}
    backend._segment_only_worker_exits = set()
    backend._repository_instruction_provenance = {}
    backend._checkpoint_tasks = set()
    backend._execution_durable_tasks = {}
    backend._metric_recorder = None
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
    assert checkpoint.input.binding_digest == execution.binding_digest
    assert checkpoint.input.binding == execution.binding
    assert checkpoint.input.user_prompt_codec == "text"
    assert checkpoint.input.mode == execution.mode
    assert checkpoint.input.planning is execution.planning
    assert checkpoint.input.thinking == execution.thinking


@pytest.mark.parametrize(
    ("error", "expected"),
    (
        (ValueError("business"), False),
        (AIError(ErrorCode.OUTPUT_VALIDATION_FAILED), False),
        (AIError(ErrorCode.STORAGE_INTEGRITY_ERROR), True),
        (AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE), True),
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
