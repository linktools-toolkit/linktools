#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local execution failure and worker supervision coverage."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._output import bind_output
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ExecutionRequest
from linktools.ai.runtime._execution import CancelEffectOutcome, ExecutionStartIdentity
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime.state import (
    ExecutionRecord,
    LoadedContextMessage,
    LoadedModelContext,
)
from linktools.ai.spec import AgentSpec
from pydantic_ai.messages import ModelRequest, UserPromptPart


def _binding_snapshot() -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("default", 1, "default"),
        agent_digest="b" * 64,
        output_schema_id=output.schema_id,
        output_schema_revision=output.schema_revision,
        output_schema_fingerprint=output.schema_fingerprint,
        local_runtime_capability_descriptors=(),
        binding_digest="a" * 64,
        global_runtime_capability_descriptors=(),
    )


def _binding() -> object:
    snapshot = _binding_snapshot()
    definition = SimpleNamespace(
        digest=snapshot.agent_digest,
        spec=SimpleNamespace(id="default", allow_tools=()),
    )
    return SimpleNamespace(
        digest=snapshot.binding_digest, snapshot=snapshot, definition=definition
    )


class _Executions:
    def __init__(self, record: ExecutionRecord) -> None:
        self.record = record

    async def get(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord:
        del execution_id, tenant_id
        return self.record

    async def claim_next_agent_run(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        expected_agent_run_sequence: int,
    ) -> ExecutionRecord:
        del execution_id, tenant_id, expected_revision, expected_agent_run_sequence
        self.record = replace(
            self.record,
            agent_run_sequence=self.record.agent_run_sequence + 1,
        )
        return self.record


class _ExecutionState:
    def __init__(self, record: ExecutionRecord) -> None:
        self.executions = _Executions(record)


class _Metrics:
    def count(self, metric: str, **labels: str) -> None:
        del metric, labels

    def operation(self, domain: str, target: str, result: str, started_at: float) -> None:
        del domain, target, result, started_at


class _Executor:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def execute(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise self.error


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
        planning=False,
        thinking=False,
        binding=_binding_snapshot(),
    )


def _backend(error: Exception) -> LocalExecutionBackend:
    record = _record()
    backend = object.__new__(LocalExecutionBackend)
    backend._execution = _ExecutionState(record)
    binding = _binding()
    backend._catalog = SimpleNamespace(
        root_ids=("default",),
        binding=lambda digest: binding,
    )
    backend._recovery_enabled = False
    backend._namespace = "test"
    backend._tenant_id = "tenant"
    backend._steps = object()
    backend._executor = _Executor(error)
    backend._execution_root = Path(".")
    backend._memory_store_factory = None
    backend._subagent_dispatcher = None
    backend._metrics = _Metrics()
    backend._tasks = {}
    backend._worker_failures = {}
    backend._captured_usage = {}
    backend._history = lambda execution: _empty_history(execution)
    return backend


async def _empty_history(execution: ExecutionRecord) -> list[object]:
    del execution
    return []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("messages", "expected_replacement"),
    (
        ((), False),
        ((ModelRequest(parts=(UserPromptPart(content="prior"),)),), True),
    ),
)
async def test_session_prompt_replacement_requires_prior_conversation_history(
    messages: tuple[ModelRequest, ...],
    expected_replacement: bool,
) -> None:
    backend = _backend(ValueError("model output failed"))
    record = replace(
        _record(),
        session_id="session",
        lineage_kind=ExecutionLineageKind.SESSION_RESUME,
    )
    backend._execution.executions.record = record

    async def session_get(session_id: str, *, tenant_id: str) -> object:
        del session_id, tenant_id
        return SimpleNamespace(history_id="history")

    class _Steps:
        async def load_loaded_model_context(
            self,
            domain: object,
            owner_id: str,
        ) -> LoadedModelContext:
            del domain, owner_id
            return LoadedModelContext(
                tuple(LoadedContextMessage(message, None) for message in messages)
            )

    class _Executor:
        def __init__(self) -> None:
            self.replacement: bool | None = None

        async def execute(self, *args: object, **kwargs: object) -> object:
            del args
            self.replacement = kwargs["replace_history_system_prompt"]
            raise ValueError("model output failed")

    executor = _Executor()
    backend._conversation = SimpleNamespace(
        sessions=SimpleNamespace(get=session_get),
    )
    backend._steps = _Steps()
    backend._executor = executor

    async def commit_failure(
        execution: ExecutionRecord,
        error: Exception,
        *,
        run_id: str | None = None,
    ) -> None:
        del error, run_id
        backend._execution.executions.record = replace(
            execution,
            status=ExecutionStatus.FAILED,
        )

    backend._commit_failure = commit_failure
    await backend._run(
        ExecutionRequest("prompt", Principal("owner", "tenant"), "idempotency"),
        record,
    )

    assert executor.replacement is expected_replacement


@pytest.mark.asyncio
async def test_prepare_start_persists_binding_identity_in_recovery_input() -> None:
    backend = _backend(ValueError("unused"))
    execution = replace(_record(), status=ExecutionStatus.PENDING_START)
    commands = _StartCommands(execution)
    backend._accepting = True
    backend._recovery_enabled = True
    backend._runtime_commands = commands

    started = await backend.prepare_start(
        ExecutionRequest("prompt", Principal("owner", "tenant"), "idempotency"),
        execution,
        ExecutionStartIdentity("scope", "key", "request"),
    )

    assert started.status is ExecutionStatus.STARTED
    checkpoint = commands.recovery_checkpoint
    assert checkpoint is not None
    assert checkpoint.input.binding_digest == execution.binding_digest
    assert checkpoint.input.binding == execution.binding


@pytest.mark.asyncio
async def test_local_business_failure_commits_failed_without_escaping() -> None:
    backend = _backend(ValueError("model output failed"))
    failures: list[Exception] = []

    async def commit_failure(
        execution: ExecutionRecord,
        error: Exception,
        *,
        run_id: str | None = None,
    ) -> None:
        del run_id
        failures.append(error)
        backend._execution.executions.record = replace(
            execution,
            status=ExecutionStatus.FAILED,
        )

    backend._commit_failure = commit_failure
    await backend._run(
        ExecutionRequest("prompt", Principal("owner", "tenant"), "idempotency"),
        _record(),
    )

    assert len(failures) == 1
    assert isinstance(failures[0], ValueError)
    assert backend._execution.executions.record.agent_run_sequence == 1


@pytest.mark.asyncio
async def test_local_infrastructure_failure_escapes_without_failed_commit() -> None:
    backend = _backend(AIError(ErrorCode.STORAGE_INTEGRITY_ERROR))
    committed = False

    async def commit_failure(*args: object, **kwargs: object) -> None:
        nonlocal committed
        del args, kwargs
        committed = True

    backend._commit_failure = commit_failure
    with pytest.raises(AIError) as error:
        await backend._run(
            ExecutionRequest("prompt", Principal("owner", "tenant"), "idempotency"),
            _record(),
        )

    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert not committed


@pytest.mark.asyncio
async def test_local_worker_failure_is_consumed_and_observable() -> None:
    backend = object.__new__(LocalExecutionBackend)
    backend._tenant_id = "tenant"
    backend._tasks = {}
    backend._captured_usage = {}
    backend._worker_failures = {}

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


@pytest.mark.asyncio
async def test_local_old_worker_callback_cannot_clear_new_owner() -> None:
    backend = object.__new__(LocalExecutionBackend)
    backend._tasks = {}
    backend._captured_usage = {}
    backend._worker_failures = {}

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
    (ExecutionStatus.SUCCEEDED, ExecutionStatus.CANCELLED, ExecutionStatus.CANCELLING),
)
async def test_local_cancel_without_worker_confirms_durable_terminal_state(
    status: ExecutionStatus,
) -> None:
    backend = _backend(ValueError("unused"))
    current = replace(_record(), status=status)
    backend._execution.executions.record = current

    assert await backend.cancel(current) == CancelEffectOutcome.CONFIRMED


@pytest.mark.asyncio
async def test_local_cancel_before_worker_coroutine_starts_confirms_side_effect_stop() -> None:
    backend = _backend(ValueError("unused"))
    current = replace(_record(), status=ExecutionStatus.CANCELLING)
    backend._execution.executions.record = current

    async def wait() -> None:
        await asyncio.Event().wait()

    task = asyncio.create_task(wait())
    backend._tasks["execution"] = task
    outcome = await backend.cancel(current)

    assert outcome is CancelEffectOutcome.CONFIRMED
    assert backend._tasks == {}
