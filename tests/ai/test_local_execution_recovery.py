#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local execution failure and worker supervision coverage."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ExecutionRequest
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime.state import ExecutionRecord


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
    def operation(self, domain: str, target: str, result: str, started_at: float) -> None:
        del domain, target, result, started_at


class _Executor:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def execute(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise self.error


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
    )


def _backend(error: Exception) -> LocalExecutionBackend:
    record = _record()
    backend = object.__new__(LocalExecutionBackend)
    backend._execution = _ExecutionState(record)
    backend._catalog = SimpleNamespace(
        definition=lambda digest: SimpleNamespace(
            digest=digest,
            spec=SimpleNamespace(
                id="default",
                allow_tools=(),
                allow_skills=(),
            ),
        )
    )
    backend._recovery_enabled = False
    backend._namespace = "test"
    backend._steps = object()
    backend._executor = _Executor(error)
    backend._execution_root = Path(".")
    backend._memory_store_factory = None
    backend._subagent_dispatcher = None
    backend._metrics = _Metrics()
    backend._tasks = {}
    backend._captured_usage = {}
    backend._history = lambda execution: _empty_history(execution)
    return backend


async def _empty_history(execution: ExecutionRecord) -> list[object]:
    del execution
    return []


@pytest.mark.asyncio
async def test_local_business_failure_commits_failed_without_escaping() -> None:
    backend = _backend(ValueError("model output failed"))
    failures: list[Exception] = []

    async def commit_failure(execution: ExecutionRecord, error: Exception, *, run_id: str | None = None) -> None:
        del execution, run_id
        failures.append(error)

    backend._commit_failure = commit_failure
    await backend._run(
        ExecutionRequest("prompt", Principal("owner", "tenant"), "idempotency"),
        _record(),
    )

    assert len(failures) == 1
    assert isinstance(failures[0], ValueError)


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
    task.add_done_callback(backend._task_done)
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)

    failure = backend.worker_failure("execution", tenant_id="tenant")
    assert failure is not None
    assert failure.code is ErrorCode.STORAGE_INTEGRITY_ERROR
