#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for Runtime-owned cancellation quiescence."""

import asyncio
from types import SimpleNamespace

import pytest

from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._execution import (
    CancelEffectOutcome,
    DefaultExecutionService,
)
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.task._local import LocalTaskGraphLauncher
from linktools.ai.task._service_impl import DefaultTaskService


class _ExecutionRecords:
    def __init__(self, current: object) -> None:
        self.current = current

    async def get(self, execution_id: str, *, tenant_id: str) -> object:
        del execution_id, tenant_id
        return self.current


class _LiveBroker:
    def complete(self, execution_id: str) -> None:
        del execution_id


class _Executor:
    @property
    def pending_background_tasks(self) -> tuple[asyncio.Task[object], ...]:
        return ()


def _local_backend(current: object) -> LocalExecutionBackend:
    backend = object.__new__(LocalExecutionBackend)
    backend._execution = SimpleNamespace(executions=_ExecutionRecords(current))
    backend._tenant_id = "tenant"
    backend._tasks = {}
    backend._worker_failures = {}
    backend._captured_usage = {}
    backend._terminal_events = {}
    backend._pending_audit_events = {}
    backend._pending_audit_locks = {}
    backend._approval_pause_segments = {}
    backend._segment_only_worker_exits = set()
    backend._repository_instruction_provenance = {}
    backend._checkpoint_tasks = set()
    backend._execution_durable_tasks = {}
    backend._worker_cancel_requests = set()
    backend._live_broker = _LiveBroker()
    backend._executor = _Executor()
    backend._subagent_dispatcher = None
    backend._accepting = True
    return backend


def _execution_service() -> DefaultExecutionService:
    service = object.__new__(DefaultExecutionService)
    service._handoff_states = {}
    service._handoff_condition = asyncio.Condition()
    service._detached_cancel_finalizers = set()
    service._detached_cancel_failure = None
    return service


@pytest.mark.asyncio
async def test_local_checkpoint_wait_settles_owned_commit_before_restoring_cancellation() -> None:
    backend = _local_backend(SimpleNamespace())
    started = asyncio.Event()
    release = asyncio.Event()

    async def commit() -> str:
        started.set()
        await release.wait()
        return "committed"

    owned = asyncio.create_task(commit())
    caller = asyncio.create_task(
        backend._await_checkpoint_task(
            owned,
            label="test",
            execution_id="execution",
        )
    )
    await started.wait()
    caller.cancel()
    await asyncio.sleep(0)

    assert not caller.done()
    assert not owned.done()

    release.set()
    value, cancellation = await caller
    assert value == "committed"
    assert isinstance(cancellation, asyncio.CancelledError)
    assert backend._checkpoint_tasks == set()
    assert backend._execution_durable_tasks.get("execution", set()) == set()


@pytest.mark.asyncio
async def test_local_cancel_waits_for_owned_worker_durable_settlement() -> None:
    current = SimpleNamespace(
        execution_id="execution",
        tenant_id="tenant",
        status=ExecutionStatus.CANCELLING,
    )
    backend = _local_backend(current)
    cancelled = asyncio.Event()
    release = asyncio.Event()

    async def worker() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
            backend._execution.executions.current = SimpleNamespace(
                execution_id="execution",
                tenant_id="tenant",
                status=ExecutionStatus.CANCELLED,
            )
            raise

    worker_task = asyncio.create_task(worker())
    backend._tasks["execution"] = worker_task
    cancel_task = asyncio.create_task(backend.cancel(current))

    await cancelled.wait()
    await asyncio.sleep(0)
    assert not cancel_task.done()

    release.set()
    assert await cancel_task is CancelEffectOutcome.CONFIRMED
    assert backend._tasks == {}
    assert backend._execution.executions.current.status is ExecutionStatus.CANCELLED


@pytest.mark.asyncio
async def test_local_cancel_does_not_repeat_cancel_during_owned_cleanup() -> None:
    current = SimpleNamespace(
        execution_id="execution",
        tenant_id="tenant",
        status=ExecutionStatus.CANCELLING,
    )
    backend = _local_backend(current)
    first_cancel = asyncio.Event()
    release = asyncio.Event()
    cancellation_count = 0

    async def worker() -> None:
        nonlocal cancellation_count
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_count += 1
            first_cancel.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancellation_count += 1
                raise
            backend._execution.executions.current = SimpleNamespace(
                execution_id="execution",
                tenant_id="tenant",
                status=ExecutionStatus.CANCELLED,
            )
            raise

    worker_task = asyncio.create_task(worker())
    backend._tasks["execution"] = worker_task
    first = asyncio.create_task(backend.cancel(current))
    await first_cancel.wait()

    second = asyncio.create_task(backend.cancel(current))
    await asyncio.sleep(0)
    assert not first.done()
    assert not second.done()
    assert cancellation_count == 1

    release.set()
    assert await first is CancelEffectOutcome.CONFIRMED
    assert await second is CancelEffectOutcome.CONFIRMED
    assert cancellation_count == 1


@pytest.mark.asyncio
async def test_local_close_drains_owned_worker_and_finalizer_before_guarding_background_work() -> None:
    current = SimpleNamespace(
        execution_id="execution",
        tenant_id="tenant",
        status=ExecutionStatus.STARTED,
    )
    backend = _local_backend(current)
    worker_cancelled = asyncio.Event()
    release_worker = asyncio.Event()
    finalizer_started = asyncio.Event()
    release_finalizer = asyncio.Event()

    async def worker() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            worker_cancelled.set()
            await release_worker.wait()
            raise

    async def finalizer() -> None:
        finalizer_started.set()
        await release_finalizer.wait()

    worker_task = asyncio.create_task(worker())
    backend._tasks["execution"] = worker_task
    finalizer_task = asyncio.create_task(finalizer())

    class Executor:
        @property
        def pending_background_tasks(self) -> tuple[asyncio.Task[object], ...]:
            return () if finalizer_task.done() else (finalizer_task,)

    backend._executor = Executor()
    close_task = asyncio.create_task(backend.close())

    await worker_cancelled.wait()
    await finalizer_started.wait()
    await asyncio.sleep(0)
    assert not close_task.done()

    release_worker.set()
    await asyncio.sleep(0)
    assert not close_task.done()

    release_finalizer.set()
    await close_task
    assert backend._tasks == {}


@pytest.mark.asyncio
async def test_execution_service_caller_cancel_detaches_owned_finalizer_until_preflight() -> None:
    service = _execution_service()
    started = asyncio.Event()
    release = asyncio.Event()

    async def finalizer(execution_id: str, request: object) -> object:
        del execution_id, request
        started.set()
        await release.wait()
        return SimpleNamespace(cancelled=True)

    service._cancel_finalizer_with_handoff = finalizer
    request = SimpleNamespace(principal=SimpleNamespace(tenant_id="tenant"))
    caller = asyncio.create_task(service.cancel("execution", request))
    await started.wait()
    caller.cancel()

    with pytest.raises(asyncio.CancelledError):
        await caller

    preflight = asyncio.create_task(service.preflight_close())
    await asyncio.sleep(0)
    assert not preflight.done()

    release.set()
    await preflight
    assert service._detached_cancel_finalizers == set()


@pytest.mark.asyncio
async def test_execution_service_preflight_surfaces_detached_cancel_failure() -> None:
    service = _execution_service()

    async def fail() -> object:
        raise RuntimeError("cancel finalizer failed")

    task = asyncio.create_task(fail())
    service._detach_cancel_finalizer(task, "execution")
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)

    with pytest.raises(AIError) as error:
        await service.preflight_close()
    assert error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    assert error.value.safe_details["phase"] == "execution_cancel_finalizer"


@pytest.mark.asyncio
async def test_task_service_preflight_drains_owned_finalizers() -> None:
    service = object.__new__(DefaultTaskService)
    release = asyncio.Event()

    async def finalizer() -> None:
        await release.wait()

    task = asyncio.create_task(finalizer())
    service._detached_finalizers = {task}
    service._detached_finalizer_failure = None

    preflight = asyncio.create_task(service.preflight_close())
    await asyncio.sleep(0)
    assert not preflight.done()

    release.set()
    await preflight


@pytest.mark.asyncio
async def test_task_service_preflight_surfaces_detached_finalizer_failure() -> None:
    service = object.__new__(DefaultTaskService)
    service._detached_finalizers = set()
    service._detached_finalizer_failure = None

    async def fail() -> None:
        raise RuntimeError("finalizer failed")

    task = asyncio.create_task(fail())
    service._detach_finalizer(task, "graph")
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)

    with pytest.raises(AIError) as error:
        await service.preflight_close()
    assert error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    assert error.value.safe_details["phase"] == "task_service_finalizer"


@pytest.mark.asyncio
async def test_task_launcher_shutdown_drains_runner_owned_cancellation_cleanup() -> None:
    release = asyncio.Event()

    async def cleanup() -> None:
        await release.wait()

    cleanup_task = asyncio.create_task(cleanup())

    class Runner:
        @property
        def pending_background_tasks(self) -> tuple[asyncio.Task[object], ...]:
            return () if cleanup_task.done() else (cleanup_task,)

        @property
        def background_failure(self) -> None:
            return None

    launcher = object.__new__(LocalTaskGraphLauncher)
    launcher._accepting = True
    launcher._graphs = {}
    launcher._wait_observations = {}
    launcher._detached_tasks = set()
    launcher._runner = Runner()

    shutdown = asyncio.create_task(launcher.shutdown())
    await asyncio.sleep(0)
    assert not shutdown.done()

    release.set()
    await shutdown


@pytest.mark.asyncio
async def test_task_launcher_shutdown_drains_detached_owned_cleanup() -> None:
    launcher = object.__new__(LocalTaskGraphLauncher)
    launcher._accepting = True
    launcher._graphs = {}
    launcher._wait_observations = {}
    launcher._detached_tasks = set()
    launcher._runner = SimpleNamespace()
    release = asyncio.Event()

    async def cleanup() -> None:
        await release.wait()

    task = asyncio.create_task(cleanup())
    launcher._detach(task, "owned cleanup")

    shutdown = asyncio.create_task(launcher.shutdown())
    await asyncio.sleep(0)
    assert not shutdown.done()

    release.set()
    await shutdown