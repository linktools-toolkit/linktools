#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Invocation capture retains callable execution ownership on interrupted writes."""

import asyncio
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus, TaskStatus, service_principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import CaptureInputRequest, Runtime, RuntimeStorage
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext, TaskRef
from .test_task_mixed_node_reliability import _TaskTestModels


@pytest.mark.asyncio
@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("cancel_write", [False, True])
async def test_invocation_capture_interruption_retains_terminal_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deferred: bool, cancel_write: bool,
) -> None:
    storage = RuntimeStorage.filesystem(tmp_path)
    principal = service_principal("default", "owner")
    calls = []

    async def handler(context: TaskNodeContext[None]) -> str:
        calls.append("run")
        return "ok"

    async def cancel(context: TaskNodeContext[None]) -> None:
        calls.append("cancel")

    task = Task("capture.work", handler, cancel=cancel, effect_policy="none")
    async with Runtime.open("capture", models=_TaskTestModels(), storage=storage) as runtime:
        objects = storage.object_store(RuntimeDomain.TASK)
        original_put = objects.put
        entered = asyncio.Event()

        async def interrupted_put(key, *args, **kwargs):
            if key.startswith("v1/input-capture/invocation/"):
                entered.set()
                if cancel_write:
                    await asyncio.Event().wait()
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
            return await original_put(key, *args, **kwargs)

        monkeypatch.setattr(objects, "put", interrupted_put)
        engine = runtime.tasks.bind(task)
        graph = TaskGraph("capture", (TaskNode("work", task=TaskRef.deferred_input() if deferred else task),))
        run = await engine.start(graph, principal=principal, idempotency_key="capture")
        await asyncio.wait_for(entered.wait(), 5)
        bound = (await run.state()).node_states[0]
        assert bound.execution_id is not None
        if cancel_write:
            await asyncio.wait_for(run.cancel(idempotency_key="cancel"), 5)
        result = await run.wait(timeout_seconds=5)
        state = (await run.state()).node_states[0]
        assert state.execution_id == bound.execution_id
        assert result.status is (TaskStatus.CANCELLED if cancel_write else TaskStatus.FAILED)
        if not cancel_write:
            assert state.error_code == ErrorCode.STORAGE_UNAVAILABLE.value
        view = await runtime.executions.inspect(state.execution_id, principal=principal)
        assert view.status is ExecutionStatus.CANCELLED
        assert view.task_attempt == 0
        monkeypatch.setattr(objects, "put", original_put)
        await run.cancel(idempotency_key="cancel-again")
        same = await engine.start(graph, principal=principal, idempotency_key="capture")
        assert (await same.state()).node_states[0].execution_id == state.execution_id
        assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_recovery", [False, True])
async def test_invocation_capture_failed_cleanup_retains_recoverable_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_recovery: bool,
) -> None:
    storage = RuntimeStorage.filesystem(tmp_path)
    principal = service_principal("default", "owner")
    calls = []

    async def handler(context: TaskNodeContext[None]) -> str:
        calls.append("run")
        return "ok"

    async def cancel(context: TaskNodeContext[None]) -> None:
        calls.append("cancel")

    task = Task("capture.work", handler, cancel=cancel, effect_policy="none")
    async with Runtime.open("capture", models=_TaskTestModels(), storage=storage) as runtime:
        objects = storage.object_store(RuntimeDomain.TASK)
        original_put = objects.put
        original_cancel = runtime._execution_service.cancel_task
        cleanup_failed = asyncio.Event()

        async def failed_put(key, *args, **kwargs):
            if key.startswith("v1/input-capture/invocation/"):
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
            return await original_put(key, *args, **kwargs)

        async def failed_cancel(*args, **kwargs):
            cleanup_failed.set()
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

        monkeypatch.setattr(objects, "put", failed_put)
        monkeypatch.setattr(runtime._execution_service, "cancel_task", failed_cancel)
        run = await runtime.tasks.bind(task).start(
            TaskGraph("capture", (TaskNode("work", task=task),)),
            principal=principal, idempotency_key="capture",
        )
        await asyncio.wait_for(cleanup_failed.wait(), 5)
        async def recovery_state():
            while True:
                node = (await run.state()).node_states[0]
                if node.error_code == ErrorCode.STORAGE_RECOVERY_REQUIRED.value:
                    return node
                await asyncio.sleep(0.01)
        state = await asyncio.wait_for(recovery_state(), 5)
        assert state.execution_id is not None
        view = await runtime.executions.inspect(state.execution_id, principal=principal)
        assert view.status is ExecutionStatus.STARTED
        assert view.task_attempt == 0
        monkeypatch.setattr(objects, "put", original_put)
        monkeypatch.setattr(runtime._execution_service, "cancel_task", original_cancel)
        assert (await run.wait(timeout_seconds=5)).status is TaskStatus.RECOVERY_REQUIRED
        if cancel_recovery:
            execution = await run.execution("work")
            await asyncio.wait_for(execution.cancel(idempotency_key="recover-cancel"), 5)
        else:
            await run.recover(idempotency_key="recover")
        result = await run.wait(timeout_seconds=5)
        assert result.status is (TaskStatus.CANCELLED if cancel_recovery else TaskStatus.SUCCEEDED)
        assert (await run.state()).node_states[0].execution_id == state.execution_id
        view = await runtime.executions.inspect(state.execution_id, principal=principal)
        assert view.status is (ExecutionStatus.CANCELLED if cancel_recovery else ExecutionStatus.SUCCEEDED)
        assert calls == ([] if cancel_recovery else ["run"])
        if not cancel_recovery:
            assert await run.result("work") == "ok"
            capture = await runtime.executions.capture_input(
                state.execution_id, CaptureInputRequest(principal, "capture-recovered", "clean"),
            )
            assert capture is not None


@pytest.mark.asyncio
async def test_invocation_capture_repeated_cancellation_waits_for_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = RuntimeStorage.filesystem(tmp_path)
    principal = service_principal("default", "owner")
    calls = []

    async def handler(context: TaskNodeContext[None]) -> str:
        calls.append("run")
        return "ok"

    async def cancel(context: TaskNodeContext[None]) -> None:
        calls.append("cancel")

    task = Task("capture.work", handler, cancel=cancel, effect_policy="none")
    async with Runtime.open("capture", models=_TaskTestModels(), storage=storage) as runtime:
        objects = storage.object_store(RuntimeDomain.TASK)
        original_put = objects.put
        original_cancel = runtime._execution_service.cancel_task
        entered = asyncio.Event()
        cleanup_entered = asyncio.Event()
        release_cleanup = asyncio.Event()
        capture_owner = None

        async def pending_put(key, *args, **kwargs):
            nonlocal capture_owner
            if key.startswith("v1/input-capture/invocation/"):
                capture_owner = asyncio.current_task()
                entered.set()
                await asyncio.Event().wait()
            return await original_put(key, *args, **kwargs)

        async def pending_cancel(*args, **kwargs):
            cleanup_entered.set()
            await release_cleanup.wait()
            return await original_cancel(*args, **kwargs)

        monkeypatch.setattr(objects, "put", pending_put)
        monkeypatch.setattr(runtime._execution_service, "cancel_task", pending_cancel)
        run = await runtime.tasks.bind(task).start(
            TaskGraph("capture", (TaskNode("work", task=task),)),
            principal=principal, idempotency_key="capture",
        )
        await asyncio.wait_for(entered.wait(), 5)
        cancellation = asyncio.create_task(run.cancel(idempotency_key="cancel"))
        try:
            await asyncio.wait_for(cleanup_entered.wait(), 5)
            assert capture_owner is not None
            for _ in range(2):
                capture_owner.cancel()
                await asyncio.sleep(0)
                assert not capture_owner.done()
            state = (await run.state()).node_states[0]
            assert state.execution_id is not None
            view = await runtime.executions.inspect(state.execution_id, principal=principal)
            assert view.status is ExecutionStatus.STARTED
        finally:
            release_cleanup.set()
            await asyncio.wait_for(cancellation, 5)
        assert capture_owner.cancelled()
        assert (await run.wait(timeout_seconds=5)).status is TaskStatus.CANCELLED
        view = await runtime.executions.inspect(state.execution_id, principal=principal)
        assert view.status is ExecutionStatus.CANCELLED
        assert calls == []
