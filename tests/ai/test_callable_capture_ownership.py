#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from linktools.ai.task_engine import TaskEngine

from linktools.ai.persistence import Persistence
"""Invocation capture retains callable execution ownership on interrupted writes."""
from ._runtime_test_helpers import model_features

import asyncio
from collections.abc import Mapping
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus, JsonValue, TaskStatus, service_principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.inputs import CaptureInputRequest
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext, TaskRef
from .test_task_mixed_node_reliability import _TaskTestModels


@pytest.mark.asyncio
@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("cancel_write", [False, True])
async def test_invocation_capture_interruption_retains_terminal_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deferred: bool, cancel_write: bool,
) -> None:
    storage = PersistenceStorage.filesystem(tmp_path)
    principal = service_principal("default", "owner")
    calls = []

    async def handler(context: TaskNodeContext[None]) -> str:
        calls.append("run")
        return "ok"

    async def cancel(context: TaskNodeContext[None]) -> None:
        calls.append("cancel")

    task = Task("capture.work", handler, cancel=cancel, effect_policy="none")
    async with Persistence.open('capture', storage=storage) as persistence_38, Runtime.open(persistence=persistence_38, features=model_features(_TaskTestModels())) as runtime, TaskEngine.open(persistence=persistence_38, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        objects = runtime._state.object_store(PersistenceDomain.TASK)
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
        engine = runtime_engine.with_definitions(task)
        graph = TaskGraph("capture", (TaskNode("work", task=TaskRef.deferred_input() if deferred else task),))
        run = await engine.start(graph, principal=principal, idempotency_key="capture")
        await asyncio.wait_for(entered.wait(), 5)
        bound = (await run.state()).node_states[0]
        assert bound.execution_id is not None
        if cancel_write:
            await asyncio.wait_for(run.cancel(idempotency_key="cancel"), 5)
        result = (await run.wait(timeout_seconds=5)).result
        state = (await run.state()).node_states[0]
        assert state.execution_id == bound.execution_id
        assert result.wait_status is (TaskStatus.CANCELLED if cancel_write else TaskStatus.FAILED)
        if not cancel_write:
            assert state.error_code == ErrorCode.STORAGE_UNAVAILABLE.value
        view = await runtime.executions.inspect(state.execution_id, principal=principal)
        assert view.status is ExecutionStatus.CANCELLED
        assert view.task_attempt == 0
        record = await runtime._state.execution.executions.get(state.execution_id, tenant_id=principal.tenant_id)
        assert record.dependency_hold_ids == ()
        with pytest.raises(AIError) as missing:
            await runtime_engine.inputs.capture_input(state.execution_id,
                CaptureInputRequest(principal, "failed-input"))
        assert missing.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
        monkeypatch.setattr(objects, "put", original_put)
        await run.cancel(idempotency_key="cancel-again")
        same = await engine.start(graph, principal=principal, idempotency_key="capture")
        assert (await same.state()).node_states[0].execution_id == state.execution_id
        assert calls == []
    async with Persistence.open('capture', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_80, Runtime.open(persistence=persistence_80, features=model_features(_TaskTestModels())) as runtime, TaskEngine.open(persistence=persistence_80, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        with pytest.raises(AIError) as missing:
            await runtime_engine.inputs.capture_input(state.execution_id, CaptureInputRequest(principal, "failed-input"))
        assert missing.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_recovery", [False, True])
async def test_invocation_capture_failed_cleanup_retains_recoverable_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_recovery: bool,
) -> None:
    storage = PersistenceStorage.filesystem(tmp_path)
    principal = service_principal("default", "owner")
    calls = []

    async def handler(context: TaskNodeContext[None]) -> str:
        calls.append("run")
        return "ok"

    async def cancel(context: TaskNodeContext[None]) -> None:
        calls.append("cancel")

    task = Task("capture.work", handler, cancel=cancel, effect_policy="none")
    async with Persistence.open('capture', storage=storage) as persistence_103, Runtime.open(persistence=persistence_103, features=model_features(_TaskTestModels())) as runtime, TaskEngine.open(persistence=persistence_103, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        objects = runtime._state.object_store(PersistenceDomain.TASK)
        original_put = objects.put
        original_cancel = runtime_engine._execution_service.cancel_task
        cleanup_failed = asyncio.Event()

        async def failed_put(key, *args, **kwargs):
            if key.startswith("v1/input-capture/invocation/"):
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
            return await original_put(key, *args, **kwargs)

        async def failed_cancel(*args, **kwargs):
            cleanup_failed.set()
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

        monkeypatch.setattr(objects, "put", failed_put)
        monkeypatch.setattr(runtime_engine._execution_service, "cancel_task", failed_cancel)
        run = await runtime_engine.with_definitions(task).start(
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
        with pytest.raises(AIError) as pending:
            await runtime_engine.inputs.capture_input(state.execution_id,
                CaptureInputRequest(principal, "capture-recovered", "clean"))
        assert pending.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
        monkeypatch.setattr(objects, "put", original_put)
        monkeypatch.setattr(runtime_engine._execution_service, "cancel_task", original_cancel)
        assert (await run.wait(timeout_seconds=5)).result.wait_status is TaskStatus.RECOVERY_REQUIRED
        if cancel_recovery:
            execution = await run.execution("work")
            await asyncio.wait_for(execution.cancel(idempotency_key="recover-cancel"), 5)
        else:
            await run.recover(idempotency_key="recover")
        result = (await run.wait(timeout_seconds=5)).result
        assert result.wait_status is (TaskStatus.CANCELLED if cancel_recovery else TaskStatus.SUCCEEDED)
        assert (await run.state()).node_states[0].execution_id == state.execution_id
        view = await runtime.executions.inspect(state.execution_id, principal=principal)
        assert view.status is (ExecutionStatus.CANCELLED if cancel_recovery else ExecutionStatus.SUCCEEDED)
        assert calls == ([] if cancel_recovery else ["run"])
        record = await runtime._state.execution.executions.get(state.execution_id, tenant_id=principal.tenant_id)
        assert record.dependency_hold_ids == ()
        if not cancel_recovery:
            assert await run.result("work") == "ok"
            capture = await runtime_engine.inputs.capture_input(
                state.execution_id, CaptureInputRequest(principal, "capture-recovered", "clean"),
            )
            assert capture is not None


@pytest.mark.asyncio
async def test_invocation_capture_repeated_cancellation_waits_for_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = PersistenceStorage.filesystem(tmp_path)
    principal = service_principal("default", "owner")
    calls = []

    async def handler(context: TaskNodeContext[None]) -> str:
        calls.append("run")
        return "ok"

    async def cancel(context: TaskNodeContext[None]) -> None:
        calls.append("cancel")

    task = Task("capture.work", handler, cancel=cancel, effect_policy="none")
    async with Persistence.open('capture', storage=storage) as persistence_180, Runtime.open(persistence=persistence_180, features=model_features(_TaskTestModels())) as runtime, TaskEngine.open(persistence=persistence_180, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        objects = runtime._state.object_store(PersistenceDomain.TASK)
        original_put = objects.put
        original_cancel = runtime_engine._execution_service.cancel_task
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
        monkeypatch.setattr(runtime_engine._execution_service, "cancel_task", pending_cancel)
        run = await runtime_engine.with_definitions(task).start(
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
        assert (await run.wait(timeout_seconds=5)).result.wait_status is TaskStatus.CANCELLED
        view = await runtime.executions.inspect(state.execution_id, principal=principal)
        assert view.status is ExecutionStatus.CANCELLED
        assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("invocation_written", [False, True])
async def test_callable_capture_waits_for_original_input_and_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invocation_written: bool,
) -> None:
    storage = PersistenceStorage.filesystem(tmp_path)
    principal = service_principal("default", "owner")
    producer_ran = asyncio.Event()

    async def produce(context: TaskNodeContext[None]) -> str:
        producer_ran.set()
        return "captured dependency"

    def normalize(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        return {"value": value["value"] + 1}

    def reproject(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        return {"value": value["value"] + 10}

    async def consume(context: TaskNodeContext[None]) -> JsonValue:
        return {"value": context.input["value"], "dependency": await context.read_dependency("source")}

    producer = Task("capture.producer", produce, effect_policy="none")
    consumer = Task("capture.consumer", consume, normalize=normalize, effect_policy="none")
    candidate = Task("capture.candidate", consume, normalize=reproject, effect_policy="none")
    async with Persistence.open('callable-ready', storage=storage) as persistence_256, Runtime.open(persistence=persistence_256, features=model_features(_TaskTestModels())) as runtime, TaskEngine.open(persistence=persistence_256, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        objects = runtime._state.object_store(PersistenceDomain.TASK)
        original_put = objects.put
        entered, release = asyncio.Event(), asyncio.Event()

        async def pause_invocation(key: str, *args: object, **kwargs: object) -> object:
            if key.startswith("v1/input-capture/invocation/") and producer_ran.is_set():
                result = await original_put(key, *args, **kwargs) if invocation_written else None
                entered.set()
                await release.wait()
                if invocation_written:
                    return result
            return await original_put(key, *args, **kwargs)

        monkeypatch.setattr(objects, "put", pause_invocation)
        engine = runtime_engine.with_definitions(producer, consumer, candidate)
        run = await engine.start(TaskGraph("source", (
            TaskNode("source", task=producer),
            TaskNode("consumer", ("source",), task=consumer, input={"value": 1}),
        )), principal=principal, idempotency_key="source")
        request = CaptureInputRequest(principal, "same-input")
        try:
            await asyncio.wait_for(entered.wait(), 30)
            execution = await run.execution("consumer")
            assert (await runtime.executions.inspect(execution.execution_id, principal=principal)).status is ExecutionStatus.STARTED
            if invocation_written:
                early = await runtime_engine.inputs.capture_input(execution.execution_id, request)
            else:
                for _ in range(2):
                    with pytest.raises(AIError) as pending:
                        await runtime_engine.inputs.capture_input(execution.execution_id, request)
                    assert pending.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
        finally:
            release.set()
        assert (await run.wait(timeout_seconds=30)).result.wait_status is TaskStatus.SUCCEEDED
        monkeypatch.setattr(objects, "put", original_put)
        capture = await runtime_engine.inputs.capture_input(execution.execution_id, request)
        assert await runtime_engine.inputs.capture_input(execution.execution_id, request) == capture
        if invocation_written:
            assert capture == early
        value = await runtime_engine.inputs.read_task(capture, principal=principal)
        assert dict(value.original_input) == {"value": 1}
        assert dict(value.input) == {"value": 2}
        for mode, expected in (("fixed_input", 2), ("reproject_input", 11)):
            captured = await runtime_engine.inputs.task_input(capture, principal=principal, input_mode=mode)
            replay = await engine.start(TaskGraph(mode, (TaskNode("consumer", task=candidate, input_capture=captured),)),
                principal=principal, idempotency_key=mode)
            assert (await replay.wait(timeout_seconds=30)).result.wait_status is TaskStatus.SUCCEEDED
            assert await replay.result("consumer") == {"value": expected, "dependency": "captured dependency"}
        record = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=principal.tenant_id)
        assert record.dependency_hold_ids == ()


@pytest.mark.asyncio
async def test_standalone_callable_input_capture_does_not_require_invocation(tmp_path: Path) -> None:
    from linktools.ai.task import TaskBindingContract

    principal = service_principal("default", "owner")
    async with Persistence.open('standalone-callable', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_314, Runtime.open(persistence=persistence_314, features=model_features(_TaskTestModels())) as runtime, TaskEngine.open(persistence=persistence_314, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        binding = TaskBindingContract("standalone.task", 1, "none", {"kind": "json"}, None, 1, 0)
        execution = await runtime_engine._execution_service.start_task(binding, principal=principal,
            input={"standalone": True}, idempotency_key="source", correlation={})
        request = CaptureInputRequest(principal, "same-input")
        capture = await runtime_engine.inputs.capture_input(execution.execution_id, request)
        value = await runtime_engine.inputs.read_task(capture, principal=principal)
        assert dict(value.original_input) == dict(value.input) == {"standalone": True}
        await runtime_engine._execution_service.complete_task(execution.execution_id, principal=principal, output="completed")
        assert await runtime_engine.inputs.capture_input(execution.execution_id, request) == capture
