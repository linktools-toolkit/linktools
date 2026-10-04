#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Invocation capture storage failures preserve the graph's execution ownership."""

import asyncio
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import ExecutionStatus, JsonValue, TaskStatus, service_principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import AgentTaskInput, CaptureInputRequest, Runtime, RuntimeStorage
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.task import TaskGraph, TaskNode
from ._runtime_test_helpers import _RuntimeUsageModelBinding, _UsageFunctionModel


class _ControlledModels(_RuntimeUsageModelBinding):
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.requests = 0
        self.fail = False

    def capture(self) -> "_ControlledModels":
        return self

    def resolve(self, route_id: str) -> "_ControlledModels":
        assert route_id == self.route_id
        return self

    def restore(self, payload: Mapping[str, JsonValue], *, route_id: str | None = None) -> "_ControlledModels":
        assert route_id in {None, self.route_id}
        assert dict(payload) == self.contract
        return self

    def materialize(self) -> FunctionModel:
        async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del messages, info
            self.requests += 1
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            if self.fail:
                raise RuntimeError("model request failed")
            return ModelResponse(parts=[TextPart("completed")])
        return _UsageFunctionModel(respond)


def _agents() -> CapabilityGroup:
    group = CapabilityGroup("capture-ownership")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    return group


@pytest.mark.asyncio
async def test_invocation_capture_failure_cancels_the_bound_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = _ControlledModels()
    storage = RuntimeStorage.filesystem(tmp_path)
    principal = service_principal("default", "owner")
    async with Runtime.open("capture-ownership", models=models, storage=storage, capabilities=(_agents(),)) as runtime:
        objects = storage.object_store(RuntimeDomain.TASK)
        original_put = objects.put

        async def fail_invocation(key: str, *args: object, **kwargs: object) -> object:
            if key.startswith("v1/input-capture/invocation/"):
                await models.entered.wait()
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
            return await original_put(key, *args, **kwargs)

        monkeypatch.setattr(objects, "put", fail_invocation)
        task = runtime.tasks.from_agent("capture.agent", runtime.agents.get())
        run = await runtime.tasks.bind(task).start(TaskGraph("source", (
            TaskNode("agent", task=task, input=AgentTaskInput("question")),
        )), principal=principal, idempotency_key="source")
        try:
            result = await run.wait(timeout_seconds=5)
            assert result.status is TaskStatus.FAILED
            state = await run.state()
            assert state.node_states[0].execution_id is not None
            assert state.node_states[0].error_code == ErrorCode.STORAGE_UNAVAILABLE.value
            execution = await run.execution("agent")
            await asyncio.wait_for(models.cancelled.wait(), 5)
            assert (await runtime.executions.inspect(execution.execution_id, principal=principal)).status is ExecutionStatus.CANCELLED
            await run.cancel(idempotency_key="cancel")
            assert (await runtime.executions.inspect(execution.execution_id, principal=principal)).status is ExecutionStatus.CANCELLED
            record = await storage.execution.executions.get(execution.execution_id, tenant_id=principal.tenant_id)
            assert record.dependency_hold_ids == ()
            assert models.requests == 1
        finally:
            models.release.set()


@pytest.mark.asyncio
async def test_capture_cleanup_failure_recovers_the_same_execution_and_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = _ControlledModels()
    storage = RuntimeStorage.filesystem(tmp_path)
    principal = service_principal("default", "owner")
    async with Runtime.open("capture-recovery", models=models, storage=storage, capabilities=(_agents(),)) as runtime:
        objects = storage.object_store(RuntimeDomain.TASK)
        original_put = objects.put
        original_cancel = runtime._execution_service.cancel

        async def fail_invocation(key: str, *args: object, **kwargs: object) -> object:
            if key.startswith("v1/input-capture/invocation/"):
                await models.entered.wait()
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
            return await original_put(key, *args, **kwargs)

        async def fail_cancel(*args: object, **kwargs: object) -> object:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

        monkeypatch.setattr(objects, "put", fail_invocation)
        monkeypatch.setattr(runtime._execution_service, "cancel", fail_cancel)
        task = runtime.tasks.from_agent("capture.agent", runtime.agents.get())
        run = await runtime.tasks.bind(task).start(TaskGraph("source", (
            TaskNode("agent", task=task, input=AgentTaskInput("question")),
        )), principal=principal, idempotency_key="source")
        try:
            assert (await run.wait(timeout_seconds=5)).status is TaskStatus.RECOVERY_REQUIRED
            state = await run.state()
            assert state.node_states[0].execution_id is not None
            assert state.node_states[0].safe_error_details["phase"] == "task_invocation_capture_cancel"
            execution = await run.execution("agent")
            assert (await runtime.executions.inspect(execution.execution_id, principal=principal)).status is ExecutionStatus.STARTED
            monkeypatch.setattr(objects, "put", original_put)
            monkeypatch.setattr(runtime._execution_service, "cancel", original_cancel)
            models.release.set()
            await run.recover(idempotency_key="recover")
            assert (await run.wait(timeout_seconds=5)).status is TaskStatus.SUCCEEDED
            assert (await run.execution("agent")).execution_id == execution.execution_id
            assert await run.result("agent") == {"text": "completed"}
            captured = await runtime.executions.capture_input(execution.execution_id,
                CaptureInputRequest(principal, "recovered-input"))
            invocation = await runtime._input_captures.task_input(captured, principal=principal)
            assert invocation.source_execution_id == execution.execution_id
            assert models.requests == 1
        finally:
            models.release.set()


@pytest.mark.asyncio
async def test_graph_cancel_during_invocation_capture_cancels_the_owned_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = _ControlledModels()
    storage = RuntimeStorage.filesystem(tmp_path)
    principal = service_principal("default", "owner")
    async with Runtime.open("capture-cancel", models=models, storage=storage, capabilities=(_agents(),)) as runtime:
        objects = storage.object_store(RuntimeDomain.TASK)
        original_put = objects.put
        capture_started = asyncio.Event()

        async def pause_invocation(key: str, *args: object, **kwargs: object) -> object:
            if key.startswith("v1/input-capture/invocation/") and not capture_started.is_set():
                capture_started.set()
                await asyncio.Event().wait()
            return await original_put(key, *args, **kwargs)

        monkeypatch.setattr(objects, "put", pause_invocation)
        task = runtime.tasks.from_agent("capture.agent", runtime.agents.get())
        run = await runtime.tasks.bind(task).start(TaskGraph("source", (
            TaskNode("agent", task=task, input=AgentTaskInput("question")),
        )), principal=principal, idempotency_key="source")
        try:
            await asyncio.wait_for(capture_started.wait(), 5)
            await asyncio.wait_for(models.entered.wait(), 5)
            state = await run.state()
            assert state.node_states[0].execution_id is not None
            execution = await run.execution("agent")
            assert (await run.cancel(idempotency_key="cancel")).status is TaskStatus.CANCELLED
            await asyncio.wait_for(models.cancelled.wait(), 5)
            assert (await runtime.executions.inspect(execution.execution_id, principal=principal)).status is ExecutionStatus.CANCELLED
            captured = await runtime.executions.capture_input(execution.execution_id,
                CaptureInputRequest(principal, "cancelled-input"))
            invocation = await runtime._input_captures.task_input(captured, principal=principal)
            assert invocation.source_execution_id == execution.execution_id
        finally:
            models.release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("model_failure", [False, True])
async def test_terminal_agent_task_keeps_its_invocation_capture(
    tmp_path: Path, model_failure: bool,
) -> None:
    models = _ControlledModels()
    models.fail = model_failure
    models.release.set()
    principal = service_principal("default", "owner")
    async with Runtime.open("capture-terminal", models=models, storage=RuntimeStorage.filesystem(tmp_path),
                            capabilities=(_agents(),)) as runtime:
        task = runtime.tasks.from_agent("capture.agent", runtime.agents.get())
        run = await runtime.tasks.bind(task).start(TaskGraph("source", (
            TaskNode("agent", task=task, input=AgentTaskInput("question")),
        )), principal=principal, idempotency_key="source")
        assert (await run.wait(timeout_seconds=5)).status is (TaskStatus.FAILED if model_failure else TaskStatus.SUCCEEDED)
        execution = await run.execution("agent")
        captured = await runtime.executions.capture_input(execution.execution_id,
            CaptureInputRequest(principal, "terminal-input"))
        invocation = await runtime._input_captures.task_input(captured, principal=principal)
        assert invocation.source_execution_id == execution.execution_id


@pytest.mark.asyncio
async def test_graph_cancel_waits_for_capture_failure_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = _ControlledModels()
    storage = RuntimeStorage.filesystem(tmp_path)
    principal = service_principal("default", "owner")
    async with Runtime.open("capture-cleanup", models=models, storage=storage, capabilities=(_agents(),)) as runtime:
        objects = storage.object_store(RuntimeDomain.TASK)
        original_put = objects.put
        original_cancel = runtime._execution_service.cancel
        cleanup_entered = asyncio.Event()
        cleanup_release = asyncio.Event()
        failed = False

        async def fail_invocation_once(key: str, *args: object, **kwargs: object) -> object:
            nonlocal failed
            if key.startswith("v1/input-capture/invocation/") and not failed:
                failed = True
                await models.entered.wait()
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
            return await original_put(key, *args, **kwargs)

        async def pause_cancel(*args: object, **kwargs: object) -> object:
            cleanup_entered.set()
            await cleanup_release.wait()
            return await original_cancel(*args, **kwargs)

        monkeypatch.setattr(objects, "put", fail_invocation_once)
        monkeypatch.setattr(runtime._execution_service, "cancel", pause_cancel)
        task = runtime.tasks.from_agent("capture.agent", runtime.agents.get())
        run = await runtime.tasks.bind(task).start(TaskGraph("source", (
            TaskNode("agent", task=task, input=AgentTaskInput("question")),
        )), principal=principal, idempotency_key="source")
        try:
            await asyncio.wait_for(cleanup_entered.wait(), 5)
            execution = await run.execution("agent")
            cancellation = asyncio.create_task(run.cancel(idempotency_key="cancel"))
            await asyncio.sleep(0)
            assert not cancellation.done()
            assert not models.cancelled.is_set()
            cleanup_release.set()
            assert (await asyncio.wait_for(cancellation, 5)).status is TaskStatus.CANCELLED
            assert models.cancelled.is_set()
            assert (await runtime.executions.inspect(execution.execution_id, principal=principal)).status is ExecutionStatus.CANCELLED
            assert models.requests == 1
        finally:
            cleanup_release.set()
            models.release.set()


@pytest.mark.asyncio
async def test_invocation_capture_holds_terminal_execution_until_handoff_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = _ControlledModels()
    models.release.set()
    storage = RuntimeStorage.filesystem(tmp_path)
    principal = service_principal("default", "owner")
    async with Runtime.open("capture-hold", models=models, storage=storage, capabilities=(_agents(),)) as runtime:
        objects = storage.object_store(RuntimeDomain.TASK)
        original_put = objects.put
        capture_started = asyncio.Event()
        capture_release = asyncio.Event()

        async def pause_invocation(key: str, *args: object, **kwargs: object) -> object:
            if key.startswith("v1/input-capture/invocation/"):
                capture_started.set()
                await capture_release.wait()
            return await original_put(key, *args, **kwargs)

        monkeypatch.setattr(objects, "put", pause_invocation)
        task = runtime.tasks.from_agent("capture.agent", runtime.agents.get())
        run = await runtime.tasks.bind(task).start(TaskGraph("source", (
            TaskNode("agent", task=task, input=AgentTaskInput("question")),
        )), principal=principal, idempotency_key="source")
        try:
            await asyncio.wait_for(capture_started.wait(), 5)
            execution = await run.execution("agent")
            assert (await runtime.executions.wait(execution.execution_id, principal=principal)).status is ExecutionStatus.SUCCEEDED
            assert (await runtime.executions.inspect(execution.execution_id, principal=principal)).status is ExecutionStatus.SUCCEEDED
            record = await storage.execution.executions.get(execution.execution_id, tenant_id=principal.tenant_id)
            assert record.dependency_hold_ids
            capture_release.set()
            assert (await run.wait(timeout_seconds=5)).status is TaskStatus.SUCCEEDED
            assert await run.result("agent") == {"text": "completed"}
            record = await storage.execution.executions.get(execution.execution_id, tenant_id=principal.tenant_id)
            assert record.dependency_hold_ids == ()
        finally:
            capture_release.set()
