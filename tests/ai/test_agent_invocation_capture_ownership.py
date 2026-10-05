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
            with pytest.raises(AIError) as missing:
                await runtime.executions.capture_input(execution.execution_id,
                    CaptureInputRequest(principal, "failed-input"))
            assert missing.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
            assert models.requests == 1
        finally:
            models.release.set()
    async with Runtime.open("capture-ownership", models=models, storage=RuntimeStorage.filesystem(tmp_path),
                            capabilities=(_agents(),)) as runtime:
        with pytest.raises(AIError) as missing:
            await runtime.executions.capture_input(execution.execution_id, CaptureInputRequest(principal, "failed-input"))
        assert missing.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE


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
            with pytest.raises(AIError) as pending:
                await runtime.executions.capture_input(execution.execution_id,
                    CaptureInputRequest(principal, "recovered-input"))
            assert pending.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
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
        cancellation_delivered = asyncio.Event()
        capture_owner: asyncio.Task[object] | None = None
        failed = False

        async def fail_invocation_once(key: str, *args: object, **kwargs: object) -> object:
            nonlocal failed, capture_owner
            if key.startswith("v1/input-capture/invocation/") and not failed:
                failed = True
                capture_owner = asyncio.current_task()
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
            await asyncio.wait_for(cleanup_entered.wait(), 30)
            execution = await run.execution("agent")
            assert capture_owner is not None
            original_owner_cancel = capture_owner.cancel

            def observe_cancel(msg: object = None) -> bool:
                requested = original_owner_cancel(msg)
                if requested:
                    cancellation_delivered.set()
                return requested

            monkeypatch.setattr(capture_owner, "cancel", observe_cancel)
            cancellation = asyncio.create_task(run.cancel(idempotency_key="cancel"))
            await asyncio.wait_for(cancellation_delivered.wait(), 30)
            assert not cancellation.done()
            assert not models.cancelled.is_set()
            cleanup_release.set()
            assert (await asyncio.wait_for(cancellation, 30)).status is TaskStatus.CANCELLED
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


@pytest.mark.asyncio
@pytest.mark.parametrize("invocation_written,session_id", [(False, None), (True, None), (False, "source-session")])
async def test_pending_invocation_capture_preserves_complete_replay_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invocation_written: bool, session_id: str | None,
) -> None:
    from linktools.ai.runtime import AgentTaskInputContext
    from linktools.ai.task import Task, TaskNodeContext

    models = _ControlledModels()
    models.release.set()
    storage = RuntimeStorage.filesystem(tmp_path)
    principal = service_principal("default", "owner")
    projections = []
    producer_ran = asyncio.Event()

    async def produce(context: TaskNodeContext[None]) -> str:
        producer_ran.set()
        return "Hello"

    async def project(context: AgentTaskInputContext) -> str:
        greeting = await context.result("greeting")
        projections.append((context.input["name"], greeting))
        return greeting + " " + context.input["name"]

    producer = Task("capture.greeting", produce, effect_policy="none")
    async with Runtime.open("capture-ready", models=models, storage=storage, capabilities=(_agents(),)) as runtime:
        task = runtime.tasks.from_agent("capture.agent", runtime.agents.get(), build_input=project)
        engine = runtime.tasks.bind(producer, task)
        objects = storage.object_store(RuntimeDomain.TASK)
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
        run = await engine.start(TaskGraph("source", (
            TaskNode("greeting", task=producer),
            TaskNode("agent", ("greeting",), task=task,
                     input=AgentTaskInput(parameters={"name": "Ada"}, session_id=session_id)),
        )),
            principal=principal, idempotency_key="source")
        request = CaptureInputRequest(principal, "same-input")
        try:
            await asyncio.wait_for(entered.wait(), 30)
            execution = await run.execution("agent")
            assert (await asyncio.wait_for(execution.wait(), 30)).status is ExecutionStatus.SUCCEEDED
            if invocation_written:
                early = await runtime.executions.capture_input(execution.execution_id, request)
            else:
                for _ in range(2):
                    with pytest.raises(AIError) as pending:
                        await runtime.executions.capture_input(execution.execution_id, request)
                    assert pending.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
        finally:
            release.set()
        assert (await run.wait(timeout_seconds=30)).status is TaskStatus.SUCCEEDED
        monkeypatch.setattr(objects, "put", original_put)
        capture = await runtime.executions.capture_input(execution.execution_id, request)
        assert await runtime.executions.capture_input(execution.execution_id, request) == capture
        if invocation_written:
            assert early == capture
        value = await runtime._input_captures.read_agent(capture, principal=principal)
        assert value.prompt == "Hello Ada"
        assert value.task_input.original_input["parameters"] == {"name": "Ada"}
        assert tuple(item.name for item in value.task_input.dependencies) == ("greeting",)
        for mode in ("fixed_input", "reproject_input"):
            captured = await runtime._input_captures.task_input(capture, principal=principal, input_mode=mode)
            assert await runtime._input_captures.read_dependency(captured, "greeting", principal=principal) == "Hello"
            replay = await engine.start(TaskGraph(mode, (TaskNode("agent", task=task, input_capture=captured),)),
                principal=principal, idempotency_key=mode)
            assert (await replay.wait(timeout_seconds=30)).status is TaskStatus.SUCCEEDED
            assert projections == [("Ada", "Hello")] * (1 if mode == "fixed_input" else 2)


@pytest.mark.asyncio
async def test_standalone_agent_capture_accepts_unrelated_execution_holds(tmp_path: Path) -> None:
    models = _ControlledModels()
    principal = service_principal("default", "owner")
    async with Runtime.open("standalone-capture", models=models, storage=RuntimeStorage.filesystem(tmp_path),
                            capabilities=(_agents(),)) as runtime:
        execution = await runtime.agents.get().start("standalone input", principal=principal)
        await asyncio.wait_for(models.entered.wait(), 30)
        holds = ("task-ref:dependency", "input-capture:reader")
        for hold_id in holds:
            await runtime._execution_service.acquire_dependency_hold(execution.execution_id,
                tenant_id=principal.tenant_id, hold_id=hold_id)
        try:
            models.release.set()
            assert (await asyncio.wait_for(execution.wait(), 30)).status is ExecutionStatus.SUCCEEDED
            capture = await runtime.executions.capture_input(execution.execution_id,
                CaptureInputRequest(principal, "standalone-input"))
            value = await runtime._input_captures.read_agent(capture, principal=principal)
            assert value.prompt == "standalone input"
            assert value.task_input is None
        finally:
            for hold_id in holds:
                await runtime._execution_service.release_dependency_hold(execution.execution_id,
                    tenant_id=principal.tenant_id, hold_id=hold_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_execution", [False, True])
async def test_invocation_capture_requirement_is_durable_and_idempotent(
    tmp_path: Path, agent_execution: bool,
) -> None:
    from dataclasses import replace

    from linktools.ai.runtime import ExecutionRequest
    from linktools.ai.runtime.service_api import ExecutionHandle
    from linktools.ai.runtime.state._codec import decode_domain, encode_domain
    from linktools.ai.runtime.state._contracts import ExecutionRecord
    from linktools.ai.task import TaskBindingContract

    models = _ControlledModels()
    models.release.set()
    principal = service_principal("default", "owner")
    execution_id = None
    for reopened in (False, True):
        storage = RuntimeStorage.filesystem(tmp_path)
        async with Runtime.open("capture-contract", models=models, storage=storage, capabilities=(_agents(),)) as runtime:
            agent = runtime.agents.get()
            binding = runtime._compiler.bind(runtime._compiled_agent(agent.id, agent.revision, agent.compiled))
            request = ExecutionRequest("source", principal, "source", None, "run", False, False)
            task_binding = TaskBindingContract("capture.native", 1, "none", {"kind": "json"}, None, 1, 0)

            async def start(required: bool) -> "ExecutionHandle":
                if agent_execution:
                    return await runtime._execution_service.start(binding.binding_digest, request,
                        binding_contract=binding.binding_contract, requires_task_invocation_capture=required)
                return await runtime._execution_service.start_task(task_binding, principal=principal,
                    input={"source": True}, idempotency_key="source", correlation={},
                    requires_task_invocation_capture=required)

            execution = await start(True)
            assert await start(True) == execution
            if agent_execution:
                assert await runtime._execution_service.resolve_existing(binding.binding_digest, request,
                    binding_contract=binding.binding_contract, requires_task_invocation_capture=True) == execution
                with pytest.raises(AIError) as changed_resolution:
                    await runtime._execution_service.resolve_existing(binding.binding_digest, request,
                        binding_contract=binding.binding_contract)
                assert changed_resolution.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
            if reopened:
                assert execution.execution_id == execution_id
            execution_id = execution.execution_id
            with pytest.raises(AIError) as changed:
                await start(False)
            assert changed.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
            if agent_execution:
                assert (await runtime.executions.wait(execution_id, principal=principal,
                    timeout_seconds=30)).status is ExecutionStatus.SUCCEEDED
            elif not reopened:
                await runtime._execution_service.complete_task(execution_id, principal=principal, output="completed")
            record = await storage.execution.executions.get(execution_id, tenant_id=principal.tenant_id)
            assert record.requires_task_invocation_capture is True
            assert record.dependency_hold_ids == ()
            assert decode_domain(encode_domain(record), ExecutionRecord) == record
            independent = replace(record, requires_task_invocation_capture=False)
            assert decode_domain(encode_domain(independent), ExecutionRecord) == independent
            malformed = encode_domain(record)
            malformed["fields"]["requires_task_invocation_capture"] = "true"
            with pytest.raises(AIError) as invalid:
                decode_domain(malformed, ExecutionRecord)
            assert invalid.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            with pytest.raises(AIError) as unavailable:
                await runtime.executions.capture_input(execution_id, CaptureInputRequest(principal, "input"))
            assert unavailable.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
    assert models.requests == (1 if agent_execution else 0)


@pytest.mark.asyncio
async def test_agent_task_retry_and_fork_capture_their_independent_input(tmp_path: Path) -> None:
    models = _ControlledModels()
    models.release.set()
    principal = service_principal("default", "owner")
    storage = RuntimeStorage.filesystem(tmp_path)
    async with Runtime.open("capture-lineage", models=models, storage=storage, capabilities=(_agents(),)) as runtime:
        task = runtime.tasks.from_agent("capture.agent", runtime.agents.get())
        source = await runtime.tasks.bind(task).start(TaskGraph("source", (
            TaskNode("agent", task=task, input=AgentTaskInput("source input")),
        )), principal=principal, idempotency_key="source")
        assert (await source.wait(timeout_seconds=30)).status is TaskStatus.SUCCEEDED
        execution = await source.execution("agent")
        for name, start in (("retry", execution.retry), ("fork", execution.fork)):
            independent = await start(name + " input", idempotency_key=name)
            assert (await asyncio.wait_for(independent.wait(), 30)).status is ExecutionStatus.SUCCEEDED
            captured = await runtime.executions.capture_input(independent.execution_id,
                CaptureInputRequest(principal, name, "clean"))
            value = await runtime._input_captures.read_agent(captured, principal=principal)
            assert value.prompt == name + " input"
            assert value.task_input is None
