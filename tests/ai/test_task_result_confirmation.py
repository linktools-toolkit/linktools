#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task confirmation failures preserve durable producer outcomes."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from linktools.ai.core import ExecutionStatus, TaskStatus, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.task import Task, TaskBindingContract, TaskGraph, TaskNode, TaskNodeRunResult
from linktools.ai.task.engine import TaskEngine

if TYPE_CHECKING:
    from linktools.ai.core import JsonValue
    from linktools.ai.task import TaskEffectResolution, TaskNodeInvocation, TaskNodeRunControl

OUTPUT = {"data": "x" * 80000}
CONTRACT = {"version": 1, "type": "runner", "effect_policy": "none", "output_contract": {"kind": "json"}, "reconcile": False, "config": {}}


class _OutputFault:
    def __init__(self, repository, service, patch, *, code=ErrorCode.STORAGE_UNAVAILABLE, precommit=False):
        self.armed = False
        self.injected = False
        self.error = AIError(code, safe_details={"reader": "output"})
        commit, read = repository.commit_terminal, service._port.result_record

        async def commit_once(terminal):
            if self.armed and precommit and terminal.execution.status is ExecutionStatus.SUCCEEDED:
                self.armed = False
                self.injected = True
                raise self.error
            return await commit(terminal)

        async def read_once(execution, *, tenant_id):
            if self.armed and not precommit and execution.status is ExecutionStatus.SUCCEEDED:
                self.armed = False
                self.injected = True
                raise self.error
            return await read(execution, tenant_id=tenant_id)

        patch.setattr(repository, "commit_terminal", commit_once)
        patch.setattr(service._port, "result_record", read_once)


class _CompletingRunner:
    def __init__(self, service, body, *, claim="success", after_complete=None):
        self.service, self.body = service, body
        self.claim, self.after_complete = claim, after_complete
        self.entries = 0
        self.body_calls = 0
        self.resumed = []

    async def run(self, invocation: "TaskNodeInvocation", *, control: "TaskNodeRunControl") -> TaskNodeRunResult:
        self.entries += 1
        if invocation.execution_id is not None:
            self.resumed.append(invocation.execution_id)
            return TaskNodeRunResult(canonical_sha256(OUTPUT), invocation.execution_id)
        if self.claim == "missing":
            return TaskNodeRunResult(canonical_sha256(OUTPUT), "missing")
        handle = await self.service.start_task(
            TaskBindingContract("producer", 1, "none", {"kind": "json"}, None, 1, 0),
            principal=invocation.principal, input={}, idempotency_key="producer", correlation={},
        )
        await control.bind_execution(handle.execution_id)
        if self.claim == "started":
            return TaskNodeRunResult(canonical_sha256(OUTPUT), handle.execution_id)
        if self.claim == "failed":
            await self.service.fail_task(handle.execution_id, principal=invocation.principal, error=AIError(ErrorCode.TASK_NODE_FAILED))
        else:
            self.body_calls += 1
            output = await self.body()
            await self.service.complete_task(handle.execution_id, principal=invocation.principal, output=output)
        if self.after_complete is not None:
            self.after_complete()
        return TaskNodeRunResult("0" * 64 if self.claim == "digest" else canonical_sha256(OUTPUT), handle.execution_id)

    async def wait_bound(self, invocation: "TaskNodeInvocation", execution_id: str) -> TaskNodeRunResult:
        return TaskNodeRunResult(canonical_sha256(OUTPUT), execution_id)

    async def supply_input(self, invocation: "TaskNodeInvocation", execution_id: str, value: "JsonValue") -> TaskNodeRunResult:
        raise AIError(ErrorCode.TASK_NOT_READY)

    async def resolve_effect(self, invocation: "TaskNodeInvocation", execution_id: str, resolution: "TaskEffectResolution") -> "TaskNodeRunResult | None":
        raise AIError(ErrorCode.TASK_NOT_READY)

    async def cancel(self, invocation: "TaskNodeInvocation") -> None:
        if invocation.execution_id is not None:
            await self.service.cancel_task(invocation.execution_id, principal=invocation.principal)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("kind", ("function", "runner"))
async def test_task_committed_output_read_recovers_same_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, kind: str) -> None:
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "state.sqlite")
    async with Persistence.open("confirmation", storage=storage) as state, TaskEngine.open(persistence=state) as engine:
        fault = _OutputFault(state.execution.executions, engine._execution_service, monkeypatch)
        calls = 0

        async def body():
            nonlocal calls
            calls += 1
            fault.armed = True
            return OUTPUT

        async def function(ctx):
            return await body()

        runner = _CompletingRunner(engine._execution_service, body)
        task = Task("producer", function, effect_policy="none") if kind == "function" else Task.from_runner("producer", runner, contract=CONTRACT)
        engine = engine.with_definitions(task)
        run = await engine.start(TaskGraph("graph", (TaskNode("node", task=task),)), idempotency_key="start")
        pending = (await run.wait(timeout_seconds=5)).result
        assert fault.injected
        assert pending.status is TaskStatus.RECOVERY_REQUIRED
        bound = pending.node_states[0]
        assert bound.execution_id is not None
        assert bound.safe_error_details["cause_details"] == fault.error.safe_details
        original = await state.execution.executions.get(bound.execution_id, tenant_id=state.tenant_id)
        assert original.status is ExecutionStatus.SUCCEEDED
        assert original.result.output.kind == "inline"
        assert original.result.output.decode() == OUTPUT
        await run.recover(idempotency_key="recover")
        completed = (await run.wait(timeout_seconds=5)).result
        assert completed.status is TaskStatus.SUCCEEDED
        assert completed.node_states[0].execution_id == bound.execution_id
        assert await run.result("node") == OUTPUT
        assert (await state.execution.executions.get(bound.execution_id, tenant_id=state.tenant_id)).result == original.result
        assert calls == 1
        if kind == "runner":
            assert runner.entries == 2
            assert runner.body_calls == 1
            assert runner.resumed == [bound.execution_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("function", "runner"))
@pytest.mark.parametrize("phase", ("body", "precommit"))
async def test_task_unavailable_before_terminal_confirmation_is_not_reclassified(monkeypatch: pytest.MonkeyPatch, kind: str, phase: str) -> None:
    async with Persistence.open("negative", storage=PersistenceStorage.in_memory()) as state, TaskEngine.open(persistence=state) as engine:
        fault = _OutputFault(state.execution.executions, engine._execution_service, monkeypatch, precommit=True)

        async def body():
            if phase == "body":
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE, safe_details={"business": "unavailable"})
            fault.armed = True
            return OUTPUT

        async def function(ctx):
            return await body()

        runner = _CompletingRunner(engine._execution_service, body)
        task = Task("producer", function, effect_policy="none") if kind == "function" else Task.from_runner("producer", runner, contract=CONTRACT)
        run = await engine.with_definitions(task).start(TaskGraph("graph", (TaskNode("node", task=task),)), idempotency_key="start")
        failed = (await run.wait(timeout_seconds=5)).result
        assert failed.status is TaskStatus.FAILED
        assert failed.node_states[0].error_code == ErrorCode.STORAGE_UNAVAILABLE.value
        assert failed.node_states[0].safe_error_details.get("phase") != "task_terminal_confirmation"
        producer = await state.execution.executions.get(failed.node_states[0].execution_id, tenant_id=state.tenant_id)
        assert producer.status is (ExecutionStatus.FAILED if kind == "function" and phase == "body" else ExecutionStatus.STARTED)


@pytest.mark.asyncio
async def test_failed_execution_confirmation_recovers_to_original_business_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def body(ctx):
        nonlocal calls
        calls += 1
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"business": "rejected"})

    task = Task("producer", body, effect_policy="none")
    async with Persistence.open("failed", storage=PersistenceStorage.in_memory()) as state, TaskEngine.open(task, persistence=state) as engine:
        original_read = engine._execution_service.result
        injected = False

        async def read_once(execution_id, *, principal):
            nonlocal injected
            current = await state.execution.executions.get(execution_id, tenant_id=state.tenant_id)
            if not injected and current.status is ExecutionStatus.FAILED:
                injected = True
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE, safe_details={"reader": "failed_result"})
            return await original_read(execution_id, principal=principal)

        monkeypatch.setattr(engine._execution_service, "result", read_once)
        run = await engine.start(TaskGraph("graph", (TaskNode("node", task=task),)), idempotency_key="start")
        pending = (await run.wait(timeout_seconds=5)).result
        assert injected and pending.status is TaskStatus.RECOVERY_REQUIRED
        execution_id = pending.node_states[0].execution_id
        original = await state.execution.executions.get(execution_id, tenant_id=state.tenant_id)
        assert original.status is ExecutionStatus.FAILED
        await run.recover(idempotency_key="recover")
        failed = (await run.wait(timeout_seconds=5)).result
        assert failed.status is TaskStatus.FAILED
        assert failed.node_states[0].execution_id == execution_id
        assert failed.node_states[0].error_code == ErrorCode.REQUEST_FIELD_INVALID.value
        assert (await state.execution.executions.get(execution_id, tenant_id=state.tenant_id)).result == original.result
        assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("code", (ErrorCode.STORAGE_NOT_FOUND, ErrorCode.STORAGE_INTEGRITY_ERROR))
async def test_task_determinate_result_fault_does_not_become_recovery(monkeypatch: pytest.MonkeyPatch, code: ErrorCode) -> None:
    async with Persistence.open("determinate", storage=PersistenceStorage.in_memory()) as state, TaskEngine.open(persistence=state) as engine:
        fault = _OutputFault(state.execution.executions, engine._execution_service, monkeypatch, code=code)

        async def body():
            fault.armed = True
            return OUTPUT

        runner = _CompletingRunner(engine._execution_service, body)
        task = Task.from_runner("producer", runner, contract=CONTRACT)
        run = await engine.with_definitions(task).start(TaskGraph("graph", (TaskNode("node", task=task),)), idempotency_key="start")
        failed = (await run.wait(timeout_seconds=5)).result
        assert fault.injected and failed.status is TaskStatus.FAILED
        assert failed.node_states[0].error_code == code.value
        assert runner.body_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("claim", ("missing", "started", "failed", "digest"))
async def test_runner_invalid_success_claim_remains_integrity_failure(claim: str) -> None:
    async with Persistence.open("claim", storage=PersistenceStorage.in_memory()) as state, TaskEngine.open(persistence=state) as engine:
        async def body():
            return OUTPUT
        runner = _CompletingRunner(engine._execution_service, body, claim=claim)
        task = Task.from_runner("producer", runner, contract=CONTRACT)
        run = await engine.with_definitions(task).start(TaskGraph("graph", (TaskNode("node", task=task),)), idempotency_key="start")
        failed = (await run.wait(timeout_seconds=5)).result
        assert failed.status is TaskStatus.FAILED
        assert failed.node_states[0].error_code == ErrorCode.STORAGE_INTEGRITY_ERROR.value


@pytest.mark.asyncio
@pytest.mark.parametrize("code", (ErrorCode.STORAGE_COMMIT_UNKNOWN, ErrorCode.STORAGE_RECOVERY_REQUIRED, ErrorCode.STORAGE_UNAVAILABLE))
async def test_runner_success_confirmation_preserves_unknown_outcome(monkeypatch: pytest.MonkeyPatch, code: ErrorCode) -> None:
    async with Persistence.open("claim-read", storage=PersistenceStorage.in_memory()) as state, TaskEngine.open(persistence=state) as engine:
        async def body():
            return OUTPUT
        armed = False
        fault = AIError(code, safe_details={"reader": "claimed_success"})
        read = engine._execution_reads.result
        async def read_once(*args, **kwargs):
            nonlocal armed
            if armed:
                armed = False
                raise fault
            return await read(*args, **kwargs)
        def arm():
            nonlocal armed
            armed = True
        monkeypatch.setattr(engine._execution_reads, "result", read_once)
        runner = _CompletingRunner(engine._execution_service, body, after_complete=arm)
        task = Task.from_runner("producer", runner, contract=CONTRACT)
        run = await engine.with_definitions(task).start(TaskGraph("graph", (TaskNode("node", task=task),)), idempotency_key="start")
        pending = (await run.wait(timeout_seconds=5)).result
        assert pending.status is TaskStatus.RECOVERY_REQUIRED
        bound = pending.node_states[0]
        assert bound.error_code == (ErrorCode.STORAGE_RECOVERY_REQUIRED.value if code is ErrorCode.STORAGE_UNAVAILABLE else code.value)
        original = await state.execution.executions.get(bound.execution_id, tenant_id=state.tenant_id)
        assert original.status is ExecutionStatus.SUCCEEDED
        await run.recover(idempotency_key="recover")
        assert (await run.wait(timeout_seconds=5)).result.status is TaskStatus.SUCCEEDED
        assert await run.result("node") == OUTPUT
        assert (await state.execution.executions.get(bound.execution_id, tenant_id=state.tenant_id)).result == original.result
        assert runner.entries == 2 and runner.body_calls == 1
        assert runner.resumed == [bound.execution_id]


@pytest.mark.asyncio
async def test_runner_result_confirmation_preserves_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    from linktools.ai.task import TaskNodeInvocation
    async with Persistence.open("cancel", storage=PersistenceStorage.in_memory()) as state, TaskEngine.open(persistence=state) as engine:
        async def cancelled(*args, **kwargs):
            raise asyncio.CancelledError
        monkeypatch.setattr(engine._execution_reads, "result", cancelled)
        async def body(ctx):
            return OUTPUT
        task = Task("producer", body, effect_policy="none")
        invocation = TaskNodeInvocation(TaskNode("node", task=task), "graph", engine._default_principal, {}, {})
        with pytest.raises(asyncio.CancelledError):
            await engine._task_node_runtime._complete_runner_result(invocation, TaskNodeRunResult(canonical_sha256(OUTPUT), "execution"))


@pytest.mark.asyncio
async def test_agent_result_wait_recovers_without_repeating_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_ai.models.function import FunctionModel
    from linktools.ai.inputs import AgentTaskInput
    from linktools.ai.model import create_model
    from linktools.ai.runtime import Runtime, RuntimeFeatures
    calls = 0
    async def respond(messages, info) -> AsyncIterator[str]:
        nonlocal calls
        calls += 1
        yield "x" * 80000
    model = FunctionModel(stream_function=respond)
    declaration = create_model(api="openai", model="offline-confirmation", input_types=("mime:text/plain",))
    class Binding:
        route_id = "default"
        provider = declaration.provider
        model_identity = declaration.model_identity
        model_digest = "a" * 64
        contract = dict(declaration.contract)
        def materialize(self) -> FunctionModel:
            return model
    features = RuntimeFeatures()
    features.model("default", Binding())
    features.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    async with Persistence.open("agent-confirmation", storage=PersistenceStorage.in_memory()) as state:
        captured = []
        async with Runtime.attach(persistence=state, features=features) as runtime:
            port = runtime._execution_service._port
            read = port.result_record

            async def read_once(producer, *, tenant_id):
                if not captured and producer.status is ExecutionStatus.SUCCEEDED:
                    graph = await state.task.tasks.graph_state("graph", tenant_id=state.tenant_id)
                    node = None if graph is None else graph.node_states[0]
                    if node is not None and node.execution_id == producer.execution_id:
                        captured.append(producer)
                        raise AIError(ErrorCode.STORAGE_UNAVAILABLE, safe_details={"reader": "agent_output"})
                return await read(producer, tenant_id=tenant_id)

            monkeypatch.setattr(port, "result_record", read_once)
            task = Task.from_agent("producer", runtime.agents.get("default"))
            async with TaskEngine.open(task, persistence=state) as engine:
                run = await engine.start(TaskGraph("graph", (TaskNode("node", task=task, input=AgentTaskInput("respond")),)), principal=runtime.default_principal, idempotency_key="start")
                pending = (await run.wait(timeout_seconds=5)).result
                assert captured and pending.status is TaskStatus.RECOVERY_REQUIRED
                assert pending.node_states[0].execution_id == captured[0].execution_id
                assert pending.node_states[0].safe_error_details["cause_details"] == {"reader": "agent_output"}
                await run.recover(idempotency_key="recover")
                completed = (await run.wait(timeout_seconds=5)).result
                assert completed.status is TaskStatus.SUCCEEDED
                assert completed.node_states[0].execution_id == captured[0].execution_id
                assert await run.result("node") == {"text": "x" * 80000}
                assert calls == 1
                assert (await state.execution.executions.get(captured[0].execution_id, tenant_id=state.tenant_id)).result == captured[0].result
