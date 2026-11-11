#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Terminal producer facts survive temporarily unavailable confirmation reads."""

from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus, OperationStatus, Principal, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.storage import FilesystemObjectStore
from linktools.ai.task import (
    Task, TaskBindingContract, TaskEffectResolution, TaskGraph, TaskInputSupplyRequest, TaskNode,
)
from linktools.ai.task._runner import TaskNodeRunError
from linktools.ai.task.engine import TaskEngine


def _assert_confirmation(error: AIError, cause: AIError, phase: str) -> None:
    assert error.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    assert error.__cause__ is cause
    assert error.safe_details["phase"] == phase
    assert error.safe_details["cause"] == ErrorCode.STORAGE_UNAVAILABLE.value
    assert error.safe_details["cause_details"] == cause.safe_details


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ("effect", "input"))
@pytest.mark.parametrize("fault_owner", ("service", "task"))
async def test_control_confirmation_retries_same_receipt_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control: str, fault_owner: str,
) -> None:
    store = FilesystemObjectStore(tmp_path / "objects")
    storage = PersistenceStorage.sqlite(tmp_path / "state.sqlite", object_store=store)
    principal = Principal("actor", "tenant")
    calls = 0
    operation_id = "控" * 85 + "r"

    async def body(ctx):
        nonlocal calls
        calls += 1
        raise RuntimeError("external outcome is unknown")

    task = Task("external", body, effect_policy="non_replay_safe")
    node = TaskNode("node", task=task) if control == "effect" else TaskNode.wait("node")
    async with Persistence.open("confirmation", storage=storage, tenant_id="tenant") as state, TaskEngine.open(task, persistence=state) as engine:
        run = await engine.start(TaskGraph("graph", (node,)), principal=principal, idempotency_key="start")
        initial = (await run.wait(timeout_seconds=3)).result
        bound = initial.node_states[0]
        expected_pending = TaskStatus.RECOVERY_REQUIRED if control == "effect" else TaskStatus.WAITING
        expected_graph = TaskStatus.RECOVERY_REQUIRED if control == "effect" else TaskStatus.RUNNING
        assert initial.status is expected_graph
        assert bound.status is expected_pending
        assert bound.execution_id is not None
        value = {"data": "x" * 80000}

        async def apply(value):
            if control == "effect":
                return await run.resolve_effect(
                    "node", bound.fence, TaskEffectResolution("applied", value), idempotency_key=operation_id,
                )
            return await run.resume("node", TaskInputSupplyRequest(principal, bound.execution_id, value, operation_id))

        owner = engine._execution_service if fault_owner == "service" else engine._execution_reads
        read = owner.result
        fault = AIError(ErrorCode.STORAGE_UNAVAILABLE, safe_details={"reader": fault_owner, "nested": {"attempt": 1}})
        pending = True

        async def read_once(*args, **kwargs):
            nonlocal pending
            if pending:
                pending = False
                raise fault
            return await read(*args, **kwargs)

        monkeypatch.setattr(owner, "result", read_once)
        with pytest.raises(AIError) as raised:
            await apply(value)
        error = raised.value
        assert isinstance(error, TaskNodeRunError) is (fault_owner == "task")
        cause = error.__cause__
        assert isinstance(cause, AIError)
        _assert_confirmation(error, cause, "task_terminal_confirmation" if fault_owner == "service" else "task_result_confirmation")
        receipt = await state.task.operations.get(operation_id, tenant_id="tenant")
        assert receipt.operation_id == operation_id
        assert receipt.status is OperationStatus.RUNNING
        graph = await run.state()
        assert graph.status is expected_graph
        assert graph.node_states[0].status is expected_pending
        assert graph.node_states[0].execution_id == bound.execution_id
        assert graph.node_states[0].fence == bound.fence
        producer = await state.execution.executions.get(bound.execution_id, tenant_id="tenant")
        assert producer.status is ExecutionStatus.SUCCEEDED
        original_result = producer.result

        assert (await apply(value)).status is TaskStatus.SUCCEEDED
        receipt = await state.task.operations.get(operation_id, tenant_id="tenant")
        assert receipt.status is OperationStatus.SUCCEEDED
        assert await run.result("node") == value
        producer = await state.execution.executions.get(bound.execution_id, tenant_id="tenant")
        assert producer.result == original_result
        assert (await run.state()).node_states[0].execution_id == bound.execution_id
        assert calls == (1 if control == "effect" else 0)
        with pytest.raises(AIError) as conflict:
            await apply({"data": "different"})
        assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("success", "failure"))
async def test_terminal_write_and_replay_preserve_result_and_original_failure(
    monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    principal = Principal("actor", "default")
    async with Persistence.open("terminal", storage=PersistenceStorage.in_memory()) as state, TaskEngine.open(persistence=state) as engine:
        service = engine._execution_service
        started = await service.start_task(
            TaskBindingContract("callable", 1, "none", {"kind": "json"}, None, 1, 0),
            principal=principal, input={}, idempotency_key="start", correlation={},
        )
        failure = AIError(ErrorCode.TASK_NODE_FAILED, safe_details={"business": "rejected"})

        async def settle(changed=False):
            if outcome == "success":
                return await service.complete_task(started.execution_id, principal=principal, output="other" if changed else "accepted")
            return await service.fail_task(
                started.execution_id, principal=principal,
                error=AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) if changed else failure,
            )

        read = service.result
        for _ in range(2):
            fault = AIError(ErrorCode.STORAGE_UNAVAILABLE, safe_details={"reader": "metadata", "nested": {"attempt": 1}})

            async def unavailable(*args, **kwargs):
                raise fault

            with monkeypatch.context() as patch:
                patch.setattr(service, "result", unavailable)
                with pytest.raises(AIError) as raised:
                    await settle()
            assert type(raised.value) is AIError
            _assert_confirmation(raised.value, fault, "task_terminal_confirmation")
            assert raised.value.safe_details["execution_id"] == started.execution_id
            result = await read(started.execution_id, principal=principal)
            assert result.status is (ExecutionStatus.SUCCEEDED if outcome == "success" else ExecutionStatus.FAILED)
            if outcome == "success":
                assert result.output == "accepted"
            else:
                assert result.error_code == failure.code.value
                assert result.safe_error_details == failure.safe_details
        assert (await settle()).status is result.status
        with pytest.raises(AIError) as conflict:
            await settle(changed=True)
        assert conflict.value.code is ErrorCode.STORAGE_CONFLICT


@pytest.mark.asyncio
async def test_supply_final_projection_is_confirmation_but_initial_load_and_public_read_are_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    principal = Principal("actor", "default")
    store = FilesystemObjectStore(tmp_path / "objects")
    storage = PersistenceStorage.sqlite(tmp_path / "state.sqlite", object_store=store)
    async with Persistence.open("projection", storage=storage) as state, TaskEngine.open(persistence=state) as engine:
        service = engine._execution_service
        started = await service.start_task(
            TaskBindingContract("callable", 1, "none", {"kind": "json"}, None, 1, 0),
            principal=principal, input={}, idempotency_key="start", correlation={},
        )
        await service.defer_task_input(started.execution_id, principal=principal, wait_id="wait")
        fault = AIError(ErrorCode.STORAGE_UNAVAILABLE, safe_details={"reader": "projection"})

        async def unavailable(*args, **kwargs):
            raise fault

        with monkeypatch.context() as patch:
            patch.setattr(service, "load_authorized", unavailable)
            with pytest.raises(AIError) as initial:
                await service.supply_task_input(started.execution_id, principal=principal, value="accepted")
        assert initial.value is fault
        assert (await service.inspect(started.execution_id, principal=principal)).status is ExecutionStatus.WAITING_DEFERRED
        with monkeypatch.context() as patch:
            patch.setattr(service, "_store_task_output", unavailable)
            with pytest.raises(AIError) as precommit:
                await service.supply_task_input(started.execution_id, principal=principal, value="x" * 80000)
        assert precommit.value is fault
        assert (await service.inspect(started.execution_id, principal=principal)).status is ExecutionStatus.WAITING_DEFERRED
        with monkeypatch.context() as patch:
            patch.setattr(service, "inspect", unavailable)
            with pytest.raises(AIError) as projection:
                await service.supply_task_input(started.execution_id, principal=principal, value="accepted")
        _assert_confirmation(projection.value, fault, "task_terminal_confirmation")
        assert (await service.supply_task_input(started.execution_id, principal=principal, value="accepted")).status is ExecutionStatus.SUCCEEDED
        with monkeypatch.context() as patch:
            patch.setattr(service, "load_authorized", unavailable)
            with pytest.raises(AIError) as public:
                await service.result(started.execution_id, principal=principal)
        assert public.value is fault


@pytest.mark.asyncio
async def test_supply_conflict_readback_preserves_unknown_terminal_state(monkeypatch: pytest.MonkeyPatch) -> None:
    principal = Principal("actor", "default")
    async with Persistence.open("conflict-confirmation", storage=PersistenceStorage.in_memory()) as state, TaskEngine.open(persistence=state) as engine:
        service = engine._execution_service
        started = await service.start_task(
            TaskBindingContract("callable", 1, "none", {"kind": "json"}, None, 1, 0),
            principal=principal, input={}, idempotency_key="start", correlation={},
        )
        await service.defer_task_input(started.execution_id, principal=principal, wait_id="wait")
        complete = service._complete_task_output
        fault = AIError(ErrorCode.STORAGE_UNAVAILABLE)

        async def concurrent_completion(*args, **kwargs):
            await complete(*args, **kwargs)
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        async def unavailable(*args, **kwargs):
            raise fault

        with monkeypatch.context() as patch:
            patch.setattr(service, "_complete_task_output", concurrent_completion)
            patch.setattr(service, "inspect", unavailable)
            with pytest.raises(AIError) as raised:
                await service.supply_task_input(started.execution_id, principal=principal, value="accepted")
        _assert_confirmation(raised.value, fault, "task_terminal_confirmation")
        assert (await service.supply_task_input(started.execution_id, principal=principal, value="accepted")).status is ExecutionStatus.SUCCEEDED
