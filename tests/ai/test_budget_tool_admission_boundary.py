#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared usage and tool dispatch become one confirmed admission fact."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest
import pytest_asyncio
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.tools import ToolDefinition
from pydantic_ai_harness.step_persistence import StepEvent, StepPersistence
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.capability import AgentContext
from linktools.ai.core import ExecutionStatus, RunBudget, ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionRecoveryDetail
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import Persistence, PersistenceStorage, SqlStepStore
from linktools.ai.persistence._runtime_commands import StateCommands
from linktools.ai.persistence.contracts import ToolBoundaryReceipt
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._tool import RuntimeToolOperationBridge
from linktools.ai.storage import InMemoryObjectStore, PayloadPolicy, StoredPayload

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features, tool_run_context
from .test_history_projection_conformance import _record
from .test_tool_execution_boundary import _request


@pytest_asyncio.fixture(params=("memory", "local_sqlite", "sqlite"))
async def admission_state(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Persistence]:
    if request.param == "memory":
        storage = PersistenceStorage.in_memory()
    elif request.param == "local_sqlite":
        storage = PersistenceStorage.from_root(tmp_path / "state", exclusive_writer=True)
    else:
        storage = PersistenceStorage.sqlite(tmp_path / "state.db")
    async with Persistence.open("budget-admission", tenant_id="tenant", storage=storage) as state:
        yield state


def _commands(state: Persistence) -> StateCommands:
    return StateCommands(
        state.execution.executions, namespace="budget-admission", events=state.execution.events,
        checkpoint=state.checkpoint, tools=state.checkpoint.tools, budgets=state.execution.budgets,
        background_tasks=set(),
    )


async def _execution(state: Persistence, *, name: str = "execution", limit: int = 1):
    await state.execution.budgets.ensure("shared", RunBudget(tool_calls=limit))
    record = replace(
        _record(ExecutionStatus.STARTED, 1), execution_id=name, root_execution_id=name,
        producer_generation=1, producer_claim_id="worker", budget_scope_id="shared",
        recovery=ExecutionRecoveryDetail(agent_run_id="run"),
    )
    await state.execution.executions.create_execution(record)
    return await state.execution.executions.get_observed(name, tenant_id="tenant")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("budget", "native"))
async def test_rejected_boundary_rolls_back_usage_control_effect_and_native(
    admission_state: Persistence, failure: str,
) -> None:
    state = admission_state
    observed = await _execution(state, limit=0 if failure == "budget" else 1)
    native_writes = []

    async def write(group):
        native_writes.append("started")
        usage = await state.execution.budgets.read_in_transaction(
            group.transaction(state.execution.budgets.state_store), "shared",
        )
        effect = await state.checkpoint.tools.get_operation_in_transaction(
            group.transaction(state.checkpoint.tools.state_store), "operation", tenant_id="tenant",
        )
        assert usage.tool_calls == 1 and effect.status is ToolOperationStatus.CLAIMED
        raise RuntimeError("native admission failed")

    with pytest.raises((AIError, RuntimeError)) as rejected:
        await _commands(state).commit_tool_admission_boundary(
            _request(), execution=observed, known_new=True, write=write,
        )
    if failure == "budget":
        assert isinstance(rejected.value, AIError)
        assert rejected.value.code is ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED
    elif isinstance(rejected.value, AIError):
        assert rejected.value.code is ErrorCode.INTERNAL_ERROR
        assert isinstance(rejected.value.__cause__, RuntimeError)
    assert native_writes == ([] if failure == "budget" else ["started"])
    assert await state.execution.executions.get_observed("execution", tenant_id="tenant") == observed
    assert await state.checkpoint.tools.get_operation("operation", tenant_id="tenant") is None
    assert (await state.execution.budgets.read("shared")).tool_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("split", (False, True))
async def test_lost_ack_confirms_same_budget_and_effect_without_second_charge(
    admission_state: Persistence, monkeypatch: pytest.MonkeyPatch, split: bool,
) -> None:
    state = admission_state
    observed = await _execution(state)
    commands = _commands(state)
    group = state.execution.executions.state_store.storage_group
    mutate = group.mutate
    lost = False

    async def lose_ack(stores, callback):
        nonlocal lost
        result = await mutate(stores, callback)
        if not lost:
            lost = True
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        return result

    monkeypatch.setattr(group, "mutate", lose_ack)
    if split:
        effect = await commands.commit_tool_admission(_request())
    else:
        receipt = await commands.commit_tool_admission_boundary(_request(), execution=observed, known_new=True)
        effect = receipt.operation
        assert receipt.budget.tool_calls == 1
    assert effect.value.status is ToolOperationStatus.CLAIMED
    assert (await state.execution.budgets.read("shared")).tool_calls == 1
    current = await state.execution.executions.get_observed("execution", tenant_id="tenant")
    with pytest.raises(AIError) as repeated:
        if split:
            await commands.commit_tool_admission(_request(), observed=effect)
        else:
            await commands.commit_tool_admission_boundary(_request(), execution=current, observed=effect)
    assert repeated.value.safe_details["reason"] == "duplicate_dispatch"
    assert (await state.execution.budgets.read("shared")).tool_calls == 1


@pytest.mark.asyncio
async def test_parallel_executions_share_atomic_dispatch_limit(admission_state: Persistence) -> None:
    state = admission_state
    commands = _commands(state)
    observations = [await _execution(state, name=f"execution-{index}", limit=3) for index in range(8)]
    requests = [replace(_request(), execution_id=f"execution-{index}",
                        tool_operation_id=f"operation-{index}", tool_call_id=f"call-{index}")
                for index in range(8)]
    results = await asyncio.gather(*(
        commands.commit_tool_admission_boundary(request, execution=observed, known_new=True)
        for request, observed in zip(requests, observations, strict=True)
    ), return_exceptions=True)
    assert sum(isinstance(result, ToolBoundaryReceipt) for result in results) == 3
    assert all(isinstance(result, ToolBoundaryReceipt) or (
        isinstance(result, AIError) and result.code is ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED
    ) for result in results)
    assert (await state.execution.budgets.read("shared")).tool_calls == 3
    for request, before, result in zip(requests, observations, results, strict=True):
        effect = await state.checkpoint.tools.get_operation(request.tool_operation_id, tenant_id="tenant")
        after = await state.execution.executions.get_observed(request.execution_id, tenant_id="tenant")
        if isinstance(result, ToolBoundaryReceipt):
            assert effect.status is ToolOperationStatus.CLAIMED
            assert after.storage_revision == before.storage_revision + 1
        else:
            assert effect is None and after == before


@pytest.mark.asyncio
async def test_cancelled_admission_resolves_commit_before_propagating(
    admission_state: Persistence, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = admission_state
    observed = await _execution(state)
    group = state.execution.executions.state_store.storage_group
    mutate = group.mutate
    committed, release = asyncio.Event(), asyncio.Event()

    async def pause_ack(stores, callback):
        result = await mutate(stores, callback)
        committed.set()
        await release.wait()
        return result

    monkeypatch.setattr(group, "mutate", pause_ack)
    pending = asyncio.create_task(_commands(state).commit_tool_admission_boundary(
        _request(), execution=observed, known_new=True,
    ))
    try:
        await asyncio.wait_for(committed.wait(), timeout=2)
        pending.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, timeout=2)
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
    assert (await state.execution.budgets.read("shared")).tool_calls == 1
    assert (await state.checkpoint.tools.get_operation("operation", tenant_id="tenant")).status is ToolOperationStatus.CLAIMED
    current = await state.execution.executions.get_observed("execution", tenant_id="tenant")
    assert current.storage_revision == observed.storage_revision + 1


@pytest.mark.asyncio
async def test_new_fence_counts_once_and_cached_terminal_result_spends_nothing(admission_state: Persistence) -> None:
    state = admission_state
    commands = _commands(state)
    observed = await _execution(state, limit=2)
    first = await commands.commit_tool_admission_boundary(_request(), execution=observed, known_new=True)
    await state.checkpoint.tools.defer("operation", tenant_id="tenant", owner="worker", fence=1)
    second = await commands.commit_tool_admission_boundary(_request(), execution=first.execution)
    assert second.operation.value.fence == 2 and second.budget.tool_calls == 2
    result = await commands.commit_tool_terminal_boundary(
        "operation", tenant_id="tenant", owner="worker", fence=2, execution=second.execution,
        observed=second.operation, result_payload=StoredPayload.inline_text("certain"),
    )
    cached = await commands.commit_tool_admission_boundary(
        _request(), execution=second.execution, observed=result.operation,
    )
    assert cached.operation == result.operation
    assert (await state.execution.budgets.read("shared")).tool_calls == 2
    with pytest.raises(AIError) as stale:
        await commands.commit_tool_terminal_boundary(
            "operation", tenant_id="tenant", owner="worker", fence=1, execution=first.execution,
            observed=first.operation, result_payload=StoredPayload.inline_text("late"),
        )
    assert stale.value.code is ErrorCode.TOOL_OPERATION_CONFLICT


@pytest.mark.asyncio
@pytest.mark.parametrize("split", (False, True))
@pytest.mark.parametrize("limit", (0, 1))
async def test_real_native_leaf_observes_budget_and_effect_together(
    tmp_path: Path, split: bool, limit: int,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'runtime.db'}")
    native_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'native.db'}") if split else engine
    await provision_database(engine)
    if split:
        await provision_database(native_engine)
    native = SqlStepStore(native_engine, namespace="budget-native", tenant_id="default")
    effects = []
    state = None

    async def perform(ctx: AgentContext[None]) -> str:
        record = await state.execution.executions.get(ctx.execution_id, tenant_id="default")
        assert (await state.execution.budgets.read(record.budget_scope_id)).tool_calls == 1
        operations = await state.checkpoint.tools.list_by_execution(ctx.execution_id, tenant_id="default")
        assert len(operations) == 1 and operations[0].status is ToolOperationStatus.CLAIMED
        effects.append("effect")
        return "certain"

    async def provider(messages, info):
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("perform", {}, tool_call_id="call")])

    features = RuntimeFeatures("budget-native")
    features.capability(StepPersistence(store=native, capture_frontier=True), scope="runtime")
    features.tool(perform, effect_policy="non_replay_safe")
    features.agent("default", model="default", allow_tools=("perform",))
    try:
        async with Persistence.open("budget-native", storage=PersistenceStorage.sql(engine)) as persistence:
            state = persistence
            async with Runtime.attach(persistence=state,
                features=model_features(_Models(_UsageFunctionModel(provider)), features)) as runtime:
                execution = await runtime.agents.get().start("work", budget=RunBudget(tool_calls=limit))
                result = (await execution.wait(timeout_seconds=5)).result
                admitted = bool(limit)
                assert result.status is (ExecutionStatus.SUCCEEDED if admitted else ExecutionStatus.FAILED)
                if not limit:
                    assert result.error_code == ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED.value
                assert (await execution.budget_usage()).tool_calls == int(admitted)
                operations = await state.checkpoint.tools.list_by_execution(execution.execution_id, tenant_id="default")
                assert len(operations) == int(admitted)
                if admitted:
                    assert operations[0].status is ToolOperationStatus.COMPLETED
            runs = await native.list_runs()
            events = [item.kind for item in await native.list_events(run_id=runs[0].run_id)
                      if item.kind.startswith("tool_call")]
            assert events == (["tool_call_started", "tool_call_completed"] if admitted else [])
        assert effects == (["effect"] if admitted else [])
    finally:
        await native.close()
        if split:
            await native_engine.dispose()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("rollback", "lost_ack"))
async def test_native_started_and_budget_share_the_exact_admission_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'atomic.db'}")
    await provision_database(engine)
    native = SqlStepStore(engine, namespace="budget-admission", tenant_id="tenant")
    event = StepEvent(run_id="run", kind="tool_call_started", step_index=1,
                      tool_call_id="call", tool_name="tool", idempotency_key="tool:operation:1:started")
    try:
        async with Persistence.open("budget-admission", tenant_id="tenant",
                                    storage=PersistenceStorage.sql(engine)) as state:
            observed = await _execution(state)
            prepared = await native.prepare_step(events=(event,))
            store = state.execution.executions.state_store
            mutate = store.storage_group.mutate

            async def write(group):
                await native.write_prepared_step(group.transaction(store), prepared)
                if outcome == "rollback":
                    raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

            async def readback():
                return event in await native.list_events(run_id="run")

            async def lose_ack(stores, callback):
                await mutate(stores, callback)
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

            if outcome == "lost_ack":
                monkeypatch.setattr(store.storage_group, "mutate", lose_ack)
            operation = _commands(state).commit_tool_admission_boundary(
                _request(), execution=observed, known_new=True, write=write, readback_write=readback,
            )
            if outcome == "rollback":
                with pytest.raises(AIError) as rejected:
                    await operation
                assert rejected.value.code is ErrorCode.STORAGE_UNAVAILABLE
                assert await native.list_events(run_id="run") == []
                assert await state.checkpoint.tools.get_operation("operation", tenant_id="tenant") is None
                assert await state.execution.executions.get_observed("execution", tenant_id="tenant") == observed
                assert (await state.execution.budgets.read("shared")).tool_calls == 0
            else:
                receipt = await operation
                assert receipt.budget.tool_calls == 1
                assert receipt.operation.value.status is ToolOperationStatus.CLAIMED
                assert await native.list_events(run_id="run") == [event]
                assert (await state.execution.budgets.read("shared")).tool_calls == 1
    finally:
        await native.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_budgeted_bridge_cannot_reuse_a_claimed_dispatch_receipt(admission_state: Persistence) -> None:
    state = admission_state
    await _execution(state, limit=2)
    bridge = RuntimeToolOperationBridge(
        state.checkpoint.tools, InMemoryObjectStore(), namespace="budget-admission", tenant_id="tenant",
        execution_id="execution", agent_run_id="run", binding_digest="b" * 64, owner="worker",
        background_tasks=set(), payload_policy=PayloadPolicy(), terminal_commands=_commands(state),
        producer_generation=1, budget_scope_id="shared",
    )
    ctx, call, definition = tool_run_context(), ToolCallPart("work", {}, tool_call_id="call"), ToolDefinition(name="work")
    decision = await bridge.begin(ctx, call, definition, {}, True)
    assert decision.budget_admitted and (await state.execution.budgets.read("shared")).tool_calls == 1
    with pytest.raises(AIError) as duplicate:
        await bridge.begin(ctx, call, definition, {}, True)
    assert duplicate.value.safe_details["reason"] == "duplicate_dispatch"
    await bridge.defer(decision)
    successor = await bridge.begin(ctx, call, definition, {}, True)
    assert successor.fence == 2 and (await state.execution.budgets.read("shared")).tool_calls == 2
    await bridge.complete(successor, "certain")
    cached = await bridge.begin(ctx, call, definition, {}, True)
    assert cached.has_cached_result and cached.cached_result == "certain"
    assert (await state.execution.budgets.read("shared")).tool_calls == 2
