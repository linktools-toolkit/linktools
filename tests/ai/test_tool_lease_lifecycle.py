#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool outcomes survive elapsed time, cancellation and producer takeover."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai.exceptions import CallDeferred
from pydantic_ai.toolsets import FunctionToolset

from linktools.ai.capability import ToolCallFailed, ToolCallRetry
from linktools.ai.core import RunBudget, ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._budget import RunBudgetContext
from linktools.ai.runtime._tool import RuntimeToolOperationBridge
from linktools.ai.runtime._tool_boundary import BoundaryToolset, ManagedToolDescriptor
from linktools.ai.persistence import Persistence, PersistenceStorage, _checkpoint_repositories, _memory_transaction
from linktools.ai.persistence.commands import StateCommands
from linktools.ai.persistence._memory_transaction import _MemoryTransaction
from linktools.ai.storage import InMemoryObjectStore, PayloadPolicy

from ._runtime_test_helpers import tool_run_context, tool_with_metadata
from .test_step_projection import _execution


class _Clock:
    def __init__(self) -> None:
        self.current = datetime(2026, 1, 1, tzinfo=timezone.utc)

    async def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)
        # Memory transactions complete without I/O; drain their scheduled continuations.
        for _ in range(10):
            await asyncio.sleep(0)


@asynccontextmanager
async def _runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[Persistence, RuntimeToolOperationBridge, _Clock]]:
    clock = _Clock()

    async def now(transaction: _MemoryTransaction) -> datetime:
        del transaction
        return clock.current

    monkeypatch.setattr(_MemoryTransaction, "now", now)
    monkeypatch.setattr(
        _memory_transaction, "datetime",
        SimpleNamespace(now=lambda tz: clock.current.astimezone(tz)),
    )
    monkeypatch.setattr(
        _checkpoint_repositories, "datetime",
        SimpleNamespace(now=lambda tz: clock.current.astimezone(tz)),
    )
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace="tool-lease", tenant_id="tenant")
    bridge = RuntimeToolOperationBridge(
        state.checkpoint.tools,
        InMemoryObjectStore(),
        namespace="tool-lease",
        tenant_id="tenant",
        execution_id="execution",
        agent_run_id="run",
        binding_digest="b" * 64,
        owner="owner",
        background_tasks=set(),
        payload_policy=PayloadPolicy(),
    )
    try:
        yield state, bridge, clock
    finally:
        await state.close()


async def _call(
    bridge: RuntimeToolOperationBridge,
    handler: Callable[[], Awaitable[str]],
    *,
    replay_safe: bool = False,
    budget: RunBudgetContext | None = None,
) -> object:
    descriptor = ManagedToolDescriptor(
        effect_owner="tool_operation",
        effect_policy="replay_safe" if replay_safe else "non_replay_safe",
        tool_class="business",
    )
    boundary = BoundaryToolset(
        (FunctionToolset([tool_with_metadata(handler, descriptor)]),),
        {handler.__name__: descriptor},
        id="test.lease",
        tool_operations=bridge,
        budget=budget,
    )
    context = tool_run_context()
    tools = await boundary.get_tools(context)
    return await boundary.call_tool(handler.__name__, {}, context, tools[handler.__name__])


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_safe", (False, True))
async def test_long_leaf_retains_ownership_without_tool_expiry(
    monkeypatch: pytest.MonkeyPatch, replay_safe: bool,
) -> None:
    async with _runtime(monkeypatch) as (state, bridge, clock):
        async def effect() -> str:
            await asyncio.sleep(0)
            for seconds in (20, 20, 21):
                await clock.advance(seconds)
            return "committed"

        assert await _call(bridge, effect, replay_safe=replay_safe) == "committed"
        operation, = await bridge.list_operations()
        assert operation.status is ToolOperationStatus.COMPLETED
        assert operation.owner == "owner"
        assert operation.fence == 1
        assert operation.updated_at - operation.created_at == timedelta(seconds=61)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("signal", "expected_status"),
    (
        (ToolCallFailed("failed"), ToolOperationStatus.FAILED),
        (ToolCallRetry("retry"), ToolOperationStatus.FAILED),
        (CallDeferred(), ToolOperationStatus.PENDING),
        (RuntimeError("unknown"), ToolOperationStatus.EFFECT_UNKNOWN),
    ),
)
async def test_long_leaf_signals_preserve_actual_outcome(
    monkeypatch: pytest.MonkeyPatch,
    signal: Exception,
    expected_status: ToolOperationStatus,
) -> None:
    async with _runtime(monkeypatch) as (_, bridge, clock):
        async def effect() -> str:
            await asyncio.sleep(0)
            for seconds in (20, 20, 21):
                await clock.advance(seconds)
            raise signal

        expected_error = AIError if isinstance(signal, RuntimeError) else type(signal)
        with pytest.raises(expected_error) as raised:
            await _call(bridge, effect, replay_safe=True)
        if isinstance(signal, RuntimeError):
            assert raised.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
        operation, = await bridge.list_operations()
        assert operation.status is expected_status
        assert operation.fence == 1


@pytest.mark.asyncio
async def test_large_result_persistence_outlasts_leaf_without_tool_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _runtime(monkeypatch) as (_, bridge, clock):
        async def effect() -> str:
            for seconds in (20, 20, 21):
                await clock.advance(seconds)
            return "x" * 70000

        put = bridge._checkpoint_objects.put

        async def slow_put(*args: Any, **kwargs: Any) -> Any:
            for _ in range(4):
                await clock.advance(20)
            return await put(*args, **kwargs)

        monkeypatch.setattr(bridge._checkpoint_objects, "put", slow_put)
        assert await _call(bridge, effect) == "x" * 70000
        operation, = await bridge.list_operations()
        assert operation.status is ToolOperationStatus.COMPLETED
        assert operation.result_payload is not None
        assert operation.updated_at - operation.created_at == timedelta(seconds=141)


@pytest.mark.asyncio
@pytest.mark.parametrize("after_commit", (False, True))
@pytest.mark.parametrize(
    ("method", "signal", "expected_status"),
    (
        ("complete_payload", None, ToolOperationStatus.COMPLETED),
        ("fail_payload", ToolCallFailed("failed"), ToolOperationStatus.FAILED),
        ("defer", CallDeferred(), ToolOperationStatus.PENDING),
    ),
)
async def test_delayed_terminal_commit_and_uncertain_response_preserve_outcome(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    signal: Exception | None,
    expected_status: ToolOperationStatus,
    after_commit: bool,
) -> None:
    async with _runtime(monkeypatch) as (state, bridge, clock):
        async def effect() -> str:
            await asyncio.sleep(0)
            if signal is not None:
                raise signal
            return "known result"

        settle = getattr(state.checkpoint.tools, method)

        async def delayed_response(*args: Any, **kwargs: Any) -> Any:
            if not after_commit:
                for _ in range(4):
                    await clock.advance(20)
            result = await settle(*args, **kwargs)
            if after_commit:
                await clock.advance(20)
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return result

        monkeypatch.setattr(state.checkpoint.tools, method, delayed_response)
        if signal is None:
            assert await _call(bridge, effect, replay_safe=True) == "known result"
        else:
            with pytest.raises(type(signal)):
                await _call(bridge, effect, replay_safe=True)
        operation, = await bridge.list_operations()
        assert operation.status is expected_status


@pytest.mark.asyncio
async def test_caller_cancellation_preserves_unknown_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _runtime(monkeypatch) as (_, bridge, clock):
        started, stopped = asyncio.Event(), asyncio.Event()

        async def effect() -> str:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        task = asyncio.create_task(_call(bridge, effect))
        await asyncio.wait_for(started.wait(), 1)
        for seconds in (20, 20, 21):
            await clock.advance(seconds)
        task.cancel()
        with pytest.raises(AIError) as raised:
            await asyncio.wait_for(task, 1)
        assert raised.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
        assert stopped.is_set()
        operation, = await bridge.list_operations()
        assert operation.status is ToolOperationStatus.EFFECT_UNKNOWN
        assert operation.result_payload is None


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", (False, True))
async def test_budget_admission_failure_defers_without_external_effect(
    monkeypatch: pytest.MonkeyPatch, cancelled: bool,
) -> None:
    async with _runtime(monkeypatch) as (state, bridge, clock):
        budgets = state.execution.budgets
        await budgets.ensure("scope", RunBudget(tool_calls=2))
        started = asyncio.Event()

        async def failed_admission(*args: Any, **kwargs: Any) -> None:
            started.set()
            if cancelled:
                await asyncio.Event().wait()
            else:
                await asyncio.sleep(0)
                for seconds in (20, 20, 21):
                    await clock.advance(seconds)
                raise AIError(ErrorCode.STORAGE_CLOSED)

        monkeypatch.setattr(budgets, "admit_tool", failed_admission)
        calls = []

        async def effect() -> str:
            calls.append("effect")
            return "unexpected"

        task = asyncio.create_task(_call(
            bridge, effect, budget=RunBudgetContext(budgets, "scope", "execution", "run")
        ))
        await asyncio.wait_for(started.wait(), 1)
        if cancelled:
            for seconds in (20, 20, 21):
                await clock.advance(seconds)
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancelled else AIError) as raised:
            await asyncio.wait_for(task, 1)
        if not cancelled:
            assert raised.value.code is ErrorCode.STORAGE_CLOSED
        assert not calls
        operation, = await bridge.list_operations()
        assert operation.status is ToolOperationStatus.PENDING


async def _owned_bridge(state: Persistence) -> tuple[RuntimeToolOperationBridge, StateCommands]:
    repo = state.execution.executions
    await repo.create_execution(replace(
        _execution(), revision=1, producer_generation=1, producer_claim_id="owner",
    ))
    commands = StateCommands(
        repo, namespace="tool-lease", events=state.execution.events,
        tools=state.checkpoint.tools, background_tasks=set(),
    )

    async def require_active() -> None:
        await repo.state_store.mutate(lambda transaction: repo.require_execution_producer_in_transaction(
            transaction, "execution", expected_producer_generation=1,
        ))

    return RuntimeToolOperationBridge(
        state.checkpoint.tools, InMemoryObjectStore(), namespace="tool-lease",
        tenant_id="tenant", execution_id="execution", agent_run_id="run",
        binding_digest="b" * 64, owner="owner", background_tasks=set(),
        payload_policy=PayloadPolicy(), terminal_commands=commands,
        producer_generation=1, require_active=require_active,
    ), commands


async def _takeover(state: Persistence, commands: StateCommands) -> None:
    repo = state.execution.executions
    current = await repo.get("execution", tenant_id="tenant")
    successor = await repo.claim_next_agent_run(
        "execution", tenant_id="tenant", expected_revision=current.revision,
        expected_agent_run_seq=current.agent_run_seq, producer_claim_id="successor",
    )
    operation, = await state.checkpoint.tools.list_by_execution("execution", tenant_id="tenant")
    await commands.reconcile_tool_turn_takeover(
        operation.tool_operation_id, execution_id="execution",
        producer_generation=successor.producer_generation, producer_claim_id="successor",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_safe", (False, True))
async def test_turn_takeover_during_budget_admission_prevents_leaf_entry(
    monkeypatch: pytest.MonkeyPatch, replay_safe: bool,
) -> None:
    async with _runtime(monkeypatch) as (state, _, _):
        bridge, commands = await _owned_bridge(state)
        budgets = state.execution.budgets
        await budgets.ensure("scope", RunBudget(tool_calls=2))
        admit = budgets.admit_tool

        async def takeover_during_admission(*args: Any, **kwargs: Any) -> Any:
            admitted = await admit(*args, **kwargs)
            await _takeover(state, commands)
            return admitted

        monkeypatch.setattr(budgets, "admit_tool", takeover_during_admission)
        calls: list[str] = []

        async def effect() -> str:
            calls.append("effect")
            return "unexpected"

        with pytest.raises(AIError):
            await _call(bridge, effect, replay_safe=replay_safe,
                        budget=RunBudgetContext(budgets, "scope", "execution", "run"))
        assert calls == []
        operation, = await bridge.list_operations()
        assert operation.status is (ToolOperationStatus.PENDING if replay_safe
                                     else ToolOperationStatus.EFFECT_UNKNOWN)
        assert operation.fence == 1
        assert operation.result_payload is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", (False, True))
async def test_known_strict_leaf_outcome_settles_after_turn_takeover(
    monkeypatch: pytest.MonkeyPatch, failed: bool,
) -> None:
    async with _runtime(monkeypatch) as (state, _, _):
        bridge, commands = await _owned_bridge(state)

        async def effect() -> str:
            await _takeover(state, commands)
            unknown, = await bridge.list_operations()
            assert unknown.status is ToolOperationStatus.EFFECT_UNKNOWN
            assert unknown.owner == "owner"
            if failed:
                raise ToolCallFailed("known failure")
            return "known result"

        if failed:
            with pytest.raises(ToolCallFailed):
                await _call(bridge, effect)
        else:
            assert await _call(bridge, effect) == "known result"
        operation, = await bridge.list_operations()
        assert operation.status is (ToolOperationStatus.FAILED if failed
                                     else ToolOperationStatus.COMPLETED)
        assert operation.owner == "owner"
        assert operation.fence == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "signal", "expected_status"),
    (("complete_payload", None, ToolOperationStatus.COMPLETED),
     ("fail_payload", ToolCallFailed("failed"), ToolOperationStatus.FAILED),
     ("defer", CallDeferred(), ToolOperationStatus.PENDING)),
)
async def test_cancellation_during_terminal_commit_preserves_known_leaf_outcome(
    monkeypatch: pytest.MonkeyPatch, method: str, signal: Exception | None,
    expected_status: ToolOperationStatus,
) -> None:
    async with _runtime(monkeypatch) as (state, bridge, _):
        started, release = asyncio.Event(), asyncio.Event()
        settle = getattr(state.checkpoint.tools, method)

        async def delayed_commit(*args: Any, **kwargs: Any) -> Any:
            started.set()
            await release.wait()
            return await settle(*args, **kwargs)

        monkeypatch.setattr(state.checkpoint.tools, method, delayed_commit)

        async def effect() -> str:
            if signal is not None:
                raise signal
            return "known result"

        task = asyncio.create_task(_call(bridge, effect, replay_safe=True))
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        operation, = await bridge.list_operations()
        assert operation.status is expected_status
        assert (operation.result_payload is not None) == (expected_status is ToolOperationStatus.COMPLETED)


@pytest.mark.asyncio
async def test_slow_budget_admission_preserves_current_turn_tool_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _runtime(monkeypatch) as (state, _, clock):
        bridge, _ = await _owned_bridge(state)
        budgets = state.execution.budgets
        await budgets.ensure("scope", RunBudget(tool_calls=2))
        admit = budgets.admit_tool

        async def slow_admission(*args: Any, **kwargs: Any) -> Any:
            result = await admit(*args, **kwargs)
            await clock.advance(120)
            operation, = await bridge.list_operations()
            assert operation.status is ToolOperationStatus.CLAIMED
            assert operation.producer_generation == 1
            return result

        monkeypatch.setattr(budgets, "admit_tool", slow_admission)

        async def effect() -> str:
            return "committed"

        assert await _call(
            bridge, effect, budget=RunBudgetContext(budgets, "scope", "execution", "run"),
        ) == "committed"
        operation, = await bridge.list_operations()
        assert operation.status is ToolOperationStatus.COMPLETED
        assert operation.owner == "owner"
        assert operation.fence == 1


@pytest.mark.asyncio
async def test_cancellation_during_large_result_put_preserves_exact_known_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _runtime(monkeypatch) as (_, bridge, _):
        result = "x" * (80 * 1024)
        entered_put, release_put = asyncio.Event(), asyncio.Event()
        put = bridge._checkpoint_objects.put
        effects: list[str] = []

        async def paused_put(*args: Any, **kwargs: Any) -> Any:
            entered_put.set()
            await release_put.wait()
            return await put(*args, **kwargs)

        async def effect() -> str:
            effects.append("committed external effect")
            return result

        monkeypatch.setattr(bridge._checkpoint_objects, "put", paused_put)
        task = asyncio.create_task(_call(bridge, effect))
        try:
            await asyncio.wait_for(entered_put.wait(), 1)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            release_put.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
            operation, = await bridge.list_operations()
            assert operation.status is ToolOperationStatus.COMPLETED
            assert operation.result_payload is not None
            assert operation.result_payload.kind == "object"
            assert (operation.owner, operation.fence) == ("owner", 1)
            assert await _call(bridge, effect) == result
            assert effects == ["committed external effect"]
        finally:
            release_put.set()
            await asyncio.gather(task, return_exceptions=True)
