#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool leaf lifetime keeps its durable claim live and preserves fencing."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai.exceptions import CallDeferred
from pydantic_ai.toolsets import FunctionToolset

from linktools.ai.capability import ToolCallFailed, ToolCallRetry
from linktools.ai.core import RunBudget, ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import _tool_boundary as boundary_module
from linktools.ai.runtime._budget import RunBudgetContext
from linktools.ai.runtime._tool import RuntimeToolOperationBridge, ToolOperationDecision
from linktools.ai.runtime._tool_boundary import BoundaryToolset, ManagedToolDescriptor
from linktools.ai.runtime.state import RuntimeStorage
from linktools.ai.runtime.state._memory_transaction import _MemoryTransaction
from linktools.ai.storage import InMemoryObjectStore, PayloadPolicy

from ._runtime_test_helpers import tool_run_context, tool_with_metadata


class _Clock:
    def __init__(self) -> None:
        self.current = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.waiters: list[tuple[datetime, asyncio.Future[None]]] = []

    async def sleep(self, delay: float) -> None:
        future = asyncio.get_running_loop().create_future()
        entry = (self.current + timedelta(seconds=delay), future)
        self.waiters.append(entry)
        try:
            await future
        finally:
            self.waiters.remove(entry)

    async def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)
        for deadline, future in tuple(self.waiters):
            if deadline <= self.current and not future.done():
                future.set_result(None)
        # Memory transactions complete without I/O; drain their scheduled continuations.
        for _ in range(10):
            await asyncio.sleep(0)


@asynccontextmanager
async def _runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[RuntimeStorage, RuntimeToolOperationBridge, _Clock]]:
    clock = _Clock()

    async def now(transaction: _MemoryTransaction) -> datetime:
        del transaction
        return clock.current

    monkeypatch.setattr(_MemoryTransaction, "now", now)
    monkeypatch.setattr(
        boundary_module, "asyncio", SimpleNamespace(**{**vars(asyncio), "sleep": clock.sleep})
    )
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="tool-lease", tenant_id="tenant")
    bridge = RuntimeToolOperationBridge(
        state.recovery.tools,
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
        assert not clock.waiters


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
async def test_long_leaf_renews_live_claim_before_terminal_commit(
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
        assert operation.lease_expires_at is None
        assert await state.recovery.tools.reconcile_expired_claim(
            operation.tool_operation_id, tenant_id="tenant"
        ) == operation


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
async def test_long_leaf_signals_settle_with_renewed_claim(
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
        assert operation.lease_expires_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize("suppress_cancellation", (False, True))
@pytest.mark.parametrize("failure", (AIError(ErrorCode.STORAGE_CLOSED), asyncio.CancelledError()))
async def test_renewal_failure_stops_leaf_and_never_commits_success(
    monkeypatch: pytest.MonkeyPatch, suppress_cancellation: bool, failure: BaseException,
) -> None:
    async with _runtime(monkeypatch) as (_, bridge, clock):
        started, stopped = asyncio.Event(), asyncio.Event()

        async def effect() -> str:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if not suppress_cancellation:
                    raise
                return "cannot commit after lease loss"
            finally:
                stopped.set()

        async def failed_renew(decision: ToolOperationDecision) -> ToolOperationDecision:
            del decision
            raise failure

        monkeypatch.setattr(bridge, "renew", failed_renew)
        task = asyncio.create_task(_call(bridge, effect))
        await asyncio.wait_for(started.wait(), 1)
        await clock.advance(20)
        with pytest.raises(AIError) as raised:
            await asyncio.wait_for(task, 1)
        assert raised.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
        assert stopped.is_set()
        operation, = await bridge.list_operations()
        assert operation.status is ToolOperationStatus.EFFECT_UNKNOWN
        assert operation.result_payload is None


@pytest.mark.asyncio
async def test_claim_renewal_covers_large_result_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _runtime(monkeypatch) as (_, bridge, clock):
        async def effect() -> str:
            for seconds in (20, 20, 21):
                await clock.advance(seconds)
            return "x" * 70000

        put = bridge._recovery_objects.put

        async def slow_put(*args: Any, **kwargs: Any) -> Any:
            for _ in range(4):
                await clock.advance(20)
            return await put(*args, **kwargs)

        monkeypatch.setattr(bridge._recovery_objects, "put", slow_put)
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
async def test_terminal_commit_renews_and_wins_racing_heartbeat(
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

        settle = getattr(state.recovery.tools, method)

        async def delayed_response(*args: Any, **kwargs: Any) -> Any:
            if not after_commit:
                for _ in range(4):
                    await clock.advance(20)
            result = await settle(*args, **kwargs)
            if after_commit:
                await clock.advance(20)
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return result

        monkeypatch.setattr(state.recovery.tools, method, delayed_response)
        if signal is None:
            assert await _call(bridge, effect, replay_safe=True) == "known result"
        else:
            with pytest.raises(type(signal)):
                await _call(bridge, effect, replay_safe=True)
        operation, = await bridge.list_operations()
        assert operation.status is expected_status
        assert operation.lease_expires_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("signal", "expected_status"),
    (
        (None, ToolOperationStatus.COMPLETED),
        (ToolCallFailed("failed"), ToolOperationStatus.FAILED),
        (CallDeferred(), ToolOperationStatus.PENDING),
    ),
)
async def test_cancellation_during_heartbeat_cleanup_preserves_known_leaf_outcome(
    monkeypatch: pytest.MonkeyPatch,
    signal: Exception | None,
    expected_status: ToolOperationStatus,
) -> None:
    async with _runtime(monkeypatch) as (_, bridge, clock):
        started, release = asyncio.Event(), asyncio.Event()
        renewal_started = asyncio.Event()
        cleanup_started, cleanup_release = asyncio.Event(), asyncio.Event()
        renew = bridge.renew

        async def effect() -> str:
            started.set()
            await release.wait()
            if signal is not None:
                raise signal
            return "known result"

        async def gated_renew(decision: ToolOperationDecision) -> ToolOperationDecision:
            renewed = await renew(decision)
            renewal_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup_started.set()
                await cleanup_release.wait()
            return renewed

        monkeypatch.setattr(bridge, "renew", gated_renew)
        task = asyncio.create_task(_call(bridge, effect, replay_safe=True))
        await asyncio.wait_for(started.wait(), 1)
        await clock.advance(20)
        await asyncio.wait_for(renewal_started.wait(), 1)
        release.set()
        await asyncio.wait_for(cleanup_started.wait(), 1)
        for _ in range(2):
            task.cancel()
            await clock.advance(0)
        assert not task.done()
        cleanup_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        operation, = await bridge.list_operations()
        assert operation.status is expected_status
        assert operation.lease_expires_at is None
        assert (operation.result_payload is not None) == (
            expected_status is ToolOperationStatus.COMPLETED
        )


@pytest.mark.asyncio
async def test_expired_non_replay_safe_claim_is_not_revived(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _runtime(monkeypatch) as (state, bridge, clock):
        started, stopped = asyncio.Event(), asyncio.Event()

        async def effect() -> str:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        task = asyncio.create_task(_call(bridge, effect))
        await asyncio.wait_for(started.wait(), 1)
        await clock.advance(61)
        with pytest.raises(AIError) as raised:
            await asyncio.wait_for(task, 1)
        assert raised.value.code is ErrorCode.TOOL_OPERATION_CONFLICT
        assert stopped.is_set()
        operation, = await bridge.list_operations()
        reconciled = await state.recovery.tools.reconcile_expired_claim(
            operation.tool_operation_id, tenant_id="tenant"
        )
        assert reconciled.status is ToolOperationStatus.EFFECT_UNKNOWN
        assert reconciled.fence == 1
        assert reconciled.result_payload is None


@pytest.mark.asyncio
async def test_renewal_cannot_overwrite_successor_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _runtime(monkeypatch) as (state, bridge, clock):
        started, stopped = asyncio.Event(), asyncio.Event()

        async def effect() -> str:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        task = asyncio.create_task(_call(bridge, effect, replay_safe=True))
        await asyncio.wait_for(started.wait(), 1)
        operation, = await bridge.list_operations()
        clock.current += timedelta(seconds=61)
        successor = await state.recovery.tools.claim(
            operation.tool_operation_id, tenant_id="tenant", owner="successor", lease_seconds=60
        )
        await clock.advance(0)
        with pytest.raises(AIError) as raised:
            await asyncio.wait_for(task, 1)
        assert raised.value.code is ErrorCode.TOOL_OPERATION_CONFLICT
        assert stopped.is_set()
        current, = await bridge.list_operations()
        assert current == successor
        assert current.status is ToolOperationStatus.CLAIMED
        assert current.owner == "successor"
        assert current.fence == 2
        assert current.result_payload is None


@pytest.mark.asyncio
async def test_caller_cancellation_stops_heartbeat_and_preserves_unknown_effect(
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
async def test_budget_admission_keeps_effect_claim_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _runtime(monkeypatch) as (state, bridge, clock):
        budgets = state.execution.budgets
        await budgets.ensure("scope", RunBudget(tool_calls=2))
        admit = budgets.admit_tool

        async def slow_admission(*args: Any, **kwargs: Any) -> Any:
            result = await admit(*args, **kwargs)
            await asyncio.sleep(0)
            for seconds in (20, 20, 21):
                await clock.advance(seconds)
                operation, = await bridge.list_operations()
                reconciled = await state.recovery.tools.reconcile_expired_claim(
                    operation.tool_operation_id, tenant_id="tenant"
                )
                assert reconciled.status is ToolOperationStatus.CLAIMED
            return result

        monkeypatch.setattr(budgets, "admit_tool", slow_admission)

        async def effect() -> str:
            return "committed"

        assert await _call(
            bridge, effect, budget=RunBudgetContext(budgets, "scope", "execution", "run")
        ) == "committed"
        operation, = await bridge.list_operations()
        assert operation.status is ToolOperationStatus.COMPLETED


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_safe", (False, True))
async def test_budget_admission_cannot_dispatch_after_claim_loss(
    monkeypatch: pytest.MonkeyPatch, replay_safe: bool,
) -> None:
    async with _runtime(monkeypatch) as (state, bridge, clock):
        budgets = state.execution.budgets
        await budgets.ensure("scope", RunBudget(tool_calls=2))
        admit = budgets.admit_tool
        expected = None

        async def slow_admission(*args: Any, **kwargs: Any) -> Any:
            nonlocal expected
            result = await admit(*args, **kwargs)
            # No event-loop turn: the heartbeat cannot detect the stolen claim first.
            clock.current += timedelta(seconds=61)
            operation, = await bridge.list_operations()
            if replay_safe:
                expected = await state.recovery.tools.claim(
                    operation.tool_operation_id, tenant_id="tenant",
                    owner="successor", lease_seconds=60,
                )
            else:
                expected = await state.recovery.tools.reconcile_expired_claim(
                    operation.tool_operation_id, tenant_id="tenant"
                )
            return result

        monkeypatch.setattr(budgets, "admit_tool", slow_admission)
        calls = []

        async def effect() -> str:
            calls.append("effect")
            return "unexpected"

        with pytest.raises(AIError) as raised:
            await _call(
                bridge, effect, replay_safe=replay_safe,
                budget=RunBudgetContext(budgets, "scope", "execution", "run"),
            )
        assert raised.value.code is ErrorCode.TOOL_OPERATION_CONFLICT
        assert not calls
        operation, = await bridge.list_operations()
        assert operation == expected


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
        assert operation.lease_expires_at is None


@pytest.mark.asyncio
async def test_budget_admission_suppressing_lease_loss_cannot_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _runtime(monkeypatch) as (state, bridge, clock):
        budgets = state.execution.budgets
        await budgets.ensure("scope", RunBudget(tool_calls=2))
        started = asyncio.Event()

        async def admission(*args: Any, **kwargs: Any) -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass

        async def failed_renew(decision: ToolOperationDecision) -> ToolOperationDecision:
            raise AIError(ErrorCode.STORAGE_CLOSED)

        monkeypatch.setattr(budgets, "admit_tool", admission)
        monkeypatch.setattr(bridge, "renew", failed_renew)
        calls = []

        async def effect() -> str:
            calls.append("effect")
            return "unexpected"

        task = asyncio.create_task(_call(
            bridge, effect, budget=RunBudgetContext(budgets, "scope", "execution", "run")
        ))
        await asyncio.wait_for(started.wait(), 1)
        await clock.advance(20)
        with pytest.raises(AIError) as raised:
            await asyncio.wait_for(task, 1)
        assert raised.value.code is ErrorCode.STORAGE_CLOSED
        assert not calls
        operation, = await bridge.list_operations()
        assert operation.status is ToolOperationStatus.PENDING
