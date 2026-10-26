#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent task producers borrow persistence without creating Agent runtimes."""

import asyncio
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionContext, Principal, RunBudget, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.task_engine import TaskEngine
from linktools.ai.task import Task, TaskExpander, TaskGraph, TaskNode, TaskNodeContext, TaskExpansionContext


@pytest.mark.asyncio
@pytest.mark.parametrize("durable", [False, True])
async def test_function_graph_expansion_and_definition_views_share_only_their_owner(tmp_path: Path, durable: bool) -> None:
    seen = []

    async def root(context: TaskNodeContext[str]) -> str:
        seen.append((context.app, context.principal))
        return "ready"

    async def child(context: TaskNodeContext[str]) -> str:
        return context.app

    first = Task("first", root, effect_policy="none")
    second = Task("second", child, effect_policy="none")

    def expand(context: TaskExpansionContext) -> tuple[TaskNode, ...]:
        return (TaskNode("child", task=second, dependencies=(context.source_node.node_id,)),)

    expander = TaskExpander("expand", expand)
    storage = PersistenceStorage.filesystem(tmp_path) if durable else PersistenceStorage.in_memory()
    principal = Principal("caller", "tenant")
    async with Persistence.open("independent", storage=storage, tenant_id="tenant") as persistence:
        async with TaskEngine.open(first, expander, persistence=persistence, context=ExecutionContext("application")) as engine:
            bound = engine.with_definitions(second)
            assert bound.shares_execution_owner(engine)
            assert len(engine.definitions) == 1
            graph = TaskGraph("expanded", (TaskNode("root", task=first, expander=expander),))
            run = await bound.start(graph, principal=principal, idempotency_key="run", budget=RunBudget(model_requests=3))
            result = await run.wait(timeout_seconds=10)
            assert result.result.status is TaskStatus.SUCCEEDED
            assert await run.result("child") == "application"
            assert seen == [("application", principal)]
            assert (await run.budget_usage()).limits == RunBudget(model_requests=3)
            events = [event async for event in await run.replay_events()]
            assert events
            await bound.close()
            assert persistence.ready
            empty = await engine.start(TaskGraph("empty", ()), principal=principal, idempotency_key="empty")
            assert (await empty.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
        assert persistence.ready
        async with TaskEngine.open(first, second, expander, persistence=persistence) as reader:
            assert not reader.shares_execution_owner(engine)
            run = await reader.get("expanded", principal=principal)
            assert await run.result("child") == "application"


@pytest.mark.asyncio
async def test_closing_one_task_producer_cancels_its_work_without_closing_peer_or_persistence() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow(context: TaskNodeContext[None]) -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "unreachable"

    async def fast(context: TaskNodeContext[None]) -> str:
        return "peer-alive"

    slow_task = Task("slow", slow, effect_policy="none")
    fast_task = Task("fast", fast, effect_policy="none")
    async with Persistence.open("shared", storage=PersistenceStorage.in_memory()) as persistence:
        async with TaskEngine.open(fast_task, persistence=persistence) as peer:
            async with TaskEngine.open(slow_task, persistence=persistence) as owner:
                await owner.start(TaskGraph("slow", (TaskNode("slow", task=slow_task),)), idempotency_key="slow")
                await asyncio.wait_for(started.wait(), 5)
            assert cancelled.is_set()
            assert persistence.ready
            run = await peer.start(TaskGraph("fast", (TaskNode("fast", task=fast_task),)), idempotency_key="fast")
            assert (await run.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
            assert await run.result("fast") == "peer-alive"
            with pytest.raises(AIError) as closed:
                await owner.get("slow")
            assert closed.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
            assert closed.value.retryable is False


@pytest.mark.asyncio
async def test_task_producers_own_independent_recorders_over_one_dataset() -> None:
    from linktools.ai.core import CommitPolicy
    from linktools.ai.observe import MetricRecorder, Metrics
    from linktools.ai.storage import BatchWriteScopeFactory, InMemoryCacheBackend

    recorders: list[MetricRecorder] = []

    class RecordingMetrics(Metrics):
        def recorder(self, *, batch_scopes: BatchWriteScopeFactory | None = None) -> MetricRecorder:
            recorder = super().recorder(batch_scopes=batch_scopes)
            recorders.append(recorder)
            return recorder

    async def work(context: TaskNodeContext[None]) -> str:
        return "done"

    task = Task("work", work, effect_policy="none")
    metrics = RecordingMetrics.in_memory(commit=CommitPolicy.batch(
        buffer=InMemoryCacheBackend(max_bytes=2**20), batch_size=128, flush_interval=60,
    ))
    async with Persistence.open("metrics", storage=PersistenceStorage.in_memory()) as persistence:
        async with TaskEngine.open(task, persistence=persistence, metrics=metrics) as peer:
            async with TaskEngine.open(task, persistence=persistence, metrics=metrics) as first:
                first_run = await first.start(TaskGraph("first", (TaskNode("n", task=task),)), idempotency_key="first")
                assert (await first_run.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
            assert not recorders[1].status().accepting
            assert recorders[0].status().accepting
            peer_run = await peer.start(TaskGraph("peer", (TaskNode("n", task=task),)), idempotency_key="peer")
            assert (await peer_run.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
        assert recorders[0].status().persisted > 0
        assert recorders[1].status().persisted > 0
        third = metrics.recorder()
        assert third.status().accepting
        await third.close()


@pytest.mark.asyncio
async def test_reopening_task_engine_preserves_admitted_principal_and_execution_binding(tmp_path: Path) -> None:
    from linktools.ai.task import TaskEffectResolution

    original = Principal("original", "tenant")
    operator = Principal("recovery-operator", "tenant")
    seen: list[Principal] = []

    async def effect(context: TaskNodeContext[None]) -> str:
        seen.append(context.principal)
        if len(seen) == 1:
            raise OSError("effect not confirmed")
        return context.principal.principal_id

    task = Task("effect", effect, effect_policy="non_replay_safe")
    async with Persistence.open("recovery", storage=PersistenceStorage.filesystem(tmp_path), tenant_id="tenant") as persistence:
        async with TaskEngine.open(task, persistence=persistence) as engine:
            run = await engine.start(TaskGraph("effect", (TaskNode("n", task=task, max_attempts=2),)), principal=original, idempotency_key="effect")
            assert (await run.wait(timeout_seconds=10)).result.status is TaskStatus.RECOVERY_REQUIRED
            state = await run.state()
            execution_id = state.node_states[0].execution_id
    async with Persistence.open("recovery", storage=PersistenceStorage.filesystem(tmp_path), tenant_id="tenant") as persistence:
        async with TaskEngine.open(task, persistence=persistence) as engine:
            run = await engine.get("effect", principal=operator)
            state = await run.state()
            assert state.node_states[0].execution_id == execution_id
            await run.resolve_effect("n", state.node_states[0].fence, TaskEffectResolution("not_applied"), idempotency_key="verified-not-applied")
            assert (await run.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
            assert await run.result("n") == original.principal_id
            assert seen == [original, original]
            assert (await run.state()).node_states[0].execution_id == execution_id


@pytest.mark.asyncio
async def test_execution_handles_remain_bound_to_their_producer_lifetime() -> None:
    async def work(context: TaskNodeContext[None]) -> str:
        return "alive"

    task = Task("work", work, effect_policy="none")
    async with Persistence.open("handle-lifetime", storage=PersistenceStorage.in_memory()) as persistence:
        async with TaskEngine.open(task, persistence=persistence) as peer:
            async with TaskEngine.open(task, persistence=persistence) as first:
                first_run = await first.start(TaskGraph("first", (TaskNode("n", task=task),)), idempotency_key="first")
                await first_run.wait(timeout_seconds=10)
                first_handle = await first_run.execution("n")
                assert (await first_handle.wait()).result.output == "alive"
            with pytest.raises(AIError) as closed:
                await first_handle.cancel(idempotency_key="closed")
            assert closed.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
            assert closed.value.retryable is False
            peer_run = await peer.start(TaskGraph("peer", (TaskNode("n", task=task),)), idempotency_key="peer")
            await peer_run.wait(timeout_seconds=10)
            peer_handle = await peer_run.execution("n")
            assert (await peer_handle.wait()).result.output == "alive"
            assert persistence.ready
