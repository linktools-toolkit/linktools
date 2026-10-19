#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model-boundary observations share one bounded execution/head snapshot."""

from linktools.ai.runtime.state import RuntimeState
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.runtime.state._execution_repositories import ExecutionRepositoryImpl
from linktools.ai.runtime.state._step_contracts import AgentRunCheckpoint, AgentRunRecord
from linktools.ai.storage import InMemoryCacheBackend, _write_cache

from .test_step_projection import _execution


@asynccontextmanager
async def _state(backend: str, path: Path, cache=None):
    if backend == "memory":
        state = RuntimeState(RuntimeStorage.in_memory(cache=cache))
    elif backend == "filesystem":
        state = RuntimeState(RuntimeStorage.filesystem(path, cache=cache))
    else:
        state = RuntimeState(RuntimeStorage.sqlite(path / "state.db", cache=cache))
    await state.initialize(namespace="model-control", tenant_id="tenant")
    try:
        yield state
    finally:
        await state.close()


async def _seed(state, *, restart: bool = False):
    repository = state.execution.executions
    execution = replace(_execution(), revision=1, restart_checkpoints=restart, durable_trace=False)
    await repository.create_with_history_head(execution)
    await repository.state_store.mutate(
        lambda tx: repository.admit_history_producer_in_transaction(tx, execution)
    )
    state.run_store.bind_execution_producer(
        "run", execution_id="execution", producer_generation=1,
        restart_checkpoints=restart, durable_trace=False,
    )
    await state.run_store.register_agent_run(AgentRunRecord("run"), execution_id="execution")
    return execution


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_model_pair_uses_one_cached_source_capture(
    tmp_path: Path, backend: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _state(backend, tmp_path, InMemoryCacheBackend(max_bytes=1000000)) as state:
        execution = await _seed(state)
        repository = state.execution.executions
        source = repository.state_store
        original = source.read
        reads = 0

        async def read(callback):
            nonlocal reads
            reads += 1
            return await original(callback)

        monkeypatch.setattr(source, "read", read)
        pair = await repository.observe_execution_control("execution", tenant_id="tenant")
        assert pair[0] == execution
        assert pair[1].producer_generation == 1
        assert await repository.observe_execution_control("execution", tenant_id="tenant") == pair
        assert reads == 1


async def _advance_pair(repository):
    async def mutate(tx):
        execution = await repository.get_in_transaction(tx, "execution", tenant_id="tenant")
        head_record = await tx.get_record(repository._key("execution_history_head", "execution"))
        head = await repository.get_history_head("execution", tenant_id="tenant")
        changed = replace(execution, revision=execution.revision + 1, status=ExecutionStatus.CANCELLING)
        await repository.compare_and_swap(
            "execution", tenant_id="tenant", expected_revision=execution.revision, next_record=changed,
        )
        changed_head = replace(head, revision=head.revision + 1, producer_generation=head.producer_generation + 1)
        await repository.replace_history_head_in_transaction(tx, head_record, changed_head)
        return changed, changed_head

    return await repository.state_store.mutate(mutate)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_no_restart_model_boundaries_reuse_the_pair(
    tmp_path: Path, backend: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _state(backend, tmp_path, InMemoryCacheBackend(max_bytes=1000000)) as state:
        await _seed(state)
        source = state.execution.executions.state_store
        original = source.read
        reads = 0

        async def read(callback):
            nonlocal reads
            reads += 1
            return await original(callback)

        monkeypatch.setattr(source, "read", read)
        for step in (1, 2):
            await state.run_store.save_checkpoint(
                AgentRunCheckpoint("run", step, []), execution_id="execution", producer_generation=1,
            )
        assert reads == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_pair_without_cache_reads_source_each_time(
    tmp_path: Path, backend: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _state(backend, tmp_path) as state:
        await _seed(state)
        repository = state.execution.executions
        original = repository.state_store.read
        reads = 0

        async def read(callback):
            nonlocal reads
            reads += 1
            return await original(callback)

        monkeypatch.setattr(repository.state_store, "read", read)
        first = await repository.observe_execution_control("execution", tenant_id="tenant")
        assert await repository.observe_execution_control("execution", tenant_id="tenant") == first
        assert reads == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_pair_inside_transaction_observes_source_not_cached_values(tmp_path: Path, backend: str) -> None:
    async with _state(backend, tmp_path, InMemoryCacheBackend(max_bytes=1000000)) as state:
        await _seed(state)
        repository = state.execution.executions
        before = await repository.observe_execution_control("execution", tenant_id="tenant")

        async def mutate(tx):
            changed = await _advance_pair(repository)
            assert changed != before
            assert await repository.observe_execution_control("execution", tenant_id="tenant") == changed
            return changed

        changed = await repository.state_store.mutate(mutate)
        assert await repository.observe_execution_control("execution", tenant_id="tenant") == changed


@pytest.mark.asyncio
async def test_pair_late_fill_cannot_replace_newer_local_commit(tmp_path: Path) -> None:
    started, release = asyncio.Event(), asyncio.Event()

    class Cache(InMemoryCacheBackend):
        blocked = False

        async def put(self, key, content):
            if self.blocked:
                self.blocked = False
                started.set()
                await release.wait()
            await super().put(key, content)

    cache = Cache(max_bytes=1000000)
    async with _state("memory", tmp_path, cache) as state:
        await _seed(state)
        repository = state.execution.executions
        cache.blocked = True
        reading = asyncio.create_task(repository.observe_execution_control("execution", tenant_id="tenant"))
        try:
            await asyncio.wait_for(started.wait(), 2)
            changed = await _advance_pair(repository)
            release.set()
            old = await reading
            assert old[0].status is ExecutionStatus.STARTED and old[1].producer_generation == 1
            assert await repository.observe_execution_control("execution", tenant_id="tenant") == changed
        finally:
            release.set()
            await reading


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("sqlite",))
async def test_remote_pair_change_observed_by_original_capture_deadline(
    tmp_path: Path, backend: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    monkeypatch.setattr(_write_cache, "time", lambda: clock[0])

    class Cache(InMemoryCacheBackend):
        delay = False

        async def put(self, key, content):
            if self.delay:
                self.delay = False
                clock[0] = 102.0
            await super().put(key, content)

    cache = Cache(max_bytes=1000000)
    async with _state(backend, tmp_path, cache) as first:
        await _seed(first)
        repository = first.execution.executions
        cache.delay = True
        old = await repository.observe_execution_control("execution", tenant_id="tenant")
        assert clock[0] == 102.0
        async with _state(backend, tmp_path, cache) as second:
            changed = await _advance_pair(second.execution.executions)
            assert await repository.observe_execution_control("execution", tenant_id="tenant") == old
            clock[0] = 103.0
            assert await repository.observe_execution_control("execution", tenant_id="tenant") == changed


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("execution", "execution_history_head"))
@pytest.mark.parametrize("damage", ("missing", "identity", "kind"))
async def test_missing_or_corrupt_pair_fails_closed(tmp_path: Path, kind: str, damage: str) -> None:
    async with _state("memory", tmp_path, InMemoryCacheBackend(max_bytes=1000000)) as state:
        await _seed(state)
        repository = state.execution.executions

        async def mutate(tx):
            key = repository._key(kind, "execution")
            stored = await tx.get_record(key)
            if damage == "missing":
                await tx.delete_record(key)
            else:
                if damage == "kind":
                    changed = replace(stored, kind="unexpected")
                    await tx.delete_record(key)
                    await tx.insert_record(changed)
                    return
                else:
                    if kind == "execution":
                        value = await repository.get_in_transaction(tx, "execution", tenant_id="tenant")
                        value = replace(value, execution_id="foreign", root_execution_id="foreign")
                    else:
                        value = replace(await repository.get_history_head("execution", tenant_id="tenant"), execution_id="foreign")
                    changed = replace(repository._stored(kind, "execution", value), storage_revision=stored.storage_revision + 1)
                assert await tx.replace_record(changed, expected_storage_revision=stored.storage_revision)

        await repository.state_store.mutate(mutate)
        with pytest.raises(AIError) as raised:
            await state.run_store._require_active_execution_producer(
                agent_run_id="run", execution_id="execution", producer_generation=1, allow_cancelling=False,
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_custom_repository_retains_original_strong_snapshot(tmp_path: Path) -> None:
    async with _state("memory", tmp_path, InMemoryCacheBackend(max_bytes=1000000)) as state:
        await _seed(state)
        source = state.execution.executions.state_store
        calls = []

        class Custom(ExecutionRepositoryImpl):
            async def observe_execution_control(self, *args, **kwargs):
                raise AssertionError("custom owner is not optimized")

            async def get_history_head(self, *args, **kwargs):
                from linktools.ai.runtime.state._store import active_state_transaction
                assert active_state_transaction(source) is not None
                calls.append("head")
                return await super().get_history_head(*args, **kwargs)

            async def get(self, *args, **kwargs):
                from linktools.ai.runtime.state._store import active_state_transaction
                assert active_state_transaction(source) is not None
                calls.append("execution")
                return await super().get(*args, **kwargs)

        state.run_store._execution_repository = Custom(source, namespace="model-control", tenant_id="tenant")
        for _ in range(2):
            await state.run_store._require_active_execution_producer(agent_run_id="run", execution_id="execution", producer_generation=1)
        assert calls == ["head", "execution", "head", "execution"]


@pytest.mark.asyncio
async def test_next_model_boundary_sees_remote_cancellation_at_three_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    monkeypatch.setattr(_write_cache, "time", lambda: clock[0])
    cache = InMemoryCacheBackend(max_bytes=1000000)
    async with _state("sqlite", tmp_path, cache) as first:
        execution = await _seed(first)
        await first.run_store.save_checkpoint(
            AgentRunCheckpoint("run", 1, []), execution_id="execution", producer_generation=1,
        )
        async with _state("sqlite", tmp_path, cache) as second:
            await second.execution.executions.compare_and_swap(
                "execution", tenant_id="tenant", expected_revision=execution.revision,
                next_record=replace(execution, revision=execution.revision + 1, status=ExecutionStatus.CANCELLING),
            )
            clock[0] = 102.9
            await first.run_store.save_checkpoint(
                AgentRunCheckpoint("run", 2, []), execution_id="execution", producer_generation=1,
            )
            clock[0] = 103.0
            with pytest.raises(AIError) as raised:
                await first.run_store.save_checkpoint(
                    AgentRunCheckpoint("run", 3, []), execution_id="execution", producer_generation=1,
                )
            assert raised.value.code is ErrorCode.EXECUTION_CANCELLED


@pytest.mark.asyncio
@pytest.mark.parametrize("delegated", (False, True), ids=("agent", "delegated-agent"))
async def test_task_owned_model_boundaries_do_not_use_bounded_observation(
    tmp_path: Path, delegated: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from .test_task_agent_persistence import test_task_agent_trace_survives_terminal_release_and_reopen

    original = RuntimeStorage.sqlite

    def sqlite(*args, **kwargs):
        kwargs["cache"] = InMemoryCacheBackend(max_bytes=1000000)
        return original(*args, **kwargs)

    async def unexpected(*args, **kwargs):
        raise AssertionError("Task-owned checkpoints require the durable recovery path")

    monkeypatch.setattr(RuntimeStorage, "sqlite", sqlite)
    monkeypatch.setattr(ExecutionRepositoryImpl, "observe_execution_control", unexpected)
    await test_task_agent_trace_survives_terminal_release_and_reopen(tmp_path, delegated)


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", (False, True))
@pytest.mark.parametrize("trace", (False, True))
async def test_transient_route_uses_frozen_restart_guarantee(
    restart: bool, trace: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.runtime import RuntimeStoragePlan
    from linktools.ai.storage import StorageRoute

    state = RuntimeState(RuntimeStorage.from_plan(
        RuntimeStoragePlan(execution=StorageRoute.transient(cache=InMemoryCacheBackend(max_bytes=1000000))),
    ))
    await state.initialize(namespace="model-control", tenant_id="tenant")
    try:
        repository = state.execution.executions
        execution = replace(_execution(), revision=1, restart_checkpoints=restart, durable_trace=trace)
        await repository.create_with_history_head(execution)
        await repository.state_store.mutate(
            lambda tx: repository.admit_history_producer_in_transaction(tx, execution)
        )
        state.run_store.bind_execution_producer(
            "run", execution_id="execution", producer_generation=1,
            restart_checkpoints=restart, durable_trace=trace,
        )
        await state.run_store.register_agent_run(AgentRunRecord("run"), execution_id="execution")
        original = repository.observe_execution_control
        observations = 0

        async def observe(*args, **kwargs):
            nonlocal observations
            observations += 1
            assert not restart
            return await original(*args, **kwargs)

        monkeypatch.setattr(repository, "observe_execution_control", observe)
        for step in (1, 2):
            await state.run_store.save_checkpoint(
                AgentRunCheckpoint("run", step, []), execution_id="execution", producer_generation=1,
            )
        assert observations == (0 if restart else 2)
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("delegated", (False, True), ids=("agent", "delegated-agent"))
async def test_volatile_task_and_descendant_keep_strong_model_boundaries(
    delegated: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
    from linktools.ai.capability import RuntimeFeatures
    from linktools.ai.core import TaskStatus
    from linktools.ai.runtime import AgentTaskInput, Runtime, RuntimeStoragePlan
    from linktools.ai.storage import StorageRoute
    from linktools.ai.task import TaskGraph, TaskNode
    from ._runtime_test_helpers import _Models, _UsageFunctionModel

    async def unexpected(*args, **kwargs):
        raise AssertionError("volatile Task still retains strong model control")

    async def model(messages, info):
        has_return = any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)
        if delegated and not has_return and "delegate_task" in {tool.name for tool in info.function_tools}:
            return ModelResponse(parts=[ToolCallPart(
                "delegate_task", {"subagent_id": "child", "task": "child answer"}, tool_call_id="child",
            )])
        return ModelResponse(parts=[TextPart("answer")])

    monkeypatch.setattr(ExecutionRepositoryImpl, "observe_execution_control", unexpected)
    group = RuntimeFeatures("application")
    group.agent("default", model="default", allow_tools=(), allow_subagents=("child",) if delegated else ())
    group.agent("child", model="default", allow_tools=(), allow_subagents=())
    storage = RuntimeStorage.from_plan(
        RuntimeStoragePlan(execution=StorageRoute.transient(cache=InMemoryCacheBackend(max_bytes=1000000))),
    )
    async with Runtime.open(
        "transient-task-control", models=_Models(_UsageFunctionModel(model)),
        storage=storage, features=group,
    ) as runtime:
        task = runtime.tasks.from_agent("answer", runtime.agents.get())
        definition = TaskGraph("graph", (TaskNode("answer", task=task, input=AgentTaskInput("question")),))
        graph = await runtime.tasks.bind(task).start(definition, idempotency_key="once")
        assert (await graph.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_boundary", (False, True), ids=("expired-model-boundary", "unexpired-tool-boundary"))
async def test_remote_cancel_stops_model_retry_and_strong_tool_admission(
    tmp_path: Path, tool_boundary: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic import BaseModel
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from linktools.ai.capability import AgentContext, RuntimeFeatures
    from linktools.ai.runtime import Runtime
    from linktools.ai.runtime.service_api import CancelExecutionRequest
    from ._runtime_test_helpers import _Models, _UsageFunctionModel

    class Answer(BaseModel):
        answer: str

    clock = [100.0]
    monkeypatch.setattr(_write_cache, "time", lambda: clock[0])
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    effects = []

    async def work(_context: AgentContext[None]) -> str:
        effects.append("effect")
        return "done"

    async def model(messages, info):
        calls.append(1)
        entered.set()
        await release.wait()
        return ModelResponse(parts=[ToolCallPart(
            "work" if tool_boundary else info.output_tools[0].name, {}, tool_call_id="call",
        )])

    group = RuntimeFeatures("application")
    group.tool(work, effect_policy="replay_safe")
    group.agent("default", model="default", allow_tools=("work",))
    database = tmp_path / "cancel.db"
    models = _Models(_UsageFunctionModel(model))
    async with Runtime.open(
        "model-pair-cancel", models=models, features=group,
        storage=RuntimeStorage.sqlite(database, cache=InMemoryCacheBackend(max_bytes=1000000)),
    ) as owner, Runtime.open(
        "model-pair-cancel", models=models, features=group,
        storage=RuntimeStorage.sqlite(database, cache=InMemoryCacheBackend(max_bytes=1000000)),
    ) as other:
        execution = await owner.agents.get().start("question", output=Answer, idempotency_key="once")
        try:
            await asyncio.wait_for(entered.wait(), 5)
            await other.executions.cancel(
                execution.execution_id, CancelExecutionRequest(owner.default_principal, "remote-cancel"),
            )
            clock[0] = 100.5 if tool_boundary else 103.0
            release.set()
            result = (await execution.wait(timeout_seconds=5)).result
            assert result.status is ExecutionStatus.CANCELLED
            assert calls == [1]
            assert effects == []
        finally:
            release.set()
