#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Cached execution inspection never substitutes for producer dispatch authority."""

from linktools.ai.persistence import Persistence
from linktools.ai.task.engine import TaskEngine
from linktools.ai.task import Task
from ._runtime_test_helpers import model_features
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence._execution_repositories import ExecutionRepositoryImpl
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint, AgentRunRecord
from linktools.ai.storage import InMemoryCacheBackend, _write_cache

from .test_step_projection import _execution


@asynccontextmanager
async def _state(backend: str, path: Path, cache=None):
    if backend == "memory":
        state = Persistence(PersistenceStorage.in_memory(cache=cache))
    else:
        state = Persistence(PersistenceStorage.sqlite(path / "state.db", cache=cache))
    await state.initialize(namespace="model-control", tenant_id="tenant")
    try:
        yield state
    finally:
        await state.close()


async def _seed(state, *, restart: bool = False):
    repository = state.execution.executions
    execution = replace(_execution(), revision=1, producer_generation=1, producer_claim_id="owner-claim",
                        restart_checkpoints=restart, durable_trace=False)
    await repository.create_execution(execution)
    state.run_store.bind_execution_producer(
        "run", execution_id="execution", producer_generation=1,
        restart_checkpoints=restart, durable_trace=False,
    )
    await state.run_store.register_agent_run(AgentRunRecord("run"), execution_id="execution")
    return execution


def _forbid_cached_dispatch_authority(monkeypatch):
    from contextvars import ContextVar
    from linktools.ai.persistence.runs import RuntimeAgentRunStore

    checking = ContextVar("checking_dispatch", default=False)
    checks = []
    original_guard = RuntimeAgentRunStore.require_active_producer
    original_control = ExecutionRepositoryImpl.require_active_producer
    original_get = ExecutionRepositoryImpl.get

    async def require_active(self, agent_run_id, *, authoritative=False):
        token = checking.set(True)
        try:
            await original_guard(self, agent_run_id, authoritative=authoritative)
            checks.append(agent_run_id)
        finally:
            checking.reset(token)

    async def control(self, *args, **kwargs):
        assert not checking.get() or kwargs["fresh"], "dispatch authority cannot come from the ordinary read cache"
        return await original_control(self, *args, **kwargs)

    async def get(self, *args, **kwargs):
        assert not checking.get(), "dispatch authority cannot fall back to an ordinary cached read"
        return await original_get(self, *args, **kwargs)

    monkeypatch.setattr(RuntimeAgentRunStore, "require_active_producer", require_active)
    monkeypatch.setattr(ExecutionRepositoryImpl, "require_active_producer", control)
    monkeypatch.setattr(ExecutionRepositoryImpl, "get", get)
    return checks


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_execution_inspection_reuses_one_cached_source_capture(
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
        observed = await repository.get("execution", tenant_id="tenant")
        assert observed == execution
        assert observed.producer_generation == 1
        first_reads = reads
        assert first_reads <= 1
        assert await repository.get("execution", tenant_id="tenant") == observed
        assert reads == first_reads


async def _advance_execution(repository, *, takeover: bool = True):
    async def mutate(tx):
        execution = await repository.get_in_transaction(tx, "execution", tenant_id="tenant")
        changed = replace(
            execution, revision=execution.revision + 1,
            status=ExecutionStatus.STARTED if takeover else ExecutionStatus.CANCELLING,
            producer_generation=execution.producer_generation + int(takeover),
            producer_claim_id="replacement-claim" if takeover else execution.producer_claim_id,
        )
        return await repository.compare_and_swap_in_transaction(
            tx, "execution", tenant_id="tenant", expected_revision=execution.revision, next_record=changed,
        )

    return await repository.state_store.mutate(mutate)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_unscoped_dispatch_reads_fresh_source_at_every_boundary(
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
        for _ in range(2):
            await state.run_store.require_active_producer("run")
        assert reads == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_execution_inspection_without_cache_reads_source_each_time(
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
        first = await repository.get("execution", tenant_id="tenant")
        assert await repository.get("execution", tenant_id="tenant") == first
        assert reads == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_execution_inside_transaction_observes_source_not_cached_values(tmp_path: Path, backend: str) -> None:
    async with _state(backend, tmp_path, InMemoryCacheBackend(max_bytes=1000000)) as state:
        await _seed(state)
        repository = state.execution.executions
        before = await repository.get("execution", tenant_id="tenant")

        async def mutate(tx):
            changed = await _advance_execution(repository)
            assert changed != before
            assert await repository.get("execution", tenant_id="tenant") == changed
            return changed

        changed = await repository.state_store.mutate(mutate)
        assert await repository.get("execution", tenant_id="tenant") == changed


@pytest.mark.asyncio
async def test_execution_late_cache_fill_cannot_replace_newer_local_commit(tmp_path: Path) -> None:
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
        reading = asyncio.create_task(repository.get("execution", tenant_id="tenant"))
        try:
            await asyncio.wait_for(started.wait(), 2)
            changed = await _advance_execution(repository)
            release.set()
            old = await reading
            assert old.status is ExecutionStatus.STARTED and old.producer_generation == 1
            assert await repository.get("execution", tenant_id="tenant") == changed
        finally:
            release.set()
            await reading


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("sqlite",))
async def test_remote_inspection_change_observed_by_original_cache_capture_deadline(
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
        await asyncio.sleep(0)
        cache.delay = True
        old = await repository.get_fresh("execution", tenant_id="tenant")
        assert clock[0] == 102.0
        async with _state(backend, tmp_path, InMemoryCacheBackend(max_bytes=1000000)) as second:
            changed = await _advance_execution(second.execution.executions)
            assert await repository.get("execution", tenant_id="tenant") == old
            clock[0] = 103.0
            assert await repository.get("execution", tenant_id="tenant") == changed


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("damage", ("missing", "identity", "kind"))
async def test_missing_or_corrupt_execution_fails_closed(tmp_path: Path, backend: str, damage: str) -> None:
    async with _state(backend, tmp_path, InMemoryCacheBackend(max_bytes=1000000)) as state:
        await _seed(state)
        repository = state.execution.executions

        async def mutate(tx):
            key = repository._key("execution", ("execution",))
            stored = await tx.get_record(key)
            if damage == "missing":
                await tx.delete_record(key)
            elif damage == "kind":
                await tx.delete_record(key)
                await tx.insert_record(replace(stored, kind="unexpected"))
            else:
                value = await repository.get_in_transaction(tx, "execution", tenant_id="tenant")
                value = replace(value, execution_id="foreign", root_execution_id="foreign")
                changed = replace(repository._stored("execution", ("execution",), value),
                                  storage_revision=stored.storage_revision + 1)
                assert await tx.replace_record(changed, expected_storage_revision=stored.storage_revision)

        await repository.state_store.mutate(mutate)
        with pytest.raises(AIError) as raised:
            await state.run_store.require_active_producer("run")
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_custom_repository_retains_fresh_source_authority(tmp_path: Path) -> None:
    async with _state("memory", tmp_path, InMemoryCacheBackend(max_bytes=1000000)) as state:
        await _seed(state)
        source = state.execution.executions.state_store
        calls = []

        class Custom(ExecutionRepositoryImpl):
            async def get(self, *args, **kwargs):
                raise AssertionError("ordinary cached reads cannot authorize dispatch")

            async def require_active_producer(self, *args, **kwargs):
                assert kwargs["fresh"]
                calls.append("execution")
                return await super().require_active_producer(*args, **kwargs)

        state.run_store._execution_repository = Custom(source, namespace="model-control", tenant_id="tenant")
        for _ in range(2):
            await state.run_store.require_active_producer("run")
        assert calls == ["execution", "execution"]


@pytest.mark.asyncio
@pytest.mark.parametrize("takeover", (False, True), ids=("cancel", "takeover"))
async def test_next_unscoped_dispatch_rejects_remote_change_without_cache_delay(
    tmp_path: Path, takeover: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_write_cache, "time", lambda: 100.0)
    async with _state("sqlite", tmp_path, InMemoryCacheBackend(max_bytes=1000000)) as first:
        execution = await _seed(first)
        repository = first.execution.executions
        assert await repository.get("execution", tenant_id="tenant") == execution
        await first.run_store.require_active_producer("run")
        async with _state("sqlite", tmp_path, InMemoryCacheBackend(max_bytes=1000000)) as second:
            changed = await _advance_execution(second.execution.executions, takeover=takeover)
            assert await repository.get("execution", tenant_id="tenant") == execution
            with pytest.raises(AIError) as raised:
                await first.run_store.require_active_producer("run")
            assert raised.value.code is (ErrorCode.STORAGE_CONFLICT if takeover else ErrorCode.EXECUTION_CANCELLED)
            observed = await repository.get_observed("execution", tenant_id="tenant")
            assert observed.value == changed


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("changed", (False, True), ids=("dispatch-fence", "publish-boundary"))
async def test_confirmed_execution_receipts_advance_without_source_reads(
    tmp_path: Path, backend: str, changed: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy import event

    async with _state(backend, tmp_path, InMemoryCacheBackend(max_bytes=1000000)) as state:
        await _seed(state)
        repository = state.execution.executions
        source = repository.state_store
        observed = await repository.get_observed("execution", tenant_id="tenant")
        statements = []

        async def forbidden_read(callback):
            raise AssertionError("confirmed boundary receipts do not need a preliminary source read")

        def query(connection, cursor, statement, parameters, context, executemany):
            statements.append(statement.lstrip().split()[0].upper())

        engine = source.storage_group.context.engine.sync_engine if backend == "sqlite" else None
        if engine is not None:
            event.listen(engine, "before_cursor_execute", query)
        try:
            with monkeypatch.context() as patch:
                patch.setattr(source, "read", forbidden_read)
                for _ in range(2):
                    previous = observed
                    target = replace(previous.value, revision=previous.value.revision + 1) if changed else previous.value
                    observed = await source.mutate(lambda tx: repository.commit_boundary_in_transaction(
                        tx, previous, next_record=target, expected_producer_generation=1,
                        expected_producer_claim_id="owner-claim",
                    ))
                    assert observed.value == target
                    assert observed.storage_revision == previous.storage_revision + 1
        finally:
            if engine is not None:
                event.remove(engine, "before_cursor_execute", query)
        assert await repository.get_observed("execution", tenant_id="tenant") == observed
        if backend == "sqlite":
            assert statements.count("SELECT") == 0
            assert statements.count("UPDATE") == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("takeover", (False, True), ids=("cancel", "takeover"))
@pytest.mark.parametrize("changed", (False, True), ids=("dispatch-fence", "publish-boundary"))
async def test_stale_boundary_receipt_rejects_remote_control_change(
    tmp_path: Path, takeover: bool, changed: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_write_cache, "time", lambda: 100.0)
    async with _state("sqlite", tmp_path, InMemoryCacheBackend(max_bytes=1000000)) as first:
        current = await _seed(first)
        repository = first.execution.executions
        observed = await repository.get_observed("execution", tenant_id="tenant")
        assert await repository.get("execution", tenant_id="tenant") == current
        async with _state("sqlite", tmp_path, InMemoryCacheBackend(max_bytes=1000000)) as second:
            changed_execution = await _advance_execution(second.execution.executions, takeover=takeover)
            fresh = await second.execution.executions.get_observed("execution", tenant_id="tenant")
            assert await repository.get("execution", tenant_id="tenant") == current
            target = replace(current, revision=current.revision + 1) if changed else current
            with pytest.raises(AIError) as raised:
                await repository.state_store.mutate(lambda tx: repository.commit_boundary_in_transaction(
                    tx, observed, next_record=target, expected_producer_generation=1,
                    expected_producer_claim_id="owner-claim",
                ))
            assert raised.value.code is ErrorCode.STORAGE_CONFLICT
            assert await repository.get_observed("execution", tenant_id="tenant") == fresh
            assert fresh.value == changed_execution


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("status", (ExecutionStatus.CANCELLING, ExecutionStatus.RECOVERY_REQUIRED))
async def test_late_observation_tail_does_not_restore_revoked_dispatch(
    tmp_path: Path, backend: str, status: ExecutionStatus,
) -> None:
    async with _state(backend, tmp_path, InMemoryCacheBackend(max_bytes=1000000)) as state:
        current = await _seed(state)
        repository = state.execution.executions
        revoked = await repository.compare_and_swap(
            "execution", tenant_id="tenant", expected_revision=current.revision,
            next_record=replace(current, revision=current.revision + 1, status=status),
        )
        tail = AgentRunCheckpoint("run", 1, [], state="interrupted")
        await state.run_store.save_checkpoint(tail, execution_id="execution", producer_generation=1)
        assert await state.run_store.latest_checkpoint(agent_run_id="run", include_interrupted=True) == tail
        assert await repository.get_fresh("execution", tenant_id="tenant") == revoked
        with pytest.raises(AIError) as raised:
            await state.run_store.require_active_producer("run")
        assert raised.value.code is (ErrorCode.EXECUTION_CANCELLED if status is ExecutionStatus.CANCELLING
                                     else ErrorCode.STORAGE_CONFLICT)


@pytest.mark.asyncio
@pytest.mark.parametrize("guard", ("execution", "generation", "unbound", "sealed"))
async def test_observation_staging_preserves_exact_local_owner_and_freeze(
    tmp_path: Path, guard: str,
) -> None:
    async with _state("memory", tmp_path) as state:
        await _seed(state)
        store = state.run_store
        execution_id, generation = "execution", 1
        if guard == "execution":
            execution_id = "foreign"
        elif guard == "generation":
            store.bind_execution_producer("run", execution_id="execution", producer_generation=2,
                                         restart_checkpoints=False, durable_trace=False)
        elif guard == "unbound":
            store._execution_producers.pop("run")
        else:
            store._terminal_seals["run"] = object()
        try:
            with pytest.raises(AIError) as raised:
                await store.save_checkpoint(AgentRunCheckpoint("run", 1, []),
                                            execution_id=execution_id, producer_generation=generation)
            assert raised.value.code is ErrorCode.STORAGE_CONFLICT
            assert await store.latest_checkpoint(agent_run_id="run", include_interrupted=True) is None
        finally:
            store._terminal_seals.pop("run", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("delegated", (False, True), ids=("agent", "delegated-agent"))
@pytest.mark.parametrize("native_restart", (False, True))
async def test_task_owned_model_boundaries_keep_source_or_receipt_authority(
    tmp_path: Path, delegated: bool, native_restart: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from .test_task_agent_persistence import test_task_agent_trace_survives_terminal_release_and_reopen
    original = PersistenceStorage.sqlite
    checks = _forbid_cached_dispatch_authority(monkeypatch)

    def sqlite(*args, **kwargs):
        kwargs["cache"] = InMemoryCacheBackend(max_bytes=1000000)
        return original(*args, **kwargs)

    monkeypatch.setattr(PersistenceStorage, "sqlite", sqlite)
    await test_task_agent_trace_survives_terminal_release_and_reopen(tmp_path, delegated, native_restart)
    assert len(set(checks)) >= (2 if delegated else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", (False, True))
@pytest.mark.parametrize("trace", (False, True))
async def test_transient_route_uses_frozen_restart_guarantee(
    restart: bool, trace: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.persistence import PersistencePlan
    from linktools.ai.storage import StorageRoute

    state = Persistence(PersistenceStorage.from_plan(
        PersistencePlan(execution=StorageRoute.transient(cache=InMemoryCacheBackend(max_bytes=1000000))),
    ))
    await state.initialize(namespace="model-control", tenant_id="tenant")
    try:
        repository = state.execution.executions
        execution = replace(_execution(), revision=1, producer_generation=1, producer_claim_id="owner-claim",
                            restart_checkpoints=restart, durable_trace=trace)
        await repository.create_execution(execution)
        state.run_store.bind_execution_producer(
            "run", execution_id="execution", producer_generation=1,
            restart_checkpoints=restart, durable_trace=trace,
        )
        await state.run_store.register_agent_run(AgentRunRecord("run"), execution_id="execution")
        original = repository.require_active_producer
        observations = 0

        async def observe(*args, **kwargs):
            nonlocal observations
            observations += 1
            assert kwargs["fresh"]
            return await original(*args, **kwargs)

        monkeypatch.setattr(repository, "require_active_producer", observe)
        for _ in range(2):
            await state.run_store.require_active_producer("run")
        assert observations == 2
        assert state.run_store.execution_persistence("run") == (True, restart, trace)
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("delegated", (False, True), ids=("agent", "delegated-agent"))
async def test_volatile_task_and_descendant_keep_strong_model_boundaries(
    delegated: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
    from linktools.ai.runtime import RuntimeFeatures
    from linktools.ai.core import TaskStatus
    from linktools.ai.inputs import AgentTaskInput
    from linktools.ai.runtime import Runtime
    from linktools.ai.persistence import PersistencePlan
    from linktools.ai.storage import StorageRoute
    from linktools.ai.task import TaskGraph, TaskNode
    from ._runtime_test_helpers import _Models, _UsageFunctionModel

    checks = _forbid_cached_dispatch_authority(monkeypatch)

    async def model(messages, info):
        has_return = any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)
        if delegated and not has_return and "delegate_task" in {tool.name for tool in info.function_tools}:
            return ModelResponse(parts=[ToolCallPart(
                "delegate_task", {"subagent_id": "child", "task": "child answer"}, tool_call_id="child",
            )])
        return ModelResponse(parts=[TextPart("answer")])

    group = RuntimeFeatures("application")
    group.agent("default", model="default", allow_tools=(), allow_subagents=("child",) if delegated else ())
    group.agent("child", model="default", allow_tools=(), allow_subagents=())
    storage = PersistenceStorage.from_plan(
        PersistencePlan(execution=StorageRoute.transient(cache=InMemoryCacheBackend(max_bytes=1000000))),
    )
    async with Persistence.open('transient-task-control', storage=storage) as persistence_417, Runtime.attach(persistence=persistence_417, features=model_features(_Models(_UsageFunctionModel(model)), group)) as runtime, TaskEngine.open(persistence=persistence_417) as engine:
        task = Task.from_agent("answer", runtime.agents.get())
        definition = TaskGraph("graph", (TaskNode("answer", task=task, input=AgentTaskInput("question")),))
        graph = await engine.with_definitions(task).start(definition, idempotency_key="once")
        assert (await graph.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
        assert len(set(checks)) >= (2 if delegated else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_boundary", (False, True), ids=("output-boundary", "tool-boundary"))
async def test_remote_cancel_is_seen_before_next_output_or_tool_boundary(
    tmp_path: Path, tool_boundary: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic import BaseModel
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from linktools.ai.capability import AgentContext
    from linktools.ai.runtime import RuntimeFeatures
    from linktools.ai.runtime import Runtime
    from linktools.ai.execution import CancelExecutionRequest
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
    async with Persistence.open('model-control-cancel', storage=PersistenceStorage.sqlite(database, cache=InMemoryCacheBackend(max_bytes=1000000))) as persistence_466, Runtime.attach(persistence=persistence_466, features=model_features(models, group)) as owner, Persistence.open('model-control-cancel', storage=PersistenceStorage.sqlite(database, cache=InMemoryCacheBackend(max_bytes=1000000))) as persistence_469, Runtime.attach(persistence=persistence_469, features=model_features(models, group)) as other:
        execution = await owner.agents.get().start("question", output=Answer, idempotency_key="once")
        try:
            await asyncio.wait_for(entered.wait(), 5)
            await other.executions.cancel(
                execution.execution_id, CancelExecutionRequest(owner.default_principal, "remote-cancel"),
            )
            release.set()
            result = (await execution.wait(timeout_seconds=5)).result
            assert result.status is ExecutionStatus.CANCELLED
            assert calls == [1]
            assert effects == []
        finally:
            release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_restart_producer_batches_control_without_using_cached_authority(
    tmp_path: Path, backend: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy import event
    from sqlalchemy.engine import Engine

    cache = InMemoryCacheBackend(max_bytes=1000000)
    async with _state(backend, tmp_path, cache) as state:
        await _seed(state, restart=True)
        repository = state.execution.executions
        await repository.get("execution", tenant_id="tenant")
        selects = []

        cache_reads = []
        original_get = cache.get

        async def cache_read(key: bytes) -> bytes | None:
            cache_reads.append(key)
            return await original_get(key)

        def query(connection, cursor, statement, parameters, context, executemany):
            if statement.lstrip().upper().startswith("SELECT"):
                selects.append(statement)

        monkeypatch.setattr(cache, "get", cache_read)
        event.listen(Engine, "before_cursor_execute", query)
        try:
            await state.run_store.require_active_producer("run")
        finally:
            event.remove(Engine, "before_cursor_execute", query)
        assert cache_reads == []
        if backend == "sqlite":
            assert len(selects) == 1
        await _advance_execution(repository)
        with pytest.raises(AIError) as raised:
            await state.run_store.require_active_producer("run")
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        assert cache_reads == []
