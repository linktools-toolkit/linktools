#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Observation batching preserves authoritative barriers and bounded pending work."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart

from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.runtime import RuntimeDomain
from linktools.ai.runtime.state import RuntimeState, _steps, _trace_persistence
from linktools.ai.runtime.state._step_contracts import AgentRunCheckpoint, AgentRunRecord, StepEvent
from linktools.ai.runtime._transcript_staging import StagedTranscript

from .test_incremental_projection_boundaries import _storage
from .test_step_projection import _execution


_DEFAULT_CACHE = object()


async def _setup(tmp_path: Path, backend: str, *, cache=_DEFAULT_CACHE):
    state = RuntimeState(_storage(backend, tmp_path))
    if cache is _DEFAULT_CACHE:
        cache = InMemoryCacheBackend(max_bytes=65536)
    await state.initialize(
        namespace="observation-batches", tenant_id="tenant",
        module_routes={RuntimeDomain.TRACE: state.plan.route(RuntimeDomain.TRACE).with_cache(cache)},
    )
    execution = replace(_execution(), revision=1)
    repository = state.execution.executions
    await repository.create_with_history_head(execution)
    await repository.state_store.mutate(
        lambda transaction: repository.admit_history_producer_in_transaction(transaction, execution)
    )
    state.run_store.trace_persistence.configure(
        "run",
        batch_size=_trace_persistence.DEFAULT_TRACE_BATCH_SIZE,
        flush_interval=_trace_persistence.DEFAULT_TRACE_FLUSH_INTERVAL,
    )
    state.run_store.bind_execution_producer("run", execution_id="execution", producer_generation=1)
    await state.run_store.register_agent_run(AgentRunRecord("run"), execution_id="execution")
    return state


async def _observe(store, index: int, *, run_id: str = "run", execution_id: str = "execution") -> None:
    await store.append_event(StepEvent(run_id, "TOOL_CALL_STARTED", index, tool_call_id=str(index), tool_name="work"))
    await store.flush_execution_projection(run_id, execution_id=execution_id, producer_generation=1, deferred=True)


async def _add_run(state, run_id: str, execution_id: str) -> None:
    execution = replace(_execution(), execution_id=execution_id, root_execution_id=execution_id, revision=1)
    repository = state.execution.executions
    await repository.create_with_history_head(execution)
    await repository.state_store.mutate(
        lambda transaction: repository.admit_history_producer_in_transaction(transaction, execution)
    )
    state.run_store.trace_persistence.configure(
        run_id,
        batch_size=_trace_persistence.DEFAULT_TRACE_BATCH_SIZE,
        flush_interval=_trace_persistence.DEFAULT_TRACE_FLUSH_INTERVAL,
    )
    state.run_store.bind_execution_producer(run_id, execution_id=execution_id, producer_generation=1)
    await state.run_store.register_agent_run(AgentRunRecord(run_id), execution_id=execution_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("barrier", ("checkpoint", "close"))
async def test_recovery_checkpoint_keeps_observations_queued_until_public_drain(
    tmp_path: Path, backend: str, barrier: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_trace_persistence, "DEFAULT_TRACE_FLUSH_INTERVAL", 60.0)
    state = await _setup(tmp_path, backend)
    try:
        store = state.run_store
        repository = state.execution.executions
        before = await repository.get_history_head("execution", tenant_id="tenant")
        archive = store.read_store(RuntimeDomain.TRACE)
        for index in range(8):
            await _observe(store, index)
        assert await archive.list_events(agent_run_id="run") == []
        message = ModelRequest(parts=[UserPromptPart("one shared boundary")])
        if barrier == "checkpoint":
            store.stage_transcript("run", StagedTranscript((message,)))
            await store.save_checkpoint(
                AgentRunCheckpoint("run", 1, [message], transcript_message_count_before=0),
                execution_id="execution", producer_generation=1,
            )
            saved = await store.read_store(RuntimeDomain.CHECKPOINT).latest_checkpoint(agent_run_id="run")
            assert saved is not None and saved.messages == [message]
            assert await archive.latest_checkpoint(agent_run_id="run") is None
            assert [item async for item in archive.iter_messages(agent_run_id="run")] == []
            assert await archive.list_events(agent_run_id="run") == []
            assert await repository.get_history_head("execution", tenant_id="tenant") == before
        else:
            await store.preflight_close()
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        if barrier == "checkpoint":
            saved = await archive.latest_checkpoint(agent_run_id="run")
            assert saved is not None and saved.messages == [message]
        after = await repository.get_history_head("execution", tenant_id="tenant")
        assert after.revision == before.revision + 1
        assert len(await archive.list_events(agent_run_id="run")) == 8
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert await repository.get_history_head("execution", tenant_id="tenant") == after
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_boundary_pressure_flushes_a_single_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = await _setup(tmp_path, "memory")
    try:
        archive = state.run_store.read_store(RuntimeDomain.TRACE)
        published = asyncio.Event()
        original = archive.sync_prepared_projection

        async def watched(*args, **kwargs):
            await original(*args, **kwargs)
            published.set()

        monkeypatch.setattr(archive, "sync_prepared_projection", watched)
        for index in range(_trace_persistence.DEFAULT_TRACE_BATCH_SIZE - 1):
            await _observe(state.run_store, index)
        assert await archive.list_events(agent_run_id="run") == []
        await _observe(state.run_store, _trace_persistence.DEFAULT_TRACE_BATCH_SIZE - 1)
        await asyncio.wait_for(published.wait(), 2)
        assert len(await archive.list_events(agent_run_id="run")) == _trace_persistence.DEFAULT_TRACE_BATCH_SIZE
        await state.run_store.preflight_close()
        assert not state.run_store._background_tasks
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_staging_during_scheduled_commit_gets_another_bounded_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_trace_persistence, "DEFAULT_TRACE_FLUSH_INTERVAL", 0.01)
    state = await _setup(tmp_path, "memory")
    entered = asyncio.Event()
    release = asyncio.Event()
    second = asyncio.Event()
    archive = state.run_store.read_store(RuntimeDomain.TRACE)
    original = archive.sync_prepared_projection
    count = 0

    async def held(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 1:
            entered.set()
            await release.wait()
        await original(*args, **kwargs)
        if count == 2:
            second.set()

    monkeypatch.setattr(archive, "sync_prepared_projection", held)
    try:
        await _observe(state.run_store, 1)
        await asyncio.wait_for(entered.wait(), 2)
        await _observe(state.run_store, 2)
        assert state.run_store.trace_persistence.offsets["run"].observation_task is not None
        assert state.run_store.trace_persistence._buffers["run"].scope.status().pending > 0
        release.set()
        await asyncio.wait_for(second.wait(), 2)
        await state.run_store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert [item.tool_call_id for item in await archive.list_events(agent_run_id="run")] == ["1", "2"]
        assert count == 2
    finally:
        release.set()
        await state.close()


@pytest.mark.asyncio
async def test_cancelling_drain_settles_active_observation_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_trace_persistence, "DEFAULT_TRACE_FLUSH_INTERVAL", 0.01)
    state = await _setup(tmp_path, "memory")
    entered = asyncio.Event()
    release = asyncio.Event()
    archive = state.run_store.read_store(RuntimeDomain.TRACE)
    original = archive.sync_prepared_projection
    task = None

    async def held(*args, **kwargs):
        await original(*args, **kwargs)
        entered.set()
        await release.wait()

    monkeypatch.setattr(archive, "sync_prepared_projection", held)
    try:
        await _observe(state.run_store, 1)
        await asyncio.wait_for(entered.wait(), 2)
        task = asyncio.create_task(state.run_store.flush_execution_projection(
            "run", execution_id="execution", producer_generation=1,
        ))
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(await archive.list_events(agent_run_id="run")) == 1
        assert not state.run_store._durability_flights
    finally:
        release.set()
        if task is not None and not task.done():
            await asyncio.gather(task, return_exceptions=True)
        await state.close()


@pytest.mark.asyncio
async def test_ongoing_observations_do_not_reset_first_dirty_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_trace_persistence, "DEFAULT_TRACE_FLUSH_INTERVAL", 0.05)
    state = await _setup(tmp_path, "memory")
    archive = state.run_store.read_store(RuntimeDomain.TRACE)
    original = archive.sync_prepared_projection
    committed = asyncio.Event()
    stop = asyncio.Event()

    async def watched(*args, **kwargs):
        await original(*args, **kwargs)
        committed.set()

    async def produce():
        index = 0
        while not stop.is_set():
            await _observe(state.run_store, index)
            index += 1
            await asyncio.sleep(0.015)

    monkeypatch.setattr(archive, "sync_prepared_projection", watched)
    producer = asyncio.create_task(produce())
    try:
        await asyncio.wait_for(committed.wait(), 1)
        assert not producer.done()
        assert 1 <= len(await archive.list_events(agent_run_id="run")) < _trace_persistence.DEFAULT_TRACE_BATCH_SIZE
    finally:
        stop.set()
        await producer
        await state.close()


@pytest.mark.asyncio
async def test_transient_observations_do_not_schedule_durability_tasks() -> None:
    from linktools.ai.storage import StorageRetentionMode
    from linktools.ai.runtime.state._step_archive import InMemoryStepArchive, StagingAgentRunStore

    store = _steps.RuntimeAgentRunStore(
        StagingAgentRunStore(),
        conversation_archive=InMemoryStepArchive(RuntimeDomain.CONVERSATION),
        execution_archive=None, recovery_archive=None,
        conversation_retention=StorageRetentionMode.VOLATILE,
        execution_retention=StorageRetentionMode.VOLATILE,
        recovery_retention=StorageRetentionMode.VOLATILE,
    )
    await store.initialize()
    try:
        await store.register_agent_run(AgentRunRecord("run"))
        for index in range(_trace_persistence.DEFAULT_TRACE_BATCH_SIZE * 2):
            await _observe(store, index)
        assert not store._background_tasks
        assert len(await store.list_events(agent_run_id="run")) == _trace_persistence.DEFAULT_TRACE_BATCH_SIZE * 2
    finally:
        await store.preflight_close()
        await store.close()


@pytest.mark.asyncio
async def test_recovery_checkpoint_does_not_force_a_public_filesystem_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.storage import FilesystemJournal

    monkeypatch.setattr(_trace_persistence, "DEFAULT_TRACE_FLUSH_INTERVAL", 60.0)
    state = await _setup(tmp_path, "filesystem")
    commits = 0
    original = FilesystemJournal.stage

    def staged(self, *args, **kwargs):
        nonlocal commits
        commits += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(FilesystemJournal, "stage", staged)
    try:
        message = ModelRequest(parts=[UserPromptPart("one physical publication")])
        store = state.run_store
        store.stage_transcript("run", StagedTranscript((message,)))
        await _observe(store, 1)
        await store.save_checkpoint(
            AgentRunCheckpoint("run", 1, [message], transcript_message_count_before=0),
            execution_id="execution", producer_generation=1,
        )
        assert commits == 1
        saved = await store.read_store(RuntimeDomain.CHECKPOINT).latest_checkpoint(agent_run_id="run")
        assert saved is not None and saved.messages == [message]
        public = store.read_store(RuntimeDomain.TRACE)
        assert await public.latest_checkpoint(agent_run_id="run") is None
        assert [item async for item in public.iter_messages(agent_run_id="run")] == []
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert commits == 2
        saved = await public.latest_checkpoint(agent_run_id="run")
        assert saved is not None and saved.messages == [message]
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert commits == 2
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_run_buffers_publish_independently_and_remain_idle_after_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    monkeypatch.setattr(_trace_persistence, "DEFAULT_TRACE_FLUSH_INTERVAL", 0.05)
    state = await _setup(tmp_path, backend)
    await _add_run(state, "other-run", "other-execution")
    archive = state.run_store.read_store(RuntimeDomain.TRACE)
    observed = set()
    committed = asyncio.Event()
    original = archive.sync_prepared_projection

    async def watched(run, **kwargs):
        await original(run, **kwargs)
        observed.add(run.agent_run_id)
        if len(observed) == 2:
            committed.set()

    monkeypatch.setattr(archive, "sync_prepared_projection", watched)
    try:
        await _observe(state.run_store, 1)
        scope = state.run_store.trace_persistence._buffers["run"].scope
        await _observe(state.run_store, 2, run_id="other-run", execution_id="other-execution")
        assert state.run_store.trace_persistence._buffers["other-run"].scope is not scope
        await asyncio.wait_for(committed.wait(), 5)
        await state.run_store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        await state.run_store.flush_execution_projection("other-run", execution_id="other-execution", producer_generation=1)
        assert [event.tool_call_id for event in await archive.list_events(agent_run_id="run")] == ["1"]
        assert [event.tool_call_id for event in await archive.list_events(agent_run_id="other-run")] == ["2"]
        before = await state.execution.executions.get_history_head("execution", tenant_id="tenant")
        await asyncio.sleep(0.12)
        assert await state.execution.executions.get_history_head("execution", tenant_id="tenant") == before
        assert not state.run_store._background_tasks
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_slow_run_does_not_hold_another_runs_flush_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_trace_persistence, "DEFAULT_TRACE_FLUSH_INTERVAL", 0.01)
    state = await _setup(tmp_path, "memory")
    await _add_run(state, "other-run", "other-execution")
    archive = state.run_store.read_store(RuntimeDomain.TRACE)
    entered = asyncio.Event()
    release = asyncio.Event()
    completed = asyncio.Event()
    original = state.run_store.commit_captured_execution_projection

    async def held(captured, flight, **kwargs):
        if captured.run.agent_run_id == "run":
            entered.set()
            await release.wait()
        await original(captured, flight, **kwargs)
        if captured.run.agent_run_id == "other-run":
            completed.set()

    monkeypatch.setattr(state.run_store, "commit_captured_execution_projection", held)
    try:
        await _observe(state.run_store, 1)
        await asyncio.wait_for(entered.wait(), 2)
        await _observe(state.run_store, 2, run_id="other-run", execution_id="other-execution")
        await asyncio.wait_for(completed.wait(), 2)
        await asyncio.wait_for(state.run_store.flush_execution_projection(
            "other-run", execution_id="other-execution", producer_generation=1,
        ), 2)
        assert not release.is_set()
        assert [event.tool_call_id for event in await archive.list_events(agent_run_id="other-run")] == ["2"]
        assert await archive.list_events(agent_run_id="run") == []
    finally:
        release.set()
        await state.close()


@pytest.mark.asyncio
async def test_failed_scheduled_run_does_not_stop_siblings_or_its_required_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_trace_persistence, "DEFAULT_TRACE_FLUSH_INTERVAL", 0.01)
    state = await _setup(tmp_path, "memory")
    await _add_run(state, "other-run", "other-execution")
    archive = state.run_store.read_store(RuntimeDomain.TRACE)
    failed = asyncio.Event()
    completed = asyncio.Event()
    original = archive.sync_prepared_projection

    async def fail_once(run, **kwargs):
        if run.agent_run_id == "run" and not failed.is_set():
            failed.set()
            raise RuntimeError("publication unavailable before commit")
        await original(run, **kwargs)
        if run.agent_run_id == "other-run":
            completed.set()

    monkeypatch.setattr(archive, "sync_prepared_projection", fail_once)
    try:
        await _observe(state.run_store, 1)
        await _observe(state.run_store, 2, run_id="other-run", execution_id="other-execution")
        await asyncio.wait_for(failed.wait(), 2)
        await asyncio.wait_for(completed.wait(), 2)
        await state.run_store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert [event.tool_call_id for event in await archive.list_events(agent_run_id="run")] == ["1"]
        assert [event.tool_call_id for event in await archive.list_events(agent_run_id="other-run")] == ["2"]
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_runtime_close_drains_all_queued_runs_before_buffer_deadlines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_trace_persistence, "DEFAULT_TRACE_FLUSH_INTERVAL", 60.0)
    state = await _setup(tmp_path, "memory")
    await _add_run(state, "other-run", "other-execution")
    archive = state.run_store.read_store(RuntimeDomain.TRACE)
    try:
        await _observe(state.run_store, 1)
        await _observe(state.run_store, 2, run_id="other-run", execution_id="other-execution")
        await asyncio.wait_for(state.run_store.preflight_close(), 5)
        assert [event.tool_call_id for event in await archive.list_events(agent_run_id="run")] == ["1"]
        assert [event.tool_call_id for event in await archive.list_events(agent_run_id="other-run")] == ["2"]
        assert not state.run_store._background_tasks
        assert not state.run_store._durability_flights
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ("release", "rebind", "pressure"))
async def test_buffer_callback_settles_before_release_rebind_or_pressure_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    monkeypatch.setattr(_trace_persistence, "DEFAULT_TRACE_FLUSH_INTERVAL", 0.01)
    state = await _setup(tmp_path, "memory")
    store = state.run_store
    archive = store.read_store(RuntimeDomain.TRACE)
    entered = asyncio.Event()
    release = asyncio.Event()
    original = archive.sync_prepared_projection

    async def held(*args, **kwargs):
        if not entered.is_set():
            entered.set()
            await release.wait()
        await original(*args, **kwargs)

    monkeypatch.setattr(archive, "sync_prepared_projection", held)
    barrier = None
    try:
        await _observe(store, 1)
        await asyncio.wait_for(entered.wait(), 2)
        if change == "pressure":
            for index in range(2, _trace_persistence.DEFAULT_TRACE_BATCH_SIZE + 1):
                await _observe(store, index)
        barrier = asyncio.create_task(store.flush_execution_projection(
            "run", execution_id="execution", producer_generation=1,
        ))
        await asyncio.sleep(0)
        assert not barrier.done()
        release.set()
        await asyncio.wait_for(barrier, 2)
        if change != "pressure":
            await store.release_staging_many(candidate_agent_run_ids=("run",), execution_id="execution")
            if change == "rebind":
                repository = state.execution.executions
                await repository.state_store.mutate(
                    lambda tx: repository.admit_history_producer_in_transaction(tx, replace(_execution(), revision=2))
                )
                store.trace_persistence.configure("run", flush_interval=60.0)
                store.bind_execution_producer("run", execution_id="execution", producer_generation=2)
                run = await archive.get_agent_run(agent_run_id="run")
                assert run is not None
                await store.register_agent_run(run, execution_id="execution")
                await store.append_event(StepEvent("run", "TOOL_CALL_STARTED", 2, tool_call_id="2", tool_name="work"))
                await store.flush_execution_projection(
                    "run", execution_id="execution", producer_generation=2, deferred=True,
                )
        await store.preflight_close()
        expected = (
            [str(index) for index in range(1, _trace_persistence.DEFAULT_TRACE_BATCH_SIZE + 1)]
            if change == "pressure" else ["1", "2"] if change == "rebind" else ["1"]
        )
        assert [event.tool_call_id for event in await archive.list_events(agent_run_id="run")] == expected
        assert not store.trace_persistence.dirty
        assert not store._durability_flights
        before = await state.execution.executions.get_history_head("execution", tenant_id="tenant")
        await asyncio.sleep(0.03)
        assert await state.execution.executions.get_history_head("execution", tenant_id="tenant") == before
    finally:
        release.set()
        if barrier is not None:
            await asyncio.gather(barrier, return_exceptions=True)
        await state.close()
