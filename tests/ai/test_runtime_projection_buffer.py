#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Observation batching requires a cache with an actual write buffer."""

from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.runtime import RuntimeDomain, RuntimeStorage
from linktools.ai.runtime.state._step_contracts import AgentRunRecord, StepEvent

from .test_step_projection import _execution


async def _setup(tmp_path: Path, backend: str, cache: object | None):
    if backend == "memory":
        state = RuntimeStorage.in_memory(cache=cache)
    elif backend == "filesystem":
        state = RuntimeStorage.filesystem(tmp_path / "state", cache=cache)
    else:
        state = RuntimeStorage.sqlite(tmp_path / "state.db", cache=cache)
    await state.initialize(namespace="projection-buffer", tenant_id="tenant")
    execution = replace(_execution(), revision=1)
    repository = state.execution.executions
    await repository.create_with_history_head(execution)
    await repository.state_store.mutate(
        lambda tx: repository.admit_history_producer_in_transaction(tx, execution)
    )
    state.run_store.trace_persistence.configure("run", cache=cache, flush_interval=60.0)
    state.run_store.bind_execution_producer("run", execution_id="execution", producer_generation=1)
    await state.run_store.register_agent_run(AgentRunRecord("run"), execution_id="execution")
    return state


async def _observe(state, index: int = 1) -> None:
    await state.run_store.append_event(StepEvent("run", "TOOL_CALL_STARTED", index, tool_call_id=str(index)))
    await state.run_store.flush_execution_projection(
        "run", execution_id="execution", producer_generation=1, deferred=True,
    )


class _HotContentCache:
    """Only the optional hot-content capability, with no queue operations."""

    def __init__(self) -> None:
        self.entries: dict[str, bytes] = {}

    async def get(self, key: str) -> bytes | None:
        return self.entries.get(key)

    async def put(self, key: str, content: bytes) -> None:
        self.entries[key] = content

    async def delete(self, key: str) -> None:
        self.entries.pop(key, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("hot_cache", (False, True))
async def test_observations_without_write_buffer_persist_at_each_boundary(
    tmp_path: Path, backend: str, hot_cache: bool,
) -> None:
    state = await _setup(tmp_path, backend, _HotContentCache() if hot_cache else None)
    try:
        archive = state.run_store.read_store(RuntimeDomain.TRACE)
        await _observe(state)
        assert len(await archive.list_events(agent_run_id="run")) == 1
        await _observe(state, 2)
        assert len(await archive.list_events(agent_run_id="run")) == 2
        await state.run_store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert len(await archive.list_events(agent_run_id="run")) == 2
    finally:
        await state.close()


def test_observation_policy_keeps_approved_window_and_threshold() -> None:
    from linktools.ai.runtime.state import _trace_persistence

    assert _trace_persistence.DEFAULT_TRACE_FLUSH_INTERVAL == 3.0
    assert _trace_persistence.DEFAULT_TRACE_BATCH_SIZE == 128


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_required_drain_repeated_cancellation_waits_for_source_confirmation(
    tmp_path: Path, backend: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    state = await _setup(tmp_path, backend, InMemoryCacheBackend(max_bytes=65536))
    archive = state.run_store.read_store(RuntimeDomain.TRACE)
    original = archive.sync_prepared_projection
    entered, release = asyncio.Event(), asyncio.Event()

    async def held(*args, **kwargs):
        await original(*args, **kwargs)
        entered.set()
        await release.wait()

    monkeypatch.setattr(archive, "sync_prepared_projection", held)
    await _observe(state)
    task = asyncio.create_task(state.run_store.flush_execution_projection(
        "run", execution_id="execution", producer_generation=1,
    ))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
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
        await asyncio.gather(task, return_exceptions=True)
        await state.close()
