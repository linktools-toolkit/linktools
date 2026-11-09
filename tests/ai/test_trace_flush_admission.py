#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A running Trace boundary cannot reject a concurrent, valid admission."""

import asyncio
from pathlib import Path

import pytest

from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence._step_contracts import StepEvent
from linktools.ai.storage import InMemoryCacheBackend, WriteBufferEntry

from .test_observation_coalescing import _observe, _setup


class _HeldReadBuffer(InMemoryCacheBackend):
    def __init__(self) -> None:
        super().__init__(max_bytes=8 * 1024 * 1024)
        self.read_started = asyncio.Event()
        self.resume_read = asyncio.Event()
        self.hold_read = False

    async def read_batch(self, scope: str, *, limit: int) -> tuple[WriteBufferEntry, ...]:
        if self.hold_read:
            self.read_started.set()
            await self.resume_read.wait()
        return await super().read_batch(scope, limit=limit)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_admission", [False, True])
async def test_flush_settles_before_concurrent_admission(
    tmp_path: Path, cancel_admission: bool,
) -> None:
    buffer = _HeldReadBuffer()
    state = await _setup(tmp_path, "sqlite", cache=buffer)
    store = state.run_store
    tasks = []
    try:
        store.trace_persistence.configure("run", buffer=buffer, batch_size=128, flush_interval=60)
        await _observe(store, 1)
        buffer.hold_read = True
        flush = asyncio.create_task(store.flush_execution_projection(
            "run", execution_id="execution", producer_generation=1,
        ))
        tasks.append(flush)
        await asyncio.wait_for(buffer.read_started.wait(), 2)
        staged = asyncio.Event()

        async def admit() -> None:
            await store.append_event(StepEvent("run", "TOOL_CALL_STARTED", 2, tool_call_id="2", tool_name="work"))
            staged.set()
            await store.flush_execution_projection(
                "run", execution_id="execution", producer_generation=1, deferred=True,
            )

        admission = asyncio.create_task(admit())
        tasks.append(admission)
        await asyncio.wait_for(staged.wait(), 2)
        if cancel_admission:
            admission.cancel()
        buffer.resume_read.set()
        await asyncio.wait_for(flush, 5)
        if cancel_admission:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(admission, 5)
        else:
            await asyncio.wait_for(admission, 5)
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        archive = store.read_store(PersistenceDomain.TRACE)
        assert [event.tool_call_id for event in await archive.list_events(agent_run_id="run")] == ["1", "2"]
        await _observe(store, 3)
        assert [event.tool_call_id for event in await archive.list_events(agent_run_id="run")] == ["1", "2"]
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert [event.tool_call_id for event in await archive.list_events(agent_run_id="run")] == ["1", "2", "3"]
    finally:
        buffer.resume_read.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await state.close()
