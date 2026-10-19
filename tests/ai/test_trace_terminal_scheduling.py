#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Terminal observations retain facts until an authoritative publication boundary."""

import asyncio
from pathlib import Path

import pytest

from linktools.ai.capability import TraceCapability
from linktools.ai.core import CommitPolicy
from linktools.ai.runtime.state import RuntimeDomain

from .test_observation_coalescing import _observe, _setup


def test_trace_accepts_terminal_policy() -> None:
    assert TraceCapability(commit=CommitPolicy.terminal()).commit == CommitPolicy.terminal()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_terminal_trace_retains_staging_until_explicit_flush(tmp_path: Path, backend: str) -> None:
    state = await _setup(tmp_path, backend)
    try:
        store = state.run_store
        store.trace_persistence.configure("run", terminal=True)
        archive = store.read_store(RuntimeDomain.TRACE)
        for index in range(1, 4):
            await _observe(store, index)
        assert await archive.list_events(agent_run_id="run") == []
        assert "run" in store.trace_persistence.dirty
        assert store.trace_persistence._buffers == {}
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert len(await archive.list_events(agent_run_id="run")) == 3
        assert "run" not in store.trace_persistence.dirty
        await _observe(store, 4)
        assert len(await archive.list_events(agent_run_id="run")) == 3
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert len(await archive.list_events(agent_run_id="run")) == 4
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_terminal_trace_known_rollback_preserves_pending_facts(tmp_path: Path, backend: str, monkeypatch: pytest.MonkeyPatch) -> None:
    state = await _setup(tmp_path, backend)
    try:
        store = state.run_store
        store.trace_persistence.configure("run", terminal=True)
        archive = store.read_store(RuntimeDomain.TRACE)
        await _observe(store, 1)
        group = archive.state_store.storage_group
        mutate = group.mutate

        async def fail_before_commit(stores, callback):
            raise RuntimeError("source rejected mutation")

        monkeypatch.setattr(group, "mutate", fail_before_commit)
        with pytest.raises(RuntimeError, match="source rejected mutation"):
            await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert "run" in store.trace_persistence.dirty
        assert await archive.list_events(agent_run_id="run") == []
        monkeypatch.setattr(group, "mutate", mutate)
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert len(await archive.list_events(agent_run_id="run")) == 1
        assert "run" not in store.trace_persistence.dirty
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_terminal_trace_ignores_batch_pressure_but_close_drains(tmp_path: Path, backend: str) -> None:
    state = await _setup(tmp_path, backend)
    try:
        store = state.run_store
        store.trace_persistence.configure("run", terminal=True)
        archive = store.read_store(RuntimeDomain.TRACE)
        for index in range(1, 130):
            await _observe(store, index)
        assert await archive.list_events(agent_run_id="run") == []
        assert "run" in store.trace_persistence.dirty
        assert store.trace_persistence._buffers == {}
        await store.preflight_close()
        assert len(await archive.list_events(agent_run_id="run")) == 129
        assert "run" not in store.trace_persistence.dirty
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("interruption", ("lost_ack", "cancel"))
async def test_terminal_trace_settles_confirmed_commit_before_interruption(
    tmp_path: Path, backend: str, interruption: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = await _setup(tmp_path, backend)
    entered, release = asyncio.Event(), asyncio.Event()
    try:
        store = state.run_store
        store.trace_persistence.configure("run", terminal=True)
        archive = store.read_store(RuntimeDomain.TRACE)
        await _observe(store, 1)
        group = archive.state_store.storage_group
        mutate = group.mutate
        commits = 0
        depth = 0

        async def interrupted_mutation(stores, callback):
            nonlocal commits, depth
            if depth:
                return await mutate(stores, callback)
            if interruption == "cancel":
                entered.set()
                await release.wait()
            depth += 1
            try:
                result = await mutate(stores, callback)
            finally:
                depth -= 1
            commits += 1
            if interruption == "lost_ack":
                raise RuntimeError("commit acknowledgement lost")
            return result

        monkeypatch.setattr(group, "mutate", interrupted_mutation)
        flush = asyncio.create_task(store.flush_execution_projection(
            "run", execution_id="execution", producer_generation=1,
        ))
        if interruption == "cancel":
            await asyncio.wait_for(entered.wait(), 2)
            flush.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await flush
        else:
            await flush
        assert commits == 1
        assert len(await archive.list_events(agent_run_id="run")) == 1
        assert "run" not in store.trace_persistence.dirty
    finally:
        release.set()
        await state.close()
