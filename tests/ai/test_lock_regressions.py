#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for lock ownership and filesystem coordination."""

import asyncio
from pathlib import Path

import pytest
import linktools.ai.storage._lock as lock_module
from linktools.ai.runtime.state._steps import _RunHistoryLock
from linktools.ai.storage import FilesystemLeaseCoordinator, KeyedAsyncLock


@pytest.mark.asyncio
async def test_cancelled_run_history_waiter_does_not_clear_holder_state() -> None:
    history_lock = _RunHistoryLock()
    holder_ready = asyncio.Event()
    waiter_cancelled = asyncio.Event()
    nested_acquired = asyncio.Event()
    release_holder = asyncio.Event()

    async def holder() -> None:
        async with history_lock.hold("run"):
            holder_ready.set()
            await waiter_cancelled.wait()
            async with history_lock.hold("run"):
                nested_acquired.set()
            await release_holder.wait()

    async def waiter() -> None:
        async with history_lock.hold("run"):
            raise AssertionError("cancelled waiter unexpectedly acquired the lock")

    holder_task = asyncio.create_task(holder())
    await holder_ready.wait()
    waiter_task = asyncio.create_task(waiter())
    while history_lock._entries["run"].references < 2:
        await asyncio.sleep(0)

    waiter_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter_task
    waiter_cancelled.set()
    await asyncio.wait_for(nested_acquired.wait(), 1)
    release_holder.set()
    await holder_task

    assert history_lock._entries == {}


@pytest.mark.asyncio
async def test_keyed_async_lock_releases_idle_entries() -> None:
    lock = KeyedAsyncLock()

    for index in range(32):
        key = f"key-{index}"
        await lock.acquire(key)
        await lock.release(key)

    assert lock._locks == {}


def test_fence_write_failure_keeps_previous_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "lease.fence"
    path.write_text("7", encoding="utf-8")
    original_write = lock_module.write_bytes_atomic

    def fail_write(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("write failed")

    monkeypatch.setattr(lock_module, "write_bytes_atomic", fail_write)
    with pytest.raises(OSError):
        lock_module._next_fence(path)
    assert path.read_text(encoding="utf-8") == "7"

    monkeypatch.setattr(lock_module, "write_bytes_atomic", original_write)
    assert lock_module._next_fence(path) == 8
    assert path.read_text(encoding="utf-8") == "8"


@pytest.mark.asyncio
async def test_filesystem_lease_timeout_includes_local_lock_wait(tmp_path: Path) -> None:
    coordinator = FilesystemLeaseCoordinator(tmp_path / "leases", lease_seconds=5)
    lease = await coordinator.acquire("key", timeout=1)
    try:
        contender = asyncio.create_task(coordinator.acquire("key", timeout=0.05))
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(contender, 0.5)
        assert contender.done()
        assert not contender.cancelled()
    finally:
        await coordinator.release(lease)
