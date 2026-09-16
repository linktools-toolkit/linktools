#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lock ownership, fencing, and lease deadline invariants."""

import asyncio
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pytest
from filelock import FileLock
import linktools.ai.storage._files as files_module
import linktools.ai.storage._lock as lock_module
from linktools.ai.runtime.state._steps import _RunHistoryLock
from linktools.ai.storage import FilesystemLeaseCoordinator, KeyedAsyncLock, Lease


@pytest.mark.asyncio
async def test_cancelled_run_history_waiter_does_not_clear_holder_state() -> None:
    history_lock = _RunHistoryLock()
    holder_ready = asyncio.Event()
    waiter_started = asyncio.Event()
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
        waiter_started.set()
        async with history_lock.hold("run"):
            raise AssertionError("cancelled waiter unexpectedly acquired the lock")

    holder_task = asyncio.create_task(holder())
    waiter_task = None
    try:
        await asyncio.wait_for(holder_ready.wait(), 2)
        waiter_task = asyncio.create_task(waiter())
        await asyncio.wait_for(waiter_started.wait(), 2)
        waiter_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter_task
        waiter_cancelled.set()
        await asyncio.wait_for(nested_acquired.wait(), 2)
        release_holder.set()
        await holder_task
        assert history_lock._entries == {}
    finally:
        waiter_cancelled.set()
        release_holder.set()
        if waiter_task is not None:
            waiter_task.cancel()
            await asyncio.gather(waiter_task, return_exceptions=True)
        await asyncio.gather(holder_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_keyed_async_lock_releases_idle_entries() -> None:
    lock = KeyedAsyncLock()
    for index in range(32):
        key = f"key-{index}"
        await lock.acquire(key)
        await lock.release(key)
    assert lock._locks == {}


@pytest.mark.asyncio
async def test_cancelled_keyed_waiter_preserves_holder_and_handoff() -> None:
    lock = KeyedAsyncLock()
    await lock.acquire("key")
    started = asyncio.Event()

    async def contender() -> None:
        started.set()
        await lock.acquire("key")
        await lock.release("key")

    waiter = asyncio.create_task(contender())
    await started.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    with pytest.raises(RuntimeError, match="recursive"):
        await lock.acquire("key", timeout=1)
    next_waiter = asyncio.create_task(contender())
    await asyncio.sleep(0)
    await lock.release("key")
    await asyncio.wait_for(next_waiter, 2)
    assert lock._locks == {}


def test_fence_replace_failure_keeps_previous_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "lease.fence"
    path.write_text("7", encoding="utf-8")
    original_replace = files_module.os.replace

    def fail_replace(source: str, destination: str | Path) -> None:
        if Path(destination) == path:
            raise OSError("replace failed")
        original_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(files_module.os, "replace", fail_replace)
        with pytest.raises(OSError):
            lock_module._next_fence(path)
    assert path.read_text(encoding="utf-8") == "7"
    assert lock_module._next_fence(path) == 8
    assert not tuple(tmp_path.glob(".lease.fence.*"))


@pytest.mark.parametrize("value", ("", "-1"))
def test_corrupt_fence_is_not_reinitialized(tmp_path: Path, value: str) -> None:
    path = tmp_path / "lease.fence"
    path.write_text(value, encoding="utf-8")
    with pytest.raises(ValueError):
        lock_module._next_fence(path)
    assert path.read_text(encoding="utf-8") == value


def _issue_fence(path: str) -> int:
    return lock_module._next_fence(Path(path))


def test_fence_is_unique_across_processes(tmp_path: Path) -> None:
    path = str(tmp_path / "lease.fence")
    with ProcessPoolExecutor(
        max_workers=2,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        values = tuple(executor.map(_issue_fence, (path,) * 8))
    assert sorted(values) == list(range(1, 9))


@pytest.mark.asyncio
async def test_filesystem_lease_timeout_includes_local_lock_wait(tmp_path: Path) -> None:
    coordinator = FilesystemLeaseCoordinator(tmp_path / "leases", lease_seconds=5)
    lease = await coordinator.acquire("key", timeout=1)
    try:
        contender = asyncio.create_task(coordinator.acquire("key", timeout=0.05))
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(contender, 0.5)
        assert not contender.cancelled()
    finally:
        await coordinator.release(lease)


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", (".guard", ".fence.lock"))
async def test_filesystem_lease_timeout_includes_file_lock_wait(
    tmp_path: Path, suffix: str,
) -> None:
    coordinator = FilesystemLeaseCoordinator(tmp_path, lease_seconds=5)
    guard = tmp_path / (lock_module._lease_name("key") + suffix)
    with FileLock(str(guard), thread_local=False):
        contender = asyncio.create_task(coordinator.acquire("key", timeout=0.05))
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(contender, 0.5)
        assert not contender.cancelled()
    pending = tuple(lock_module._DETACHED_LOCK_TASKS)
    if pending:
        await asyncio.wait_for(asyncio.gather(*pending), 2)
    assert not tuple(tmp_path.glob("*.lease"))
    lease = await coordinator.acquire("key", timeout=1)
    await coordinator.release(lease)


@pytest.mark.asyncio
async def test_expired_queued_lease_attempt_does_not_acquire(tmp_path: Path) -> None:
    coordinator = FilesystemLeaseCoordinator(tmp_path, lease_seconds=5)
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()

    def occupy_worker() -> None:
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        loop.set_default_executor(executor)
        occupied = loop.run_in_executor(executor, occupy_worker)
        try:
            await asyncio.wait_for(started.wait(), 2)
            contender = asyncio.create_task(coordinator.acquire("key", timeout=0.05))
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(contender, 0.5)
            assert not contender.cancelled()
        finally:
            release.set()
            await occupied
            pending = tuple(lock_module._DETACHED_LOCK_TASKS)
            if pending:
                await asyncio.wait_for(asyncio.gather(*pending), 2)
        assert not tuple(tmp_path.glob("*.lease"))
        lease = await coordinator.acquire("key", timeout=1)
        await coordinator.release(lease)


@pytest.mark.asyncio
async def test_cancelled_acquire_releases_a_late_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = FilesystemLeaseCoordinator(tmp_path, lease_seconds=5)
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    original_attempt = coordinator._try_acquire

    def delayed_result(key: str, deadline: float) -> Lease | None:
        lease = original_attempt(key, deadline)
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        return lease

    monkeypatch.setattr(coordinator, "_try_acquire", delayed_result)
    task = asyncio.create_task(coordinator.acquire("key", timeout=1))
    try:
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        pending = tuple(lock_module._DETACHED_LOCK_TASKS)
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending), 2)
    assert not tuple(tmp_path.glob("*.lease"))
    monkeypatch.setattr(coordinator, "_try_acquire", original_attempt)
    lease = await coordinator.acquire("key", timeout=1)
    await coordinator.release(lease)
