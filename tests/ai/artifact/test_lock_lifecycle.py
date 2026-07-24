#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""InProcessArtifactDigestCoordinator registry lifecycle: the per-digest lock
registry is refcounted and bounded. Every exit path -- normal release, holder
cancellation, waiter cancellation -- returns its reference, and an entry whose
reference count reaches zero and whose lock is free is deleted. A process that
churns through many unique digests must NOT accumulate stale entries."""

import asyncio

import pytest

from linktools.ai.artifact.coordination import InProcessArtifactDigestCoordinator
from linktools.ai.artifact.digest import ArtifactDigest

_VALID = "a" * 64


def _digest(n: int) -> ArtifactDigest:
    # 64 lowercase hex chars varying by n (n < 16 keeps it hex).
    return ArtifactDigest.parse(format(n, "x").rjust(64, "0"))


@pytest.mark.asyncio
async def test_release_deletes_entry():
    coord = InProcessArtifactDigestCoordinator()
    async with coord.hold(ArtifactDigest.parse(_VALID)):
        assert coord.active_entry_count == 1
    assert coord.active_entry_count == 0


@pytest.mark.asyncio
async def test_multiple_waiters_then_all_release_leaves_no_entry():
    coord = InProcessArtifactDigestCoordinator()
    d = ArtifactDigest.parse(_VALID)
    entered = asyncio.Event()
    proceed = asyncio.Event()

    async def _hold_then_wait():
        async with coord.hold(d):
            entered.set()
            await proceed.wait()

    holder = asyncio.create_task(_hold_then_wait())
    await entered.wait()
    # Two waiters queue on the same digest while the holder holds it: the entry
    # stays registered (references counts holder + waiters).
    w1 = asyncio.create_task(_async_noop_hold(coord, d))
    w2 = asyncio.create_task(_async_noop_hold(coord, d))
    await asyncio.sleep(0)  # let waiters register on the lock
    assert coord.active_entry_count == 1  # one digest, three references
    proceed.set()
    await holder
    await w1
    await w2
    assert coord.active_entry_count == 0


async def _async_noop_hold(coord, d):
    async with coord.hold(d):
        pass


@pytest.mark.asyncio
async def test_waiter_cancellation_decrements_reference():
    coord = InProcessArtifactDigestCoordinator()
    d = ArtifactDigest.parse(_VALID)
    entered = asyncio.Event()
    proceed = asyncio.Event()

    async def _hold():
        async with coord.hold(d):
            entered.set()
            await proceed.wait()

    holder = asyncio.create_task(_hold())
    await entered.wait()
    # A waiter that gets cancelled while queued must still return its reference.
    waiter = asyncio.create_task(_async_noop_hold(coord, d))
    await asyncio.sleep(0)
    waiter.cancel()
    try:
        await waiter
    except asyncio.CancelledError:
        pass
    # Holder still active -> entry remains, but the cancelled waiter's reference
    # is gone.
    assert coord.active_entry_count == 1
    proceed.set()
    await holder
    assert coord.active_entry_count == 0


@pytest.mark.asyncio
async def test_holder_cancellation_releases_lock_and_reaps_entry():
    coord = InProcessArtifactDigestCoordinator()
    d = ArtifactDigest.parse(_VALID)
    started = asyncio.Event()

    async def _hold_forever():
        async with coord.hold(d):
            started.set()
            await asyncio.Event().wait()  # never resolves -> cancelled mid-hold

    holder = asyncio.create_task(_hold_forever())
    await started.wait()
    holder.cancel()
    try:
        await holder
    except asyncio.CancelledError:
        pass
    # Holder cancelled: the lock is released and the entry reaped, so a fresh
    # acquisition succeeds immediately and the registry is empty afterwards.
    async with coord.hold(d):
        pass
    assert coord.active_entry_count == 0


@pytest.mark.asyncio
async def test_many_unique_digests_leave_registry_empty():
    coord = InProcessArtifactDigestCoordinator()
    for i in range(10000):
        async with coord.hold(_digest(i)):
            pass
    assert coord.active_entry_count == 0


@pytest.mark.asyncio
async def test_same_digest_is_mutually_exclusive():
    coord = InProcessArtifactDigestCoordinator()
    d = ArtifactDigest.parse(_VALID)
    order: "list[str]" = []

    async def _critical(name: str):
        async with coord.hold(d):
            order.append(f"{name}-in")
            await asyncio.sleep(0)
            order.append(f"{name}-out")

    await asyncio.gather(_critical("a"), _critical("b"), _critical("c"))
    # Each critical section is fully nested (in/out pairs never interleave).
    assert order == ["a-in", "a-out", "b-in", "b-out", "c-in", "c-out"]


@pytest.mark.asyncio
async def test_different_digests_run_in_parallel():
    coord = InProcessArtifactDigestCoordinator()
    inflight = 0
    max_inflight = 0
    lock = asyncio.Lock()

    async def _hold(d: ArtifactDigest):
        nonlocal inflight, max_inflight
        async with coord.hold(d):
            async with lock:
                inflight += 1
                max_inflight = max(max_inflight, inflight)
            await asyncio.sleep(0.02)
            async with lock:
                inflight -= 1

    await asyncio.gather(*(_hold(_digest(i)) for i in range(8)))
    # Different digests do not contend -> several run concurrently.
    assert max_inflight >= 2
    assert coord.active_entry_count == 0
