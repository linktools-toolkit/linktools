#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""BUG-01 (v5 guide §8): Runtime crash-recovery initialization must be
serialized and must stay retryable after a failure.

Before the fix the flag was flipped before recovery ran, so (a) two concurrent
first-callers both proceeded past the guard, and (b) a recovery that raised left
the flag set, so no later entry point ever retried. Both tests fail against the
old single-flag implementation."""

import asyncio

import pytest

from linktools.ai.runtime import Runtime
from linktools.ai.storage.facade import FilesystemStorage


def _runtime(tmp_path) -> Runtime:
    return Runtime.build(storage=FilesystemStorage(root=tmp_path))


@pytest.mark.asyncio
async def test_concurrent_recovery_runs_recover_exactly_once(tmp_path):
    rt = _runtime(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()
    calls = {"n": 0}

    async def recover():
        calls["n"] += 1
        started.set()
        await asyncio.wait_for(release.wait(), timeout=5)

    rt._components.commit_coordinator.recover_incomplete_commits = recover

    t1 = asyncio.create_task(rt._coordinator._ensure_recovered())
    t2 = asyncio.create_task(rt._coordinator._ensure_recovered())
    await asyncio.wait_for(started.wait(), timeout=5)

    # The second caller must NOT have returned early while recovery is still in
    # flight -- it is serialized behind the lock.
    assert not t1.done()
    assert not t2.done()

    release.set()
    await asyncio.gather(t1, t2)

    assert calls["n"] == 1, "recover must run exactly once under concurrency"
    assert rt._coordinator._recovery_done is True


@pytest.mark.asyncio
async def test_failed_recovery_stays_retryable(tmp_path):
    rt = _runtime(tmp_path)
    calls = {"n": 0}

    async def recover():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient recovery failure")

    rt._components.commit_coordinator.recover_incomplete_commits = recover

    with pytest.raises(RuntimeError):
        await rt._coordinator._ensure_recovered()

    # The flag must NOT be set after a failure, so the next entry point retries.
    assert rt._coordinator._recovery_done is False

    await rt._coordinator._ensure_recovered()
    assert rt._coordinator._recovery_done is True
    assert calls["n"] == 2, "a failed recovery must be retried"
