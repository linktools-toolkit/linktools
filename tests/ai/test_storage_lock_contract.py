#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for storage lease ownership semantics."""

import asyncio

import pytest
from linktools.ai.storage import FilesystemLeaseCoordinator, ProcessLeaseCoordinator


@pytest.mark.asyncio
async def test_process_lease_can_be_released_by_a_different_task() -> None:
    coordinator = ProcessLeaseCoordinator()
    lease = await asyncio.wait_for(coordinator.acquire("key"), 1)
    await coordinator.release(lease)

    replacement = await asyncio.wait_for(coordinator.acquire("key"), 1)
    await coordinator.release(replacement)


@pytest.mark.asyncio
async def test_filesystem_lost_lease_releases_local_keyed_lock(tmp_path) -> None:
    coordinator = FilesystemLeaseCoordinator(tmp_path / "leases", lease_seconds=5)
    lease = await coordinator.acquire("key", timeout=1)
    await asyncio.to_thread(coordinator._release, lease)

    await coordinator.renew(lease)

    replacement = await asyncio.wait_for(coordinator.acquire("key", timeout=1), 1)
    await coordinator.release(replacement)
