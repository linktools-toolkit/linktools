#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Single-process lease coordinator with monotonically increasing fences."""

import asyncio
from datetime import datetime, timedelta

from ...foundation.errors import StorageConflictError
from .protocols import Lease, assert_active, claim, is_expired, release, renew


class LocalLeaseCoordinator:
    """Coordinate named leases in one event loop."""

    def __init__(self) -> None:
        self._leases: "dict[str, Lease]" = {}
        self._lock = asyncio.Lock()

    async def acquire(self, key: str, *, owner: str, now: datetime, duration: timedelta) -> Lease:
        async with self._lock:
            current = self._leases.get(key, Lease())
            next_lease = claim(current, owner=owner, now=now, duration=duration)
            self._leases[key] = next_lease
            return next_lease

    async def renew(self, key: str, *, owner: str, fence: int, now: datetime, duration: timedelta) -> Lease:
        async with self._lock:
            current = self._leases.get(key, Lease())
            next_lease = renew(current, owner=owner, fence=fence, now=now, duration=duration)
            self._leases[key] = next_lease
            return next_lease

    async def release(self, key: str, *, owner: str, fence: int, now: datetime) -> Lease:
        async with self._lock:
            current = self._leases.get(key, Lease())
            assert_active(current, owner=owner, fence=fence, now=now)
            next_lease = release(current)
            self._leases[key] = next_lease
            return next_lease


class ProcessLocalLeaseCoordinator(LocalLeaseCoordinator):
    """Canonical v8 name for the single-process coordinator."""


__all__ = ["LocalLeaseCoordinator", "Lease", "ProcessLocalLeaseCoordinator", "StorageConflictError", "is_expired"]
