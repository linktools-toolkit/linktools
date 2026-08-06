#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure lease and fencing rules shared by Run, Task, and Tool stores."""


from dataclasses import dataclass
from datetime import timezone
from ...foundation.errors import StorageConflictError

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime, timedelta

@dataclass(frozen=True, slots=True)
class Lease:
    owner: "str | None" = None
    fence: int = 0
    expires_at: "datetime | None" = None


class LeaseCoordinator(Protocol):
    async def acquire(self, key: str, *, owner: str, now: "datetime", duration: "timedelta") -> Lease: ...
    async def renew(self, key: str, *, owner: str, fence: int, now: "datetime", duration: "timedelta") -> Lease: ...
    async def release(self, key: str, *, owner: str, fence: int, now: "datetime") -> Lease: ...


def is_expired(lease: Lease, now: "datetime") -> bool:
    if lease.expires_at is None:
        return False
    expires_at = lease.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return expires_at <= now


def claim(lease: Lease, *, owner: str, now: "datetime", duration: "timedelta") -> Lease:
    if lease.owner is not None and lease.owner != owner and not is_expired(lease, now):
        raise StorageConflictError("lease is owned by another active worker")
    return Lease(owner, lease.fence + 1, now + duration)


def renew(lease: Lease, *, owner: str, fence: int, now: "datetime", duration: "timedelta") -> Lease:
    assert_active(lease, owner=owner, fence=fence, now=now)
    return Lease(owner, fence, now + duration)


def assert_active(lease: Lease, *, owner: str, fence: int, now: "datetime") -> None:
    if lease.owner != owner or lease.fence != fence or lease.expires_at is None or is_expired(lease, now):
        raise StorageConflictError("lease is not active for this owner and fence")


def release(lease: Lease) -> Lease:
    return Lease(None, lease.fence, None)


__all__ = ["Lease", "LeaseCoordinator", "assert_active", "claim", "is_expired", "release", "renew"]
