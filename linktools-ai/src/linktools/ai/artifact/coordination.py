#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Artifact digest coordination: serialize the put (blob + record) path and the
orphan-sweep (re-check + delete) path per SHA-256 digest.

Without coordination, a sweeper that reads "is this digest referenced?" while a
put is between its blob write and its record create sees the blob as unreferenced
and deletes it -- corrupting the still-in-flight put. Holding the per-digest lock
across both the blob reuse AND the record create (put side), and across the
re-stat/re-age/re-reference check AND the delete (sweep side), makes the two
flows mutually exclusive per digest: the sweeper either sees the record the put
created, or blocks on the lock until the put finishes and then sees it.

The Protocol is the contract; ``InProcessArtifactDigestCoordinator`` is the
process-local implementation (single-process Memory/Filesystem/SqlAlchemy
storage and tests). A multi-worker object-store deployment injects a distributed
implementation (Redis named lock, DB advisory lock, ...); the core ships none --
when a distributed coordinator is required but not provided, construction fails
closed rather than falling back to a lockless mode that would reintroduce the
race."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Protocol, runtime_checkable

from ..errors import ArtifactError
from ..storage.features import CoordinationScope
from .digest import ArtifactDigest


class UnsupportedArtifactCoordinationError(ArtifactError):
    """Raised when a coordinator that cannot provide the required coordination
    scope is constructed (e.g. a filesystem flock coordinator on a non-POSIX
    platform), or when a deployment that needs a distributed coordinator did not
    inject one. Fail-closed: never silently degrade to a lockless fallback."""


@runtime_checkable
class ArtifactDigestCoordinator(Protocol):
    """Per-digest mutual exclusion for the artifact put/sweep race.

    ``hold(digest)`` takes the validated :class:`ArtifactDigest` value object --
    a bare string is never accepted, so an unvalidated digest cannot become a
    coordination key. The lock is scoped to a single SHA-256 digest: the same
    digest serializes, different digests run in parallel (no global bottleneck).

    ``scope`` declares the coordination range this coordinator actually
    provides -- PROCESS_LOCAL (this process only) or DISTRIBUTED (spans workers/
    processes). The Runtime multi-worker gate reads this to refuse a
    process-local coordinator under a topology that shares ArtifactStore across
    workers, exactly as it does for the Job Lease coordinator's own scope."""

    scope: CoordinationScope

    @asynccontextmanager
    async def hold(self, digest: ArtifactDigest) -> AsyncIterator[None]:
        ...
        yield


@dataclass
class _LockEntry:
    """One in-process digest lock plus a reference count of every coroutine
    currently relying on it -- the holder AND each waiter. The entry is reaped
    when the count returns to zero and the lock is free, so the registry cannot
    grow without bound over the lifetime of a long-running process."""

    lock: asyncio.Lock
    references: int = 0


class InProcessArtifactDigestCoordinator:
    """Process-local digest mutex: one ``asyncio.Lock`` per digest, refcounted.
    Declares PROCESS_LOCAL scope only -- it coordinates within a single process,
    so a multi-worker deployment MUST inject a distributed coordinator instead.
    Used by in-repo Memory/Filesystem/SqlAlchemy storage and by tests.

    The registry is bounded: every ``hold`` registers (references +1, counting
    the holder and each waiter), and every exit path -- normal release, holder
    cancellation, waiter cancellation -- decrements. An entry whose count
    reaches zero and whose lock is free is deleted, so a process that churns
    through many unique digests does not accumulate stale entries."""

    scope = CoordinationScope.PROCESS_LOCAL

    def __init__(self) -> None:
        self._entries: "dict[str, _LockEntry]" = {}
        self._registry_lock = asyncio.Lock()

    @property
    def active_entry_count(self) -> int:
        """Read-only count of currently-registered digest entries (held or
        awaited). Returns to zero when no coroutine is relying on any lock."""
        return len(self._entries)

    @asynccontextmanager
    async def hold(self, digest: ArtifactDigest) -> AsyncIterator[None]:
        key = digest.value
        async with self._registry_lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = _LockEntry(lock=asyncio.Lock())
                self._entries[key] = entry
            # references counts this coroutine (the about-to-be holder/waiter);
            # every exit path below decrements it.
            entry.references += 1
        try:
            async with entry.lock:
                yield
        finally:
            # Reached on normal release AND on cancellation (holder or waiter):
            # the per-digest lock is released by ``async with`` on the way out,
            # and the registry reference this coroutine held is returned. When
            # no coroutine references the entry and the lock is free, the entry
            # is reaped so the registry stays bounded.
            async with self._registry_lock:
                entry.references -= 1
                if entry.references <= 0 and not entry.lock.locked():
                    self._entries.pop(key, None)


__all__: "list[str]" = (
    "ArtifactDigestCoordinator",
    "InProcessArtifactDigestCoordinator",
    "UnsupportedArtifactCoordinationError",
)
