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
from typing import AsyncIterator, Protocol, runtime_checkable

from ..errors import ArtifactError


class UnsupportedArtifactCoordinationError(ArtifactError):
    """Raised when a coordinator that cannot provide the required coordination
    scope is constructed (e.g. a filesystem flock coordinator on a non-POSIX
    platform), or when a deployment that needs a distributed coordinator did not
    inject one. Fail-closed: never silently degrade to a lockless fallback."""


@runtime_checkable
class ArtifactDigestCoordinator(Protocol):
    """Per-digest mutual exclusion for the artifact put/sweep race.

    ``hold(digest)`` is an async context manager; the lock is scoped to a single
    SHA-256 digest -- the same digest serializes, different digests run in
    parallel (no global serialization bottleneck)."""

    @asynccontextmanager
    async def hold(self, digest: str) -> AsyncIterator[None]:
        ...
        yield


class InProcessArtifactDigestCoordinator:
    """Process-local digest mutex: one ``asyncio.Lock`` per digest. Declares
    PROCESS_LOCAL scope only -- it coordinates within a single process, so a
    multi-worker deployment MUST inject a distributed coordinator instead. Used
    by in-repo Memory/Filesystem/SqlAlchemy storage and by tests."""

    def __init__(self) -> None:
        self._locks: "dict[str, asyncio.Lock]" = {}
        self._registry_guard = asyncio.Lock()

    @asynccontextmanager
    async def hold(self, digest: str) -> AsyncIterator[None]:
        async with self._registry_guard:
            lock = self._locks.get(digest)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[digest] = lock
        async with lock:
            yield


__all__: "list[str]" = (
    "ArtifactDigestCoordinator",
    "InProcessArtifactDigestCoordinator",
    "UnsupportedArtifactCoordinationError",
)
