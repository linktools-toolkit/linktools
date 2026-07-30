#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Version-query Protocol for content-addressed, revisioned backends.

Distinct from ``revision.py`` (the single-load metadata REPLACE/PATCH concern):
this is the point-in-time history surface -- list a key's versions and read the
value in effect at a given revision. A backend that retains a permanent change
log implements this Protocol; backends that keep no history (e.g. a local
directory backend) simply omit it, and callers gate access with an
``isinstance(backend, VersionedStorage)`` capability check."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, TypeVar, runtime_checkable

RevisionT = TypeVar("RevisionT")
KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")


@dataclass(frozen=True, slots=True)
class VersionSummary:
    """One historical version of a key: the revision at which it took effect, its
    content etag, ``object_id`` (the content-addressed blob's sha256, or ``None``
    for a tombstone deletion), the row's ``created_at`` timestamp, and whether
    that version is a deletion."""

    revision: int
    etag: "str | None"
    object_id: "str | None"
    created_at: datetime
    deleted: bool


@runtime_checkable
class VersionedStorage(Protocol[RevisionT, KeyT, ValueT]):
    async def list_versions(self, key: KeyT) -> tuple[VersionSummary, ...]: ...

    async def get_at_revision(
        self, key: KeyT, revision: RevisionT
    ) -> "ValueT | None": ...


__all__ = ["VersionSummary", "VersionedStorage"]
