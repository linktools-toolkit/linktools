#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Version-query Protocol for content-addressed, revisioned backends.

Distinct from ``revision.py`` (the single-load metadata REPLACE/PATCH concern):
this is the point-in-time history surface -- list a key's versions and read the
value in effect at a given revision or version. A backend that retains a
permanent change log implements this Protocol; backends that keep no history
(e.g. a local directory backend) simply omit it, and callers gate access with
an ``isinstance(backend, VersionedStorage)`` capability check."""

from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

RevisionT = TypeVar("RevisionT")
KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")


@dataclass(frozen=True, slots=True)
class VersionSummary:
    """One historical version of a key: the revision at which it took effect, its
    content etag, ``object_id`` (the content-addressed blob's sha256, or ``None``
    for a tombstone deletion), the row's ``created_at`` timestamp, whether that
    version is a deletion, and ``version`` -- the value's own declared version
    number at that point in time (the same field surfaced on the value's info,
    e.g. ``SpecDocumentInfo.version``), not a synthesized history ordinal."""

    revision: int
    version: "int | None"
    etag: "str | None"
    object_id: "str | None"
    created_at: "datetime"
    deleted: bool


@runtime_checkable
class VersionedStorage(Protocol[RevisionT, KeyT, ValueT]):
    async def list_versions(self, key: KeyT) -> "tuple[VersionSummary, ...]": ...

    async def get_at_revision(
        self, key: KeyT, revision: RevisionT
    ) -> "ValueT | None": ...

    async def get_at_version(self, key: KeyT, version: int) -> "ValueT | None":
        """Read the value of ``key`` recorded under its own declared
        ``version`` number (the same number as the value's info, e.g.
        ``SpecDocumentInfo.version`` -- not a synthesized history ordinal).
        When more than one history record shares that version number, the
        most recent one wins. Returns None when no record of ``key`` carries
        that version, or the matching record is a deletion."""
        ...


__all__ = ["VersionSummary", "VersionedStorage"]
