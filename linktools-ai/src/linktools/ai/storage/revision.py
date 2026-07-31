#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Single-load metadata protocol and per-layer metadata views.

A backend returns its current revision and the entries that changed since a
caller-held revision in one ``load_metadata`` call. This replaces the old
multi-stage ``current_revision -> list_changes -> current_revision`` round
trip: a single load either REPLACES the whole entry set (first load, or the
caller's revision is too old to patch) or PATCHES it against a known prior
state. See ``.docs/linktools_ai_storage_composition_revision_io_optimization_spec.md``.

``LayerMetadataView`` wraps one backend's metadata with single-flight refresh
so N concurrent readers trigger at most one backend load and a cancelled
caller never publishes a half-loaded state."""


import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

RevisionT = TypeVar("RevisionT")
KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")
InfoT = TypeVar("InfoT")


class MetadataLoadMode(StrEnum):
    REPLACE = "replace"
    PATCH = "patch"


@dataclass(frozen=True, slots=True)
class StorageChange(Generic[KeyT, InfoT]):
    """One entry's effective state at the loaded revision.

    ``current`` is the entry's info, or ``None`` when the entry was removed:
    a removal is represented as a tombstone in the change set, not a missing
    row, so a PATCH can delete a key from a caller's entry map."""

    key: KeyT
    current: "InfoT | None"


@dataclass(frozen=True, slots=True)
class MetadataLoad(Generic[RevisionT, KeyT, InfoT]):
    """One backend load: the revision the data was read at, whether the
    changes fully replace or only patch the caller's state, and the change
    set (every entry for REPLACE, only diffs for PATCH)."""

    revision: RevisionT
    mode: MetadataLoadMode
    changes: "tuple[StorageChange[KeyT, InfoT], ...]"


@dataclass(frozen=True, slots=True)
class MetadataState(Generic[RevisionT, KeyT, InfoT]):
    """A materialized entry map keyed by business key at a revision."""

    revision: RevisionT
    entries: "Mapping[KeyT, InfoT]"


@runtime_checkable
class StorageReader(Protocol[KeyT, ValueT, InfoT]):
    async def get(self, key: KeyT) -> "ValueT | None": ...

    async def list_info(self) -> "tuple[InfoT, ...]": ...


@runtime_checkable
class BatchStorageReader(Protocol[KeyT, ValueT]):
    async def get_many(
        self,
        keys: "tuple[KeyT, ...]",
    ) -> "Mapping[KeyT, ValueT]": ...


@runtime_checkable
class StorageMetadataBackend(Protocol[RevisionT, KeyT, InfoT]):
    async def load_metadata(
        self,
        after_revision: "RevisionT | None",
    ) -> "MetadataLoad[RevisionT, KeyT, InfoT]":
        ...

    async def head_revision(self) -> RevisionT:
        """The current revision without loading any entry/change data.

        A cheap probe a caller uses to decide whether a held state is still
        current (e.g. against a cached revision) before paying for a
        ``load_metadata`` diff. Distinct from ``load_metadata``: it returns a
        single scalar, never touches the change set, and is the point at which
        an external revision cache (file/redis, downstream) is validated."""
        ...


class LayerRefreshPolicy(StrEnum):
    STATIC = "static"
    REVISIONED = "revisioned"
    ALWAYS = "always"


StorageInitializer = Callable[..., Awaitable[None]]


@runtime_checkable
class RevisionSource(Protocol):
    async def current(self) -> "int | str | None":
        """The current revision, or ``None`` when unknown/stale.

        A view consults this before paying for ``load_metadata``: when the
        returned revision equals its held state's revision, the held state is
        reused and no metadata load is issued. ``None`` means "I don't know"
        (a cache miss on an external source) -- the caller must then load.

        The default implementation probes the backend's ``head_revision``; a
        downstream system injects a redis/file-backed source so multiple
        processes share one revision signal (cheap, cross-process)."""
        ...

    async def revision_bumped(self, revision: "int | str") -> None:
        """Called by the composition AFTER a write commits and the revision
        advanced to ``revision``.

        A caching source uses this to refresh/publish its held revision (e.g.
        redis SET + PUBLISH so cross-machine subscribers invalidate within ms
        rather than waiting on a TTL). The default source no-ops: it reads
        ``head_revision`` live and never caches, so it is never stale. ``None``
        is never passed -- the composition probes head once post-commit to get
        the concrete new revision before calling this."""
        ...


class _BackendHeadRevisionSource:
    """Default :class:`RevisionSource`: probes the backend's ``head_revision``.

    Always live (no cache), so it is always correct for single-process use; it
    merely replaces the heavier ``load_metadata`` JOIN with a single-row head
    read when the revision has not changed. Returns ``None`` for a backend that
    is not a :class:`StorageMetadataBackend` (no head to probe)."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    async def current(self) -> "int | str | None":
        if isinstance(self._backend, StorageMetadataBackend):
            return await self._backend.head_revision()
        return None

    async def revision_bumped(self, revision: "int | str") -> None:
        # No-op: this source reads head_revision live on every current() call,
        # so it is never stale and needs no post-write notification.
        return None


def apply_metadata_load(
    current: "MetadataState[RevisionT, KeyT, InfoT] | None",
    load: "MetadataLoad[RevisionT, KeyT, InfoT]",
    *,
    info_key: "Callable[[InfoT], KeyT]",
) -> "MetadataState[RevisionT, KeyT, InfoT]":
    """Fold one backend ``load`` into a caller state. REPLACE discards the
    prior state; PATCH applies only the listed changes (a ``None`` current
    removes the key). Entries are keyed by ``StorageChange.key`` (the
    backend's declared business key), not re-derived from the info, so the
    key is stable whether or not the info happens to encode it. The result is
    always a plain dict the caller owns."""
    if load.mode is MetadataLoadMode.REPLACE:
        entries: "dict[KeyT, InfoT]" = {}
        for change in load.changes:
            if change.current is not None:
                entries[change.key] = change.current
        return MetadataState(load.revision, entries)
    entries = dict(current.entries) if current is not None else {}
    for change in load.changes:
        if change.current is None:
            entries.pop(change.key, None)
        else:
            entries[change.key] = change.current
    return MetadataState(load.revision, entries)


class LayerMetadataView:
    """Single-flight metadata refresh for one backend (primary or layer).

    - STATIC: loads once, then serves the cached state forever.
    - REVISIONED: ``load_metadata(current_revision | None)``; a same-revision
      load is served from the cached state without a backend call (handled by
      the backend returning an empty PATCH, see spec contract 5).
    - ALWAYS: ``list_info()`` every refresh, tagged by a local generation so
      callers see a fresh effective revision each time.

    Concurrent refreshes collapse into one backend load: an caller that
    observed a stale epoch before acquiring the lock returns whatever the
    in-flight load published instead of loading again. A cancelled caller
    never publishes a half state -- the load is awaited to completion by the
    lock holder, and non-lock holders only read the published result."""

    def __init__(
        self,
        backend: Any,
        policy: LayerRefreshPolicy,
        *,
        info_key: "Callable[[Any], Any]",
        initializer: "StorageInitializer | None" = None,
        revision_source: "RevisionSource | None" = None,
    ) -> None:
        if policy is LayerRefreshPolicy.REVISIONED and not isinstance(
            backend, StorageMetadataBackend
        ):
            raise ValueError(
                "a revisioned layer requires a StorageMetadataBackend"
            )
        self.backend = backend
        self.policy = policy
        self.info_key = info_key
        self.initializer = initializer
        self.revision_source = revision_source
        self._state: "MetadataState[Any, Any, Any] | None" = None
        self._lock = asyncio.Lock()
        self._epoch = 0
        self._always_generation = 0

    async def initialize(self, *args: object) -> None:
        if self.initializer is not None:
            await self.initializer(*args)

    async def refresh(self) -> "MetadataState[Any, Any, Any] | None":
        if self.policy is LayerRefreshPolicy.STATIC and self._state is not None:
            return self._state
        observed = self._epoch
        async with self._lock:
            if self._state is not None and self._epoch != observed:
                return self._state
            return await self._refresh_locked()

    async def _refresh_locked(self) -> "MetadataState[Any, Any, Any] | None":
        if (
            self.revision_source is not None
            and self._state is not None
            and isinstance(self.backend, StorageMetadataBackend)
        ):
            # Cheap short-circuit: ask the source whether the held state's
            # revision is still current. When it is, reuse the held state and
            # skip load_metadata entirely (this is the path that intercepts a
            # hot get() loop). None/changed -> fall through to a real load.
            current = await self.revision_source.current()
            if current is not None and current == self._state.revision:
                return self._state
        if isinstance(self.backend, StorageMetadataBackend):
            # REVISIONED patches against the held revision; STATIC loads a full
            # snapshot once (refresh()'s early return serves it thereafter).
            after = None if self.policy is LayerRefreshPolicy.STATIC else (
                None if self._state is None else self._state.revision
            )
            load = await self.backend.load_metadata(after)
            state = apply_metadata_load(self._state, load, info_key=self.info_key)
            self._state = state
            self._epoch += 1
            return state
        # Non-metadata reader (ALWAYS): reload the full entry set every refresh
        # under a fresh generation so the effective revision always changes.
        infos = await self.backend.list_info()
        self._always_generation += 1
        revision: Any = ("always", self._always_generation)
        entries = {self.info_key(info): info for info in infos}
        state = MetadataState(revision, entries)
        self._state = state
        self._epoch += 1
        return state

    async def head_revision(self) -> Any:
        """The backend's current revision with no entry/change data loaded.

        REVISIONED/STATIC backends are ``StorageMetadataBackend`` instances with
        a cheap head probe; a STATIC backend's head is immutable and always
        matches the held state, so a fresh probe is both correct and cheap.
        ALWAYS layers (plain ``StorageReader``) have no such probe, so this
        returns ``None`` and the composition falls back to a full refresh for
        accuracy."""
        if isinstance(self.backend, StorageMetadataBackend):
            return await self.backend.head_revision()
        return None

    def invalidate(self) -> None:
        """Drop the cached state so the next refresh reloads from the backend.
        Used after a write on a primary that cannot serve a reliable patch
        (the unversioned case)."""
        self._state = None
        self._epoch += 1


__all__ = [
    "BatchStorageReader",
    "LayerMetadataView",
    "LayerRefreshPolicy",
    "MetadataLoad",
    "MetadataLoadMode",
    "MetadataState",
    "StorageChange",
    "StorageInitializer",
    "StorageMetadataBackend",
    "StorageReader",
    "apply_metadata_load",
]
