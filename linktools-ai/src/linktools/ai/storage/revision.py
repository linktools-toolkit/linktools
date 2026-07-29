"""Optional revision, delta, and metadata snapshot capabilities."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Generic, Mapping, Protocol, TypeVar

EntryT = TypeVar("EntryT")
ChangeT = TypeVar("ChangeT")
KeyT = TypeVar("KeyT")
Revision = int | str


class SnapshotRequired(Exception):
    """The change source cannot provide a complete delta."""


class RevisionSource(Protocol):
    async def current_revision(self) -> Revision: ...


class ChangeSource(Protocol[ChangeT]):
    async def list_changes(
        self,
        *,
        after_revision: Revision,
        through_revision: Revision,
    ) -> tuple[ChangeT, ...]: ...


class CompositeRevisionSource:
    def __init__(self, *sources: RevisionSource) -> None:
        if not sources:
            raise ValueError("at least one revision source is required")
        self._sources = sources

    async def current_revision(self) -> str:
        values = tuple(
            [await source.current_revision() for source in self._sources]
        )
        payload = "|".join(
            f"{index}:{type(source).__qualname__}:{value}"
            for index, (source, value) in enumerate(
                zip(self._sources, values, strict=True)
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MetadataState(Generic[KeyT, EntryT]):
    revision: Revision | None
    entries: Mapping[KeyT, EntryT]
    cacheable: bool = True


class MetadataSnapshot(Generic[KeyT, EntryT, ChangeT]):
    """Revision-gated metadata with optional delta refresh."""

    def __init__(
        self,
        reader: Any,
        *,
        revision: RevisionSource | None = None,
        changes: ChangeSource[ChangeT] | None = None,
        always_refresh: bool = False,
        entry_key: Callable[[EntryT], KeyT] = lambda entry: entry.path,
        change_key: Callable[[ChangeT], KeyT] = lambda change: change.path,
        change_value: Callable[[ChangeT], EntryT | None] = lambda change: change.info,
    ) -> None:
        if changes is not None and revision is None:
            raise ValueError("a change source requires a revision source")
        self.reader = reader
        self.revision = revision
        self.changes = changes
        self.always_refresh = always_refresh
        self.entry_key = entry_key
        self.change_key = change_key
        self.change_value = change_value
        self._state: MetadataState[KeyT, EntryT] | None = None
        self._refresh_lock = asyncio.Lock()

    async def _target_revision(self) -> Revision | None:
        if self.revision is None:
            return None
        return await self.revision.current_revision()

    async def _full(
        self,
        revision: Revision | None,
        *,
        cacheable: bool = True,
    ) -> MetadataState[KeyT, EntryT]:
        values = await self.reader.list_info()
        return MetadataState(
            revision,
            {self.entry_key(value): value for value in values},
            cacheable,
        )

    async def refresh(self) -> MetadataState[KeyT, EntryT] | None:
        target = await self._target_revision()
        if (
            not self.always_refresh
            and target is not None
            and self._state is not None
            and self._state.revision == target
        ):
            return self._state
        async with self._refresh_lock:
            for _attempt in range(3):
                target = await self._target_revision()
                if (
                    not self.always_refresh
                    and target is not None
                    and self._state is not None
                    and self._state.revision == target
                ):
                    return self._state
                current = self._state
                try:
                    if (
                        current is None
                        or target is None
                        or self.always_refresh
                        or self.changes is None
                    ):
                        candidate = await self._full(target)
                    else:
                        delta = await self.changes.list_changes(
                            after_revision=current.revision,
                            through_revision=target,
                        )
                        if len(delta) > max(1, int(len(current.entries) * 0.25)):
                            candidate = await self._full(target)
                        else:
                            entries = dict(current.entries)
                            for change in delta:
                                key = self.change_key(change)
                                value = self.change_value(change)
                                if value is None:
                                    entries.pop(key, None)
                                else:
                                    entries[key] = value
                            candidate = MetadataState(target, entries)
                except SnapshotRequired:
                    candidate = await self._full(target)
                if self.revision is None:
                    self._state = candidate
                    return candidate
                if await self._target_revision() == target:
                    self._state = candidate
                    return candidate
            # The source stayed unstable through the bounded retries. Serve one
            # uncached repository read for this request without publishing it.
            return await self._full(
                await self._target_revision(),
                cacheable=False,
            )

    async def get(self, key: KeyT) -> EntryT | None:
        state = await self.refresh()
        return None if state is None else state.entries.get(key)


class RevisionCacheSource(Protocol):
    async def revision(self) -> str: ...

    async def list_ids(self, suffix: str) -> tuple[str, ...]: ...

    async def read(self, path: str) -> str: ...


class RevisionCacheCodec(Protocol, Generic[EntryT]):
    def decode(self, item_id: str, raw: str) -> EntryT: ...


class RevisionCache(Generic[EntryT]):
    """Revision-invalidated cache for parsed source items."""

    def __init__(
        self,
        source: RevisionCacheSource,
        codec: RevisionCacheCodec[EntryT],
        *,
        suffix: str = ".md",
        source_name: str | None = None,
        metrics: Any | None = None,
    ) -> None:
        self._source = source
        self._codec = codec
        self._suffix = suffix
        self._source_name = source_name or type(source).__name__
        self._cache: dict[tuple[str, str], EntryT] = {}
        self._cached_revision: str | None = None
        self._ids: tuple[str, ...] | None = None
        self._refresh_lock = asyncio.Lock()
        self._inflight: dict[tuple[str, str], asyncio.Future[EntryT]] = {}
        self._metrics = metrics

    @property
    def source_name(self) -> str:
        return self._source_name

    async def _ensure_fresh(self) -> str:
        async with self._refresh_lock:
            revision = await self._source.revision()
            if revision != self._cached_revision:
                self._cache.clear()
                self._ids = None
                self._cached_revision = revision
                if self._metrics is not None:
                    self._metrics.counter("spec_revision_refresh_total")
            return revision

    async def list_ids(self) -> tuple[str, ...]:
        await self._ensure_fresh()
        if self._ids is None:
            self._ids = await self._source.list_ids(self._suffix)
        return self._ids

    async def get(self, item_id: str) -> EntryT:
        revision = await self._ensure_fresh()
        key = (item_id, revision)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        future = self._inflight.get(key)
        if future is None:
            future = asyncio.get_running_loop().create_future()
            self._inflight[key] = future
            try:
                value = self._codec.decode(
                    item_id,
                    await self._source.read(f"{item_id}{self._suffix}"),
                )
                self._cache[key] = value
                future.set_result(value)
            except BaseException as exc:
                future.set_exception(exc)
            finally:
                self._inflight.pop(key, None)
        return await future


__all__ = [
    "ChangeSource",
    "CompositeRevisionSource",
    "MetadataSnapshot",
    "MetadataState",
    "Revision",
    "RevisionCache",
    "RevisionCacheCodec",
    "RevisionCacheSource",
    "RevisionSource",
    "SnapshotRequired",
]
