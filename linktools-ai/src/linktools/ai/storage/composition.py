"""Storage topology and consistency assembly shared by domain stores."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Generic, Protocol, TypeVar, cast, runtime_checkable

from ..errors import StorageCapabilityError
from .cache import ContentCache, ContentCacheKey, read_cache, write_cache
from .multi import MultiBackend, StorageLayer, StorageReader
from .revision import (
    ChangeSource,
    CompositeRevisionSource,
    MetadataSnapshot,
    MetadataState,
    Revision,
    RevisionSource,
)

KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")
InfoT = TypeVar("InfoT")
ChangeT = TypeVar("ChangeT")
PrimaryT = TypeVar("PrimaryT")
WriterT = TypeVar("WriterT")


class StorageAdapter(Protocol[KeyT, ValueT, InfoT, ChangeT]):
    def info_key(self, info: InfoT) -> KeyT: ...

    def change_key(self, change: ChangeT) -> KeyT: ...

    def change_value(self, change: ChangeT) -> InfoT | None: ...

    def value_info(self, value: ValueT) -> InfoT: ...


class StorageCacheAdapter(Protocol[KeyT, ValueT, InfoT]):
    def cache_key(self, key: KeyT, info: InfoT) -> ContentCacheKey: ...

    def cache_content(self, value: ValueT) -> bytes: ...

    def from_cache(self, info: InfoT, content: bytes) -> ValueT: ...


@runtime_checkable
class StorageInitializer(Protocol):
    async def initialize_storage(self, *args: object) -> None: ...


class StorageComposition(
    Generic[PrimaryT, KeyT, ValueT, InfoT, ChangeT, WriterT]
):
    """Compose optional storage capabilities without imposing them on backends."""

    def __init__(
        self,
        primary: PrimaryT,
        *,
        writer: WriterT | None = None,
        overlays: tuple[StorageLayer[KeyT, ValueT, InfoT], ...] = (),
        revision: RevisionSource | None = None,
        changes: ChangeSource[ChangeT] | None = None,
        cache: ContentCache | None = None,
        adapter: StorageAdapter[KeyT, ValueT, InfoT, ChangeT] | None = None,
        cache_adapter: StorageCacheAdapter[KeyT, ValueT, InfoT] | None = None,
        preload_batch_size: int = 100,
        preload_concurrency: int = 8,
    ) -> None:
        if changes is not None and revision is None:
            raise ValueError("a change source requires a primary revision source")
        if adapter is None and (overlays or revision or changes or cache):
            raise ValueError("composed storage capabilities require an adapter")
        if cache is not None and cache_adapter is None:
            raise ValueError("a content cache requires a cache adapter")
        if preload_batch_size < 1:
            raise ValueError("preload_batch_size must be positive")
        if preload_concurrency < 1:
            raise ValueError("preload_concurrency must be positive")
        self.primary = primary
        self.writer = writer
        self.overlays = overlays
        self.cache = cache
        self.adapter = adapter
        self.cache_adapter = cache_adapter
        self.preload_batch_size = preload_batch_size
        self.preload_concurrency = preload_concurrency
        self._preloaded: dict[KeyT, ContentCacheKey] = {}
        self._primary_revision = revision
        self._changes = changes
        self.reader: MultiBackend[KeyT, ValueT, InfoT] | None = None
        self.revision: RevisionSource | None = None
        self.snapshot: MetadataSnapshot[KeyT, InfoT, ChangeT] | None = None
        if adapter is not None:
            self.reader = MultiBackend(
                cast("StorageReader[KeyT, ValueT, InfoT]", primary),
                overlays,
                info_key=adapter.info_key,
            )
            sources = (
                *((revision,) if revision is not None else ()),
                *self.reader.overlay_revisions,
            )
            if len(sources) == 1:
                self.revision = sources[0]
            elif sources:
                self.revision = CompositeRevisionSource(*sources)
            self.snapshot = MetadataSnapshot(
                self.reader,
                revision=self.revision,
                changes=changes if not overlays else None,
                always_refresh=self.reader.always_refresh,
                entry_key=adapter.info_key,
                change_key=adapter.change_key,
                change_value=adapter.change_value,
            )

    @property
    def backend(self) -> PrimaryT:
        return self.primary

    async def initialize(self, *args: object) -> None:
        components = (
            (self.primary, None),
            (self.writer, None),
            *((layer.reader, layer.initializer) for layer in self.overlays),
            (self._primary_revision, None),
            (self._changes, None),
            *((layer.revision, None) for layer in self.overlays),
        )
        initialized: set[int] = set()
        for component, initializer in components:
            if component is None or id(component) in initialized:
                continue
            initialized.add(id(component))
            if initializer is not None:
                await initializer(*args)
            elif isinstance(component, StorageInitializer):
                await component.initialize_storage(*args)

    def require_writer(self) -> WriterT:
        if self.writer is None:
            raise StorageCapabilityError("storage is read-only")
        return self.writer

    def _require_reader(
        self,
    ) -> tuple[
        MultiBackend[KeyT, ValueT, InfoT],
        MetadataSnapshot[KeyT, InfoT, ChangeT],
        StorageAdapter[KeyT, ValueT, InfoT, ChangeT],
    ]:
        if self.reader is None or self.snapshot is None or self.adapter is None:
            raise StorageCapabilityError("storage composition has no reader adapter")
        return self.reader, self.snapshot, self.adapter

    async def refresh(
        self,
        *,
        preload: bool = False,
    ) -> MetadataState[KeyT, InfoT] | None:
        _, snapshot, _ = self._require_reader()
        state = await snapshot.refresh()
        if not preload or state is None:
            return state
        return await self._preload(state)

    async def _preload(
        self,
        state: MetadataState[KeyT, InfoT],
    ) -> MetadataState[KeyT, InfoT]:
        reader, snapshot, adapter = self._require_reader()
        cache_adapter = self.cache_adapter
        if self.cache is None or cache_adapter is None:
            raise StorageCapabilityError(
                "storage composition has no content cache"
            )
        if self.revision is None:
            raise StorageCapabilityError(
                "content preload requires a revision source"
            )
        if not state.cacheable:
            return state
        for _attempt in range(3):
            if not state.cacheable:
                return state
            identities = {
                key: cache_adapter.cache_key(key, info)
                for key, info in state.entries.items()
            }
            keys = tuple(
                [
                    key
                    for key, identity in identities.items()
                    if (
                        self._preloaded.get(key) != identity
                        or await read_cache(self.cache, identity) is None
                    )
                ]
            )
            loaded: dict[KeyT, ValueT] = {}
            for batch in self._batches(keys):
                loaded.update(
                    await reader.get_many(
                        batch,
                        concurrency=self.preload_concurrency,
                    )
                )
            valid = {
                key: value
                for key, value in loaded.items()
                if (
                    key in state.entries
                    and adapter.value_info(value) == state.entries[key]
                )
            }
            if await self.revision.current_revision() != state.revision:
                state = await snapshot.refresh()
                if state is None:
                    return state
                if not state.cacheable:
                    return state
                continue
            written = await self._write_many(valid, identities)
            self._preloaded = {
                key: identity
                for key, identity in identities.items()
                if key not in keys or key in written
            }
            return state
        return MetadataState(
            state.revision,
            state.entries,
            cacheable=False,
        )

    def _batches(
        self,
        keys: tuple[KeyT, ...],
    ) -> Iterable[tuple[KeyT, ...]]:
        size = self.preload_batch_size
        for offset in range(0, len(keys), size):
            yield keys[offset : offset + size]

    async def _write_many(
        self,
        values: dict[KeyT, ValueT],
        identities: dict[KeyT, ContentCacheKey],
    ) -> set[KeyT]:
        cache_adapter = self.cache_adapter
        if cache_adapter is None:
            return set()
        written: set[KeyT] = set()
        for key, value in values.items():
            if await write_cache(
                self.cache,
                identities[key],
                cache_adapter.cache_content(value),
            ):
                written.add(key)
        return written

    async def get_info(self, key: KeyT) -> InfoT | None:
        state = await self.refresh()
        return None if state is None else state.entries.get(key)

    async def list_info(self, *, preload: bool = False) -> tuple[InfoT, ...]:
        state = await self.refresh(preload=preload)
        if state is None:
            return ()
        return tuple(
            state.entries[key]
            for key in sorted(state.entries, key=lambda value: str(value))
        )

    async def current_revision(self) -> Revision | None:
        if self.revision is None:
            return None
        return await self.revision.current_revision()

    async def get(self, key: KeyT) -> ValueT | None:
        reader, _, adapter = self._require_reader()
        state = await self.refresh()
        info = None if state is None else state.entries.get(key)
        if info is None:
            return await reader.get(key)
        cache_adapter = self.cache_adapter
        cache_key = (
            None
            if cache_adapter is None
            else cache_adapter.cache_key(key, info)
        )
        if state.cacheable and cache_adapter is not None:
            cached = await read_cache(self.cache, cache_key)
            if cached is not None:
                return cache_adapter.from_cache(info, cached)
        value = await reader.get(key)
        if value is None or adapter.value_info(value) != info:
            return None
        if state.cacheable and cache_adapter is not None:
            await write_cache(
                self.cache,
                cache_key,
                cache_adapter.cache_content(value),
            )
        return value


__all__ = [
    "StorageAdapter",
    "StorageCacheAdapter",
    "StorageComposition",
    "StorageInitializer",
]
