#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Storage topology assembly: per-layer metadata, merge, read, write.

``StorageComposition`` is the unified base for every domain store. It wires a
primary reader with optional ordered layers (primary-first fallback), a writer,
and a content cache, and owns: parallel per-layer metadata refresh, owner-aware
entry merge, effective revision, owner-directed ``get``/``get_many`` with the
metadata-miss rule, preload via ``contains_many``, and write post-processing.

Layers are ordered fallbacks; earlier readers win."""

import asyncio
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, runtime_checkable
from ..errors import StorageFeatureSupportError
from .cache import ContentCache, contains_many, read_cache, write_cache
from .multi import BatchStorageWriter, StorageReader, batch_get
from .revision import (
    LayerMetadataView,
    LayerRefreshPolicy,
    RevisionSource,
    StorageInitializer,
    StorageMetadataBackend,
    _BackendHeadRevisionSource,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from .cache import ContentCacheKey

KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")
InfoT = TypeVar("InfoT")
WriterT = TypeVar("WriterT")


@runtime_checkable
class StorageAdapter(Protocol[KeyT, ValueT, InfoT]):
    def info_key(self, info: InfoT) -> KeyT: ...

    def value_info(self, value: ValueT) -> InfoT: ...


@runtime_checkable
class StorageCacheAdapter(Protocol[KeyT, ValueT, InfoT]):
    def cache_key(self, key: KeyT, info: InfoT) -> "ContentCacheKey": ...

    def cache_content(self, value: ValueT) -> bytes: ...

    def from_cache(self, info: InfoT, content: bytes) -> ValueT: ...


@dataclass(frozen=True, slots=True)
class StorageLayer(Generic[KeyT, ValueT, InfoT]):
    """One read-only layer behind the primary. ``backend`` is a
    :class:`StorageReader`; ``refresh`` selects how its metadata is loaded."""

    backend: "StorageReader[KeyT, ValueT, InfoT]"
    refresh: LayerRefreshPolicy = LayerRefreshPolicy.STATIC
    initializer: "StorageInitializer | None" = None


@dataclass(frozen=True, slots=True)
class EffectiveMetadataState(Generic[KeyT, InfoT]):
    """The merged view across primary + layers at one refresh. ``owners`` maps
    each key to the index of the layer (0 = primary) that won it, so a read
    goes straight to that backend instead of probing each layer in order."""

    revision: "int | str"
    entries: "Mapping[KeyT, InfoT]"
    owners: "Mapping[KeyT, int]"


def _primary_policy(primary: object) -> LayerRefreshPolicy:
    # A primary that can serve metadata is REVISIONED (patch-able); otherwise
    # its metadata is reloaded fully every refresh (ALWAYS).
    return (
        LayerRefreshPolicy.REVISIONED
        if isinstance(primary, StorageMetadataBackend)
        else LayerRefreshPolicy.ALWAYS
    )


def _no_adapter_info_key(info: object) -> object:
    # Placeholder for a StorageComposition built with no adapter: every method
    # that would exercise a LayerMetadataView's refresh (and thus call this)
    # first calls _require_adapter(), which raises before getting here. Raises
    # rather than guessing at the info's shape, so a future change that breaks
    # that guarantee fails loudly instead of silently mis-keying entries.
    raise StorageFeatureSupportError("storage composition has no adapter")


class StorageComposition(Generic[KeyT, ValueT, InfoT, WriterT]):
    """Compose a primary reader with layers, a writer, and a content cache."""

    def __init__(
        self,
        primary: "StorageReader[KeyT, ValueT, InfoT]",
        *,
        writer: "WriterT | None" = None,
        layers: "tuple[StorageLayer[KeyT, ValueT, InfoT], ...]" = (),
        adapter: "StorageAdapter[KeyT, ValueT, InfoT] | None" = None,
        revision_source: "RevisionSource | None" = None,
        cache: "ContentCache | None" = None,
        cache_adapter: "StorageCacheAdapter[KeyT, ValueT, InfoT] | None" = None,
        cache_concurrency: int = 16,
        preload_batch_size: int = 100,
        preload_concurrency: int = 8,
    ) -> None:
        if adapter is None and (layers or cache):
            raise ValueError("composed storage features require an adapter")
        if cache is not None and cache_adapter is None:
            raise ValueError("a content cache requires a cache adapter")
        if preload_batch_size < 1:
            raise ValueError("preload_batch_size must be positive")
        if preload_concurrency < 1:
            raise ValueError("preload_concurrency must be positive")
        if cache_concurrency < 1:
            raise ValueError("cache_concurrency must be positive")
        self.primary = primary
        self.writer = writer
        self.layers = layers
        self.cache = cache
        self.adapter = adapter
        self.cache_adapter = cache_adapter
        self.preload_batch_size = preload_batch_size
        self.preload_concurrency = preload_concurrency
        self.cache_concurrency = cache_concurrency
        info_key = adapter.info_key if adapter is not None else _no_adapter_info_key
        # Auto-wire the default revision source (cheap head_revision probe)
        # when none is injected: a plain SpecStore(backend) then short-circuits
        # unchanged-revision refreshes for free. Layers do not get a source --
        # the primary carries the revisioned patch load that benefits.
        primary_source = revision_source or _BackendHeadRevisionSource(primary)
        self._views: "tuple[LayerMetadataView, ...]" = (
            LayerMetadataView(
                primary,
                _primary_policy(primary),
                info_key=info_key,
                revision_source=primary_source,
            ),
            *(
                LayerMetadataView(
                    layer.backend,
                    layer.refresh,
                    info_key=info_key,
                    initializer=layer.initializer,
                )
                for layer in layers
            ),
        )
        # Marker per key of the cache identity already preloaded into the cache.
        self._preloaded: "dict[KeyT, ContentCacheKey]" = {}

    @property
    def primary_view(self) -> LayerMetadataView:
        return self._views[0]

    async def initialize(self, *args: object) -> None:
        seen: "set[int]" = set()
        for view in self._views:
            backend = view.backend
            if id(backend) in seen:
                continue
            seen.add(id(backend))
            await view.initialize(*args)

    def require_writer(self) -> WriterT:
        if self.writer is None:
            raise StorageFeatureSupportError("storage is read-only")
        return self.writer

    def _require_adapter(
        self,
    ) -> "tuple[StorageAdapter[KeyT, ValueT, InfoT], StorageCacheAdapter[KeyT, ValueT, InfoT] | None]":
        if self.adapter is None:
            raise StorageFeatureSupportError("storage composition has no adapter")
        return self.adapter, self.cache_adapter

    async def refresh(self) -> "EffectiveMetadataState[KeyT, InfoT] | None":
        adapter, _ = self._require_adapter()
        states = await asyncio.gather(*(view.refresh() for view in self._views))
        if all(state is None for state in states):
            return None
        entries: "dict[KeyT, InfoT]" = {}
        owners: "dict[KeyT, int]" = {}
        for index, state in enumerate(states):
            if state is None:
                continue
            for key, info in state.entries.items():
                if key not in entries:
                    entries[key] = info
                    owners[key] = index
        return EffectiveMetadataState(
            self._effective_revision(
                tuple(state.revision if state is not None else None for state in states)
            ),
            entries,
            owners,
        )

    def _effective_revision(
        self,
        revisions: "tuple[Any, ...]",
    ) -> "int | str":
        # Single primary: its revision directly. Multiple layers: a canonical
        # hash over every loaded layer's revision so any change in any layer
        # changes the effective revision. ALWAYS layers use a local generation,
        # so they always change the effective revision.
        loaded = tuple(revision for revision in revisions if revision is not None)
        if len(loaded) <= 1:
            return loaded[0] if loaded else 0
        payload = "|".join(str(revision) for revision in loaded)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def current_revision(self) -> "int | str":
        # Cheap path: probe each layer's head revision without loading any
        # entry/change data (the point at which an external revision cache is
        # validated). A layer with no cheap probe (ALWAYS) forces a full refresh.
        heads = await asyncio.gather(*(view.head_revision() for view in self._views))
        if any(head is None for head in heads):
            state = await self.refresh()
            return 0 if state is None else state.revision
        return self._effective_revision(heads)

    async def get(self, key: KeyT) -> "ValueT | None":
        return await self._get_with_retry(key, retried=False)

    async def _get_with_retry(self, key: KeyT, *, retried: bool) -> "ValueT | None":
        adapter, cache_adapter = self._require_adapter()
        state = await self.refresh()
        if state is None or key not in state.entries:
            # metadata is loaded and this key is absent: do NOT probe backends.
            return None
        owner = state.owners[key]
        info = state.entries[key]
        backend = self._views[owner].backend
        cache_key = (
            None if cache_adapter is None else cache_adapter.cache_key(key, info)
        )
        if self.cache is not None and cache_adapter is not None:
            cached = await read_cache(self.cache, cache_key)
            if cached is not None:
                return cache_adapter.from_cache(info, cached)
        value = await backend.get(key)
        if value is None:
            return None
        if adapter.value_info(value) != info:
            # Content raced (metadata is stale). Invalidate the owner view so
            # the recursive _get_with_retry's refresh() loads fresh metadata
            # instead of being short-circuited by a revision source that still
            # holds the stale revision. Retry once; a second mismatch means the
            # state is genuinely inconsistent -- treat as absent rather than
            # serving a wrong version.
            if not retried:
                self._views[owner].invalidate()
                return await self._get_with_retry(key, retried=True)
            return None
        if self.cache is not None and cache_adapter is not None:
            await write_cache(self.cache, cache_key, cache_adapter.cache_content(value))
        return value

    async def get_many(self, keys: "tuple[KeyT, ...]") -> "dict[KeyT, ValueT]":
        if not keys:
            return {}
        adapter, cache_adapter = self._require_adapter()
        state = await self.refresh()
        result: "dict[KeyT, ValueT]" = {}
        if state is None:
            return result
        wanted = tuple(dict.fromkeys(k for k in keys if k in state.entries))
        if not wanted:
            return result
        if self.cache is not None and cache_adapter is not None:

            async def _read_one(key: KeyT) -> "tuple[KeyT, bytes | None]":
                return key, await read_cache(
                    self.cache, cache_adapter.cache_key(key, state.entries[key])
                )

            miss: "list[KeyT]" = []
            for key, cached in await asyncio.gather(*(_read_one(k) for k in wanted)):
                if cached is not None:
                    result[key] = cache_adapter.from_cache(state.entries[key], cached)
                else:
                    miss.append(key)
        else:
            miss = list(wanted)
        if not miss:
            return result
        loaded = await self._load_by_owner(tuple(miss), state)
        # Track owners whose content raced the held metadata (stale). After the
        # loop, invalidate each once so the next read loads fresh metadata
        # instead of re-serving the stale state. The mismatched keys are omitted
        # from this batch's result (the caller sees them as absent and re-reads);
        # this mirrors _get_with_retry's contract without a per-key recursive
        # reload, which would defeat the batch path's purpose.
        raced_owners: "set[int]" = set()
        for key in miss:
            value = loaded.get(key)
            if value is None or key not in state.entries:
                continue
            if adapter.value_info(value) != state.entries[key]:
                raced_owners.add(state.owners[key])
                continue
            result[key] = value
            if self.cache is not None and cache_adapter is not None:
                await write_cache(
                    self.cache,
                    cache_adapter.cache_key(key, state.entries[key]),
                    cache_adapter.cache_content(value),
                )
        for owner in raced_owners:
            self._views[owner].invalidate()
        return result

    def _group_by_owner(
        self,
        keys: "tuple[KeyT, ...]",
        state: "EffectiveMetadataState[KeyT, InfoT]",
    ) -> "dict[int, tuple[KeyT, ...]]":
        groups: "dict[int, list[KeyT]]" = {}
        for key in keys:
            groups.setdefault(state.owners[key], []).append(key)
        return {owner: tuple(keys) for owner, keys in groups.items()}

    async def _load_by_owner(
        self,
        keys: "tuple[KeyT, ...]",
        state: "EffectiveMetadataState[KeyT, InfoT]",
        *,
        batch_size: "int | None" = None,
    ) -> "dict[KeyT, ValueT]":
        # Load every key in ``keys`` from its owning layer's backend, grouping by
        # owner and running the owner groups (and their sub-batches) in parallel.
        # Shared by get_many and preload. ``batch_size`` splits each owner's
        # group into bounded chunks (preload uses it to cap SQL IN-clause size);
        # None reads a whole group in one batch_get.
        loaded: "dict[KeyT, ValueT]" = {}
        by_owner = self._group_by_owner(keys, state)

        async def _owner_load(owner: int, group: "tuple[KeyT, ...]") -> None:
            loaded.update(
                await batch_get(
                    self._views[owner].backend,
                    group,
                    concurrency=self.preload_concurrency,
                )
            )

        batches = (
            ((owner, group) for owner, group in by_owner.items())
            if batch_size is None
            else (
                (owner, chunk)
                for owner, group in by_owner.items()
                for chunk in self._batches(group)
            )
        )
        await asyncio.gather(*(_owner_load(owner, batch) for owner, batch in batches))
        return loaded

    async def list_info(self, *, preload: bool = False) -> "tuple[InfoT, ...]":
        state = await self.refresh()
        if state is None:
            return ()
        if preload:
            await self._preload(state)
        return tuple(
            state.entries[key]
            for key in sorted(state.entries, key=lambda value: str(value))
        )

    async def list_info_with_owners(
        self, *, preload: bool = False
    ) -> "tuple[tuple[InfoT, int], ...]":
        """Same entries as :meth:`list_info`, each paired with its owning layer
        index (0 = primary, >0 = a fallback layer). A caller that needs to tell
        primary-managed entries apart from layer-provided ones (e.g. DB-customized
        vs builtin-default, when a filesystem layer supplies read-through
        defaults) uses this instead of :meth:`list_info`."""
        state = await self.refresh()
        if state is None:
            return ()
        if preload:
            await self._preload(state)
        return tuple(
            (state.entries[key], state.owners.get(key, 0))
            for key in sorted(state.entries, key=lambda value: str(value))
        )

    async def _preload(self, state: "EffectiveMetadataState[KeyT, InfoT]") -> None:
        adapter, cache_adapter = self._require_adapter()
        if self.cache is None or cache_adapter is None:
            return
        identities = {
            key: cache_adapter.cache_key(key, info)
            for key, info in state.entries.items()
        }
        present = await contains_many(self.cache, tuple(identities.values()))
        missing = tuple(
            key
            for key, identity in identities.items()
            if identity not in present or self._preloaded.get(key) != identity
        )
        if not missing:
            return
        loaded = await self._load_by_owner(
            missing, state, batch_size=self.preload_batch_size
        )
        # Bounded concurrent cache puts.
        semaphore = asyncio.Semaphore(self.cache_concurrency)
        verified: "set[KeyT]" = set()

        async def _cache(key: KeyT) -> None:
            async with semaphore:
                value = loaded.get(key)
                if value is None or adapter.value_info(value) != state.entries[key]:
                    return
                if await write_cache(
                    self.cache,
                    identities[key],
                    cache_adapter.cache_content(value),
                ):
                    verified.add(key)

        await asyncio.gather(*(_cache(key) for key in missing))
        # Verify the cache actually holds each written identity before marking
        # it preloaded: a cache ``put`` may silently drop an oversized blob
        # (returns without raising, so write_cache reports True) yet store
        # nothing. Without this re-check such a key would be permanently marked
        # preloaded and never re-attempted, silently degrading to a per-read
        # origin fetch forever.
        if verified:
            actually_present = await contains_many(
                self.cache, tuple(identities[key] for key in missing if key in verified)
            )
        else:
            actually_present = frozenset()
        for key in missing:
            if key in verified and identities[key] in actually_present:
                self._preloaded[key] = identities[key]

    def _batches(self, keys: "tuple[KeyT, ...]") -> "Iterable[tuple[KeyT, ...]]":
        size = self.preload_batch_size
        for offset in range(0, len(keys), size):
            yield keys[offset : offset + size]

    # ---- writes --------------------------------------------------------

    async def put(self, value: ValueT) -> ValueT:
        writer = self.require_writer()
        result, revision = await writer.put(value)
        adapter, _ = self._require_adapter()
        await self._after_put(result, adapter, revision)
        return result

    async def delete(self, key: KeyT) -> None:
        writer = self.require_writer()
        revision = await writer.delete(key)
        await self._after_delete(key, revision)

    async def reset(self, values: "tuple[ValueT, ...]") -> None:
        writer = self.require_writer()
        revision = await writer.reset(values)
        await self._after_reset(revision)

    async def apply_batch(
        self,
        puts: "tuple[ValueT, ...]",
        deletes: "tuple[KeyT, ...]",
    ) -> None:
        # When the writer declares the BatchStorageWriter capability, delegate to
        # its atomic apply_batch (one transaction, one revision) and run the
        # _after_batch hook once. Otherwise fall back to per-op put/delete
        # through this composition's own put()/delete() -- each carries its own
        # _after_* hook, so preloaded markers and revision-source notification
        # stay correct, at the cost of N separate transactions/round trips.
        writer = self.require_writer()
        if isinstance(writer, BatchStorageWriter):
            revision = await writer.apply_batch(puts, deletes)
            adapter, _ = self._require_adapter()
            put_keys = tuple(
                adapter.info_key(adapter.value_info(value)) for value in puts
            )
            await self._after_batch(put_keys, deletes, revision)
            return
        for value in puts:
            await self.put(value)
        for key in deletes:
            await self.delete(key)

    async def _after_put(
        self,
        value: ValueT,
        adapter: "StorageAdapter[KeyT, ValueT, InfoT]",
        revision: "Any | None",
    ) -> None:
        # Clear the preloaded marker for the written key so a later preload
        # re-reads it. A revisioned primary keeps its old state so the next
        # refresh fetches the patch from the prior revision; an unversioned
        # primary invalidates fully.
        key = adapter.info_key(adapter.value_info(value))
        self._preloaded.pop(key, None)
        if self.primary_view.policy is not LayerRefreshPolicy.REVISIONED:
            self.primary_view.invalidate()
        await self._notify_revision_source(revision)

    async def _after_delete(self, key: KeyT, revision: "Any | None") -> None:
        self._preloaded.pop(key, None)
        if self.primary_view.policy is not LayerRefreshPolicy.REVISIONED:
            self.primary_view.invalidate()
        await self._notify_revision_source(revision)

    async def _after_reset(self, revision: "Any | None") -> None:
        self._preloaded.clear()
        if self.primary_view.policy is not LayerRefreshPolicy.REVISIONED:
            self.primary_view.invalidate()
        await self._notify_revision_source(revision)

    async def _after_batch(
        self,
        put_keys: "tuple[KeyT, ...]",
        delete_keys: "tuple[KeyT, ...]",
        revision: "Any | None",
    ) -> None:
        # Clear preloaded markers for every touched key (puts + deletes) so a
        # later preload re-reads them; a revisioned primary keeps its old state
        # so the next refresh fetches the patch, an unversioned primary
        # invalidates fully. One revision-source notification per batch.
        for key in put_keys:
            self._preloaded.pop(key, None)
        for key in delete_keys:
            self._preloaded.pop(key, None)
        if self.primary_view.policy is not LayerRefreshPolicy.REVISIONED:
            self.primary_view.invalidate()
        await self._notify_revision_source(revision)

    async def _notify_revision_source(self, revision: "Any | None") -> None:
        # Skipped when no source is wired; the default source no-ops here (it
        # reads head live).
        source = self.primary_view.revision_source
        if source is None:
            return
        if revision is not None:
            await source.revision_bumped(revision)


__all__ = [
    "EffectiveMetadataState",
    "StorageAdapter",
    "StorageCacheAdapter",
    "StorageComposition",
    "StorageLayer",
]
