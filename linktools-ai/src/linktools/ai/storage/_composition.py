#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ordered, revision-aware storage composition."""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, cast

from linktools.core import environ

from ..errors import ErrorCode, AIError
from ..core import canonical_sha256
from ._cache import ContentCache, contains_many, read_cache, write_cache
from ._layer import LayerRefreshPolicy, StorageLayer, StorageWriteVisibility
from ._contracts import (
    BatchStorageReader,
    BatchStorageWriter,
    InitializableStorage,
    PreloadResult,
    ReadableStorageBackend,
    StorageBatchFailure,
    StorageBatchPartialError,
    StorageBatchResult,
    StorageChange,
    StorageDeleteResult,
    StorageOperation,
    StoragePutResult,
    StorageStatReader,
    StorageOwnedInfo,
    StorageResetResult,
    StorageWriter,
    VersionSummary,
    VersionedStorage,
)
from ._revision import StorageRevisionSource, LayerMetadataView, RevisionSource

DomainKeyT = TypeVar("DomainKeyT")
DomainValueT = TypeVar("DomainValueT")
StorageKeyT = TypeVar("StorageKeyT")
StorageValueT = TypeVar("StorageValueT")
InfoT = TypeVar("InfoT")
EntryRevisionT = TypeVar("EntryRevisionT")
StoreRevisionT = TypeVar("StoreRevisionT")

_logger = environ.get_logger("ai.storage.composition")


class StorageAdapter(
    Protocol[DomainKeyT, DomainValueT, StorageKeyT, StorageValueT, InfoT],
):
    def to_storage_key(self, key: DomainKeyT) -> StorageKeyT: ...
    def from_storage_value(self, value: StorageValueT) -> DomainValueT: ...
    def to_storage_value(self, value: DomainValueT) -> StorageValueT: ...
    def validate_value(self, key: DomainKeyT, value: DomainValueT, info: InfoT) -> None: ...


class CacheAdapter(Protocol[DomainKeyT, DomainValueT, InfoT]):
    def cache_key(self, key: DomainKeyT, info: InfoT) -> str: ...
    def to_cache(self, value: DomainValueT) -> bytes: ...
    def from_cache(self, value: bytes) -> DomainValueT: ...


@dataclass(frozen=True, slots=True)
class EffectiveMetadataState(Generic[DomainKeyT, InfoT, StoreRevisionT]):
    revision: StoreRevisionT
    entries: "Mapping[DomainKeyT, InfoT]"
    owners: "Mapping[DomainKeyT, int]"


class StorageComposition(
    Generic[
        DomainKeyT,
        DomainValueT,
        StorageKeyT,
        StorageValueT,
        InfoT,
        EntryRevisionT,
        StoreRevisionT,
    ]
):
    """Own layer merge, cache, refresh, preload, write and version semantics."""

    def __init__(
        self,
        primary: 'ReadableStorageBackend[StorageKeyT, StorageValueT, InfoT, StoreRevisionT]',
        *,
        writer: 'StorageWriter[StorageKeyT, StorageValueT, InfoT, EntryRevisionT, StoreRevisionT] | None' = None,
        write_visibility: StorageWriteVisibility = StorageWriteVisibility.READABLE,
        layers: 'Sequence[StorageLayer[StorageKeyT, StorageValueT, InfoT, StoreRevisionT]]' = (),
        adapter: 'StorageAdapter[DomainKeyT, DomainValueT, StorageKeyT, StorageValueT, InfoT] | None' = None,
        revision_source: 'RevisionSource[StoreRevisionT] | None' = None,
        cache: 'ContentCache | None' = None,
        cache_adapter: 'CacheAdapter[DomainKeyT, DomainValueT, InfoT] | None' = None,
        cache_concurrency: int = 16,
        preload_batch_size: int = 100,
        preload_concurrency: int = 8,
    ) -> None:
        layer_values = tuple(layers)
        if adapter is None and layer_values:
            raise ValueError("storage adapter is required for layers")
        if adapter is None and cache is not None:
            raise ValueError("storage adapter is required for cache")
        if cache is not None and cache_adapter is None:
            raise ValueError("cache_adapter is required when cache is configured")
        if cache_concurrency < 1 or preload_batch_size < 1 or preload_concurrency < 1:
            raise ValueError("storage concurrency and batch settings must be positive")
        layer_ids = tuple(layer.id for layer in layer_values)
        if any(not layer_id for layer_id in layer_ids) or len(set(layer_ids)) != len(layer_ids):
            raise ValueError("layer ids must be non-empty and unique")
        if any(layer.backend is primary for layer in layer_values):
            raise ValueError("primary backend cannot be repeated as a layer")
        if write_visibility is StorageWriteVisibility.READABLE and writer is not None:
            if writer is not primary and all(writer is not layer.backend for layer in layer_values):
                raise ValueError("readable writer must be one of the read backends")
        if write_visibility is StorageWriteVisibility.EXTERNAL and writer is not None:
            if writer is primary or any(writer is layer.backend for layer in layer_values):
                raise ValueError("external writer must not be readable")
        if writer is None and write_visibility is not StorageWriteVisibility.READABLE:
            raise ValueError("read-only compositions must use READABLE visibility")
        self.primary = primary
        self.writer = writer
        self.layers = layer_values
        self.adapter = adapter
        self.cache = cache
        self.cache_adapter = cache_adapter
        self._write_visibility = write_visibility
        self.cache_concurrency = cache_concurrency
        self.preload_batch_size = preload_batch_size
        self.preload_concurrency = preload_concurrency
        self._cache_tasks: dict[str, asyncio.Task[DomainValueT | None]] = {}
        self._cache_task_lock = asyncio.Lock()
        source = revision_source or StorageRevisionSource(primary)
        self._views = (
            LayerMetadataView(primary, LayerRefreshPolicy.REVISIONED, revision_source=source),
            *(LayerMetadataView(layer.backend, layer.refresh) for layer in layer_values),
        )
        self._preloaded: dict[DomainKeyT, str] = {}
        self._initialize_lock = asyncio.Lock()
        self._initialized = False

    @property
    def write_visibility(self) -> StorageWriteVisibility:
        return self._write_visibility

    @property
    def writer_is_primary(self) -> bool:
        return self.writer is self.primary

    @property
    def primary_view(self) -> 'LayerMetadataView[StorageKeyT, StorageValueT, InfoT, StoreRevisionT]':
        return self._views[0]

    async def initialize(self) -> None:
        async with self._initialize_lock:
            if self._initialized:
                return
            seen: set[int] = set()
            backends = tuple(view.backend for view in self._views)
            if self.writer is not None:
                backends = (self.writer, *backends)
            for backend in backends:
                identity = id(backend)
                if identity in seen:
                    continue
                seen.add(identity)
                if isinstance(backend, InitializableStorage):
                    await backend.initialize()
            self._initialized = True
        _logger.info("storage composition initialized: layers=%s", len(self._views))

    async def refresh(self) -> StoreRevisionT:
        states = await asyncio.gather(*(view.refresh() for view in self._views))
        entries: dict[DomainKeyT, InfoT] = {}
        owners: dict[DomainKeyT, int] = {}
        revisions: list[str] = []
        for index, state in enumerate(states):
            revisions.append(str(state.revision))
            for storage_key, info in state.entries.items():
                domain_key = self._domain_key(storage_key, info)
                if domain_key not in entries:
                    entries[domain_key] = info
                    owners[domain_key] = index
        return EffectiveMetadataState(self._effective_revision(revisions, states[0].revision), entries, owners).revision

    async def _state(self) -> 'EffectiveMetadataState[DomainKeyT, InfoT, StoreRevisionT]':
        states = await asyncio.gather(*(view.refresh() for view in self._views))
        entries: dict[DomainKeyT, InfoT] = {}
        owners: dict[DomainKeyT, int] = {}
        revisions: list[str] = []
        for index, state in enumerate(states):
            revisions.append(str(state.revision))
            for storage_key, info in state.entries.items():
                domain_key = self._domain_key(storage_key, info)
                if domain_key not in entries:
                    entries[domain_key] = info
                    owners[domain_key] = index
        revision = self._effective_revision(revisions, states[0].revision)
        return EffectiveMetadataState(revision, entries, owners)

    def _domain_key(self, storage_key: StorageKeyT, info: InfoT) -> DomainKeyT:
        return cast(DomainKeyT, storage_key)

    def _to_storage_key(self, key: DomainKeyT) -> StorageKeyT:
        if self.adapter is None:
            return cast(StorageKeyT, key)
        return self.adapter.to_storage_key(key)

    def _from_storage_value(self, value: StorageValueT) -> DomainValueT:
        if self.adapter is None:
            return cast(DomainValueT, value)
        return self.adapter.from_storage_value(value)

    def _to_storage_value(self, value: DomainValueT) -> StorageValueT:
        if self.adapter is None:
            return cast(StorageValueT, value)
        return self.adapter.to_storage_value(value)

    def _validate_value(self, key: DomainKeyT, value: DomainValueT, info: InfoT) -> None:
        if self.adapter is not None:
            self.adapter.validate_value(key, value, info)

    def _effective_revision(self, revisions: 'Sequence[str]', primary: StoreRevisionT) -> StoreRevisionT:
        if len(revisions) == 1:
            return primary
        return cast(StoreRevisionT, canonical_sha256(list(revisions)))

    async def current_revision(self) -> StoreRevisionT:
        heads = await asyncio.gather(*(view.head_revision() for view in self._views))
        if any(head is None for head in heads):
            return (await self._state()).revision
        return self._effective_revision([str(head) for head in heads], cast(StoreRevisionT, heads[0]))

    async def get(self, key: DomainKeyT) -> 'DomainValueT | None':
        state = await self._state()
        return await self._get_from_state(key, state, retried=False)

    async def stat(self, key: DomainKeyT) -> 'InfoT | None':
        state = await self._state()
        info = state.entries.get(key)
        if info is None:
            return None
        backend = self._views[state.owners[key]].backend
        if isinstance(backend, StorageStatReader):
            origin = await backend.stat(self._to_storage_key(key))
            if origin is None:
                raise AIError(
                    ErrorCode.STORAGE_INTEGRITY_ERROR,
                    "storage metadata points to a missing origin",
                    safe_details={"key_digest": canonical_sha256(str(key)), "layer": self._owner_id(state.owners[key])},
                )
            if origin != info:
                raise AIError(
                    ErrorCode.STORAGE_INTEGRITY_ERROR,
                    "storage metadata and origin disagree",
                    safe_details={"key_digest": canonical_sha256(str(key)), "layer": self._owner_id(state.owners[key])},
                )
        return info

    async def _get(self, key: DomainKeyT, *, retried: bool) -> 'DomainValueT | None':
        state = await self._state()
        return await self._get_from_state(key, state, retried=retried)

    async def get_many(self, keys: 'Sequence[DomainKeyT]') -> 'tuple[DomainValueT | None, ...]':
        if not keys:
            return ()
        state = await self._state()
        unique = tuple(dict.fromkeys(keys))
        values = await self._get_many_from_state(unique, state, retried=False)
        return tuple(values[key] for key in keys)

    async def _get_from_state(
        self,
        key: DomainKeyT,
        state: 'EffectiveMetadataState[DomainKeyT, InfoT, StoreRevisionT]',
        *,
        retried: bool,
    ) -> 'DomainValueT | None':
        info = state.entries.get(key)
        if info is None:
            return None
        owner = state.owners[key]
        cache_key = self.cache_adapter.cache_key(key, info) if self.cache_adapter is not None else None
        if self.cache is not None and cache_key is not None:
            cached_value = await self._read_cache_value(key, info, cache_key)
            if cached_value is not None:
                return cached_value
        storage_key = self._to_storage_key(key)
        backend = self._views[owner].backend
        value = await backend.get(storage_key)
        if value is None:
            self._views[owner].invalidate()
            if retried:
                _logger.error(
                    "storage metadata has no origin value: key=%s",
                    key,
                    extra={"error_code": ErrorCode.STORAGE_OWNER_MISMATCH.value},
                )
                raise AIError(
                    ErrorCode.STORAGE_INTEGRITY_ERROR,
                    "storage metadata points to a missing origin",
                    safe_details={"key_digest": canonical_sha256(str(key))},
                )
            refreshed = await self._state()
            return await self._get_from_state(key, refreshed, retried=True)
        domain_value = self._from_storage_value(value)
        try:
            self._validate_value(key, domain_value, info)
        except Exception:
            self._views[owner].invalidate()
            if retried:
                _logger.error(
                    "storage owner mismatch after refresh: key=%s",
                    key,
                    extra={"error_code": ErrorCode.STORAGE_OWNER_MISMATCH.value},
                )
                raise AIError(
                    ErrorCode.STORAGE_INTEGRITY_ERROR,
                    "storage metadata and origin disagree",
                    safe_details={"key_digest": canonical_sha256(str(key))},
                )
            refreshed = await self._state()
            return await self._get_from_state(key, refreshed, retried=True)
        if self.cache is not None and cache_key is not None:
            await write_cache(self.cache, cache_key, self.cache_adapter.to_cache(domain_value))
        return domain_value

    async def _get_many_from_state(
        self,
        keys: 'Sequence[DomainKeyT]',
        state: 'EffectiveMetadataState[DomainKeyT, InfoT, StoreRevisionT]',
        *,
        retried: bool,
    ) -> 'dict[DomainKeyT, DomainValueT | None]':
        values: dict[DomainKeyT, DomainValueT | None] = {key: None for key in keys}
        misses: list[DomainKeyT] = []
        semaphore = asyncio.Semaphore(self.cache_concurrency)

        async def _cache_read(key: DomainKeyT) -> None:
            info = state.entries.get(key)
            if info is None or self.cache is None or self.cache_adapter is None:
                misses.append(key)
                return
            cache_key = self.cache_adapter.cache_key(key, info)
            async with semaphore:
                value = await self._read_cache_value(key, info, cache_key)
            if value is None:
                misses.append(key)
            else:
                values[key] = value

        await asyncio.gather(*(_cache_read(key) for key in keys))
        misses = [key for key in misses if key in state.entries]
        loaded = await self._load_origins(misses, state)
        raced: set[int] = set()
        for key in misses:
            raw = loaded.get(key)
            if raw is None:
                raced.add(state.owners[key])
                continue
            info = state.entries[key]
            value = self._from_storage_value(raw)
            try:
                self._validate_value(key, value, info)
            except Exception:
                raced.add(state.owners[key])
                continue
            values[key] = value
            if self.cache is not None and self.cache_adapter is not None:
                await write_cache(
                    self.cache,
                    self.cache_adapter.cache_key(key, info),
                    self.cache_adapter.to_cache(value),
                )
        if raced:
            for owner in raced:
                self._views[owner].invalidate()
            if not retried:
                refreshed = await self._state()
                return await self._get_many_from_state(keys, refreshed, retried=True)
            for key in misses:
                if state.owners.get(key) in raced:
                    _logger.error(
                        "storage owner mismatch after batch refresh: key=%s",
                        key,
                        extra={"error_code": ErrorCode.STORAGE_OWNER_MISMATCH.value},
                    )
                    raise AIError(
                        ErrorCode.STORAGE_INTEGRITY_ERROR,
                        "storage metadata and origin disagree",
                        safe_details={"key_digest": canonical_sha256(str(key))},
                    )
        return values

    async def _read_cache_value(
        self,
        key: DomainKeyT,
        info: InfoT,
        cache_key: str,
    ) -> 'DomainValueT | None':
        if self.cache is None or self.cache_adapter is None:
            return None
        async with self._cache_task_lock:
            task = self._cache_tasks.get(cache_key)
            if task is None:
                task = asyncio.create_task(self._load_cache_value(key, info, cache_key))
                self._cache_tasks[cache_key] = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done() and self._cache_tasks.get(cache_key) is task:
                self._cache_tasks.pop(cache_key, None)

    async def _load_cache_value(
        self,
        key: DomainKeyT,
        info: InfoT,
        cache_key: str,
    ) -> 'DomainValueT | None':
        cached = await read_cache(self.cache, cache_key)
        if cached is None:
            return None
        try:
            value = self.cache_adapter.from_cache(cached)
            self._validate_value(key, value, info)
            return value
        except asyncio.CancelledError:
            raise
        except Exception:
            _logger.warning(
                "storage cache entry failed validation: key=%s",
                key,
                extra={"error_code": ErrorCode.STORAGE_CACHE_CORRUPT.value},
            )
            await self._delete_cache(cache_key)
            return None

    async def _load_origins(
        self,
        keys: 'Sequence[DomainKeyT]',
        state: 'EffectiveMetadataState[DomainKeyT, InfoT, StoreRevisionT]',
    ) -> 'dict[DomainKeyT, StorageValueT]':
        loaded: dict[DomainKeyT, StorageValueT] = {}
        grouped: dict[int, list[DomainKeyT]] = {}
        for key in keys:
            if key in state.entries:
                grouped.setdefault(state.owners[key], []).append(key)
        semaphore = asyncio.Semaphore(self.preload_concurrency)

        async def _load_group(owner: int, group: 'Sequence[DomainKeyT]') -> None:
            backend = self._views[owner].backend
            storage_keys = tuple(self._to_storage_key(key) for key in group)
            if isinstance(backend, BatchStorageReader):
                async with semaphore:
                    raw_values = await backend.get_many(storage_keys)
                for key, storage_key in zip(group, storage_keys):
                    raw = raw_values.get(storage_key)
                    if raw is not None:
                        loaded[key] = raw
                return

            async def _load_one(key: DomainKeyT) -> None:
                async with semaphore:
                    raw = await backend.get(self._to_storage_key(key))
                if raw is not None:
                    loaded[key] = raw

            await asyncio.gather(*(_load_one(key) for key in group))

        await asyncio.gather(*(_load_group(owner, group) for owner, group in grouped.items()))
        return loaded

    async def _delete_cache(self, cache_key: str) -> None:
        if self.cache is None:
            return
        try:
            await self.cache.delete(cache_key)
        except asyncio.CancelledError:
            raise
        except Exception:
            _logger.debug("failed to delete corrupted storage cache entry: key=%s", cache_key)

    async def list_info(self) -> 'tuple[InfoT, ...]':
        state = await self._state()
        return tuple(state.entries[key] for key in sorted(state.entries, key=str))

    async def list_info_with_owners(self) -> 'tuple[StorageOwnedInfo[InfoT], ...]':
        state = await self._state()
        return tuple(
            StorageOwnedInfo(
                state.entries[key],
                self._owner_id(state.owners[key]),
                self.writer is self._views[state.owners[key]].backend,
            )
            for key in sorted(state.entries, key=str)
        )

    def _owner_id(self, index: int) -> str:
        return "primary" if index == 0 else self.layers[index - 1].id

    async def preload(self, keys: 'Sequence[DomainKeyT] | None' = None) -> 'PreloadResult[StoreRevisionT]':
        state = await self._state()
        selected = tuple(dict.fromkeys(keys)) if keys is not None else tuple(state.entries)
        if self.cache is None or self.cache_adapter is None:
            return PreloadResult(state.revision, len(selected), 0, 0, len(selected))
        identities = {
            key: self.cache_adapter.cache_key(key, state.entries[key])
            for key in selected
            if key in state.entries
        }
        marked = {
            key
            for key, identity in identities.items()
            if self._preloaded.get(key) == identity
        }
        pending_identities = tuple(
            identity
            for key, identity in identities.items()
            if key not in marked
        )
        present = await contains_many(self.cache, pending_identities)
        already_cached = len(marked) + sum(1 for identity in pending_identities if identity in present)
        for key, identity in identities.items():
            if key in marked or identity in present:
                self._preloaded[key] = identity
        missing = tuple(
            key
            for key, identity in identities.items()
            if key not in marked and identity not in present
        )
        loaded_count = 0
        failed = len(selected) - len(identities)
        for offset in range(0, len(missing), self.preload_batch_size):
            batch = missing[offset : offset + self.preload_batch_size]
            raw_values = await self._load_origins(batch, state)
            writes: list[DomainKeyT] = []
            for key in batch:
                raw = raw_values.get(key)
                if raw is None:
                    failed += 1
                    continue
                value = self._from_storage_value(raw)
                try:
                    self._validate_value(key, value, state.entries[key])
                except Exception:
                    failed += 1
                    continue
                await write_cache(
                    self.cache,
                    identities[key],
                    self.cache_adapter.to_cache(value),
                )
                writes.append(key)
            if writes:
                present_after = await contains_many(
                    self.cache,
                    tuple(identities[key] for key in writes),
                )
                for key in writes:
                    if identities[key] in present_after:
                        self._preloaded[key] = identities[key]
                        loaded_count += 1
                    else:
                        failed += 1
        return PreloadResult(
            state.revision,
            len(selected),
            loaded_count,
            already_cached,
            failed,
        )

    def _require_writer(self) -> 'StorageWriter[StorageKeyT, StorageValueT, InfoT, EntryRevisionT, StoreRevisionT]':
        if self.writer is None:
            raise AIError(ErrorCode.STORAGE_READ_ONLY)
        return self.writer

    async def put(
        self,
        key: DomainKeyT,
        value: DomainValueT,
        *,
        expected_entry_revision: 'EntryRevisionT | None' = None,
    ) -> 'StoragePutResult[InfoT, EntryRevisionT, StoreRevisionT]':
        writer = self._require_writer()
        storage_key = self._to_storage_key(key)
        result = await writer.put(storage_key, self._to_storage_value(value), expected_entry_revision=expected_entry_revision)
        self._validate_value(key, value, result.info)
        self._after_put(storage_key, result.info, result.store_revision, key)
        if result.changed:
            await self._notify_revision(result.store_revision)
        return result

    async def delete(
        self,
        key: DomainKeyT,
        *,
        expected_entry_revision: 'EntryRevisionT | None' = None,
    ) -> 'StorageDeleteResult[DomainKeyT, EntryRevisionT, StoreRevisionT]':
        writer = self._require_writer()
        result = await writer.delete(self._to_storage_key(key), expected_entry_revision=expected_entry_revision)
        self._preloaded.pop(key, None)
        if result.deleted:
            self._invalidate_writer_view()
            await self._notify_revision(result.store_revision)
        return StorageDeleteResult(key, result.deleted, result.entry_revision, result.store_revision)

    async def reset(self) -> 'StorageResetResult[StoreRevisionT]':
        writer = self._require_writer()
        result = await writer.reset()
        self._preloaded.clear()
        if result.deleted_count:
            self._invalidate_writer_view()
            await self._notify_revision(result.store_revision)
        return result

    async def apply_batch(
        self,
        changes: 'Sequence[StorageChange[DomainKeyT, DomainValueT, EntryRevisionT]]',
        *,
        expected_store_revision: 'StoreRevisionT | None' = None,
    ) -> 'StorageBatchResult[InfoT, DomainKeyT, EntryRevisionT, StoreRevisionT]':
        self._validate_batch(changes)
        writer = self._require_writer()
        if isinstance(writer, BatchStorageWriter):
            storage_changes = tuple(
                StorageChange(change.operation, self._to_storage_key(change.key), None if change.value is None else self._to_storage_value(change.value), change.expected_entry_revision)
                for change in changes
            )
            result = await writer.apply_batch(storage_changes, expected_store_revision=expected_store_revision)
            self._validate_writer_batch_result(changes, storage_changes, result)
            for change in changes:
                self._preloaded.pop(change.key, None)
            invalidate = False
            for change, storage_change, item in zip(changes, storage_changes, result.results):
                if isinstance(item, StoragePutResult):
                    self._after_put(storage_change.key, item.info, item.store_revision, change.key)
                elif item.deleted:
                    invalidate = True
            if invalidate:
                self._invalidate_writer_view()
            if any(
                isinstance(item, StoragePutResult) and item.changed
                or isinstance(item, StorageDeleteResult) and item.deleted
                for item in result.results
            ):
                await self._notify_revision(result.store_revision)
            mapped_results = tuple(
                StorageDeleteResult(
                    changes[index].key,
                    item.deleted,
                    item.entry_revision,
                    item.store_revision,
                )
                if isinstance(item, StorageDeleteResult)
                else item
                for index, item in enumerate(result.results)
            )
            return StorageBatchResult(result.store_revision, result.atomic, mapped_results)
        if expected_store_revision is not None:
            current = await self.current_revision()
            if current != expected_store_revision:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        results: list[StoragePutResult[InfoT, EntryRevisionT, StoreRevisionT] | StorageDeleteResult[DomainKeyT, EntryRevisionT, StoreRevisionT]] = []
        revision = expected_store_revision
        for index, change in enumerate(changes):
            try:
                if change.operation is StorageOperation.PUT:
                    value = cast(DomainValueT, change.value)
                    item = await self.put(
                        change.key,
                        value,
                        expected_entry_revision=change.expected_entry_revision,
                    )
                else:
                    item = await self.delete(
                        change.key,
                        expected_entry_revision=change.expected_entry_revision,
                    )
            except Exception as exc:
                failure_revision = revision if revision is not None else await self.current_revision()
                error_code = (
                    exc.code.value
                    if isinstance(exc, AIError)
                    else ErrorCode.STORAGE_BATCH_PARTIAL_FAILURE.value
                )
                failure = StorageBatchFailure(
                    index,
                    error_code,
                    failure_revision,
                    tuple(results),
                )
                raise StorageBatchPartialError(failure) from exc
            results.append(item)
            revision = item.store_revision
        if revision is None:
            revision = await self.current_revision()
        return StorageBatchResult(revision, False, tuple(results))

    def _validate_writer_batch_result(
        self,
        domain_changes: 'Sequence[StorageChange[DomainKeyT, DomainValueT, EntryRevisionT]]',
        changes: 'Sequence[StorageChange[StorageKeyT, StorageValueT, EntryRevisionT]]',
        result: 'StorageBatchResult[InfoT, StorageKeyT, EntryRevisionT, StoreRevisionT]',
    ) -> None:
        if len(domain_changes) != len(changes) or len(result.results) != len(changes):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "storage batch result count mismatch")
        for domain_change, change, item in zip(domain_changes, changes, result.results):
            if change.operation is StorageOperation.PUT:
                if not isinstance(item, StoragePutResult):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "storage batch put result mismatch")
                self._validate_value(domain_change.key, cast(DomainValueT, domain_change.value), item.info)
            elif not isinstance(item, StorageDeleteResult) or item.key != change.key:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "storage batch delete result mismatch")

    def _validate_batch(self, changes: 'Sequence[StorageChange[DomainKeyT, DomainValueT, EntryRevisionT]]') -> None:
        seen: set[DomainKeyT] = set()
        for change in changes:
            if change.operation not in {StorageOperation.PUT, StorageOperation.DELETE}:
                raise ValueError(f"unsupported storage operation: {change.operation}")
            if change.key in seen:
                raise AIError(ErrorCode.STORAGE_BATCH_DUPLICATE_KEY)
            seen.add(change.key)
            if change.operation is StorageOperation.PUT and change.value is None:
                raise ValueError("PUT changes require a value")
            if change.operation is StorageOperation.DELETE and change.value is not None:
                raise ValueError("DELETE changes cannot contain a value")

    def _after_put(
        self,
        storage_key: StorageKeyT,
        info: InfoT,
        revision: StoreRevisionT,
        domain_key: DomainKeyT,
    ) -> None:
        self._preloaded.pop(domain_key, None)
        if self._write_visibility is not StorageWriteVisibility.READABLE or self.writer is None:
            return
        for view in self._views:
            if view.backend is self.writer:
                view.apply_write(storage_key, info, revision)

    def _invalidate_writer_view(self) -> None:
        if self._write_visibility is not StorageWriteVisibility.READABLE or self.writer is None:
            return
        for view in self._views:
            if view.backend is self.writer:
                view.invalidate()

    async def _notify_revision(self, revision: StoreRevisionT) -> None:
        source = self.primary_view.revision_source
        if source is not None:
            try:
                await source.revision_bumped(revision)
            except Exception:
                self.primary_view.invalidate()
                _logger.warning(
                    "storage revision notification failed",
                    extra={"error_code": ErrorCode.STORAGE_REVISION_NOTIFY_FAILED.value},
                    exc_info=environ.debug,
                )

    async def list_versions(self, key: DomainKeyT) -> 'tuple[VersionSummary[EntryRevisionT], ...]':
        state = await self._state()
        owners = tuple(range(len(self._views)))
        collected: dict[tuple[str, str, bool], VersionSummary[EntryRevisionT]] = {}
        for owner in owners:
            backend = self._views[owner].backend
            if not isinstance(backend, VersionedStorage):
                continue
            versions = tuple(await backend.list_versions(self._to_storage_key(key)))
            for version in versions:
                collected[(str(version.entry_revision), version.digest, version.deleted)] = version
        if collected:
            return tuple(sorted(collected.values(), key=lambda version: str(version.entry_revision), reverse=True))
        if key not in state.owners:
            raise AIError(ErrorCode.ASSET_VERSION_OWNER_UNKNOWN)
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)

    async def get_at_revision(self, key: DomainKeyT, entry_revision: EntryRevisionT) -> 'DomainValueT | None':
        state = await self._state()
        owners = tuple(range(len(self._views)))
        for owner in owners:
            backend = self._views[owner].backend
            if not isinstance(backend, VersionedStorage):
                continue
            versions = tuple(await backend.list_versions(self._to_storage_key(key)))
            if any(version.entry_revision == entry_revision for version in versions):
                value = await backend.get_at_revision(self._to_storage_key(key), entry_revision)
                return None if value is None else self._from_storage_value(value)
        raise AIError(ErrorCode.ASSET_VERSION_OWNER_UNKNOWN if key not in state.owners else ErrorCode.STORAGE_VERSION_UNSUPPORTED)

    async def get_at_version(self, key: DomainKeyT, version: int) -> 'DomainValueT | None':
        state = await self._state()
        owners = tuple(range(len(self._views)))
        for owner in owners:
            backend = self._views[owner].backend
            if not isinstance(backend, VersionedStorage):
                continue
            versions = tuple(await backend.list_versions(self._to_storage_key(key)))
            if any(str(item.entry_revision.value) == str(version) for item in versions):
                value = await backend.get_at_version(self._to_storage_key(key), version)
                return None if value is None else self._from_storage_value(value)
        raise AIError(ErrorCode.ASSET_VERSION_OWNER_UNKNOWN if key not in state.owners else ErrorCode.STORAGE_VERSION_UNSUPPORTED)


__all__ = [
    "CacheAdapter",
    "EffectiveMetadataState",
    "StorageAdapter",
    "StorageComposition",
    "StorageLayer",
]
