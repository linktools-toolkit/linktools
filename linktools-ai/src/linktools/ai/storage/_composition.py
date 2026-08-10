#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ordered, revision-aware storage overlay."""

import asyncio
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, cast

from linktools.core import environ

from ..core import canonical_sha256
from ..errors import AIError, ErrorCode
from ._cache import ContentCache, contains_many, read_cache, write_cache
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
    StorageEntryRevision,
    StorageEntryStatus,
    StorageEntryStatusInfo,
    StorageOperation,
    StorageOwnedInfo,
    StoragePutResult,
    StorageResetResult,
    StorageRevision,
    StorageStatReader,
    StorageWriter,
    VersionedStorage,
    VersionSummary,
)
from ._layer import LayerRefreshPolicy, StorageLayer
from ._revision import LayerMetadataView, RevisionSource, StorageRevisionSource

KeyT = TypeVar("KeyT", bound=Hashable)
ValueT = TypeVar("ValueT")
InfoT = TypeVar("InfoT")

_logger = environ.get_logger("ai.storage.overlay")


def _is_deleted(info: InfoT) -> bool:
    return isinstance(info, StorageEntryStatusInfo) and info.status is StorageEntryStatus.DELETED


def _is_reset(info: InfoT) -> bool:
    return isinstance(info, StorageEntryStatusInfo) and info.status is StorageEntryStatus.RESET


class StorageValueValidator(Protocol[KeyT, ValueT, InfoT]):
    def validate_value(self, key: KeyT, value: ValueT, info: InfoT) -> None: ...


class CacheAdapter(Protocol[KeyT, ValueT, InfoT]):
    def cache_key(self, key: KeyT, info: InfoT) -> str: ...
    def to_cache(self, value: ValueT) -> bytes: ...
    def from_cache(self, value: bytes) -> ValueT: ...


@dataclass(frozen=True, slots=True)
class EffectiveMetadataState(Generic[KeyT, InfoT]):
    revision: StorageRevision
    entries: "Mapping[KeyT, InfoT]"
    owners: "Mapping[KeyT, int]"


@dataclass(frozen=True, slots=True)
class StorageLocation(Generic[KeyT, ValueT, InfoT]):
    key: KeyT
    info: InfoT
    backend: "ReadableStorageBackend[KeyT, ValueT, InfoT]"
    layer: str
    writable: bool


class StorageOverlay(Generic[KeyT, ValueT, InfoT]):
    """Own layer merge, cache, refresh, preload, write and version semantics."""

    def __init__(
        self,
        primary: 'ReadableStorageBackend[KeyT, ValueT, InfoT]',
        *,
        writer: 'StorageWriter[KeyT, ValueT, InfoT] | None' = None,
        layers: 'Sequence[StorageLayer[KeyT, ValueT, InfoT]]' = (),
        validator: 'StorageValueValidator[KeyT, ValueT, InfoT] | None' = None,
        revision_source: 'RevisionSource | None' = None,
        cache: 'ContentCache | None' = None,
        cache_adapter: 'CacheAdapter[KeyT, ValueT, InfoT] | None' = None,
        cache_concurrency: int = 16,
        preload_batch_size: int = 100,
        preload_concurrency: int = 8,
    ) -> None:
        layer_values = tuple(layers)
        if cache is not None and cache_adapter is None:
            raise ValueError("cache_adapter is required when cache is configured")
        if cache_concurrency < 1 or preload_batch_size < 1 or preload_concurrency < 1:
            raise ValueError("storage concurrency and batch settings must be positive")
        layer_ids = tuple(layer.id for layer in layer_values)
        if any(not layer_id for layer_id in layer_ids) or len(set(layer_ids)) != len(layer_ids):
            raise ValueError("layer ids must be non-empty and unique")
        if any(layer.backend is primary for layer in layer_values):
            raise ValueError("primary backend cannot be repeated as a layer")
        if writer is not None and writer is not primary and all(writer is not layer.backend for layer in layer_values):
            raise ValueError("writer must be one of the read backends")
        self.primary = primary
        self.writer = writer
        self.layers = layer_values
        self.validator = validator
        self.cache = cache
        self.cache_adapter = cache_adapter
        self.cache_concurrency = cache_concurrency
        self.preload_batch_size = preload_batch_size
        self.preload_concurrency = preload_concurrency
        self._cache_tasks: dict[str, asyncio.Task[ValueT | None]] = {}
        self._cache_task_lock = asyncio.Lock()
        source = revision_source or StorageRevisionSource(primary)
        self._views = (
            LayerMetadataView(primary, LayerRefreshPolicy.REVISIONED, revision_source=source),
            *(LayerMetadataView(layer.backend, layer.refresh) for layer in layer_values),
        )
        self._preloaded: dict[KeyT, str] = {}
        self._initialize_lock = asyncio.Lock()
        self._initialized = False

    @property
    def writer_is_primary(self) -> bool:
        return self.writer is self.primary

    @property
    def writable(self) -> bool:
        """Return whether generic storage mutations have a configured writer."""
        return self.writer is not None

    def is_writable_backend(self, backend: "ReadableStorageBackend[KeyT, ValueT, InfoT]") -> bool:
        """Return whether a backend is the overlay's writable backend."""
        return self.writer is backend

    @property
    def primary_view(self) -> 'LayerMetadataView[KeyT, ValueT, InfoT]':
        return self._views[0]

    @property
    def backends(self) -> 'tuple[ReadableStorageBackend[KeyT, ValueT, InfoT], ...]':
        """Return the primary backend followed by ordered layer backends."""
        return tuple(view.backend for view in self._views)

    async def initialize(self) -> None:
        async with self._initialize_lock:
            if self._initialized:
                return
            seen: set[int] = set()
            for backend in (view.backend for view in self._views):
                identity = id(backend)
                if identity in seen:
                    continue
                seen.add(identity)
                if isinstance(backend, InitializableStorage):
                    await backend.initialize()
            self._initialized = True
        _logger.info("storage overlay initialized: layers=%s", len(self._views))

    async def refresh(self) -> StorageRevision:
        states = await asyncio.gather(*(view.refresh() for view in self._views))
        entries: dict[KeyT, InfoT] = {}
        owners: dict[KeyT, int] = {}
        revisions: list[str] = []
        for index, state in enumerate(states):
            revisions.append(str(state.revision))
            for key, info in state.entries.items():
                if _is_reset(info):
                    continue
                if key not in entries:
                    entries[key] = info
                    owners[key] = index
        return EffectiveMetadataState(self._effective_revision(revisions, states[0].revision), entries, owners).revision

    async def _state(self) -> 'EffectiveMetadataState[KeyT, InfoT]':
        states = await asyncio.gather(*(view.refresh() for view in self._views))
        entries: dict[KeyT, InfoT] = {}
        owners: dict[KeyT, int] = {}
        revisions: list[str] = []
        for index, state in enumerate(states):
            revisions.append(str(state.revision))
            for key, info in state.entries.items():
                if _is_reset(info):
                    continue
                if key not in entries:
                    entries[key] = info
                    owners[key] = index
        revision = self._effective_revision(revisions, states[0].revision)
        return EffectiveMetadataState(revision, entries, owners)

    def _validate_value(self, key: KeyT, value: ValueT, info: InfoT) -> None:
        if self.validator is not None:
            self.validator.validate_value(key, value, info)

    def _effective_revision(self, revisions: 'Sequence[str]', primary: StorageRevision) -> StorageRevision:
        if len(revisions) == 1:
            return primary
        return StorageRevision(canonical_sha256(list(revisions)))

    async def current_revision(self) -> StorageRevision:
        heads = await asyncio.gather(*(view.head_revision() for view in self._views))
        if any(head is None for head in heads):
            return (await self._state()).revision
        return self._effective_revision([str(head) for head in heads], cast(StorageRevision, heads[0]))

    async def get(self, key: KeyT) -> 'ValueT | None':
        state = await self._state()
        return await self._get_from_state(key, state, retried=False)

    async def stat(self, key: KeyT) -> 'InfoT | None':
        state = await self._state()
        info = state.entries.get(key)
        if info is None:
            return None
        backend = self._views[state.owners[key]].backend
        if isinstance(backend, StorageStatReader):
            origin = await backend.stat(key)
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

    async def locate(self, key: KeyT) -> 'StorageLocation[KeyT, ValueT, InfoT] | None':
        """Return the effective owner selected by layer precedence for one key."""
        state = await self._state()
        info = state.entries.get(key)
        if info is None:
            return None
        owner = state.owners[key]
        backend = self._views[owner].backend
        return StorageLocation(key, info, backend, self._owner_id(owner), self.is_writable_backend(backend))

    def invalidate(self) -> None:
        """Discard materialized metadata and preload markers after an external mutation."""
        for view in self._views:
            view.invalidate()
        self._preloaded.clear()

    async def _get(self, key: KeyT, *, retried: bool) -> 'ValueT | None':
        state = await self._state()
        return await self._get_from_state(key, state, retried=retried)

    async def get_many(self, keys: 'Sequence[KeyT]') -> 'tuple[ValueT | None, ...]':
        if not keys:
            return ()
        state = await self._state()
        unique = tuple(dict.fromkeys(keys))
        values = await self._get_many_from_state(unique, state, retried=False)
        return tuple(values[key] for key in keys)

    async def _get_from_state(
        self,
        key: KeyT,
        state: 'EffectiveMetadataState[KeyT, InfoT]',
        *,
        retried: bool,
    ) -> 'ValueT | None':
        info = state.entries.get(key)
        if info is None or _is_deleted(info):
            return None
        owner = state.owners[key]
        cache_key = self.cache_adapter.cache_key(key, info) if self.cache_adapter is not None else None
        if self.cache is not None and cache_key is not None:
            cached_value = await self._read_cache_value(key, info, cache_key)
            if cached_value is not None:
                return cached_value
        backend = self._views[owner].backend
        value = await backend.get(key)
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
        try:
            self._validate_value(key, value, info)
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
            await write_cache(self.cache, cache_key, self.cache_adapter.to_cache(value))
        return value

    async def _get_many_from_state(
        self,
        keys: 'Sequence[KeyT]',
        state: 'EffectiveMetadataState[KeyT, InfoT]',
        *,
        retried: bool,
    ) -> 'dict[KeyT, ValueT | None]':
        values: dict[KeyT, ValueT | None] = {key: None for key in keys}
        misses: list[KeyT] = []
        semaphore = asyncio.Semaphore(self.cache_concurrency)

        async def _cache_read(key: KeyT) -> None:
            info = state.entries.get(key)
            if info is None or _is_deleted(info):
                return
            if self.cache is None or self.cache_adapter is None:
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
        misses = [key for key in misses if key in state.entries and not _is_deleted(state.entries[key])]
        loaded = await self._load_origins(misses, state)
        raced: set[int] = set()
        for key in misses:
            raw = loaded.get(key)
            if raw is None:
                raced.add(state.owners[key])
                continue
            info = state.entries[key]
            value = raw
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
        key: KeyT,
        info: InfoT,
        cache_key: str,
    ) -> 'ValueT | None':
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
        key: KeyT,
        info: InfoT,
        cache_key: str,
    ) -> 'ValueT | None':
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
        keys: 'Sequence[KeyT]',
        state: 'EffectiveMetadataState[KeyT, InfoT]',
    ) -> 'dict[KeyT, ValueT]':
        loaded: dict[KeyT, ValueT] = {}
        grouped: dict[int, list[KeyT]] = {}
        for key in keys:
            if key in state.entries:
                grouped.setdefault(state.owners[key], []).append(key)
        semaphore = asyncio.Semaphore(self.preload_concurrency)

        async def _load_group(owner: int, group: 'Sequence[KeyT]') -> None:
            backend = self._views[owner].backend
            if isinstance(backend, BatchStorageReader):
                async with semaphore:
                    raw_values = await backend.get_many(group)
                for key in group:
                    raw = raw_values.get(key)
                    if raw is not None:
                        loaded[key] = raw
                return

            async def _load_one(key: KeyT) -> None:
                async with semaphore:
                    raw = await backend.get(key)
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
                self.is_writable_backend(self._views[state.owners[key]].backend),
            )
            for key in sorted(state.entries, key=str)
        )

    def _owner_id(self, index: int) -> str:
        return "primary" if index == 0 else self.layers[index - 1].id

    async def preload(self, keys: 'Sequence[KeyT] | None' = None) -> PreloadResult:
        state = await self._state()
        selected = tuple(dict.fromkeys(keys)) if keys is not None else tuple(state.entries)
        if self.cache is None or self.cache_adapter is None:
            return PreloadResult(state.revision, len(selected), 0, 0, len(selected))
        identities = {
            key: self.cache_adapter.cache_key(key, state.entries[key])
            for key in selected
            if key in state.entries and not _is_deleted(state.entries[key])
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
        failed = sum(1 for key in selected if key not in state.entries)
        for offset in range(0, len(missing), self.preload_batch_size):
            batch = missing[offset : offset + self.preload_batch_size]
            raw_values = await self._load_origins(batch, state)
            writes: list[KeyT] = []
            for key in batch:
                raw = raw_values.get(key)
                if raw is None:
                    failed += 1
                    continue
                value = raw
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

    def _require_writer(self) -> 'StorageWriter[KeyT, ValueT, InfoT]':
        if self.writer is None:
            raise AIError(ErrorCode.STORAGE_READ_ONLY)
        return self.writer

    async def put(
        self,
        key: KeyT,
        value: ValueT,
        *,
        expected_entry_revision: 'StorageEntryRevision | None' = None,
    ) -> 'StoragePutResult[InfoT]':
        writer = self._require_writer()
        result = await writer.put(key, value, expected_entry_revision=expected_entry_revision)
        self._validate_value(key, value, result.info)
        self._after_put(key, result.info, result.store_revision)
        if result.changed:
            await self._notify_revision(result.store_revision)
        return result

    async def delete(
        self,
        key: KeyT,
        *,
        expected_entry_revision: 'StorageEntryRevision | None' = None,
    ) -> 'StorageDeleteResult[KeyT]':
        writer = self._require_writer()
        result = await writer.delete(key, expected_entry_revision=expected_entry_revision)
        self._preloaded.pop(key, None)
        if result.deleted:
            self._invalidate_writer_view()
            await self._notify_revision(result.store_revision)
        return result

    async def reset(
        self,
        key: KeyT,
        *,
        expected_entry_revision: 'StorageEntryRevision | None' = None,
    ) -> 'StorageResetResult[KeyT]':
        writer = self._require_writer()
        result = await writer.reset(key, expected_entry_revision=expected_entry_revision)
        self._preloaded.pop(key, None)
        if result.reset:
            self._invalidate_writer_view()
            await self._notify_revision(result.store_revision)
        return result

    async def apply_batch(
        self,
        changes: 'Sequence[StorageChange[KeyT, ValueT]]',
        *,
        expected_store_revision: 'StorageRevision | None' = None,
    ) -> 'StorageBatchResult[InfoT, KeyT]':
        self._validate_batch(changes)
        writer = self._require_writer()
        if isinstance(writer, BatchStorageWriter):
            result = await writer.apply_batch(changes, expected_store_revision=expected_store_revision)
            self._validate_writer_batch_result(changes, result)
            for change in changes:
                self._preloaded.pop(change.key, None)
            invalidate = False
            for change, item in zip(changes, result.results):
                if isinstance(item, StoragePutResult):
                    self._after_put(change.key, item.info, item.store_revision)
                elif (
                    isinstance(item, StorageDeleteResult)
                    and item.deleted
                    or isinstance(item, StorageResetResult)
                    and item.reset
                ):
                    invalidate = True
            if invalidate:
                self._invalidate_writer_view()
            if any(
                isinstance(item, StoragePutResult) and item.changed
                or isinstance(item, StorageDeleteResult) and item.deleted
                or isinstance(item, StorageResetResult) and item.reset
                for item in result.results
            ):
                await self._notify_revision(result.store_revision)
            return result
        if expected_store_revision is not None:
            current = await self.current_revision()
            if current != expected_store_revision:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        results: list[StoragePutResult[InfoT] | StorageDeleteResult[KeyT] | StorageResetResult[KeyT]] = []
        revision = expected_store_revision
        for index, change in enumerate(changes):
            try:
                if change.operation is StorageOperation.PUT:
                    value = cast(ValueT, change.value)
                    item = await self.put(
                        change.key,
                        value,
                        expected_entry_revision=change.expected_entry_revision,
                    )
                elif change.operation is StorageOperation.DELETE:
                    item = await self.delete(
                        change.key,
                        expected_entry_revision=change.expected_entry_revision,
                    )
                else:
                    item = await self.reset(
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
        changes: 'Sequence[StorageChange[KeyT, ValueT]]',
        result: 'StorageBatchResult[InfoT, KeyT]',
    ) -> None:
        if len(result.results) != len(changes):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "storage batch result count mismatch")
        for change, item in zip(changes, result.results):
            if change.operation is StorageOperation.PUT:
                if not isinstance(item, StoragePutResult):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "storage batch put result mismatch")
                self._validate_value(change.key, cast(ValueT, change.value), item.info)
            elif change.operation is StorageOperation.DELETE and (
                not isinstance(item, StorageDeleteResult) or item.key != change.key
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "storage batch delete result mismatch")
            elif change.operation is StorageOperation.RESET and (
                not isinstance(item, StorageResetResult) or item.key != change.key
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "storage batch reset result mismatch")

    def _validate_batch(self, changes: 'Sequence[StorageChange[KeyT, ValueT]]') -> None:
        seen: set[KeyT] = set()
        for change in changes:
            if change.operation not in {StorageOperation.PUT, StorageOperation.DELETE, StorageOperation.RESET}:
                raise ValueError(f"unsupported storage operation: {change.operation}")
            if change.key in seen:
                raise AIError(ErrorCode.STORAGE_BATCH_DUPLICATE_KEY)
            seen.add(change.key)
            if change.operation is StorageOperation.PUT and change.value is None:
                raise ValueError("PUT changes require a value")
            if change.operation in {StorageOperation.DELETE, StorageOperation.RESET} and change.value is not None:
                raise ValueError(f"{change.operation.value} changes cannot contain a value")

    def _after_put(
        self,
        key: KeyT,
        info: InfoT,
        revision: StorageRevision,
    ) -> None:
        self._preloaded.pop(key, None)
        if self.writer is None:
            return
        for view in self._views:
            if view.backend is self.writer:
                view.apply_write(key, info, revision)

    def _invalidate_writer_view(self) -> None:
        if self.writer is None:
            return
        for view in self._views:
            if view.backend is self.writer:
                view.invalidate()

    async def _notify_revision(self, revision: StorageRevision) -> None:
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

    async def list_versions(self, key: KeyT) -> 'tuple[VersionSummary, ...]':
        state = await self._state()
        owners = tuple(range(len(self._views)))
        collected: dict[tuple[int, str, StorageEntryStatus], VersionSummary] = {}
        for owner in owners:
            backend = self._views[owner].backend
            if not isinstance(backend, VersionedStorage):
                continue
            versions = tuple(await backend.list_versions(key))
            for version in versions:
                collected[(version.entry_revision.value, version.digest, version.status)] = version
        if collected:
            return tuple(sorted(collected.values(), key=lambda version: version.entry_revision.value, reverse=True))
        if key not in state.owners:
            raise AIError(ErrorCode.ASSET_VERSION_OWNER_UNKNOWN)
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)

    async def get_at_revision(self, key: KeyT, entry_revision: StorageEntryRevision) -> 'ValueT | None':
        state = await self._state()
        owners = tuple(range(len(self._views)))
        for owner in owners:
            backend = self._views[owner].backend
            if not isinstance(backend, VersionedStorage):
                continue
            versions = tuple(await backend.list_versions(key))
            if any(version.entry_revision == entry_revision for version in versions):
                return await backend.get_at_revision(key, entry_revision)
        raise AIError(ErrorCode.ASSET_VERSION_OWNER_UNKNOWN if key not in state.owners else ErrorCode.STORAGE_VERSION_UNSUPPORTED)

    async def get_at_version(self, key: KeyT, version: int) -> 'ValueT | None':
        state = await self._state()
        owners = tuple(range(len(self._views)))
        for owner in owners:
            backend = self._views[owner].backend
            if not isinstance(backend, VersionedStorage):
                continue
            versions = tuple(await backend.list_versions(key))
            if any(item.entry_revision.value == version for item in versions):
                return await backend.get_at_version(key, version)
        raise AIError(ErrorCode.ASSET_VERSION_OWNER_UNKNOWN if key not in state.owners else ErrorCode.STORAGE_VERSION_UNSUPPORTED)


__all__ = [
    "CacheAdapter",
    "EffectiveMetadataState",
    "StorageLayer",
    "StorageLocation",
    "StorageOverlay",
    "StorageValueValidator",
]
