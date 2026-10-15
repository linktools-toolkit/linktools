#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ordered, revision-aware storage overlay."""

import asyncio
from collections.abc import Awaitable, Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, cast

from linktools.core import environ

from ..core import JsonValue, canonical_sha256
from ..errors import AIError, ErrorCode
from ._cache import CacheBackend, contains_many, read_cache, write_cache
from ._contracts import (
    AtomicBatchStorageWriter,
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
    StorageLocatedInfo,
    StoragePutResult,
    StorageResetResult,
    StorageRevision,
    StorageStatReader,
    StorageWriter,
)
from ._layer import LayerRefreshPolicy, StorageLayer
from ._revision import (
    LayerMetadataView,
    MetadataState,
    RevisionSource,
)

KeyT = TypeVar("KeyT", bound=Hashable)
ValueT = TypeVar("ValueT")
InfoT = TypeVar("InfoT")

_logger = environ.get_logger("ai.storage.overlay")


def _is_deleted(info: InfoT) -> bool:
    return isinstance(info, StorageEntryStatusInfo) and info.status is StorageEntryStatus.DELETED


def _is_reset(info: InfoT) -> bool:
    return isinstance(info, StorageEntryStatusInfo) and info.status is StorageEntryStatus.RESET


async def _settle_task(task: "asyncio.Task[object]") -> bool:
    """Wait for owned work to finish before propagating caller cancellation."""
    cancelled = False
    while True:
        try:
            await asyncio.wait((task,))
            task.result()
            return cancelled
        except asyncio.CancelledError:
            if task.done():
                if task.cancelled():
                    raise
                task.result()
                return True
            cancelled = True


class StorageValueValidator(Protocol[KeyT, ValueT, InfoT]):
    def validate_value(self, key: KeyT, value: ValueT, info: InfoT) -> None: ...


class ContentCache(Protocol[KeyT, ValueT, InfoT]):
    """Bind a borrowed byte cache to one domain's content identity and codec."""

    @property
    def backend(self) -> CacheBackend: ...

    def cache_key(self, key: KeyT, info: InfoT) -> str: ...
    def to_cache(self, value: ValueT) -> bytes: ...
    def from_cache(self, value: bytes) -> ValueT: ...


@dataclass(frozen=True, slots=True)
class EffectiveMetadataState(Generic[KeyT, InfoT]):
    revision: StorageRevision
    entries: "Mapping[KeyT, InfoT]"
    layer_indexes: "Mapping[KeyT, int]"


@dataclass(frozen=True, slots=True)
class _OverlayMetadataCapture(Generic[KeyT, InfoT]):
    effective: EffectiveMetadataState[KeyT, InfoT]
    writer_state: "MetadataState[KeyT, InfoT] | None"


@dataclass(frozen=True, slots=True)
class StorageWriteState(Generic[InfoT]):
    effective: "StorageLocatedInfo[InfoT] | None"
    writer: InfoT | None
    writer_can_own: bool


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
        content_cache: 'ContentCache[KeyT, ValueT, InfoT] | None' = None,
        cache_concurrency: int = 16,
        preload_batch_size: int = 100,
        preload_concurrency: int = 8,
    ) -> None:
        layer_values = tuple(layers)
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
        self.content_cache = content_cache
        self.cache_concurrency = cache_concurrency
        self.preload_batch_size = preload_batch_size
        self.preload_concurrency = preload_concurrency
        self._cache_tasks: dict[tuple[int, str], asyncio.Future[tuple[ValueT | None, bytes | None]]] = {}
        self._cache_loaders: set[asyncio.Task[None]] = set()
        self._cache_task_lock = asyncio.Lock()
        self._views = (
            LayerMetadataView(
                primary,
                LayerRefreshPolicy.REVISIONED,
                revision_source=revision_source,
            ),
            *(LayerMetadataView(layer.backend, layer.refresh) for layer in layer_values),
        )
        self._writer_index = next(
            (
                index
                for index, view in enumerate(self._views)
                if view.backend is writer
            ),
            None,
        )
        if writer is not None and self._writer_index is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._initialize_lock = asyncio.Lock()
        self._initialized = False
        self._close_started = False
        self._closed = False
        self._lifecycle_backends: tuple[InitializableStorage, ...] = ()

    @property
    def writer_is_primary(self) -> bool:
        return self.writer is self.primary

    @property
    def writable(self) -> bool:
        """Return whether generic storage mutations have a configured writer."""
        return self.writer is not None

    @property
    def atomic_batch(self) -> bool:
        """Return whether the configured writer commits batches atomically."""
        return isinstance(self.writer, AtomicBatchStorageWriter) and self.writer.atomic_batch

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
            if self._closed or self._close_started:
                raise AIError(ErrorCode.STORAGE_CLOSED)
            if self._initialized:
                return
            seen: set[int] = set()
            opened: list[InitializableStorage] = []
            try:
                for backend in (view.backend for view in self._views):
                    identity = id(backend)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    if isinstance(backend, InitializableStorage):
                        opened.append(backend)
                        await backend.initialize()
            except BaseException:
                failed: list[InitializableStorage] = []
                for backend in reversed(opened):
                    try:
                        await backend.close()
                    except BaseException as cleanup_error:  # noqa: BLE001
                        failed.append(backend)
                        _logger.error(
                            "storage backend cleanup failed after initialize error: "
                            "backend=%s exception_type=%s",
                            type(backend).__name__,
                            type(cleanup_error).__name__,
                        )
                self._initialized = False
                if failed:
                    self._close_started = True
                    self._lifecycle_backends = tuple(reversed(failed))
                else:
                    self._closed = True
                raise
            self._lifecycle_backends = tuple(opened)
            self._initialized = True
        _logger.debug("storage overlay initialized: layers=%s", len(self._views))

    async def close(self) -> None:
        async with self._initialize_lock:
            if self._closed:
                return
            self._close_started = True
            self._initialized = False
            backends = self._lifecycle_backends

        first_error: BaseException | None = None
        cancelled = False
        async with self._cache_task_lock:
            tasks = tuple(self._cache_loaders)
        tasks += tuple(
            task
            for view in self._views
            if (task := view.pending_refresh_task) is not None
        )
        for task in tasks:
            try:
                cancelled = await _settle_task(cast("asyncio.Task[object]", task)) or cancelled
            except asyncio.CancelledError:
                cancelled = True
            except BaseException as error:  # noqa: BLE001
                if first_error is None:
                    first_error = error
        for backend in reversed(backends):
            task = asyncio.create_task(backend.close())
            try:
                cancelled = await _settle_task(cast("asyncio.Task[object]", task)) or cancelled
            except asyncio.CancelledError:
                cancelled = True
            except BaseException as error:  # noqa: BLE001
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
        async with self._initialize_lock:
            self._lifecycle_backends = ()
            self._closed = True
        if cancelled:
            raise asyncio.CancelledError
        _logger.debug("storage overlay closed")

    async def refresh(self) -> StorageRevision:
        return (await self._capture_metadata()).effective.revision

    async def _state(self) -> 'EffectiveMetadataState[KeyT, InfoT]':
        return (await self._capture_metadata()).effective

    async def _capture_metadata(self) -> "_OverlayMetadataCapture[KeyT, InfoT]":
        states = await asyncio.gather(*(view.refresh() for view in self._views))
        entries: dict[KeyT, InfoT] = {}
        layer_indexes: dict[KeyT, int] = {}
        revisions: list[str] = []
        for index, state in enumerate(states):
            revisions.append(str(state.revision))
            for key, info in state.entries.items():
                if _is_reset(info):
                    continue
                if key not in entries:
                    entries[key] = info
                    layer_indexes[key] = index
        revision = self._effective_revision(revisions, states[0].revision)
        writer_state = None if self._writer_index is None else states[self._writer_index]
        return _OverlayMetadataCapture(
            EffectiveMetadataState(revision, entries, layer_indexes),
            writer_state,
        )

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
        retried = False
        while True:
            info = state.entries.get(key)
            if info is None:
                return None
            layer_index = state.layer_indexes[key]
            view = self._views[layer_index]
            backend = view.backend
            if not isinstance(backend, StorageStatReader):
                return info
            origin = await backend.stat(key)
            if origin == info:
                return info
            if not retried and view.policy is not LayerRefreshPolicy.STATIC:
                view.invalidate()
                refreshed = await self._state()
                if (refreshed.entries.get(key) != info
                        or refreshed.layer_indexes.get(key) != layer_index):
                    state = refreshed
                    retried = True
                    continue
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                "storage metadata points to a missing origin" if origin is None else "storage metadata and origin disagree",
                safe_details={
                    "storage_key_digest": canonical_sha256(str(key)),
                    "layer": self._layer_id(layer_index),
                },
            )

    async def locate(self, key: KeyT) -> 'StorageLocation[KeyT, ValueT, InfoT] | None':
        """Return the effective layer selected by precedence for one key."""
        return (await self.locate_many((key,)))[0]

    async def locate_many(
        self,
        keys: 'Sequence[KeyT]',
    ) -> 'tuple[StorageLocation[KeyT, ValueT, InfoT] | None, ...]':
        """Return effective layers for keys from one metadata capture."""
        state = await self._state()
        result: list[StorageLocation[KeyT, ValueT, InfoT] | None] = []
        for key in keys:
            info = state.entries.get(key)
            if info is None:
                result.append(None)
                continue
            layer_index = state.layer_indexes[key]
            backend = self._views[layer_index].backend
            result.append(
                StorageLocation(
                    key,
                    info,
                    backend,
                    self._layer_id(layer_index),
                    self.is_writable_backend(backend),
                )
            )
        return tuple(result)

    def invalidate(self) -> None:
        """Discard materialized metadata after an external mutation."""
        for view in self._views:
            view.invalidate()

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
        layer_index = state.layer_indexes[key]
        if self.content_cache is not None:
            value = (await self._read_cached_or_origins((key,), state))[key]
        else:
            value = await self._views[layer_index].backend.get(key)
        if value is None:
            self._views[layer_index].invalidate()
            if retried:
                _logger.error(
                    "storage metadata has no origin value: key=%s",
                    key,
                    extra={"error_code": ErrorCode.STORAGE_OWNER_MISMATCH.value},
                )
                raise AIError(
                    ErrorCode.STORAGE_INTEGRITY_ERROR,
                    "storage metadata points to a missing origin",
                    safe_details={"storage_key_digest": canonical_sha256(str(key))},
                )
            return await self._get_from_state(key, await self._state(), retried=True)
        self._validate_value(key, value, info)
        return value

    async def _get_many_from_state(
        self,
        keys: 'Sequence[KeyT]',
        state: 'EffectiveMetadataState[KeyT, InfoT]',
        *,
        retried: bool,
    ) -> 'dict[KeyT, ValueT | None]':
        visible = tuple(key for key in keys if key in state.entries and not _is_deleted(state.entries[key]))
        loaded = await self._read_cached_or_origins(visible, state)
        values: dict[KeyT, ValueT | None] = {key: None for key in keys}
        raced: set[int] = set()
        missing_origins: list[KeyT] = []
        for key in visible:
            value = loaded.get(key)
            if value is None:
                raced.add(state.layer_indexes[key])
                missing_origins.append(key)
                continue
            self._validate_value(key, value, state.entries[key])
            values[key] = value
        if raced:
            for layer_index in raced:
                self._views[layer_index].invalidate()
            if not retried:
                refreshed = await self._state()
                return await self._get_many_from_state(keys, refreshed, retried=True)
            for key in missing_origins:
                _logger.error(
                    "storage layer mismatch after batch refresh: key=%s",
                    key,
                    extra={"error_code": ErrorCode.STORAGE_OWNER_MISMATCH.value},
                )
                raise AIError(
                    ErrorCode.STORAGE_INTEGRITY_ERROR,
                    "storage metadata and origin disagree",
                    safe_details={
                        "storage_key_digest": canonical_sha256(str(key)),
                    },
                )
        return values

    async def _read_cached_or_origins(
        self,
        keys: 'Sequence[KeyT]',
        state: 'EffectiveMetadataState[KeyT, InfoT]',
    ) -> 'dict[KeyT, ValueT | None]':
        if self.content_cache is None:
            loaded, errors = await self._load_origins(keys, state)
            if errors:
                raise next(iter(errors.values()))
            return loaded
        tasks: dict[KeyT, asyncio.Future[tuple[ValueT | None, bytes | None]]] = {}
        owned: dict[KeyT, asyncio.Future[tuple[ValueT | None, bytes | None]]] = {}
        async with self._cache_task_lock:
            if self._close_started:
                raise AIError(ErrorCode.STORAGE_CLOSED)
            for key in keys:
                identity = (state.layer_indexes[key], self.content_cache.cache_key(key, state.entries[key]))
                future = self._cache_tasks.get(identity)
                if future is None:
                    future = asyncio.get_running_loop().create_future()
                    self._cache_tasks[identity] = future
                    owned[key] = future

                    def discard(done: "asyncio.Future[tuple[ValueT | None, bytes | None]]", identity=identity) -> None:
                        # A cancelled waiter must not leave an unobserved origin failure.
                        if not done.cancelled():
                            done.exception()
                        if self._cache_tasks.get(identity) is done:
                            self._cache_tasks.pop(identity, None)

                    future.add_done_callback(discard)
                tasks[key] = future
            if owned:
                loader = asyncio.create_task(self._load_cache_batch(owned, state))
                self._cache_loaders.add(loader)

                def finished(done: "asyncio.Task[None]") -> None:
                    self._cache_loaders.discard(done)
                    error = asyncio.CancelledError() if done.cancelled() else done.exception()
                    if error is not None:
                        for future in owned.values():
                            if not future.done():
                                future.set_exception(error)

                loader.add_done_callback(finished)
        if not tasks:
            return {}
        # The loader owns these futures; cancelling a reader only stops its wait.
        done, _ = await asyncio.wait(tasks.values(), return_when=asyncio.FIRST_EXCEPTION)
        for future in done:
            future.result()
        loaded = [future.result() for future in tasks.values()]
        return {key: value if type(value) is bytes or content is None else self.content_cache.from_cache(content)
                for key, (value, content) in zip(tasks, loaded)}

    async def _load_cache_batch(
        self,
        pending: 'Mapping[KeyT, asyncio.Future[tuple[ValueT | None, bytes | None]]]',
        state: 'EffectiveMetadataState[KeyT, InfoT]',
    ) -> None:
        semaphore = asyncio.Semaphore(self.cache_concurrency)

        async def read(key: KeyT) -> None:
            try:
                info = state.entries[key]
                async with semaphore:
                    value, content = await self._load_cache_content(key, info, self.content_cache.cache_key(key, info))
                if content is not None:
                    pending[key].set_result((value, content))
            except (Exception, asyncio.CancelledError) as error:
                pending[key].set_exception(error)

        await asyncio.gather(*(read(key) for key in pending))
        misses = tuple(key for key, future in pending.items() if not future.done())
        async def publish(key: KeyT, value: ValueT | None, error: BaseException | None) -> None:
            if error is not None:
                pending[key].set_exception(error)
                return
            try:
                content = None
                if value is not None:
                    info = state.entries[key]
                    self._validate_value(key, value, info)
                    content = self.content_cache.to_cache(value)
                    await write_cache(self.content_cache.backend, self.content_cache.cache_key(key, info), content)
                pending[key].set_result((value, content))
            except (Exception, asyncio.CancelledError) as error:
                pending[key].set_exception(error)

        await self._load_origins(misses, state, on_read=publish)

    async def _load_cache_content(
        self,
        key: KeyT,
        info: InfoT,
        cache_key: str,
    ) -> 'tuple[ValueT | None, bytes | None]':
        cached = await read_cache(self.content_cache.backend, cache_key)
        if cached is None:
            return None, None
        try:
            value = self.content_cache.from_cache(cached)
            self._validate_value(key, value, info)
            return value, cached
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            _logger.warning(
                "storage cache entry failed validation: key=%s",
                key,
                extra={"error_code": ErrorCode.STORAGE_CACHE_CORRUPT.value},
            )
            await self._delete_cache(cache_key)
            return None, None

    async def _load_origins(
        self,
        keys: 'Sequence[KeyT]',
        state: 'EffectiveMetadataState[KeyT, InfoT]',
        *, on_read: 'Callable[[KeyT, ValueT | None, BaseException | None], Awaitable[None]] | None' = None,
    ) -> 'tuple[dict[KeyT, ValueT], dict[KeyT, BaseException]]':
        loaded: dict[KeyT, ValueT] = {}
        errors: dict[KeyT, BaseException] = {}
        grouped: dict[int, list[KeyT]] = {}
        for key in keys:
            if key in state.entries:
                grouped.setdefault(state.layer_indexes[key], []).append(key)
        semaphore = asyncio.Semaphore(self.preload_concurrency)

        async def _load_group(layer_index: int, group: 'Sequence[KeyT]') -> None:
            backend = self._views[layer_index].backend
            if isinstance(backend, BatchStorageReader):
                try:
                    async with semaphore:
                        raw_values = await backend.get_many(group)
                except (Exception, asyncio.CancelledError) as error:
                    errors.update((key, error) for key in group)
                    if on_read is not None:
                        await asyncio.gather(*(on_read(key, None, error) for key in group))
                    return
                for key in group:
                    raw = raw_values.get(key)
                    if raw is not None:
                        loaded[key] = raw
                if on_read is not None:
                    await asyncio.gather(*(on_read(key, raw_values.get(key), None) for key in group))
                return

            async def _load_one(key: KeyT) -> None:
                try:
                    async with semaphore:
                        raw = await backend.get(key)
                except (Exception, asyncio.CancelledError) as error:
                    errors[key] = error
                    if on_read is not None:
                        await on_read(key, None, error)
                    return
                if raw is not None:
                    loaded[key] = raw
                if on_read is not None:
                    await on_read(key, raw, None)

            await asyncio.gather(*(_load_one(key) for key in group))

        await asyncio.gather(
            *(_load_group(layer_index, group) for layer_index, group in grouped.items())
        )
        return loaded, errors

    async def _delete_cache(self, cache_key: str) -> None:
        if self.content_cache is None:
            return
        try:
            await self.content_cache.backend.delete(cache_key)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            _logger.debug("failed to delete corrupted storage cache entry: key=%s", cache_key)

    async def list_info(self) -> 'tuple[InfoT, ...]':
        state = await self._state()
        return tuple(state.entries[key] for key in sorted(state.entries, key=str))

    async def list_info_with_locations(self) -> 'tuple[StorageLocatedInfo[InfoT], ...]':
        state = await self._state()
        return tuple(
            StorageLocatedInfo(
                state.entries[key],
                self._layer_id(state.layer_indexes[key]),
                self.is_writable_backend(self._views[state.layer_indexes[key]].backend),
            )
            for key in sorted(state.entries, key=str)
        )

    async def write_states(
        self,
        keys: "Sequence[KeyT]",
    ) -> "Mapping[KeyT, StorageWriteState[InfoT]]":
        self._require_writer()
        capture = await self._capture_metadata()
        if capture.writer_state is None or self._writer_index is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result: dict[KeyT, StorageWriteState[InfoT]] = {}
        for key in dict.fromkeys(keys):
            layer_index = capture.effective.layer_indexes.get(key)
            effective_info = capture.effective.entries.get(key)
            effective = None
            if effective_info is not None and layer_index is not None:
                effective = StorageLocatedInfo(
                    effective_info,
                    self._layer_id(layer_index),
                    layer_index == self._writer_index,
                )
            result[key] = StorageWriteState(
                effective,
                capture.writer_state.entries.get(key),
                layer_index is None or self._writer_index <= layer_index,
            )
        return result

    def _layer_id(self, index: int) -> str:
        return "primary" if index == 0 else self.layers[index - 1].id

    async def preload(self, keys: 'Sequence[KeyT] | None' = None) -> PreloadResult:
        state = await self._state()
        selected = tuple(dict.fromkeys(keys)) if keys is not None else tuple(state.entries)
        if self.content_cache is None:
            return PreloadResult(state.revision, len(selected), 0, 0, len(selected))
        identities = {
            key: self.content_cache.cache_key(key, state.entries[key])
            for key in selected
            if key in state.entries and not _is_deleted(state.entries[key])
        }
        present = await contains_many(self.content_cache.backend, tuple(identities.values()))
        already_cached = sum(1 for identity in identities.values() if identity in present)
        missing = tuple(
            key
            for key, identity in identities.items()
            if identity not in present
        )
        loaded_count = 0
        failed = len(selected) - len(identities)
        for offset in range(0, len(missing), self.preload_batch_size):
            batch = missing[offset : offset + self.preload_batch_size]
            raw_values = await self._read_cached_or_origins(batch, state)
            loaded_keys: list[KeyT] = []
            for key in batch:
                raw = raw_values.get(key)
                if raw is None:
                    failed += 1
                    continue
                self._validate_value(key, raw, state.entries[key])
                loaded_keys.append(key)
            if loaded_keys:
                present_after = await contains_many(
                    self.content_cache.backend,
                    tuple(identities[key] for key in loaded_keys),
                )
                for key in loaded_keys:
                    if identities[key] in present_after:
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
        expected_revision: 'StorageEntryRevision | None' = None,
        metadata: 'Mapping[str, JsonValue] | None' = None,
    ) -> 'StoragePutResult[InfoT]':
        writer = self._require_writer()
        result = await writer.put(
            key,
            value,
            expected_revision=expected_revision,
            metadata=metadata,
        )
        self._validate_value(key, value, result.info)
        self._after_put(key, result.info, result.store_revision)
        if result.changed:
            await self._notify_revision(result.store_revision)
        return result

    async def delete(
        self,
        key: KeyT,
        *,
        expected_revision: 'StorageEntryRevision | None' = None,
        metadata: 'Mapping[str, JsonValue] | None' = None,
    ) -> 'StorageDeleteResult[KeyT]':
        writer = self._require_writer()
        result = await writer.delete(
            key,
            expected_revision=expected_revision,
            metadata=metadata,
        )
        if result.deleted:
            self._invalidate_writer_view()
            await self._notify_revision(result.store_revision)
        return result

    async def reset(
        self,
        key: KeyT,
        *,
        expected_revision: 'StorageEntryRevision | None' = None,
        metadata: 'Mapping[str, JsonValue] | None' = None,
    ) -> 'StorageResetResult[KeyT]':
        writer = self._require_writer()
        result = await writer.reset(
            key,
            expected_revision=expected_revision,
            metadata=metadata,
        )
        if result.reset:
            self._invalidate_writer_view()
            await self._notify_revision(result.store_revision)
        return result

    async def apply_batch(
        self,
        changes: 'Sequence[StorageChange[KeyT, ValueT]]',
        *,
        expected_revision: 'StorageRevision | None' = None,
        idempotency_key: 'str | None' = None,
        request_digest: 'str | None' = None,
    ) -> 'StorageBatchResult[InfoT, KeyT]':
        self._validate_batch(changes)
        if (idempotency_key is None) != (request_digest is None):
            raise ValueError("idempotency_key and request_digest must be provided together")
        writer = self._require_writer()
        if isinstance(writer, BatchStorageWriter):
            if idempotency_key is not None:
                receipt = await writer.batch_result(idempotency_key)
                if receipt is not None:
                    if receipt.request_digest != request_digest:
                        raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                    self._validate_writer_batch_result(changes, receipt)
                    self._invalidate_writer_view()
                    return receipt
            writer_expected_revision = None
            if expected_revision is not None:
                capture = await self._capture_metadata()
                if capture.effective.revision != expected_revision:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                if capture.writer_state is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                writer_expected_revision = capture.writer_state.revision
            result = await writer.apply_batch(
                changes,
                expected_revision=writer_expected_revision,
                idempotency_key=idempotency_key,
                request_digest=request_digest,
            )
            self._validate_writer_batch_result(changes, result)
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
        if idempotency_key is not None:
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
        if expected_revision is not None:
            current = await self.current_revision()
            if current != expected_revision:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        results: list[StoragePutResult[InfoT] | StorageDeleteResult[KeyT] | StorageResetResult[KeyT]] = []
        revision = expected_revision
        for index, change in enumerate(changes):
            try:
                if change.operation is StorageOperation.PUT:
                    value = cast(ValueT, change.value)
                    item = await self.put(
                        change.key,
                        value,
                        expected_revision=change.expected_revision,
                        metadata=change.metadata,
                    )
                elif change.operation is StorageOperation.DELETE:
                    item = await self.delete(
                        change.key,
                        expected_revision=change.expected_revision,
                        metadata=change.metadata,
                    )
                else:
                    item = await self.reset(
                        change.key,
                        expected_revision=change.expected_revision,
                        metadata=change.metadata,
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

    async def batch_result(
        self,
        idempotency_key: str,
    ) -> 'StorageBatchResult[InfoT, KeyT] | None':
        writer = self._require_writer()
        if not isinstance(writer, BatchStorageWriter):
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
        return await writer.batch_result(idempotency_key)

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
        if self._writer_index is None:
            return
        view = self._views[self._writer_index]
        source = view.revision_source
        if source is not None:
            try:
                await source.revision_bumped(revision)
            except Exception:
                view.invalidate()
                _logger.warning(
                    "storage revision notification failed",
                    extra={"error_code": ErrorCode.STORAGE_REVISION_NOTIFY_FAILED.value},
                    exc_info=True,
                )



__all__ = [
    "ContentCache",
    "EffectiveMetadataState",
    "StorageLayer",
    "StorageLocation",
    "StorageOverlay",
    "StorageValueValidator",
    "StorageWriteState",
]
