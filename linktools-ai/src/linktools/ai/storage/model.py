#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Domain-independent storage DTOs and protocols."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from collections.abc import Mapping, Sequence
from typing import Generic, Protocol, TypeVar, runtime_checkable

from ..core.errors import ErrorCode, LinktoolsAIError

KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")
InfoT = TypeVar("InfoT")
EntryRevisionT = TypeVar("EntryRevisionT")
StoreRevisionT = TypeVar("StoreRevisionT")


class StorageOperation(StrEnum):
    PUT = "PUT"
    DELETE = "DELETE"


@dataclass(frozen=True, slots=True)
class StorageChange(Generic[KeyT, ValueT, EntryRevisionT]):
    operation: StorageOperation
    key: KeyT
    value: "ValueT | None"
    expected_entry_revision: "EntryRevisionT | None"

    def __post_init__(self) -> None:
        if self.operation is StorageOperation.PUT and self.value is None:
            raise ValueError("PUT changes require a value")
        if self.operation is StorageOperation.DELETE and self.value is not None:
            raise ValueError("DELETE changes cannot contain a value")


@dataclass(frozen=True, slots=True)
class StoragePutResult(Generic[InfoT, EntryRevisionT, StoreRevisionT]):
    info: InfoT
    entry_revision: EntryRevisionT
    store_revision: StoreRevisionT
    changed: bool


@dataclass(frozen=True, slots=True)
class StorageDeleteResult(Generic[KeyT, EntryRevisionT, StoreRevisionT]):
    key: KeyT
    deleted: bool
    entry_revision: "EntryRevisionT | None"
    store_revision: StoreRevisionT


@dataclass(frozen=True, slots=True)
class StorageResetResult(Generic[StoreRevisionT]):
    store_revision: StoreRevisionT
    deleted_count: int


@dataclass(frozen=True, slots=True)
class StorageBatchFailure(Generic[InfoT, KeyT, EntryRevisionT, StoreRevisionT]):
    failed_index: int
    error_code: str
    store_revision: StoreRevisionT
    completed: "tuple[StoragePutResult[InfoT, EntryRevisionT, StoreRevisionT] | StorageDeleteResult[KeyT, EntryRevisionT, StoreRevisionT], ...]"


class StorageBatchPartialError(
    LinktoolsAIError,
    Generic[InfoT, KeyT, EntryRevisionT, StoreRevisionT],
):
    def __init__(
        self,
        failure: 'StorageBatchFailure[InfoT, KeyT, EntryRevisionT, StoreRevisionT]',
    ) -> None:
        self.failure = failure
        super().__init__(ErrorCode.STORAGE_BATCH_PARTIAL_FAILURE)


@dataclass(frozen=True, slots=True)
class StorageBatchResult(Generic[InfoT, KeyT, EntryRevisionT, StoreRevisionT]):
    store_revision: StoreRevisionT
    atomic: bool
    results: "tuple[StoragePutResult[InfoT, EntryRevisionT, StoreRevisionT] | StorageDeleteResult[KeyT, EntryRevisionT, StoreRevisionT], ...]"


class MetadataLoadMode(StrEnum):
    REPLACE = "REPLACE"
    PATCH = "PATCH"


@dataclass(frozen=True, slots=True)
class MetadataChange(Generic[KeyT, InfoT]):
    key: KeyT
    info: "InfoT | None"


@dataclass(frozen=True, slots=True)
class MetadataLoad(Generic[KeyT, InfoT, StoreRevisionT]):
    mode: MetadataLoadMode
    store_revision: StoreRevisionT
    changes: "tuple[MetadataChange[KeyT, InfoT], ...]"


@dataclass(frozen=True, slots=True)
class VersionSummary(Generic[EntryRevisionT]):
    entry_revision: EntryRevisionT
    digest: str
    size: int
    created_at: datetime
    deleted: bool = False


@dataclass(frozen=True, slots=True)
class StorageOwnedInfo(Generic[InfoT]):
    info: InfoT
    layer: str
    writable: bool


@dataclass(frozen=True, slots=True)
class PreloadResult(Generic[StoreRevisionT]):
    store_revision: StoreRevisionT
    requested: int
    loaded: int
    already_cached: int
    failed: int


@runtime_checkable
class StorageReader(Protocol[KeyT, ValueT]):
    async def get(self, key: KeyT) -> 'ValueT | None': ...


@runtime_checkable
class BatchStorageReader(Protocol[KeyT, ValueT]):
    async def get_many(self, keys: 'Sequence[KeyT]') -> 'Mapping[KeyT, ValueT]': ...


@runtime_checkable
class StorageWriter(Protocol[KeyT, ValueT, InfoT, EntryRevisionT, StoreRevisionT]):
    async def put(
        self,
        key: KeyT,
        value: ValueT,
        *,
        expected_entry_revision: 'EntryRevisionT | None' = None,
    ) -> 'StoragePutResult[InfoT, EntryRevisionT, StoreRevisionT]': ...

    async def delete(
        self,
        key: KeyT,
        *,
        expected_entry_revision: 'EntryRevisionT | None' = None,
    ) -> 'StorageDeleteResult[KeyT, EntryRevisionT, StoreRevisionT]': ...

    async def reset(self) -> 'StorageResetResult[StoreRevisionT]': ...


@runtime_checkable
class BatchStorageWriter(Protocol[KeyT, ValueT, InfoT, EntryRevisionT, StoreRevisionT]):
    async def apply_batch(
        self,
        changes: 'Sequence[StorageChange[KeyT, ValueT, EntryRevisionT]]',
        *,
        expected_store_revision: 'StoreRevisionT | None' = None,
    ) -> 'StorageBatchResult[InfoT, KeyT, EntryRevisionT, StoreRevisionT]': ...


@runtime_checkable
class StorageMetadataBackend(Protocol[KeyT, InfoT, StoreRevisionT]):
    async def head_revision(self) -> StoreRevisionT: ...

    async def load_metadata(
        self,
        after_revision: 'StoreRevisionT | None',
    ) -> 'MetadataLoad[KeyT, InfoT, StoreRevisionT]': ...


@runtime_checkable
class VersionedStorage(Protocol[KeyT, ValueT, EntryRevisionT]):
    async def list_versions(self, key: KeyT) -> 'Sequence[VersionSummary[EntryRevisionT]]': ...

    async def get_at_revision(
        self,
        key: KeyT,
        entry_revision: EntryRevisionT,
    ) -> 'ValueT | None': ...

    async def get_at_version(
        self,
        key: KeyT,
        version: int,
    ) -> 'ValueT | None': ...


@runtime_checkable
class InitializableStorage(Protocol):
    async def initialize(self) -> None: ...


@runtime_checkable
class ReadableMetadataBackend(
    StorageReader[KeyT, ValueT],
    StorageMetadataBackend[KeyT, InfoT, StoreRevisionT],
    Protocol,
):
    pass


@runtime_checkable
class StorageStatBackend(Protocol[KeyT, InfoT]):
    async def stat(self, key: KeyT) -> "InfoT | None": ...


__all__ = [
    "BatchStorageReader", "BatchStorageWriter", "InitializableStorage",
    "MetadataChange", "MetadataLoad", "MetadataLoadMode", "PreloadResult",
    "StorageBatchFailure", "StorageBatchResult", "StorageChange",
    "StorageBatchPartialError",
    "StorageDeleteResult", "StorageMetadataBackend", "StorageOperation",
    "StorageOwnedInfo", "StoragePutResult", "StorageReader", "StorageResetResult",
    "StorageWriter", "VersionSummary", "VersionedStorage", "ReadableMetadataBackend", "StorageStatBackend",
]
