#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Domain-independent storage DTOs and protocols."""

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Generic, Protocol, TypeVar, runtime_checkable

from ..errors import AIError, ErrorCode

KeyT = TypeVar("KeyT", bound=Hashable)
ValueT = TypeVar("ValueT")
InfoT = TypeVar("InfoT")


@dataclass(frozen=True, slots=True)
class StorageEntryRevision:
    value: int

    def __post_init__(self) -> None:
        if not isinstance(self.value, int) or isinstance(self.value, bool) or self.value < 1:
            raise ValueError("storage entry revision must be positive")

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class StorageRevision:
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value:
            raise ValueError("storage revision must not be empty")

    def __str__(self) -> str:
        return self.value


class StorageOperation(StrEnum):
    PUT = "PUT"
    DELETE = "DELETE"


@dataclass(frozen=True, slots=True)
class StorageChange(Generic[KeyT, ValueT]):
    operation: StorageOperation
    key: KeyT
    value: "ValueT | None"
    expected_entry_revision: "StorageEntryRevision | None"

    def __post_init__(self) -> None:
        if self.operation is StorageOperation.PUT and self.value is None:
            raise ValueError("PUT changes require a value")
        if self.operation is StorageOperation.DELETE and self.value is not None:
            raise ValueError("DELETE changes cannot contain a value")


@dataclass(frozen=True, slots=True)
class StoragePutResult(Generic[InfoT]):
    info: InfoT
    entry_revision: StorageEntryRevision
    store_revision: StorageRevision
    changed: bool


@dataclass(frozen=True, slots=True)
class StorageDeleteResult(Generic[KeyT]):
    key: KeyT
    deleted: bool
    entry_revision: "StorageEntryRevision | None"
    store_revision: StorageRevision


@dataclass(frozen=True, slots=True)
class StorageResetResult:
    store_revision: StorageRevision
    deleted_count: int


@dataclass(frozen=True, slots=True)
class StorageBatchFailure(Generic[InfoT, KeyT]):
    failed_index: int
    error_code: str
    store_revision: StorageRevision
    completed: "tuple[StoragePutResult[InfoT] | StorageDeleteResult[KeyT], ...]"


class StorageBatchPartialError(
    AIError,
    Generic[InfoT, KeyT],
):
    def __init__(
        self,
        failure: 'StorageBatchFailure[InfoT, KeyT]',
    ) -> None:
        self.failure = failure
        super().__init__(ErrorCode.STORAGE_BATCH_PARTIAL_FAILURE)


@dataclass(frozen=True, slots=True)
class StorageBatchResult(Generic[InfoT, KeyT]):
    store_revision: StorageRevision
    atomic: bool
    results: "tuple[StoragePutResult[InfoT] | StorageDeleteResult[KeyT], ...]"


class MetadataLoadMode(StrEnum):
    REPLACE = "REPLACE"
    PATCH = "PATCH"


@dataclass(frozen=True, slots=True)
class MetadataChange(Generic[KeyT, InfoT]):
    key: KeyT
    info: "InfoT | None"


@dataclass(frozen=True, slots=True)
class MetadataLoad(Generic[KeyT, InfoT]):
    mode: MetadataLoadMode
    store_revision: StorageRevision
    changes: "tuple[MetadataChange[KeyT, InfoT], ...]"


@dataclass(frozen=True, slots=True)
class VersionSummary:
    entry_revision: StorageEntryRevision
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
class PreloadResult:
    store_revision: StorageRevision
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
class StorageWriter(Protocol[KeyT, ValueT, InfoT]):
    async def put(
        self,
        key: KeyT,
        value: ValueT,
        *,
        expected_entry_revision: 'StorageEntryRevision | None' = None,
    ) -> 'StoragePutResult[InfoT]': ...

    async def delete(
        self,
        key: KeyT,
        *,
        expected_entry_revision: 'StorageEntryRevision | None' = None,
    ) -> 'StorageDeleteResult[KeyT]': ...

    async def reset(self) -> StorageResetResult: ...


@runtime_checkable
class BatchStorageWriter(Protocol[KeyT, ValueT, InfoT]):
    async def apply_batch(
        self,
        changes: 'Sequence[StorageChange[KeyT, ValueT]]',
        *,
        expected_store_revision: 'StorageRevision | None' = None,
    ) -> 'StorageBatchResult[InfoT, KeyT]': ...


@runtime_checkable
class StorageMetadataReader(Protocol[KeyT, InfoT]):
    async def head_revision(self) -> StorageRevision: ...

    async def load_metadata(
        self,
        after_revision: 'StorageRevision | None',
    ) -> 'MetadataLoad[KeyT, InfoT]': ...


@runtime_checkable
class VersionedStorage(Protocol[KeyT, ValueT]):
    async def list_versions(self, key: KeyT) -> 'Sequence[VersionSummary]': ...

    async def get_at_revision(
        self,
        key: KeyT,
        entry_revision: StorageEntryRevision,
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
class ReadableStorageBackend(
    StorageReader[KeyT, ValueT],
    StorageMetadataReader[KeyT, InfoT],
    Protocol,
):
    pass


@runtime_checkable
class StorageStatReader(Protocol[KeyT, InfoT]):
    async def stat(self, key: KeyT) -> "InfoT | None": ...


__all__ = [
    "BatchStorageReader",
    "BatchStorageWriter",
    "InitializableStorage",
    "MetadataChange",
    "MetadataLoad",
    "MetadataLoadMode",
    "PreloadResult",
    "ReadableStorageBackend",
    "StorageBatchFailure",
    "StorageBatchPartialError",
    "StorageBatchResult",
    "StorageChange",
    "StorageDeleteResult",
    "StorageEntryRevision",
    "StorageMetadataReader",
    "StorageOperation",
    "StorageOwnedInfo",
    "StoragePutResult",
    "StorageReader",
    "StorageResetResult",
    "StorageRevision",
    "StorageStatReader",
    "StorageWriter",
    "VersionSummary",
    "VersionedStorage",
]
