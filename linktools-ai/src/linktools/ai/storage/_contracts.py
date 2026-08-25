#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Domain-independent storage DTOs and protocols."""

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Generic, Protocol, TypeVar, runtime_checkable

from ..core import JsonValue
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


class StorageEntryStatus(str, Enum):
    __str__ = str.__str__
    __format__ = str.__format__
    NORMAL = "NORMAL"
    DELETED = "DELETED"
    RESET = "RESET"


class StorageOperation(str, Enum):
    __str__ = str.__str__
    __format__ = str.__format__
    PUT = "PUT"
    DELETE = "DELETE"
    RESET = "RESET"


def normalize_storage_metadata(value: Mapping[str, JsonValue] | None) -> dict[str, JsonValue]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("storage metadata must be a JSON object")
    normalized: dict[str, JsonValue] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("storage metadata keys must be non-empty strings")
        normalized[key] = _normalize_json_value(item)
    return normalized


def _normalize_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("storage metadata numbers must be finite")
        return value
    if isinstance(value, Mapping):
        return normalize_storage_metadata(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    raise TypeError("storage metadata contains an unsupported JSON value")


@dataclass(frozen=True, slots=True)
class StorageChange(Generic[KeyT, ValueT]):
    operation: StorageOperation
    key: KeyT
    value: "ValueT | None"
    expected_revision: "StorageEntryRevision | None"
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.operation is StorageOperation.PUT and self.value is None:
            raise ValueError("PUT changes require a value")
        if self.operation in {StorageOperation.DELETE, StorageOperation.RESET} and self.value is not None:
            raise ValueError(f"{self.operation.value} changes cannot contain a value")
        object.__setattr__(self, "metadata", normalize_storage_metadata(self.metadata))


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
class StorageResetResult(Generic[KeyT]):
    key: KeyT
    reset: bool
    store_revision: StorageRevision


@dataclass(frozen=True, slots=True)
class StorageBatchFailure(Generic[InfoT, KeyT]):
    failed_index: int
    error_code: str
    store_revision: StorageRevision
    completed: "tuple[StoragePutResult[InfoT] | StorageDeleteResult[KeyT] | StorageResetResult[KeyT], ...]"


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
    results: "tuple[StoragePutResult[InfoT] | StorageDeleteResult[KeyT] | StorageResetResult[KeyT], ...]"


class MetadataLoadMode(str, Enum):
    __str__ = str.__str__
    __format__ = str.__format__
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
    status: StorageEntryStatus = StorageEntryStatus.NORMAL
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", normalize_storage_metadata(self.metadata))


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
        expected_revision: 'StorageEntryRevision | None' = None,
        metadata: 'Mapping[str, JsonValue] | None' = None,
    ) -> 'StoragePutResult[InfoT]': ...

    async def delete(
        self,
        key: KeyT,
        *,
        expected_revision: 'StorageEntryRevision | None' = None,
        metadata: 'Mapping[str, JsonValue] | None' = None,
    ) -> 'StorageDeleteResult[KeyT]': ...

    async def reset(
        self,
        key: KeyT,
        *,
        expected_revision: "StorageEntryRevision | None" = None,
        metadata: "Mapping[str, JsonValue] | None" = None,
    ) -> "StorageResetResult[KeyT]":
        """Mark one writer entry RESET so a lower read layer can become effective."""
        ...


@runtime_checkable
class BatchStorageWriter(Protocol[KeyT, ValueT, InfoT]):
    async def apply_batch(
        self,
        changes: 'Sequence[StorageChange[KeyT, ValueT]]',
        *,
        expected_revision: 'StorageRevision | None' = None,
    ) -> 'StorageBatchResult[InfoT, KeyT]': ...


@runtime_checkable
class AtomicBatchStorageWriter(BatchStorageWriter[KeyT, ValueT, InfoT], Protocol):
    @property
    def atomic_batch(self) -> bool: ...


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


@runtime_checkable
class StorageEntryStatusInfo(Protocol):
    @property
    def status(self) -> StorageEntryStatus: ...


__all__ = [
    "AtomicBatchStorageWriter",
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
    "StorageEntryStatus",
    "StorageEntryStatusInfo",
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
    "normalize_storage_metadata",
]
