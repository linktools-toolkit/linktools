#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asset values, metadata and backend protocols."""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, Literal, Protocol, TypeVar, runtime_checkable

from ..core.errors import ErrorCode, LinktoolsAIError
from ..storage.model import (
    BatchStorageReader,
    BatchStorageWriter,
    StorageMetadataBackend,
    StorageReader,
    StorageWriter,
    VersionedStorage,
    StorageBatchFailure,
)

TAsset = TypeVar("TAsset", bound="AssetValue")
_EMPTY_ETAG = hashlib.sha256(b"").hexdigest()


@runtime_checkable
class AssetValue(Protocol):
    @property
    def asset_kind(self) -> str: ...

    @property
    def asset_id(self) -> str: ...


@dataclass(frozen=True, slots=True)
class AssetKey:
    kind: str
    id: str

    def __post_init__(self) -> None:
        try:
            identifier_size = len(self.id.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise ValueError("asset key is invalid") from error
        if not self.kind or not self.id or identifier_size > 512 or "\x00" in self.kind or "\x00" in self.id:
            raise ValueError("asset key is invalid")


@dataclass(frozen=True, slots=True)
class AssetRoot:
    root_id: str
    scheme: "Literal['file', 'sql', 'memory']"
    locator: str
    digest: str

    def __post_init__(self) -> None:
        if self.scheme not in {"file", "sql", "memory"} or not self.root_id or not self.locator or not self.digest:
            raise ValueError("asset root is incomplete")


@dataclass(frozen=True, slots=True)
class AssetRevision:
    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("asset revision must be non-negative")


@dataclass(frozen=True, slots=True)
class AssetStoreRevision:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("asset store revision must not be empty")


@dataclass(frozen=True, slots=True)
class AssetInfo:
    key: AssetKey
    entry_revision: AssetRevision
    store_revision: AssetStoreRevision
    etag: str
    size: int
    deleted: bool
    root_id: str
    root_digest: str
    modified_at: datetime

    def __post_init__(self) -> None:
        if self.size < 0 or not self.root_id or not self.root_digest or len(self.etag) != 64 or any(character not in "0123456789abcdef" for character in self.etag):
            raise ValueError("asset metadata is invalid")
        if self.modified_at.tzinfo is None:
            raise ValueError("asset metadata requires a timezone-aware timestamp")
        if self.deleted and (self.size != 0 or self.etag != _EMPTY_ETAG):
            raise ValueError("asset tombstone metadata is invalid")


@dataclass(frozen=True, slots=True)
class OwnedAssetInfo:
    info: AssetInfo
    layer: str
    writable: bool


@dataclass(frozen=True, slots=True)
class AssetRequest(Generic[TAsset]):
    key: AssetKey
    expected: "type[TAsset]"


@dataclass(frozen=True, slots=True)
class AssetChange:
    operation: "Literal['PUT', 'DELETE']"
    key: AssetKey
    value: "AssetValue | None"
    expected_entry_revision: "AssetRevision | None"


@dataclass(frozen=True, slots=True)
class AssetBatchResult:
    store_revision: AssetStoreRevision
    atomic: bool
    results: "tuple[AssetInfo | AssetDeleteResult, ...]"


class AssetBatchPartialError(LinktoolsAIError):
    def __init__(
        self,
        failure: 'StorageBatchFailure[AssetInfo, AssetKey, AssetRevision, AssetStoreRevision]',
    ) -> None:
        self.failure = failure
        super().__init__(ErrorCode.ASSET_BATCH_PARTIAL_FAILURE)


@dataclass(frozen=True, slots=True)
class AssetDeleteResult:
    key: AssetKey
    deleted: bool
    entry_revision: "AssetRevision | None"
    store_revision: AssetStoreRevision


@dataclass(frozen=True, slots=True)
class AssetVersion:
    entry_revision: AssetRevision
    etag: str
    size: int
    modified_at: datetime
    deleted: bool = False


@runtime_checkable
class AssetBackend(
    StorageReader[AssetKey, bytes],
    StorageWriter[AssetKey, bytes, AssetInfo, AssetRevision, AssetStoreRevision],
    StorageMetadataBackend[AssetKey, AssetInfo, AssetStoreRevision],
    Protocol,
):
    @property
    def root(self) -> AssetRoot: ...

    @property
    def writable(self) -> bool: ...


@runtime_checkable
class BatchAssetReader(BatchStorageReader[AssetKey, bytes], Protocol):
    pass


@runtime_checkable
class BatchAssetWriter(
    BatchStorageWriter[AssetKey, bytes, AssetInfo, AssetRevision, AssetStoreRevision],
    Protocol,
):
    pass


@runtime_checkable
class VersionedAssetBackend(VersionedStorage[AssetKey, bytes, AssetRevision], Protocol):
    pass


__all__ = [
    "AssetBackend", "AssetBatchPartialError", "AssetBatchResult", "AssetChange", "AssetDeleteResult", "AssetInfo",
    "AssetKey", "AssetRequest", "AssetRevision", "AssetRoot", "AssetStoreRevision",
    "AssetValue", "AssetVersion", "BatchAssetReader", "BatchAssetWriter", "OwnedAssetInfo",
    "TAsset", "VersionedAssetBackend",
]
