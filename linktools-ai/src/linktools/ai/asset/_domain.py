#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asset file identities and metadata."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from ..core import JsonValue, validate_asset_kind
from ..errors import AIError
from ..storage import (
    ReadableStorageBackend,
    StorageEntryRevision,
    StorageEntryStatus,
    StorageRevision,
    StorageWriter,
    StoredPayload,
    normalize_storage_metadata,
)


@dataclass(frozen=True, slots=True)
class AssetKey:
    kind: str
    id: str

    def __post_init__(self) -> None:
        try:
            validate_asset_kind(self.kind)
            identifier_size = len(self.id.encode("utf-8"))
        except (AIError, UnicodeEncodeError) as error:
            raise ValueError("asset key is invalid") from error
        if (
            not self.id
            or identifier_size > 512
            or "\x00" in self.id
        ):
            raise ValueError("asset key is invalid")


@dataclass(frozen=True, slots=True)
class AssetRoot:
    root_id: str
    scheme: "Literal['file', 'sql', 'memory']"
    locator: str
    digest: str

    def __post_init__(self) -> None:
        if (
            self.scheme not in {"file", "sql", "memory"}
            or not self.root_id
            or not self.locator
            or not self.digest
        ):
            raise ValueError("asset root is incomplete")


@dataclass(frozen=True, slots=True)
class AssetInfo:
    key: AssetKey
    revision: StorageEntryRevision
    store_revision: StorageRevision
    etag: str
    size: int
    status: StorageEntryStatus
    root_id: str
    root_digest: str
    modified_at: datetime
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    content: "StoredPayload | None" = None

    def __post_init__(self) -> None:
        if (
            self.size < 0
            or not self.root_id
            or not self.root_digest
            or not isinstance(self.status, StorageEntryStatus)
            or len(self.etag) != 64
            or any(character not in "0123456789abcdef" for character in self.etag)
        ):
            raise ValueError("asset metadata is invalid")
        if self.modified_at.tzinfo is None:
            raise ValueError("asset metadata requires a timezone-aware timestamp")
        object.__setattr__(self, "metadata", normalize_storage_metadata(self.metadata))
        if self.status is not StorageEntryStatus.NORMAL and (
            self.size != 0 or self.etag != hashlib.sha256(b"").hexdigest()
        ):
            raise ValueError("non-normal asset metadata must not contain file content")
        if self.content is not None and (
            self.content.digest != self.etag or self.content.size != self.size
        ):
            raise ValueError("asset content descriptor does not match metadata")


@runtime_checkable
class AssetBackend(ReadableStorageBackend[AssetKey, bytes, AssetInfo], Protocol):
    @property
    def root(self) -> AssetRoot: ...

    async def initialize(self) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class WritableAssetBackend(AssetBackend, StorageWriter[AssetKey, bytes, AssetInfo], Protocol):
    @property
    def writable(self) -> bool: ...


__all__ = ["AssetBackend", "AssetInfo", "AssetKey", "AssetRoot", "WritableAssetBackend"]
