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
    VersionedStorage,
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
class AssetVersionRef:
    """Stable reference to one immutable Asset version."""

    key: AssetKey
    source_id: str
    revision: StorageEntryRevision
    etag: str
    size: int

    def __post_init__(self) -> None:
        if not isinstance(self.key, AssetKey):
            raise TypeError("asset version key must be AssetKey")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("asset version source_id must be non-empty")
        if not isinstance(self.revision, StorageEntryRevision):
            raise TypeError("asset version revision must be StorageEntryRevision")
        if (
            not isinstance(self.etag, str)
            or len(self.etag) != 64
            or any(character not in "0123456789abcdef" for character in self.etag)
            or isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or self.size < 0
        ):
            raise ValueError("asset version integrity metadata is invalid")

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "version": 1,
            "kind": self.key.kind,
            "id": self.key.id,
            "source_id": self.source_id,
            "revision": self.revision.value,
            "etag": self.etag,
            "size": self.size,
        }

    @classmethod
    def from_payload(cls, value: object) -> "AssetVersionRef":
        if not isinstance(value, Mapping) or set(value) != {
            "version",
            "kind",
            "id",
            "source_id",
            "revision",
            "etag",
            "size",
        }:
            raise ValueError("asset version payload is invalid")
        version = value.get("version")
        if isinstance(version, bool) or version != 1:
            raise ValueError("asset version payload version is unsupported")
        kind = value.get("kind")
        identity = value.get("id")
        source_id = value.get("source_id")
        revision = value.get("revision")
        etag = value.get("etag")
        size = value.get("size")
        if (
            not isinstance(kind, str)
            or not isinstance(identity, str)
            or not isinstance(source_id, str)
            or isinstance(revision, bool)
            or not isinstance(revision, int)
            or not isinstance(etag, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
        ):
            raise ValueError("asset version payload is invalid")
        return cls(
            AssetKey(kind, identity),
            source_id,
            StorageEntryRevision(revision),
            etag,
            size,
        )


@dataclass(frozen=True, slots=True)
class AssetRoot:
    scheme: "Literal['file', 'sql', 'memory']"
    locator: str
    digest: str

    def __post_init__(self) -> None:
        if (
            self.scheme not in {"file", "sql", "memory"}
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
    root_digest: str
    modified_at: datetime
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    content: "StoredPayload | None" = None

    def __post_init__(self) -> None:
        if (
            self.size < 0
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
class AssetBackend(
    ReadableStorageBackend[AssetKey, bytes, AssetInfo],
    VersionedStorage[AssetKey, bytes],
    Protocol,
):
    @property
    def root(self) -> AssetRoot: ...

    async def initialize(self) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class WritableAssetBackend(AssetBackend, StorageWriter[AssetKey, bytes, AssetInfo], Protocol):
    @property
    def writable(self) -> bool: ...


__all__ = [
    "AssetBackend",
    "AssetInfo",
    "AssetKey",
    "AssetRoot",
    "AssetVersionRef",
    "WritableAssetBackend",
]
