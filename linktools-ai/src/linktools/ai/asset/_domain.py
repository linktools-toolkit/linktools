#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asset file identities and metadata."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from ..core import ImmutableJsonMapping, JsonValue, validate_asset_kind
from ..errors import AIError, ErrorCode
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
    """Stable reference to one immutable Asset version on a storage layer."""

    key: AssetKey
    layer_id: str
    revision: StorageEntryRevision
    digest: str
    size: int

    def __post_init__(self) -> None:
        if not isinstance(self.key, AssetKey):
            raise TypeError("asset version key must be AssetKey")
        if not isinstance(self.layer_id, str) or not self.layer_id:
            raise ValueError("asset version layer_id must be non-empty")
        if not isinstance(self.revision, StorageEntryRevision):
            raise TypeError("asset version revision must be StorageEntryRevision")
        if (
            not isinstance(self.digest, str)
            or len(self.digest) != 64
            or any(character not in "0123456789abcdef" for character in self.digest)
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
            "layer_id": self.layer_id,
            "revision": self.revision.value,
            "digest": self.digest,
            "size": self.size,
        }

    def verify_content(self, data: bytes) -> None:
        """Validate bytes against this version's existing integrity contract."""
        if not isinstance(data, bytes):
            raise TypeError("asset version content must be bytes")
        if len(data) != self.size or hashlib.sha256(data).hexdigest() != self.digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def matches_info(self, info: "AssetInfo") -> bool:
        """Check the captured key, entry revision, and byte integrity fields."""
        return (
            self.key == info.key
            and self.revision == info.revision
            and self.digest == info.digest
            and self.size == info.size
        )

    @classmethod
    def from_payload(cls, value: object) -> "AssetVersionRef":
        if not isinstance(value, Mapping) or set(value) != {
            "version",
            "kind",
            "id",
            "layer_id",
            "revision",
            "digest",
            "size",
        }:
            raise ValueError("asset version payload is invalid")
        version = value.get("version")
        if isinstance(version, bool) or version != 1:
            raise ValueError("asset version payload version is unsupported")
        kind = value.get("kind")
        identity = value.get("id")
        layer_id = value.get("layer_id")
        revision = value.get("revision")
        digest = value.get("digest")
        size = value.get("size")
        if (
            not isinstance(kind, str)
            or not isinstance(identity, str)
            or not isinstance(layer_id, str)
            or isinstance(revision, bool)
            or not isinstance(revision, int)
            or not isinstance(digest, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
        ):
            raise ValueError("asset version payload is invalid")
        return cls(
            AssetKey(kind, identity),
            layer_id,
            StorageEntryRevision(revision),
            digest,
            size,
        )


@dataclass(frozen=True, slots=True)
class AssetRoot:
    scheme: "Literal['file', 'sql', 'memory']"
    locator: str

    def __post_init__(self) -> None:
        if self.scheme not in {"file", "sql", "memory"} or not self.locator:
            raise ValueError("asset root is incomplete")


@dataclass(frozen=True, slots=True)
class AssetInfo:
    key: AssetKey
    revision: StorageEntryRevision
    store_revision: StorageRevision
    digest: str
    size: int
    status: StorageEntryStatus
    modified_at: datetime
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    content: "StoredPayload | None" = None

    def __post_init__(self) -> None:
        if (
            self.size < 0
            or not isinstance(self.status, StorageEntryStatus)
            or len(self.digest) != 64
            or any(character not in "0123456789abcdef" for character in self.digest)
        ):
            raise ValueError("asset metadata is invalid")
        if self.modified_at.tzinfo is None:
            raise ValueError("asset metadata requires a timezone-aware timestamp")
        object.__setattr__(self, "metadata", ImmutableJsonMapping(normalize_storage_metadata(self.metadata)))
        if self.status is not StorageEntryStatus.NORMAL and (
            self.size != 0 or self.digest != hashlib.sha256(b"").hexdigest()
        ):
            raise ValueError("non-normal asset metadata must not contain file content")
        if self.content is not None and (
            self.content.digest != self.digest or self.content.size != self.size
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
