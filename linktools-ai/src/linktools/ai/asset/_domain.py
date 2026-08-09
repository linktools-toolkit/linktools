#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asset container, file-entry and immutable history contracts."""

import hashlib
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, Literal, Protocol, TypeVar, runtime_checkable

from ..errors import AIError, ErrorCode

TAsset = TypeVar("TAsset", bound="AssetValue")
AssetEntryOrigin = Literal["SOURCE", "OVERRIDE", "TOMBSTONE"]
AssetDeleteOrigin = Literal["SOURCE", "OVERRIDE"]
_EMPTY_ETAG = hashlib.sha256(b"").hexdigest()


@runtime_checkable
class AssetValue(Protocol):
    @property
    def asset_kind(self) -> str: ...

    @property
    def asset_id(self) -> str: ...


def validate_rel_path(rel_path: str) -> str:
    if (
        not isinstance(rel_path, str)
        or not rel_path
        or rel_path.startswith("/")
        or rel_path.endswith("/")
        or "\\" in rel_path
        or "\x00" in rel_path
        or "//" in rel_path
        or any(unicodedata.category(character) == "Cc" for character in rel_path)
        or any(part in {"", ".", ".."} for part in rel_path.split("/"))
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "asset entry path is invalid")
    return rel_path


@dataclass(frozen=True, slots=True)
class AssetKey:
    kind: str
    id: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.kind, str)
            or not isinstance(self.id, str)
            or not self.kind
            or not self.id
            or "\x00" in self.kind
            or "\x00" in self.id
            or len(self.kind.encode("utf-8")) > 512
            or len(self.id.encode("utf-8")) > 512
        ):
            raise ValueError("asset key is invalid")


@dataclass(frozen=True, slots=True)
class AssetEntryKey:
    asset: AssetKey
    rel_path: str

    def __post_init__(self) -> None:
        validate_rel_path(self.rel_path)

    @property
    def file_id(self) -> str:
        from ..core import canonical_sha256

        return canonical_sha256({"kind": self.asset.kind, "id": self.asset.id, "rel_path": self.rel_path})


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
        if not isinstance(self.value, int) or isinstance(self.value, bool) or self.value < 1:
            raise ValueError("asset revision must be positive")


@dataclass(frozen=True, slots=True)
class AssetEntryRevision:
    value: int

    def __post_init__(self) -> None:
        if not isinstance(self.value, int) or isinstance(self.value, bool) or self.value < 1:
            raise ValueError("asset entry revision must be positive")


@dataclass(frozen=True, slots=True)
class AssetStoreRevision:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("asset store revision must not be empty")


@dataclass(frozen=True, slots=True)
class AssetInfo:
    key: AssetKey
    revision: AssetRevision
    store_revision: AssetStoreRevision
    etag: str
    size: int
    file_count: int
    deleted: bool
    deleted_by: "AssetDeleteOrigin | None"
    composition_digest: str
    source_layers: "tuple[str, ...]"
    modified_at: datetime

    def __post_init__(self) -> None:
        if self.size < 0 or self.file_count < 0 or len(self.etag) != 64 or any(character not in "0123456789abcdef" for character in self.etag):
            raise ValueError("asset metadata is invalid")
        if self.modified_at.tzinfo is None:
            raise ValueError("asset metadata requires a timezone-aware timestamp")
        if self.deleted and (self.size != 0 or self.file_count != 0 or self.etag != _EMPTY_ETAG or self.deleted_by is None):
            raise ValueError("asset tombstone metadata is invalid")
        if not self.deleted and self.deleted_by is not None:
            raise ValueError("active asset cannot have a delete origin")


@dataclass(frozen=True, slots=True)
class AssetVersion:
    revision: AssetRevision
    etag: str
    size: int
    file_count: int
    composition_digest: str
    modified_at: datetime
    deleted: bool = False
    deleted_by: "AssetDeleteOrigin | None" = None


@dataclass(frozen=True, slots=True)
class AssetEntryInfo:
    key: AssetEntryKey
    file_id: str
    entry_revision: AssetEntryRevision
    asset_revision: AssetRevision
    store_revision: AssetStoreRevision
    etag: str
    size: int
    deleted: bool
    origin: AssetEntryOrigin
    source_digest: "str | None"
    layer: str
    writable: bool
    modified_at: datetime


@dataclass(frozen=True, slots=True)
class AssetEntryVersion:
    entry_revision: AssetEntryRevision
    etag: str
    size: int
    modified_at: datetime
    deleted: bool
    origin: AssetEntryOrigin
    source_digest: "str | None"


@dataclass(frozen=True, slots=True)
class AssetEntrySnapshot:
    key: AssetEntryKey
    entry_revision: AssetEntryRevision
    etag: str
    deleted: bool
    origin: AssetEntryOrigin
    source_digest: "str | None"
    content: "bytes | None"


@dataclass(frozen=True, slots=True)
class AssetRequest(Generic[TAsset]):
    key: AssetKey
    expected: "type[TAsset]"


@dataclass(frozen=True, slots=True)
class AssetChange:
    operation: "Literal['PUT', 'DELETE']"
    key: AssetKey
    value: "AssetValue | None"
    expected_revision: "AssetRevision | None" = None


@dataclass(frozen=True, slots=True)
class AssetDeleteResult:
    key: AssetKey
    deleted: bool
    revision: "AssetRevision | None"
    store_revision: AssetStoreRevision


@dataclass(frozen=True, slots=True)
class AssetBatchResult:
    store_revision: AssetStoreRevision
    atomic: bool
    results: "tuple[AssetInfo | AssetDeleteResult, ...]"


@dataclass(frozen=True, slots=True)
class AssetEntryChange:
    operation: "Literal['PUT', 'DELETE']"
    rel_path: str
    value: "bytes | None"
    expected_entry_revision: "AssetEntryRevision | None" = None

    def __post_init__(self) -> None:
        validate_rel_path(self.rel_path)
        if self.operation == "PUT" and self.value is None:
            raise ValueError("PUT changes require a value")
        if self.operation == "DELETE" and self.value is not None:
            raise ValueError("DELETE changes cannot contain a value")


@dataclass(frozen=True, slots=True)
class AssetEntryDeleteResult:
    key: AssetEntryKey
    deleted: bool
    entry_revision: "AssetEntryRevision | None"
    asset_revision: AssetRevision
    store_revision: AssetStoreRevision


@dataclass(frozen=True, slots=True)
class AssetEntryBatchResult:
    asset: AssetKey
    revision: AssetRevision
    store_revision: AssetStoreRevision
    atomic: bool
    results: "tuple[AssetEntryInfo | AssetEntryDeleteResult, ...]"


@runtime_checkable
class AssetSource(Protocol):
    async def list_assets(self, kind: str) -> "tuple[AssetKey, ...]": ...

    async def list_files(self, asset: AssetKey) -> "tuple[str, ...]": ...

    async def read_file(self, key: AssetEntryKey) -> bytes: ...

    def identity(self, data: bytes) -> str: ...


def empty_etag() -> str:
    return _EMPTY_ETAG


__all__ = [
    "AssetChange", "AssetDeleteOrigin", "AssetDeleteResult", "AssetEntryBatchResult", "AssetEntryChange",
    "AssetEntryDeleteResult", "AssetEntryInfo", "AssetEntryKey", "AssetEntryOrigin", "AssetEntryRevision",
    "AssetEntrySnapshot", "AssetEntryVersion", "AssetInfo", "AssetKey", "AssetRequest", "AssetRevision",
    "AssetRoot", "AssetSource", "AssetStoreRevision", "AssetValue", "AssetVersion", "TAsset",
    "empty_etag", "validate_rel_path",
]
