#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asset file identities and metadata."""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from ..storage import (
    ReadableStorageBackend,
    StorageEntryRevision,
    StorageRevision,
    StorageWriter,
)


@dataclass(frozen=True, slots=True)
class AssetKey:
    kind: str
    id: str

    def __post_init__(self) -> None:
        try:
            kind_size = len(self.kind.encode("utf-8"))
            identifier_size = len(self.id.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise ValueError("asset key is invalid") from error
        if (
            not self.kind
            or not self.id
            or kind_size > 128
            or identifier_size > 512
            or "\x00" in self.kind
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
    deleted: bool
    root_id: str
    root_digest: str
    modified_at: datetime

    def __post_init__(self) -> None:
        if (
            self.size < 0
            or not self.root_id
            or not self.root_digest
            or len(self.etag) != 64
            or any(character not in "0123456789abcdef" for character in self.etag)
        ):
            raise ValueError("asset metadata is invalid")
        if self.modified_at.tzinfo is None:
            raise ValueError("asset metadata requires a timezone-aware timestamp")
        if self.deleted and (self.size != 0 or self.etag != hashlib.sha256(b"").hexdigest()):
            raise ValueError("asset tombstone metadata is invalid")


@runtime_checkable
class AssetBackend(
    ReadableStorageBackend[AssetKey, bytes, AssetInfo],
    StorageWriter[AssetKey, bytes, AssetInfo],
    Protocol,
):
    @property
    def root(self) -> AssetRoot: ...


__all__ = ["AssetBackend", "AssetInfo", "AssetKey", "AssetRoot"]
