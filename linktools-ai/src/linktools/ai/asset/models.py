#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asset domain models: AssetKind/AssetInfo/Asset, the three-state
AssetLookup (Found/Missing/Masked), paging, write options, and idempotency records."""

import json as _json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, TypeAlias

from .path import AssetPath


class AssetKind(str, Enum):
    FILE = "file"
    COLLECTION = "collection"
    DIRECTORY = "directory"


class Depth(str, Enum):
    ZERO = "0"
    ONE = "1"
    INFINITY = "infinity"


@dataclass(frozen=True, slots=True)
class AssetInfo:
    path: AssetPath
    kind: AssetKind
    etag: str
    version: int
    content_type: "str | None"
    size: int
    modified_at: datetime
    metadata: "Mapping[str, Any]" = field(default_factory=dict)
    # True for the synthesized namespace-root directory (AssetPath("/")): no
    # backend stores a root record, so the Store fabricates this AssetInfo
    # rather than reading it from any backend.
    synthetic: bool = False


@dataclass(frozen=True, slots=True)
class Asset:
    info: AssetInfo
    content: bytes

    def text(self, encoding: str = "utf-8") -> str:
        return self.content.decode(encoding)

    def json(self) -> Any:
        return _json.loads(self.content)

    @classmethod
    def from_text(
        cls, info: AssetInfo, text: str, encoding: str = "utf-8"
    ) -> "Asset":
        return cls(info=info, content=text.encode(encoding))

    @classmethod
    def from_json(cls, info: AssetInfo, value: Any) -> "Asset":
        return cls(info=info, content=_json.dumps(value).encode("utf-8"))


@dataclass(frozen=True, slots=True)
class Found:
    asset: Asset


@dataclass(frozen=True, slots=True)
class Missing:
    pass


@dataclass(frozen=True, slots=True)
class Masked:
    path: AssetPath
    version: int


AssetLookup: TypeAlias = "Found | Missing | Masked"


@dataclass(frozen=True, slots=True)
class AssetPage:
    items: "tuple[AssetInfo, ...]"
    cursor: "str | None"


@dataclass(frozen=True, slots=True)
class WriteOptions:
    idempotency_key: "str | None" = None
    if_match: "str | None" = None
    if_none_match: bool = False
    content_type: "str | None" = None
    metadata: "Mapping[str, Any]" = field(default_factory=dict)
    actor: "str | None" = None


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    key: str
    request_hash: str
    result: "AssetInfo | None"


# AssetLookupInfo is the metadata-only shape returned by raw_stat.
# It is intentionally an alias of AssetInfo: AssetInfo already
# carries exactly the metadata fields (path/kind/etag/version/content_type/
# size/modified_at/metadata) and -- crucially -- no content field. Aliasing
# rather than duplicating keeps a single source of truth for the metadata
# shape; the distinct name documents the "no content loaded" contract at the
# type level.
AssetLookupInfo: TypeAlias = AssetInfo

# MoveResult is the result shape of an atomic MOVE. The target
# asset (info + content) is what callers receive from AssetStore.move();
# aliasing Asset avoids parallel result types.
MoveResult: TypeAlias = Asset
