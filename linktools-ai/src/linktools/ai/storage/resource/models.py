#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resource domain models: ResourceKind/ResourceInfo/Resource, the three-state
ResourceLookup (Found/Missing/Masked), paging, write options, and idempotency records."""

import json as _json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, TypeAlias

from .path import ResourcePath


class ResourceKind(str, Enum):
    FILE = "file"
    COLLECTION = "collection"


class Depth(str, Enum):
    ZERO = "0"
    ONE = "1"


@dataclass(frozen=True, slots=True)
class ResourceInfo:
    path: ResourcePath
    kind: ResourceKind
    etag: str
    version: int
    content_type: "str | None"
    size: int
    modified_at: datetime
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Resource:
    info: ResourceInfo
    content: bytes

    def text(self, encoding: str = "utf-8") -> str:
        return self.content.decode(encoding)

    def json(self) -> Any:
        return _json.loads(self.content)

    @classmethod
    def from_text(cls, info: ResourceInfo, text: str, encoding: str = "utf-8") -> "Resource":
        return cls(info=info, content=text.encode(encoding))

    @classmethod
    def from_json(cls, info: ResourceInfo, value: Any) -> "Resource":
        return cls(info=info, content=_json.dumps(value).encode("utf-8"))


@dataclass(frozen=True, slots=True)
class Found:
    resource: Resource


@dataclass(frozen=True, slots=True)
class Missing:
    pass


@dataclass(frozen=True, slots=True)
class Masked:
    path: ResourcePath
    version: int


ResourceLookup: TypeAlias = "Found | Missing | Masked"


@dataclass(frozen=True, slots=True)
class ResourcePage:
    items: "tuple[ResourceInfo, ...]"
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
    result: "ResourceInfo | None"


# ResourceLookupInfo is the metadata-only shape returned by raw_stat.
# It is intentionally an alias of ResourceInfo: ResourceInfo already
# carries exactly the metadata fields (path/kind/etag/version/content_type/
# size/modified_at/metadata) and -- crucially -- no content field. Aliasing
# rather than duplicating keeps a single source of truth for the metadata
# shape; the distinct name documents the "no content loaded" contract at the
# type level.
ResourceLookupInfo: TypeAlias = ResourceInfo

# MoveResult is the result shape of an atomic MOVE. The target
# resource (info + content) is what callers receive from ResourceStore.move();
# aliasing Resource avoids parallel result types.
MoveResult: TypeAlias = Resource
