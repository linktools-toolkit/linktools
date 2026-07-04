#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MemoryResourceBackend: a dependency-free ResourceBackend used to exercise the
Protocol contract and as one of the parametrized backends in the ResourceStore
contract-test suite (Task 6)."""

import hashlib
from datetime import datetime, timezone
from typing import Mapping

from .models import Depth, Found, IdempotencyRecord, Masked, Missing, Resource, ResourceInfo, ResourceKind, ResourcePage
from .path import ResourcePath


class MemoryResourceBackend:
    def __init__(self, *, readonly: bool = False) -> None:
        self.readonly = readonly
        self._entries: "dict[str, tuple[bytes, ResourceInfo]]" = {}
        self._whiteouts: "dict[str, int]" = {}
        self._idempotency: "dict[str, IdempotencyRecord]" = {}
        self._revision = 0

    async def raw_get(self, path: ResourcePath, *, include_content: bool = True):
        key = path.value
        if key in self._entries:
            content, info = self._entries[key]
            if not include_content:
                return Found(resource=Resource(info=info, content=b""))
            return Found(resource=Resource(info=info, content=content))
        if key in self._whiteouts:
            return Masked(path=path, version=self._whiteouts[key])
        return Missing()

    async def raw_propfind(self, path: ResourcePath, *, depth: Depth, limit: int, cursor: "str | None") -> ResourcePage:
        prefix = path.value.rstrip("/") + "/"
        items = []
        for key, (_content, info) in sorted(self._entries.items()):
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix):]
            if depth == Depth.ONE and "/" in rest:
                continue
            items.append(info)
        return ResourcePage(items=tuple(items[:limit]), cursor=None)

    async def raw_put(self, path: ResourcePath, content: bytes, *, content_type: "str | None", metadata: "Mapping[str, object]"):
        key = path.value
        self._revision += 1
        prior_info = self._entries.get(key, (b"", None))[1]
        version = (prior_info.version if prior_info else 0) + 1
        info = ResourceInfo(
            path=path,
            kind=ResourceKind.FILE,
            etag=hashlib.sha256(content).hexdigest(),
            version=version,
            content_type=content_type,
            size=len(content),
            modified_at=datetime.now(timezone.utc),
            metadata=dict(metadata),
        )
        self._entries[key] = (content, info)
        self._whiteouts.pop(key, None)
        return info

    async def raw_delete(self, path: ResourcePath) -> "ResourceInfo | None":
        key = path.value
        self._revision += 1
        removed = self._entries.pop(key, None)
        prior_version = removed[1].version if removed else self._whiteouts.get(key, 0)
        self._whiteouts[key] = prior_version + 1
        return removed[1] if removed else None

    async def raw_move(self, src: ResourcePath, dst: ResourcePath) -> ResourceInfo:
        lookup = await self.raw_get(src)
        if not isinstance(lookup, Found):
            raise FileNotFoundError(f"cannot move missing resource: {src}")
        info = await self.raw_put(
            dst,
            lookup.resource.content,
            content_type=lookup.resource.info.content_type,
            metadata=lookup.resource.info.metadata,
        )
        await self.raw_delete(src)
        return info

    async def revision(self) -> int:
        return self._revision

    async def get_idempotency(self, key: str) -> "IdempotencyRecord | None":
        return self._idempotency.get(key)

    async def put_idempotency(self, record: IdempotencyRecord) -> None:
        self._idempotency[record.key] = record
