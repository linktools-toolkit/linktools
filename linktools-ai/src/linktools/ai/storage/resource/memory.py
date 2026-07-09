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
        # Keyset pagination (spec §15.2): entries iterate in sorted key order;
        # ``key > cursor`` is the resume point. Collect limit+1 so the
        # (limit+1)th path becomes next_cursor. Memory does NOT implement
        # raw_move / raw_stat -- ResourceStore falls back to the legacy path
        # for this backend, exercising the hasattr() probe.
        prefix = path.value.rstrip("/") + "/"
        items = []
        for key, (_content, info) in sorted(self._entries.items()):
            if not key.startswith(prefix):
                continue
            if cursor is not None and key <= cursor:
                continue
            rest = key[len(prefix):]
            if depth == Depth.ONE and "/" in rest:
                continue
            items.append(info)
            if len(items) > limit:
                break
        next_cursor = items[limit].path.value if len(items) > limit else None
        return ResourcePage(items=tuple(items[:limit]), cursor=next_cursor)

    async def raw_put(self, path: ResourcePath, content: bytes, *, content_type: "str | None", metadata: "Mapping[str, object]"):
        key = path.value
        self._revision += 1
        prior_entry_version = self._entries.get(key, (b"", None))[1]
        prior_version = max(
            prior_entry_version.version if prior_entry_version else 0,
            self._whiteouts.get(key, 0),
        )
        version = prior_version + 1
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

    async def revision(self) -> int:
        return self._revision

    async def get_idempotency(self, key: str) -> "IdempotencyRecord | None":
        return self._idempotency.get(key)

    async def put_idempotency(self, record: IdempotencyRecord) -> None:
        self._idempotency[record.key] = record
