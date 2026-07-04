#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ResourceBackend: the narrow, "dumb" raw-CRUD contract every backend implements.
All idempotency-key and conditional-write decision logic lives in ResourceStore, not
here -- backends only store/retrieve idempotency records verbatim via get_idempotency/
put_idempotency."""

from typing import Mapping, Protocol, runtime_checkable

from .models import Depth, IdempotencyRecord, ResourceInfo, ResourcePage
from .path import ResourcePath


@runtime_checkable
class ResourceBackend(Protocol):
    readonly: bool

    async def raw_get(self, path: ResourcePath, *, include_content: bool = True):
        ...

    async def raw_propfind(
        self,
        path: ResourcePath,
        *,
        depth: Depth,
        limit: int,
        cursor: "str | None",
    ) -> ResourcePage:
        ...

    async def raw_put(
        self,
        path: ResourcePath,
        content: bytes,
        *,
        content_type: "str | None",
        metadata: "Mapping[str, object]",
    ) -> ResourceInfo:
        ...

    async def raw_delete(self, path: ResourcePath) -> "ResourceInfo | None":
        ...

    async def raw_move(self, src: ResourcePath, dst: ResourcePath) -> ResourceInfo:
        ...

    async def revision(self) -> int:
        ...

    async def get_idempotency(self, key: str) -> "IdempotencyRecord | None":
        ...

    async def put_idempotency(self, record: IdempotencyRecord) -> None:
        ...
