#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ResourceBackend: the narrow, "dumb" raw-CRUD contract every backend implements.
All idempotency-key and conditional-write decision logic lives in ResourceStore, not
here -- backends only store/retrieve idempotency records verbatim via get_idempotency/
put_idempotency."""

from typing import Mapping, Protocol, runtime_checkable

from .models import Depth, IdempotencyRecord, Resource, ResourceInfo, ResourcePage, WriteOptions
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

    async def revision(self) -> int:
        ...

    async def get_idempotency(self, key: str) -> "IdempotencyRecord | None":
        ...

    async def put_idempotency(self, record: IdempotencyRecord) -> None:
        ...

    # -- OPTIONAL atomic checked operations (spec section 16, TOCTOU fix) --
    # These fold precondition-check + idempotency-reservation + mutate into ONE
    # backend call so the three steps cannot be interleaved by a concurrent
    # writer (the TOCTOU race the split ResourceStore orchestration has). A
    # backend that has a real transaction primitive (SqlAlchemy) implements them
    # to run all three steps inside a single transaction. The File backend
    # implements them under an in-process lock (the best a non-transactional
    # filesystem can do). The Memory backend does NOT implement them:
    # ResourceStore probes with hasattr() and, when absent, falls back to the
    # legacy 3-step orchestration -- so Memory keeps today's behavior unchanged.
    # `request_hash` is computed by ResourceStore (which owns the hash recipe)
    # and passed in so the backend can do the idempotency comparison atomically.

    async def raw_put_checked(
        self,
        path: ResourcePath,
        content: bytes,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> Resource:
        ...

    async def raw_delete_checked(
        self,
        path: ResourcePath,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> None:
        ...
