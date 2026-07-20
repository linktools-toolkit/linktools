#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AssetBackend: the narrow, "dumb" raw-CRUD contract every backend implements.
All idempotency-key and conditional-write decision logic lives in AssetStore, not
here -- backends only store/retrieve idempotency records verbatim via get_idempotency/
put_idempotency."""

from typing import Mapping, Protocol, runtime_checkable

from .models import (
    Depth,
    IdempotencyRecord,
    MoveResult,
    Asset,
    AssetInfo,
    AssetLookupInfo,
    AssetPage,
    WriteOptions,
)
from .path import AssetPath


@runtime_checkable
class AssetBackend(Protocol):
    readonly: bool

    async def raw_get(self, path: AssetPath, *, include_content: bool = True): ...

    async def raw_propfind(
        self,
        path: AssetPath,
        *,
        depth: Depth,
        limit: int,
        cursor: "str | None",
    ) -> AssetPage: ...

    async def raw_put(
        self,
        path: AssetPath,
        content: bytes,
        *,
        content_type: "str | None",
        metadata: "Mapping[str, object]",
    ) -> AssetInfo: ...

    async def raw_delete(self, path: AssetPath) -> "AssetInfo | None": ...

    async def revision(self) -> int: ...

    async def get_idempotency(self, key: str) -> "IdempotencyRecord | None": ...

    async def put_idempotency(self, record: IdempotencyRecord) -> None: ...

    # -- OPTIONAL atomic checked operations (TOCTOU fix) --
    # These fold precondition-check + idempotency-reservation + mutate into ONE
    # backend call so the three steps cannot be interleaved by a concurrent
    # writer (the TOCTOU race the split AssetStore orchestration has). A
    # backend that has a real transaction primitive (SqlAlchemy) implements them
    # to run all three steps inside a single transaction. The File backend
    # implements them under an in-process lock (the best a non-transactional
    # filesystem can do). The Memory backend does NOT implement them:
    # AssetStore probes with hasattr() and, when absent, falls back to its
    # own 3-step orchestration.
    # `request_hash` is computed by AssetStore (which owns the hash recipe)
    # and passed in so the backend can do the idempotency comparison atomically.

    async def raw_put_checked(
        self,
        path: AssetPath,
        content: bytes,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> Asset: ...

    async def raw_delete_checked(
        self,
        path: AssetPath,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> None: ...

    # -- OPTIONAL atomic MOVE --
    # MOVE is a single domain operation: the backend must NOT decompose it into
    # a public put() + delete() pair, which would expose intermediate state
    # (target written while source still live, or source gone while target not
    # yet written) and would bump the revision twice instead of once. A backend
    # that has a real transaction primitive (SqlAlchemy) implements raw_move as
    # ONE transaction. The File backend implements it under self._lock with an
    # os.replace for the data file. The Memory backend does NOT implement it:
    # AssetStore probes with hasattr() and, when absent, falls back to its
    # own put+delete orchestration.

    async def raw_move(
        self,
        source: AssetPath,
        target: AssetPath,
        *,
        options: WriteOptions,
    ) -> MoveResult: ...

    # -- OPTIONAL metadata-only stat --
    # Returns the resource metadata (path/version/etag/content_type/metadata/
    # state) WITHOUT loading the content blob. A backend that can select only
    # metadata columns (SqlAlchemy) or read only the sidecar (File) implements
    # this to avoid pulling potentially-large content into memory for callers
    # that only need metadata. AssetStore.stat() delegates when available
    # and otherwise falls back to get() + .info.

    async def raw_stat(self, path: AssetPath) -> "AssetLookupInfo | None": ...
