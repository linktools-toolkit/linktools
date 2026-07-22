#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asset backend Protocols, split by capability.

:class:`AssetReaderBackend` is the read surface every backend implements
(raw_get / raw_stat / raw_list / revision / get_idempotency). Overlay backends
are readers. :class:`AssetWriterBackend` extends it with the atomic checked
operations (raw_put_checked / raw_delete_checked / raw_move / put_idempotency)
and declares ``readonly: Literal[False]``. The AssetStore primary is always a
Writer; overlays are Readers.

Splitting the Protocols (and typing the primary as ``AssetWriterBackend``,
overlays as ``tuple[AssetReaderBackend, ...]``) removes the former
``hasattr(backend, "raw_...")`` probing: a backend either is a Writer (and the
checked ops exist by Protocol) or it is not (and it cannot be a primary).
Read-only backends simply do not implement the Writer Protocol -- ``readonly``
is a declared marker, not a runtime-guessable flag."""

from typing import Literal, Protocol, runtime_checkable

from .models import (
    Depth,
    IdempotencyRecord,
    Asset,
    AssetInfo,
    AssetLookupInfo,
    AssetPage,
    WriteOptions,
)
from .path import AssetPath


@runtime_checkable
class AssetReaderBackend(Protocol):
    """Read surface. Every backend (including read-only overlays) implements
    these."""

    async def raw_get(self, path: AssetPath, *, include_content: bool = True): ...

    async def raw_stat(self, path: AssetPath) -> "AssetLookupInfo | None": ...

    async def raw_list(
        self,
        path: AssetPath,
        *,
        depth: Depth,
        limit: int,
        cursor: "str | None",
    ) -> AssetPage: ...

    async def revision(self) -> int: ...

    async def get_idempotency(self, key: str) -> "IdempotencyRecord | None": ...


@runtime_checkable
class AssetWriterBackend(AssetReaderBackend, Protocol):
    """Write surface. The checked operations fold precondition-check +
    idempotency-reservation + mutate into ONE atomic call so the three steps
    cannot be interleaved by a concurrent writer (the TOCTOU race a split
    orchestration has). A backend that has a real transaction primitive
    (SqlAlchemy) runs all three inside one transaction; File and Memory
    serialize with a process-local lock. ``raw_move`` is a single atomic
    operation (load-source + write-target + whiteout-source + bump-revision);
    it never decomposes into a public put + delete.

    ``readonly`` is ``Literal[False]``: a read-only backend does not implement
    this Protocol at all, so it cannot be supplied as a primary."""

    readonly: Literal[False]

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

    async def raw_move_checked(
        self,
        source: AssetPath,
        target: AssetPath,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> Asset: ...

    async def put_idempotency(self, record: IdempotencyRecord) -> None: ...


__all__: "list[str]" = ["AssetReaderBackend", "AssetWriterBackend"]
