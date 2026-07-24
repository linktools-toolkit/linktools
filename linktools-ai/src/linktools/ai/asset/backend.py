#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asset backend Protocols, split by capability.

:class:`AssetReaderBackend` is the read surface every backend implements
(raw_get / raw_stat / raw_list / revision). Overlay backends are readers.
:class:`AssetWriterBackend` extends it with the atomic checked operations
(raw_put_checked / raw_delete_checked / raw_move_checked). The AssetStore
primary is always a Writer; overlays are Readers. Idempotency is NOT part of the
read surface -- it lives inside the Writer's checked operations, so a Reader
backend never has to implement or expose it.

Each backend carries a ``backend_id`` the AssetStore tags with a stable
canonical id (``primary`` / ``overlay:N``) so a multi-backend cursor can name
each backend's contribution.

Splitting the Protocols (and typing the primary as ``AssetWriterBackend``,
overlays as ``tuple[AssetReaderBackend, ...]``) removes the former
``hasattr(backend, "raw_...")`` probing: a backend either is a Writer (and the
checked ops exist by Protocol) or it is not (and it cannot be a primary).
Read-only-ness is structural -- :class:`~linktools.ai.asset.readonly.ReadOnlyAssetBackend`
implements only AssetReaderBackend (no write methods), so it does not satisfy
AssetWriterBackend and cannot be a primary. There is no ``readonly`` flag."""

from typing import Protocol, runtime_checkable

from .models import (
    Depth,
    Asset,
    AssetLookupInfo,
    AssetPage,
    WriteOptions,
)
from .path import AssetPath


@runtime_checkable
class AssetReaderBackend(Protocol):
    """Read surface. Every backend (including read-only overlays) implements
    these. ``backend_id`` is a stable identifier the AssetStore tags so a
    multi-backend listing cursor can attribute each item to its source."""

    backend_id: str

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

    async def revision(self) -> str: ...


@runtime_checkable
class AssetWriterBackend(AssetReaderBackend, Protocol):
    """Write surface. The checked operations fold precondition-check +
    idempotency-reservation + mutate into ONE atomic call so the three steps
    cannot be interleaved by a concurrent writer (the TOCTOU race a split
    orchestration has). A backend that has a real transaction primitive
    (SqlAlchemy) runs all three inside one transaction; File and Memory
    serialize with a process-local lock. ``raw_move_checked`` is a single atomic
    operation (load-source + write-target + whiteout-source + bump-revision);
    it never decomposes into a public put + delete. Idempotency is encapsulated
    INSIDE these checked writes -- there is no separate idempotency method on
    the Writer.

    A read-only backend (``ReadOnlyAssetBackend``) implements only
    :class:`AssetReaderBackend` -- it lacks these write methods, so it does not
    satisfy this Protocol and cannot be supplied as a primary. Read-only-ness is
    structural (which methods exist), not a runtime flag."""

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


__all__: "list[str]" = ["AssetReaderBackend", "AssetWriterBackend"]
