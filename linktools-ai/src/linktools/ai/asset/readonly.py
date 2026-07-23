#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ReadOnlyAssetBackend: a read-only view over any AssetReaderBackend.

Read-only-ness is structural, not a runtime flag: this wrapper exposes ONLY the
AssetReaderBackend surface (raw_get / raw_stat / raw_list / revision /
get_idempotency) and defines NO write methods. It therefore does not satisfy
AssetWriterBackend and cannot be supplied as an AssetStore primary -- the type
system rejects it, and AssetStore's runtime callable-check rejects it with a
clear error. Compose a writable backend as primary and one or more of these as
overlays."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import AssetLookupInfo, AssetPage, Depth, IdempotencyRecord
    from .path import AssetPath


class ReadOnlyAssetBackend:
    """Read-only delegate over an :class:`AssetReaderBackend`. Every read
    forwards to the inner backend; no write method exists on this class."""

    def __init__(self, inner) -> None:
        self._inner = inner

    async def raw_get(self, path: "AssetPath", *, include_content: bool = True):
        return await self._inner.raw_get(path, include_content=include_content)

    async def raw_stat(self, path: "AssetPath") -> "AssetLookupInfo | None":
        return await self._inner.raw_stat(path)

    async def raw_list(
        self,
        path: "AssetPath",
        *,
        depth: "Depth",
        limit: int,
        cursor: "str | None",
    ) -> "AssetPage":
        return await self._inner.raw_list(
            path, depth=depth, limit=limit, cursor=cursor
        )

    async def revision(self) -> int:
        return await self._inner.revision()

    async def get_idempotency(self, key: str) -> "IdempotencyRecord | None":
        return await self._inner.get_idempotency(key)


__all__: "list[str]" = ["ReadOnlyAssetBackend"]
