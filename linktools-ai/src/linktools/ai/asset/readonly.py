#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ReadOnlyAssetBackend: a read-only view over any AssetReaderBackend.

Read-only-ness is structural, not a runtime flag: this wrapper exposes ONLY the
AssetReaderBackend surface (raw_get / raw_stat / raw_list / revision) and defines
NO write methods and NO idempotency surface. It therefore does not satisfy
AssetWriterBackend and cannot be supplied as an AssetStore primary -- the type
system rejects it, and AssetStore's runtime callable-check rejects it with a
clear error. Compose a writable backend as primary and one or more of these as
overlays."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import AssetLookupInfo, AssetPage, Depth
    from .path import AssetPath


class ReadOnlyAssetBackend:
    """Read-only delegate over an :class:`AssetReaderBackend`. Every read
    forwards to the inner backend; no write method exists on this class. The
    ``backend_id`` is whatever the AssetStore tagged on the inner backend."""

    def __init__(self, inner) -> None:
        self._inner = inner
        # backend_id is settable: the AssetStore overrides it with the canonical
        # primary/overlay tag. Defaults to the inner backend's id so an unwrapped
        # read-only view still reports a stable id.
        self.backend_id = inner.backend_id

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

    async def revision(self) -> str:
        return await self._inner.revision()


__all__: "list[str]" = ["ReadOnlyAssetBackend"]
