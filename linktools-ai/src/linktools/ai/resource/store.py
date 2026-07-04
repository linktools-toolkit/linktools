#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ResourceStore: ChainMap-style multi-backend composition over ResourceBackend.

Single-key reads (get, optionally pinned to a version) use first-non-None-wins,
exactly like a ChainMap lookup. Listing reads (list) union across every backend
keyed by path, with an earlier backend's entry shadowing the same path in a later
one -- this mirrors ChainMap's own `.keys()`/iteration semantics (every map's keys
are visible; `[key]` picks the first map that has it), applied to a multi-item view.
Writes (put/delete/move/apply_batch) always target backends[0] only -- never fall
through to a fallback backend."""

from typing import TYPE_CHECKING

from .protocols import Operation, ResourceFile

if TYPE_CHECKING:
    from datetime import datetime

    from .protocols import ResourceBackend


class ResourceStore:
    def __init__(self, *backends: "ResourceBackend") -> None:
        if not backends:
            raise ValueError("ResourceStore requires at least one backend")
        self._backends = backends

    async def get(self, path: str, version: "int | None" = None) -> "ResourceFile | None":
        for backend in self._backends:
            result = await backend.get(path, version)
            if result is not None:
                return result
        return None

    async def list(self, *, pattern: "str | None" = None, since: "datetime | None" = None) -> "list[ResourceFile]":
        by_path: "dict[str, ResourceFile]" = {}
        # Iterate backends in reverse so an earlier backend's entry, written last,
        # ends up shadowing a later backend's entry for the same path.
        for backend in reversed(self._backends):
            for resource in await backend.list(pattern=pattern, since=since):
                by_path[resource.path] = resource
        return list(by_path.values())

    async def put(self, path: str, content: str, *, updated_by: str = "") -> ResourceFile:
        return await self._backends[0].put(path, content, updated_by=updated_by)

    async def delete(self, path: str, *, updated_by: str = "") -> bool:
        return await self._backends[0].delete(path, updated_by=updated_by)

    async def move(self, src_path: str, dst_path: str, *, updated_by: str = "") -> "ResourceFile | None":
        return await self._backends[0].move(src_path, dst_path, updated_by=updated_by)

    async def apply_batch(self, ops: "list[Operation]", *, updated_by: str = "") -> "list[ResourceFile]":
        return await self._backends[0].apply_batch(ops, updated_by=updated_by)

    async def revision(self) -> int:
        return await self._backends[0].revision()
