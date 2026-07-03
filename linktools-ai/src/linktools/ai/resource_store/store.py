#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ResourceStore: ChainMap-style multi-backend composition over ResourceBackend.

Single-key reads (get/get_at_version) use first-non-None-wins, exactly like a
ChainMap lookup. Listing reads (propfind/get_by_name) union across every backend
keyed by path, with an earlier backend's entry shadowing the same path in a later
one -- this mirrors ChainMap's own `.keys()`/iteration semantics (every map's keys
are visible; `[key]` picks the first map that has it), applied to a multi-item view.
Writes (put/delete/move/apply_batch) always target backends[0] only -- never fall
through to a fallback backend."""

from typing import TYPE_CHECKING

from .protocols import DeleteOp, MoveOp, Operation, PutOp, ResourceFile

if TYPE_CHECKING:
    from .protocols import ResourceBackend


class ResourceStore:
    def __init__(self, backends: "list[ResourceBackend]") -> None:
        if not backends:
            raise ValueError("ResourceStore requires at least one backend")
        self._backends = backends

    async def get(self, path: str) -> "ResourceFile | None":
        for backend in self._backends:
            result = await backend.get(path)
            if result is not None:
                return result
        return None

    async def get_at_version(self, path: str, version: int) -> "ResourceFile | None":
        for backend in self._backends:
            result = await backend.get_at_version(path, version)
            if result is not None:
                return result
        return None

    async def propfind(self, path: str) -> "list[ResourceFile]":
        return await self._union_listing(lambda backend: backend.propfind(path))

    async def get_by_name(self, namespace: str, name: str) -> "list[ResourceFile]":
        return await self._union_listing(lambda backend: backend.get_by_name(namespace, name))

    async def _union_listing(self, list_call) -> "list[ResourceFile]":
        by_path: "dict[str, ResourceFile]" = {}
        # Iterate backends in reverse so an earlier backend's entry, written last,
        # ends up shadowing a later backend's entry for the same path.
        for backend in reversed(self._backends):
            for resource in await list_call(backend):
                by_path[resource.path] = resource
        return list(by_path.values())

    async def put(self, path: str, content: str, *, updated_by: str = "engine") -> ResourceFile:
        return await self._backends[0].put(path, content, updated_by=updated_by)

    async def delete(self, path: str, *, updated_by: str = "engine") -> bool:
        return await self._backends[0].delete(path, updated_by=updated_by)

    async def move(self, src_path: str, dst_path: str, *, updated_by: str = "engine") -> "ResourceFile | None":
        return await self._backends[0].move(src_path, dst_path, updated_by=updated_by)

    async def apply_batch(self, ops: "list[Operation]", *, updated_by: str = "engine") -> "list[ResourceFile]":
        return await self._backends[0].apply_batch(ops, updated_by=updated_by)

    async def put_many(self, ops: "list[PutOp]", *, updated_by: str = "engine") -> "list[ResourceFile]":
        return await self.apply_batch(list(ops), updated_by=updated_by)

    async def delete_many(self, paths: "list[str]", *, updated_by: str = "engine") -> None:
        await self.apply_batch([DeleteOp(path=p) for p in paths], updated_by=updated_by)

    async def move_many(self, moves: "list[tuple[str, str]]", *, updated_by: str = "engine") -> "list[ResourceFile]":
        ops = [MoveOp(src_path=src, dst_path=dst) for src, dst in moves]
        return await self.apply_batch(ops, updated_by=updated_by)

    async def get_revision(self) -> int:
        return await self._backends[0].get_revision()

    async def refresh(self) -> None:
        """Pass-through to the primary backend's own internal resync, if it has one.

        ResourceStore holds no cache of its own to invalidate -- every backend is
        responsible for its own freshness (DatabaseBackend checks get_revision()
        internally on every read; InMemoryResourceBackend/FileBackend have no
        staleness concept at all). This just gives callers a single place to nudge
        that check explicitly rather than waiting for the next read.
        """
        await self._backends[0].get_revision()
