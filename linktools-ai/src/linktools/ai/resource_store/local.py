#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""InMemoryResourceBackend: single in-memory reference ResourceBackend, for tests.

Merges what used to be two separate reference doubles (InMemoryCapabilityRepository +
InMemoryCapabilityCache) into one backend -- ResourceBackend no longer separates
"repository" from "cache" (see resource_store/database.py's DatabaseBackend). Not for
production use: no persistence, no concurrency control.
"""

from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from typing import Any

from .protocols import DeleteOp, MoveOp, Operation, PutOp, ResourceFile


@dataclass
class InMemoryResourceBackend:
    _rows: "dict[str, dict[str, Any]]" = field(default_factory=dict)
    _history: "dict[str, dict[int, dict[str, Any]]]" = field(default_factory=dict)
    _revision: int = 0

    def _checksum(self, content: str) -> str:
        return sha256(content.encode("utf-8")).hexdigest()

    def _to_resource_file(self, path: str, row: "dict[str, Any]") -> ResourceFile:
        return ResourceFile(path=path, content=row["content"], version=row["version"])

    def _record_version(self, path: str, row: "dict[str, Any]") -> None:
        self._history.setdefault(path, {})[row["version"]] = dict(row)

    async def propfind(self, path: str) -> "list[ResourceFile]":
        return [
            self._to_resource_file(p, row)
            for p, row in self._rows.items()
            if p.startswith(path) and row["status"] == "active"
        ]

    async def get(self, path: str) -> "ResourceFile | None":
        row = self._rows.get(path)
        if row is None or row["status"] != "active":
            return None
        return self._to_resource_file(path, row)

    async def get_at_version(self, path: str, version: int) -> "ResourceFile | None":
        row = self._history.get(path, {}).get(version)
        if row is None:
            return None
        return self._to_resource_file(path, row)

    async def get_by_name(self, namespace: str, name: str) -> "list[ResourceFile]":
        prefix = f"/{namespace}/"
        suffix = f"/{name}"
        return [
            self._to_resource_file(p, row)
            for p, row in self._rows.items()
            if p.startswith(prefix) and p.endswith(suffix) and row["status"] == "active"
        ]

    async def put(self, path: str, content: str, *, updated_by: str = "engine") -> ResourceFile:
        checksum = self._checksum(content)
        existing = self._rows.get(path)
        if existing is not None and existing["status"] == "active" and existing["checksum"] == checksum:
            return self._to_resource_file(path, existing)
        version = (existing["version"] + 1) if existing else 1
        self._rows[path] = {
            "content": content, "checksum": checksum, "version": version,
            "status": "active", "updated_by": updated_by, "updated_at": datetime.now(),
        }
        self._record_version(path, self._rows[path])
        self._revision += 1
        return self._to_resource_file(path, self._rows[path])

    async def delete(self, path: str, *, updated_by: str = "engine") -> bool:
        existing = self._rows.get(path)
        if existing is None or existing["status"] != "active":
            return False
        existing["status"] = "deleted"
        existing["updated_by"] = updated_by
        existing["version"] += 1
        existing["updated_at"] = datetime.now()
        self._record_version(path, existing)
        self._revision += 1
        return True

    async def move(self, src_path: str, dst_path: str, *, updated_by: str = "engine") -> "ResourceFile | None":
        dst_existing = self._rows.get(dst_path)
        src_existing = self._rows.get(src_path)
        src_active = src_existing is not None and src_existing["status"] == "active"
        dst_active = dst_existing is not None and dst_existing["status"] == "active"
        if dst_active and not src_active:
            # Already applied -- idempotent retry.
            return self._to_resource_file(dst_path, dst_existing)
        if not src_active:
            return None
        new_version = src_existing["version"] + 1
        self._rows[dst_path] = {
            "content": src_existing["content"], "checksum": src_existing["checksum"],
            "version": new_version, "status": "active", "updated_by": updated_by,
            "updated_at": datetime.now(),
        }
        src_existing["status"] = "deleted"
        src_existing["updated_by"] = updated_by
        src_existing["updated_at"] = datetime.now()
        self._record_version(dst_path, self._rows[dst_path])
        self._revision += 1
        return self._to_resource_file(dst_path, self._rows[dst_path])

    async def list_since(self, since: "datetime | None") -> "list[ResourceFile]":
        if since is None:
            return [self._to_resource_file(p, row) for p, row in self._rows.items()]
        return [
            self._to_resource_file(p, row)
            for p, row in self._rows.items()
            if row["updated_at"] >= since
        ]

    async def apply_batch(self, ops: "list[Operation]", *, updated_by: str = "engine") -> "list[ResourceFile]":
        results: "dict[str, ResourceFile]" = {}
        for op in ops:
            if isinstance(op, PutOp):
                results[op.path] = await self.put(op.path, op.content, updated_by=updated_by)
            elif isinstance(op, DeleteOp):
                await self.delete(op.path, updated_by=updated_by)
                results.pop(op.path, None)
            elif isinstance(op, MoveOp):
                moved = await self.move(op.src_path, op.dst_path, updated_by=updated_by)
                results.pop(op.src_path, None)
                if moved is not None:
                    results[op.dst_path] = moved
        return list(results.values())

    async def get_revision(self) -> int:
        return self._revision
