#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FileBackend: local-filesystem ResourceBackend, readonly or readwrite.

`path` ("/{namespace}/{rest}") maps directly onto `root / {namespace} / {rest}` --
the leading slash is stripped and the remainder used as a relative filesystem path.
No version history and no idempotency tracking: every write always returns
version=1, and put()/move() always write through regardless of whether the content
already matches what's on disk. Callers that need real versioning or idempotent
writes should use DatabaseBackend instead.
"""

import shutil
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path

from .protocols import DeleteOp, MoveOp, Operation, PutOp, ResourceBackend, ResourceFile

_VERSION = 1


class FileBackend(ResourceBackend):
    def __init__(self, root: Path, readonly: bool = False) -> None:
        self.root = root
        self.readonly = readonly

    def _fs_path(self, path: str) -> Path:
        return self.root / path.lstrip("/")

    def _require_writable(self) -> None:
        if self.readonly:
            raise PermissionError("FileBackend is readonly")

    async def get(self, path: str, version: "int | None" = None) -> "ResourceFile | None":
        fs_path = self._fs_path(path)
        if not fs_path.is_file():
            return None
        # FileBackend keeps no version history -- only version=1 is ever
        # retrievable; a caller asking for any other version simply gets None.
        if version is not None and version != _VERSION:
            return None
        return ResourceFile(path=path, content=fs_path.read_text(encoding="utf-8"), version=_VERSION)

    async def list(self, *, pattern: "str | None" = None, since: "datetime | None" = None) -> "list[ResourceFile]":
        # FileBackend has no timestamp index beyond mtime -- `since` never filters
        # anything out here; only `pattern` narrows the full scan.
        results: "list[ResourceFile]" = []
        if not self.root.exists():
            return results
        for fs_file in self.root.rglob("*"):
            if not fs_file.is_file():
                continue
            rel_path = "/" + str(fs_file.relative_to(self.root)).replace("\\", "/")
            if pattern is not None and not fnmatch(rel_path, pattern):
                continue
            resource = await self.get(rel_path)
            if resource is not None:
                results.append(resource)
        return results

    async def put(self, path: str, content: str, *, updated_by: str = "") -> ResourceFile:
        self._require_writable()
        fs_path = self._fs_path(path)
        fs_path.parent.mkdir(parents=True, exist_ok=True)
        fs_path.write_text(content, encoding="utf-8")
        return ResourceFile(path=path, content=content, version=_VERSION)

    async def delete(self, path: str, *, updated_by: str = "") -> bool:
        self._require_writable()
        fs_path = self._fs_path(path)
        if not fs_path.is_file():
            return False
        fs_path.unlink()
        return True

    async def move(self, src_path: str, dst_path: str, *, updated_by: str = "") -> "ResourceFile | None":
        self._require_writable()
        src_fs = self._fs_path(src_path)
        dst_fs = self._fs_path(dst_path)
        if dst_fs.is_file() and not src_fs.is_file():
            # Already applied -- idempotent retry.
            return await self.get(dst_path)
        if not src_fs.is_file():
            return None
        content = src_fs.read_text(encoding="utf-8")
        dst_fs.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_fs), str(dst_fs))
        return ResourceFile(path=dst_path, content=content, version=_VERSION)

    async def apply_batch(self, ops: "list[Operation]", *, updated_by: str = "") -> "list[ResourceFile]":
        self._require_writable()
        applied: "list[tuple[str, Operation]]" = []
        results: "dict[str, ResourceFile]" = {}
        try:
            for op in ops:
                if isinstance(op, PutOp):
                    results[op.path] = await self.put(op.path, op.content, updated_by=updated_by)
                    applied.append(("put", op))
                elif isinstance(op, DeleteOp):
                    await self.delete(op.path, updated_by=updated_by)
                    results.pop(op.path, None)
                    applied.append(("delete", op))
                elif isinstance(op, MoveOp):
                    moved = await self.move(op.src_path, op.dst_path, updated_by=updated_by)
                    results.pop(op.src_path, None)
                    if moved is not None:
                        results[op.dst_path] = moved
                    applied.append(("move", op))
            return list(results.values())
        except Exception:
            # Best-effort rollback: true multi-file atomicity isn't achievable on a
            # plain filesystem without journaling, so this undoes already-applied
            # ops in reverse order rather than leaving a half-applied batch.
            for kind, op in reversed(applied):
                if kind == "put":
                    await self.delete(op.path, updated_by=updated_by)
                elif kind == "move":
                    await self.move(op.dst_path, op.src_path, updated_by=updated_by)
            raise

    async def revision(self) -> int:
        return 0
