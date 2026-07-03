#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FileBackend: local-filesystem ResourceBackend, readonly or readwrite.

`path` ("/{namespace}/{rest}") maps directly onto `root / {namespace} / {rest}` --
the leading slash is stripped and the remainder used as a relative filesystem path.
Version tracking is content-hash-based only (no separate version-history storage):
each write recomputes a checksum and bumps an in-process version counter kept
alongside the file as a ".meta.json" sidecar, mirroring the checksum-validated
sidecar scheme the old registry_store used for its disk L2 cache.
"""

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

from .protocols import DeleteOp, MoveOp, Operation, PutOp, ResourceFile


def _checksum(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class FileBackend:
    def __init__(self, root: Path, readonly: bool = False) -> None:
        self.root = root
        self.readonly = readonly

    def _fs_path(self, path: str) -> Path:
        return self.root / path.lstrip("/")

    def _meta_path(self, fs_path: Path) -> Path:
        return fs_path.with_suffix(fs_path.suffix + ".meta.json")

    def _read_meta(self, fs_path: Path) -> "dict | None":
        meta_path = self._meta_path(fs_path)
        if not meta_path.exists():
            return None
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def _write_meta(self, fs_path: Path, *, version: int, checksum: str) -> None:
        meta_path = self._meta_path(fs_path)
        meta_path.write_text(
            json.dumps({"version": version, "checksum": checksum, "updated_at": datetime.now().isoformat()}),
            encoding="utf-8",
        )

    def _require_writable(self) -> None:
        if self.readonly:
            raise PermissionError("FileBackend is readonly")

    async def propfind(self, path: str) -> "list[ResourceFile]":
        base = self._fs_path(path)
        if not base.exists():
            return []
        results: "list[ResourceFile]" = []
        for fs_file in base.rglob("*"):
            if not fs_file.is_file() or fs_file.name.endswith(".meta.json"):
                continue
            rel_path = "/" + str(fs_file.relative_to(self.root)).replace("\\", "/")
            resource = await self.get(rel_path)
            if resource is not None:
                results.append(resource)
        return results

    async def get(self, path: str) -> "ResourceFile | None":
        fs_path = self._fs_path(path)
        if not fs_path.is_file():
            return None
        meta = self._read_meta(fs_path)
        version = meta["version"] if meta else 1
        return ResourceFile(path=path, content=fs_path.read_text(encoding="utf-8"), version=version)

    async def get_at_version(self, path: str, version: int) -> "ResourceFile | None":
        # FileBackend keeps no version history -- only the current version is ever
        # retrievable; a caller asking for an old version simply gets None.
        current = await self.get(path)
        if current is not None and current.version == version:
            return current
        return None

    async def get_by_name(self, namespace: str, name: str) -> "list[ResourceFile]":
        results = await self.propfind(f"/{namespace}/")
        return [r for r in results if r.path.endswith(f"/{name}")]

    async def put(self, path: str, content: str, *, updated_by: str = "engine") -> ResourceFile:
        self._require_writable()
        fs_path = self._fs_path(path)
        checksum = _checksum(content)
        meta = self._read_meta(fs_path)
        if meta is not None and meta.get("checksum") == checksum and fs_path.is_file():
            return ResourceFile(path=path, content=content, version=meta["version"])
        version = (meta["version"] + 1) if meta else 1
        fs_path.parent.mkdir(parents=True, exist_ok=True)
        fs_path.write_text(content, encoding="utf-8")
        self._write_meta(fs_path, version=version, checksum=checksum)
        return ResourceFile(path=path, content=content, version=version)

    async def delete(self, path: str, *, updated_by: str = "engine") -> bool:
        self._require_writable()
        fs_path = self._fs_path(path)
        if not fs_path.is_file():
            return False
        fs_path.unlink()
        meta_path = self._meta_path(fs_path)
        meta_path.unlink(missing_ok=True)
        return True

    async def move(self, src_path: str, dst_path: str, *, updated_by: str = "engine") -> "ResourceFile | None":
        self._require_writable()
        src_fs = self._fs_path(src_path)
        dst_fs = self._fs_path(dst_path)
        if dst_fs.is_file() and not src_fs.is_file():
            # Already applied -- idempotent retry.
            return await self.get(dst_path)
        if not src_fs.is_file():
            return None
        content = src_fs.read_text(encoding="utf-8")
        src_meta = self._read_meta(src_fs)
        new_version = (src_meta["version"] + 1) if src_meta else 1
        dst_fs.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_fs), str(dst_fs))
        self._meta_path(src_fs).unlink(missing_ok=True)
        self._write_meta(dst_fs, version=new_version, checksum=_checksum(content))
        return ResourceFile(path=dst_path, content=content, version=new_version)

    async def list_since(self, since: "datetime | None") -> "list[ResourceFile]":
        # FileBackend has no timestamp index beyond mtime; local/test usage never
        # relies on incremental sync (that's DatabaseBackend's job), so this always
        # returns everything, matching a "since=None" full scan regardless of `since`.
        return await self.propfind("/")

    async def apply_batch(self, ops: "list[Operation]", *, updated_by: str = "engine") -> "list[ResourceFile]":
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

    async def get_revision(self) -> int:
        return 0
