#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FilesystemContentCache: a disk-backed ContentCache, sharded by the SHA-256
of the cache key (``aa/bb/<digest>.data`` + a ``.json`` sidecar carrying
checksum + size) so no single directory accumulates unbounded entries.

Writes are temp-file-then-``os.replace`` (atomic, no partial file is ever
visible to a reader); a read whose sidecar checksum does not match the data
file is corrupt -- both files are deleted and the read reports a miss rather
than handing back bad bytes. All blocking I/O runs via ``asyncio.to_thread``.
The cache directory can be deleted wholesale at any time with no correctness
impact (only cache misses result)."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from hashlib import sha256
from pathlib import Path


def _digest(key: str) -> str:
    return sha256(key.encode("utf-8")).hexdigest()


class FilesystemContentCache:
    def __init__(self, *, root: Path) -> None:
        self._root = Path(root)

    def _paths(self, key: str) -> "tuple[Path, Path]":
        digest = _digest(key)
        shard = self._root / digest[:2] / digest[2:4]
        return shard / f"{digest}.data", shard / f"{digest}.json"

    def _atomic_write(self, path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
        try:
            tmp_path.write_bytes(content)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _get_sync(self, key: str) -> "bytes | None":
        data_path, sidecar_path = self._paths(key)
        try:
            sidecar = json.loads(sidecar_path.read_text())
            content = data_path.read_bytes()
        except (OSError, ValueError):
            return None
        if sidecar.get("size") != len(content) or sidecar.get("checksum") != sha256(content).hexdigest():
            data_path.unlink(missing_ok=True)
            sidecar_path.unlink(missing_ok=True)
            return None
        return content

    def _put_sync(self, key: str, content: bytes) -> None:
        data_path, sidecar_path = self._paths(key)
        self._atomic_write(data_path, content)
        sidecar = {"checksum": sha256(content).hexdigest(), "size": len(content)}
        self._atomic_write(sidecar_path, json.dumps(sidecar).encode("utf-8"))

    def _delete_sync(self, key: str) -> None:
        data_path, sidecar_path = self._paths(key)
        data_path.unlink(missing_ok=True)
        sidecar_path.unlink(missing_ok=True)

    async def get(self, key: str) -> "bytes | None":
        return await asyncio.to_thread(self._get_sync, key)

    async def put(self, key: str, content: bytes) -> None:
        await asyncio.to_thread(self._put_sync, key, content)

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self._delete_sync, key)


__all__: "list[str]" = ["FilesystemContentCache"]
