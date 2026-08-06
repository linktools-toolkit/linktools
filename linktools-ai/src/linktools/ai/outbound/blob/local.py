#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Local-coding filesystem ObjectStore."""

import asyncio
import os
import tempfile
from pathlib import Path


class LocalObjectStore:
    def __init__(self, root: "str | Path") -> None:
        self._root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        path = (self._root / key).resolve()
        if path == self._root or self._root not in path.parents:
            raise ValueError("object key escapes root")
        return path

    async def put(self, key: str, content: bytes) -> str:
        path = self._path(key)
        await asyncio.to_thread(self._atomic_write, path, content)
        return key

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    async def get(self, key: str) -> bytes:
        return await asyncio.to_thread(self._path(key).read_bytes)

    async def head(self, key: str) -> object:
        return await asyncio.to_thread(self._path(key).stat)

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self._path(key).unlink, missing_ok=True)


__all__ = ["LocalObjectStore"]
