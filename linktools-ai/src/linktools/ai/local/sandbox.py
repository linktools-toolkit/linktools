#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local sandbox bound to the explicit project root."""

import asyncio
from pathlib import Path

from ..core.ids import canonical_sha256
from ..storage.files import write_bytes_atomic
from .tool import build_local_tool_map


class LocalSandbox:
    def __init__(self, root: 'str | Path') -> None:
        self._root = Path(root).expanduser().resolve()

    @property
    def fingerprint(self) -> str:
        return canonical_sha256({"kind": "trusted-local-sandbox", "root": str(self._root)})

    async def read_file(self, path: str) -> str:
        target = self._resolve(path)
        if target.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("file is too large")
        return await asyncio.to_thread(target.read_text, encoding="utf-8")

    async def write_file(self, path: str, content: str) -> None:
        target = self._resolve(path)
        if len(content.encode("utf-8")) > 4 * 1024 * 1024:
            raise ValueError("file is too large")
        await asyncio.to_thread(write_bytes_atomic, target, content.encode("utf-8"), fsync=True)

    async def run(self, command: str) -> str:
        result = await build_local_tool_map(self._root)["bash"](command)
        return str(result)

    def _resolve(self, path: str) -> Path:
        target = (self._root / path).expanduser().resolve()
        try:
            target.relative_to(self._root)
        except ValueError as exc:
            raise ValueError(f"path escapes project root: {path}") from exc
        return target


__all__ = ["LocalSandbox"]
