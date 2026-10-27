#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace byte access through an existing native environment backend."""

import asyncio

from ..errors import AIError, ErrorCode
from ._root import Workspace
from ._sandbox import SupportsClose, SupportsFileTools


class WorkspaceAccess:
    """Borrow checked byte access; ownership is explicit when the caller creates it."""

    def __init__(self, backend: SupportsFileTools | None, *, workspace: Workspace | None = None,
                 owns_backend: bool = False) -> None:
        self.backend = backend
        self._workspace = workspace
        self._owns_backend = owns_backend
        self._lock = asyncio.Lock()
        self._closed = False
        self._cleanup_complete = False

    @classmethod
    def for_workspace(cls, workspace: Workspace, *, backend: SupportsFileTools | None,
                      owns_backend: bool = False) -> "WorkspaceAccess":
        return cls(backend, workspace=workspace, owns_backend=owns_backend)

    async def canonicalize_path(self, path: str) -> str:
        if self._closed:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if self.backend is None:
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
        canonical = await self.backend.canonicalize_path(path)
        return canonical if self._workspace is None else self._workspace.validate_path(canonical)

    async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        if self._closed:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if self.backend is None:
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
        return await self.backend.read_bytes(path, max_bytes=max_bytes)

    async def close(self) -> None:
        async with self._lock:
            if self._cleanup_complete:
                return
            self._closed = True
            if self._owns_backend and isinstance(self.backend, SupportsClose):
                await self.backend.close()
            self._cleanup_complete = True


__all__ = ["WorkspaceAccess"]
