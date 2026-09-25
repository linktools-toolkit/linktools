#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace byte access backed by one lazy SandboxSession."""

import asyncio
from pathlib import Path

from ..errors import AIError, ErrorCode
from ._root import Workspace
from ._sandbox import Sandbox, SandboxSession


class WorkspaceAccess:
    """Own one lazy SandboxSession for Workspace path and byte access."""

    def __init__(
        self,
        sandbox: Sandbox,
        *,
        root: Path,
        session: SandboxSession | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._root = root
        self._workspace = workspace if workspace is not None else Workspace(root, {})
        self._session = session
        self._lock = asyncio.Lock()
        self._closed = False

    @classmethod
    def for_workspace(
        cls,
        workspace: Workspace,
        *,
        sandbox: Sandbox,
    ) -> "WorkspaceAccess":
        return cls(
            sandbox,
            root=workspace.root,
            workspace=workspace,
        )

    async def _ensure_session(self) -> SandboxSession:
        async with self._lock:
            if self._closed:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            if self._session is None:
                self._session = await self._sandbox.open(root=self._root)
            return self._session

    async def canonicalize_path(self, path: str) -> str:
        session = await self._ensure_session()
        return self._workspace.validate_path(await session.canonicalize_path(path))

    async def read_bytes(
        self,
        path: str,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        session = await self._ensure_session()
        return await session.read_bytes(path, max_bytes=max_bytes)

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            session = self._session
            self._session = None
        if session is not None:
            await session.close()


__all__ = ["WorkspaceAccess"]
