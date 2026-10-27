#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native protocol operations shared by the concrete workspace providers."""

from __future__ import annotations

import errno
import math
import posixpath
import shlex
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pydantic_ai.workspaces import CommandResult, FileEntry, WorkspaceRef

from ..errors import AIError, ErrorCode
from ._sandbox import WorkspaceOperationRejected, normalize_workspace_input_path


class _NativeWorkspaceOperations:
    """Protocol translation only; concrete providers own environment lifetime."""

    @property
    def ref(self) -> WorkspaceRef | None:
        if self._ref_id is None:
            return None
        from pydantic_ai.workspaces import WorkspaceRef

        return WorkspaceRef(provider="linktools-workspace", id=self._ref_id)

    async def working_dir(self) -> str:
        try:
            await self._ensure_open()
        except AIError as error:
            self._raise(error)
            raise
        return self.execution_root

    def _path(self, path: str) -> str:
        root = self.execution_root
        if not posixpath.isabs(path):
            return normalize_workspace_input_path(path)
        normalized = posixpath.normpath(path)
        if not posixpath.isabs(normalized) or (normalized != root and not normalized.startswith(root.rstrip("/") + "/")):
            raise PermissionError("Path is outside this workspace")
        return posixpath.relpath(normalized, root)

    async def _file(self, operation: str, path: str, *, data: bytes | None = None) -> Mapping[str, object]:
        try:
            await self._ensure_open()
            return await self.raw_filesystem(
                operation, self._path(path) if operation in {"write", "mkdir", "remove"} else path, data=data,
            )
        except AIError as error:
            self._raise(error, path, mutation=operation in {"write", "mkdir", "remove"})
            raise

    def _raise(self, error: AIError, path: str = "", *, mutation: bool = False) -> None:
        if error.code in {ErrorCode.SANDBOX_SESSION_CLOSED, ErrorCode.SANDBOX_SESSION_LOST}:
            from pydantic_ai.workspaces import WorkspaceUnavailableError

            raise WorkspaceUnavailableError("Sandbox session is unavailable") from error
        if mutation and not isinstance(error, WorkspaceOperationRejected):
            return
        if error.code == ErrorCode.AUTHORIZATION_DENIED:
            raise PermissionError("Sandbox policy refused the operation") from error
        if error.code == ErrorCode.STORAGE_NOT_FOUND:
            raise FileNotFoundError(errno.ENOENT, "Workspace path does not exist") from error

    async def read_bytes(self, path: str, *, max_bytes: int | None = None,
                         expected_state: str | None = None) -> bytes:
        try:
            await self._ensure_open()
            return await self._read_bytes(path, max_bytes=max_bytes, expected_state=expected_state)
        except AIError as error:
            if isinstance(path, str) and posixpath.isabs(path):
                self._raise(error, path)
            raise

    async def write_bytes(self, path: str, data: bytes) -> None:
        await self._file("write", path, data=data)

    async def stat(self, path: str) -> FileEntry:
        return self._entry(await self._file("stat", path))

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        result = await self._file("list", path)
        return tuple(self._entry(entry) for entry in cast(Sequence[Mapping[str, object]], result["entries"]))

    def _entry(self, entry: Mapping[str, object]) -> FileEntry:
        from pydantic_ai.workspaces import FileEntry

        return FileEntry(name=cast(str, entry["name"]), path=cast(str, entry["path"]),
                         is_dir=cast(bool, entry["is_dir"]), size=cast(int, entry["size"]))

    async def make_dir(self, path: str) -> None:
        await self._file("mkdir", path)

    async def remove(self, path: str) -> None:
        if self._path(path) == ".":
            raise ValueError("Cannot remove the working directory")
        await self._file("remove", path)

    async def exists(self, path: str) -> bool:
        return cast(bool, (await self._file("exists", path))["exists"])

    async def realpath(self, path: str) -> str:
        return cast(str, (await self._file("realpath", path))["path"])

    async def run(
        self, command: str | Sequence[str], *, shell: bool = False,
        env: Mapping[str, str] | None = None, timeout: float | None = None,
    ) -> CommandResult:
        from pydantic_ai.workspaces import CommandResult, WorkspaceOutputLimitError, WorkspaceTimeoutError

        if shell != isinstance(command, str):
            raise TypeError("Use shell=True with a shell string, otherwise an argv sequence")
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (float, int))
                                    or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be a positive finite number")
        if not shell:
            if not command:
                raise ValueError("command must not be empty")
            command = shlex.join(command)
        try:
            result = await self.execute_command(cast(str, command), environment=env, timeout_seconds=timeout)
        except AIError as error:
            self._raise(error, mutation=True)
            raise
        stdout, stderr = cast(str, result["stdout"]), cast(str, result["stderr"])
        if result["timed_out"]:
            raise WorkspaceTimeoutError("Sandbox command timed out", stdout=stdout, stderr=stderr)
        if result["output_truncated"]:
            raise WorkspaceOutputLimitError("Sandbox command exceeded its output cap", limit=50_000,
                                            stdout=stdout, stderr=stderr)
        return CommandResult(exit_code=cast(int, result["exit_code"]), stdout=stdout, stderr=stderr)


__all__ = []
