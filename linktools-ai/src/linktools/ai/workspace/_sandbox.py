#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace filesystem and process execution boundary."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..errors import AIError, ErrorCode


@dataclass(frozen=True, slots=True)
class SandboxResource:
    """One explicitly authorized read-only directory exposed to a session."""

    key: str
    source: Path

    def __post_init__(self) -> None:
        if (
            not isinstance(self.key, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.key)
        ):
            raise ValueError("sandbox resource key is invalid")
        if not isinstance(self.source, Path):
            raise TypeError("sandbox resource source must be a Path")
        source = self.source
        if not source.is_absolute() or not source.is_dir() or source.is_symlink():
            raise ValueError("sandbox resource source must be an absolute directory")


class Sandbox(Protocol):
    async def open(
        self,
        *,
        root: Path,
        resources: tuple[SandboxResource, ...] = (),
    ) -> "SandboxSession": ...


class SandboxSession(Protocol):
    def resource_path(self, key: str) -> str: ...

    async def canonicalize_path(self, path: str) -> str: ...

    async def read_bytes(
        self,
        path: str,
        *,
        max_bytes: "int | None" = None,
    ) -> bytes: ...

    async def read_file(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: "int | None" = None,
    ) -> str: ...

    async def write_file(
        self,
        path: str,
        content: str,
        *,
        expected_hash: "str | None" = None,
    ) -> str: ...

    async def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        *,
        expected_hash: "str | None" = None,
    ) -> str: ...

    async def list_directory(self, path: str = ".") -> str: ...

    async def search_files(
        self,
        pattern: str,
        *,
        path: str = ".",
        include_glob: "str | None" = None,
    ) -> str: ...

    async def find_files(
        self,
        pattern: str,
        *,
        path: str = ".",
    ) -> str: ...

    async def create_directory(self, path: str) -> str: ...

    async def file_info(self, path: str) -> str: ...

    async def run_command(
        self,
        command: str,
        *,
        timeout_seconds: "float | None" = None,
    ) -> str: ...

    async def start_command(self, command: str) -> str: ...

    async def check_command(self, command_id: str) -> str: ...

    async def stop_command(self, command_id: str) -> str: ...

    async def close(self) -> None: ...


def normalize_workspace_path(path: str) -> str:
    """Normalize one logical workspace-relative POSIX path."""
    if not isinstance(path, str) or "\x00" in path or "\\" in path:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    try:
        path.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
    if path == "":
        return "."
    if path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if path.lower().startswith("file:"):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    parts: list[str] = []
    for part in path.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        parts.append(part)
    return "." if not parts else "/".join(parts)


class DisabledSandbox:
    async def open(
        self,
        *,
        root: Path,
        resources: tuple[SandboxResource, ...] = (),
    ) -> SandboxSession:
        del root, resources
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)


__all__ = [
    "DisabledSandbox",
    "Sandbox",
    "SandboxResource",
    "SandboxSession",
    "normalize_workspace_path",
]
