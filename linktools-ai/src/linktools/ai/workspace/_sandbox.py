#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace filesystem and process execution boundary."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from ..core import JsonValue
from ..errors import AIError, ErrorCode, ErrorDiagnostics
from ._paths import validate_workspace_path


class SandboxOperationRejected(AIError):
    """Prove that one sandbox operation was rejected before external effect."""

    def __init__(
        self,
        code: ErrorCode,
        message: str = "",
        *,
        safe_details: Mapping[str, Any] | None = None,
        diagnostics: ErrorDiagnostics | None = None,
    ) -> None:
        super().__init__(
            code,
            message,
            retryable=False,
            safe_details=safe_details,  # type: ignore[arg-type]
            diagnostics=diagnostics,
        )

    @classmethod
    def from_error(cls, error: AIError) -> "SandboxOperationRejected":
        if isinstance(error, cls):
            return error
        return cls(
            error.code,
            str(error),
            safe_details=error.safe_details,
            diagnostics=error.diagnostics,
        )


@dataclass(frozen=True, slots=True)
class SandboxResource:
    """One explicitly authorized read-only directory exposed to a session."""

    id: str
    source: Path

    def __post_init__(self) -> None:
        if (
            not isinstance(self.id, str)
            or not self.id
            or not self.id.strip()
        ):
            raise ValueError("sandbox resource id is invalid")
        try:
            self.id.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise ValueError("sandbox resource id is invalid") from error
        if not isinstance(self.source, Path):
            raise TypeError("sandbox resource source must be a Path")
        source = self.source
        if not source.is_absolute() or not source.is_dir() or source.is_symlink():
            raise ValueError("sandbox resource source must be an absolute directory")


@dataclass(frozen=True, slots=True)
class ReadOnlySandboxPolicy:
    """Root-relative read rules shared by supported sandbox backends."""

    readable_paths: tuple[str, ...]
    resource_paths: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.readable_paths, (str, bytes, bytearray)):
            raise TypeError("readable_paths must be a sequence")
        readable = tuple(self.readable_paths)
        normalized = tuple(_normalize_read_pattern(value) for value in readable)
        if not isinstance(self.resource_paths, Mapping):
            raise TypeError("resource_paths must be a mapping")
        resources: dict[str, tuple[str, ...]] = {}
        for key, values in self.resource_paths.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("resource policy key is invalid")
            if isinstance(values, (str, bytes, bytearray)):
                raise TypeError("resource policy paths must be a sequence")
            values = tuple(values)
            resources[key] = tuple(_normalize_read_pattern(value) for value in values)
        object.__setattr__(self, "readable_paths", normalized)
        object.__setattr__(self, "resource_paths", MappingProxyType(resources))

    def allows(self, path: str, *, resource_key: str | None = None) -> bool:
        """Return whether one normalized path may be read."""
        parts = _read_path_parts(path)
        patterns = self._patterns(resource_key)
        return any(_match_path(pattern, parts) for pattern in patterns)

    def may_descend(self, path: str, *, resource_key: str | None = None) -> bool:
        """Return whether a directory may be traversed for visible children."""
        parts = _read_path_parts(path)
        patterns = self._patterns(resource_key)
        return any(_may_match_descendant(pattern, parts) for pattern in patterns)

    def _patterns(self, resource_key: str | None) -> tuple[tuple[str, ...], ...]:
        if resource_key is None:
            return tuple(tuple(value.split("/")) for value in self.readable_paths)
        return tuple(
            tuple(value.split("/"))
            for value in self.resource_paths.get(resource_key, ())
        )


class Sandbox(Protocol):
    async def open(
        self,
        *,
        root: Path,
        resources: tuple[SandboxResource, ...] = (),
    ) -> "SandboxSession": ...


class SandboxSession(Protocol):
    """Sandbox operation contract.

    Effectful implementations may raise ``SandboxOperationRejected`` only when
    they can prove that the rejected operation produced no external effect.
    Ordinary ``AIError`` values carry no effect-certainty guarantee.
    """

    def resource_path(self, resource_id: str) -> "str | None": ...

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


@dataclass(frozen=True, slots=True)
class SandboxResourcePath:
    """One file path inside a resource explicitly granted to a process."""

    resource_id: str
    path: str

    def __post_init__(self) -> None:
        if not isinstance(self.resource_id, str) or not self.resource_id:
            raise ValueError("sandbox resource id is invalid")
        try:
            validate_workspace_path(self.path)
        except (TypeError, ValueError) as error:
            raise ValueError("sandbox resource path is invalid") from error
        if self.path == ".":
            raise ValueError("sandbox resource path must name a file")


class SandboxStdioProcess(Protocol):
    async def write_stdin(self, data: bytes) -> None: ...

    async def read_stdout(self, max_bytes: int = 65536) -> bytes: ...

    async def close_stdin(self) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class StdioSandbox(Sandbox, Protocol):
    def stdio_execution_policy(self) -> Mapping[str, JsonValue]: ...


@runtime_checkable
class StdioSandboxSession(SandboxSession, Protocol):
    async def open_stdio_process(
        self,
        command: str,
        args: "Sequence[str | SandboxResourcePath]" = (),
        *,
        resources: "Sequence[SandboxResource]" = (),
    ) -> SandboxStdioProcess: ...


def normalize_workspace_input_path(path: str) -> str:
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
    normalized = "." if not parts else "/".join(parts)
    try:
        return validate_workspace_path(normalized)
    except ValueError as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error


class DisabledSandbox:
    async def open(
        self,
        *,
        root: Path,
        resources: tuple[SandboxResource, ...] = (),
    ) -> SandboxSession:
        del root, resources
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)


def _normalize_read_pattern(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("read policy pattern is invalid")
    if value.startswith("/") or "//" in value:
        raise ValueError("read policy pattern is invalid")
    parts = value.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise ValueError("read policy pattern is invalid")
    return value


def _read_path_parts(path: str) -> tuple[str, ...]:
    if not isinstance(path, str) or path in {"", "."}:
        return ()
    if path.startswith("/") or "\\" in path:
        raise ValueError("read policy path is invalid")
    parts = tuple(path.split("/"))
    if any(not part or part in {".", ".."} for part in parts):
        raise ValueError("read policy path is invalid")
    return parts


def _match_path(pattern: tuple[str, ...], path: tuple[str, ...]) -> bool:
    if not pattern:
        return not path
    if pattern[0] == "**":
        return _match_path(pattern[1:], path) or (
            bool(path) and _match_path(pattern, path[1:])
        )
    return bool(path) and _match_segment(pattern[0], path[0]) and _match_path(
        pattern[1:], path[1:]
    )


def _may_match_descendant(
    pattern: tuple[str, ...],
    path: tuple[str, ...],
) -> bool:
    if not path:
        return bool(pattern)
    if _match_path(pattern, path):
        return True
    if not pattern:
        return False
    if pattern[0] == "**":
        return _may_match_descendant(pattern[1:], path) or (
            bool(path) and _may_match_descendant(pattern, path[1:])
        )
    return _match_segment(pattern[0], path[0]) and _may_match_descendant(
        pattern[1:], path[1:]
    )


def _match_segment(pattern: str, value: str) -> bool:
    rows = len(pattern) + 1
    columns = len(value) + 1
    table = [[False] * columns for _ in range(rows)]
    table[0][0] = True
    for row, character in enumerate(pattern, 1):
        for column in range(columns):
            if character == "*":
                table[row][column] = table[row - 1][column] or (
                    column > 0 and table[row][column - 1]
                )
            elif column > 0 and (
                character == "?" or character == value[column - 1]
            ):
                table[row][column] = table[row - 1][column - 1]
    return table[-1][-1]


__all__ = [
    "DisabledSandbox",
    "Sandbox",
    "SandboxOperationRejected",
    "SandboxResource",
    "SandboxResourcePath",
    "SandboxSession",
    "SandboxStdioProcess",
    "StdioSandbox",
    "StdioSandboxSession",
    "ReadOnlySandboxPolicy",
    "normalize_workspace_input_path",
]
