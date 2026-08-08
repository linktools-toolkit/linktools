#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace-local tools and capability boundaries."""

import asyncio
import os
import signal
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from linktools.core import environ

from ..core.json import JsonValue
from ..core.ids import canonical_sha256
from ..core.errors import ErrorCode
from ..storage.files import write_bytes_atomic

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pydantic_ai.capabilities import AgentCapability

_logger = environ.get_logger("ai.workspace.tools")


class WorkspaceTool(Protocol):
    async def __call__(self, **kwargs: JsonValue) -> 'dict[str, JsonValue]': ...


WorkspaceToolResultValue = str | int | bool | None | list[dict[str, str]]


def build_workspace_tools(root: 'str | Path') -> 'tuple[WorkspaceTool, ...]':
    project_root = Path(root).expanduser().resolve()
    shell_slots = asyncio.Semaphore(4)

    async def list_dir(path: str = ".", recursive: bool = False) -> 'dict[str, WorkspaceToolResultValue]':
        try:
            target = _resolve_path(project_root, path)
        except ValueError:
            return {"error": "PATH_OUTSIDE_ROOT"}
        if not target.is_dir():
            return {"error": f"not a directory: {path}"}
        iterator = target.rglob("*") if recursive else target.iterdir()
        entries: list[dict[str, str]] = []
        for item in sorted(iterator, key=lambda value: value.relative_to(target).as_posix()):
            entries.append({"path": item.relative_to(project_root).as_posix(), "type": "directory" if item.is_dir() else "file"})
            if len(entries) >= 500:
                break
        _logger.info("local tool completed name=list_dir path=%s count=%s", path, len(entries))
        return {"path": target.relative_to(project_root).as_posix(), "entries": entries}

    async def read_file(path: str, max_chars: int = 6000) -> 'dict[str, WorkspaceToolResultValue]':
        try:
            target = _resolve_path(project_root, path)
        except ValueError:
            return {"error": "PATH_OUTSIDE_ROOT"}
        if not target.is_file():
            return {"error": f"not a file: {path}"}
        content = await asyncio.to_thread(target.read_text, encoding="utf-8")
        limit = max(1, int(max_chars))
        _logger.info("local tool completed name=read_file path=%s", path)
        return {"path": target.relative_to(project_root).as_posix(), "content": content[:limit], "truncated": len(content) > limit}

    async def write_file(path: str, content: str) -> 'dict[str, WorkspaceToolResultValue]':
        try:
            target = _resolve_path(project_root, path)
        except ValueError:
            return {"error": "PATH_OUTSIDE_ROOT"}
        value = content
        size = len(value.encode("utf-8"))
        if size > 4 * 1024 * 1024:
            return {"error": "file is too large"}
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(write_bytes_atomic, target, value.encode("utf-8"), fsync=True)
        _logger.info("local tool completed name=write_file path=%s bytes=%s", path, size)
        return {"path": target.relative_to(project_root).as_posix(), "bytes": size}

    async def bash(command: str, timeout_ms: 'int | None' = None) -> 'dict[str, WorkspaceToolResultValue]':
        async with shell_slots:
            if os.name != "posix":
                return {"error": ErrorCode.LOCAL_SHELL_PLATFORM_UNSUPPORTED.value}
            if not command.strip():
                return {"error": "command must not be empty"}
            timeout_value = int(timeout_ms or 60_000)
            if timeout_value < 1_000 or timeout_value > 900_000:
                return {"error": "timeout must be between 1 and 900 seconds"}
            _logger.info("local tool started name=bash command_digest=%s", canonical_sha256(command))
            process = await asyncio.create_subprocess_exec(
                "/bin/sh",
                "-lc",
                command,
                cwd=str(project_root),
                env=_shell_environment(project_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            stdout_truncated = stderr_truncated = False
            try:
                (stdout, stdout_truncated), (stderr, stderr_truncated) = await asyncio.wait_for(
                    asyncio.gather(
                        _read_limited(process.stdout),
                        _read_limited(process.stderr),
                    ),
                    timeout=timeout_value / 1000,
                )
            except asyncio.TimeoutError:
                _terminate_process_group(process.pid)
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    _kill_process_group(process.pid)
                    await process.wait()
                return {"exit_code": None, "status": "TIMEOUT", "truncated": stdout_truncated or stderr_truncated}
            except asyncio.CancelledError:
                _terminate_process_group(process.pid)
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    _kill_process_group(process.pid)
                    await process.wait()
                raise
            _logger.info("local tool completed name=bash exit_code=%s", process.returncode)
            return {
                "exit_code": process.returncode,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "truncated": stdout_truncated or stderr_truncated,
            }

    return cast(tuple[WorkspaceTool, ...], (list_dir, read_file, write_file, bash))


def build_workspace_capabilities(root: 'str | Path') -> 'tuple[AgentCapability[None], ...]':
    from pydantic_ai_harness.filesystem import FileSystem
    from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS, Shell

    project_root = Path(root).expanduser().resolve()
    _logger.info("local Harness capabilities configured root=%s", project_root)
    return (
        FileSystem(root_dir=project_root),
        Shell(cwd=project_root, denied_env_patterns=LLM_API_KEY_ENV_PATTERNS),
    )


def build_workspace_tool_map(root: 'str | Path') -> 'Mapping[str, WorkspaceTool]':
    tools = build_workspace_tools(root)
    return {
        "list_dir": tools[0],
        "read_file": tools[1],
        "write_file": tools[2],
        "bash": tools[3],
    }


def _resolve_path(root: Path, path: str) -> Path:
    target = (root / path).expanduser().resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path escapes project root: {path}") from error
    return target


async def _read_limited(stream: 'asyncio.StreamReader | None') -> 'tuple[bytes, bool]':
    if stream is None:
        return b"", False
    limit = 1024 * 1024
    chunks: list[bytes] = []
    size = 0
    truncated = False
    while True:
        chunk = await stream.read(64 * 1024)
        if not chunk:
            break
        if size < limit:
            kept = chunk[: limit - size]
            chunks.append(kept)
            size += len(kept)
            if len(chunk) > len(kept):
                truncated = True
        else:
            truncated = True
    return b"".join(chunks), truncated


def _shell_environment(project_root: Path) -> 'dict[str, str]':
    environment = {key: value for key, value in os.environ.items() if key in {"PATH", "LANG", "LC_ALL", "TERM"}}
    home = project_root / ".linktools" / "home"
    home.mkdir(parents=True, exist_ok=True)
    environment["HOME"] = str(home)
    return environment


def _terminate_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def _kill_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


__all__ = ["WorkspaceTool", "build_workspace_capabilities", "build_workspace_tool_map", "build_workspace_tools"]
