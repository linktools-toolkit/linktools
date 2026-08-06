#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local-coding tools and durable tool state."""

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from linktools.core import environ

from ..capability.tool import ToolState
from ..core.json import JsonValue
from ..storage.files import read_json, write_json_atomic

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pydantic_ai.capabilities import AgentCapability

_logger = environ.get_logger("ai.local.tool")


class LocalTool(Protocol):
    async def __call__(self, **kwargs: JsonValue) -> 'dict[str, JsonValue]': ...


LocalToolResultValue = str | int | bool | None | list[dict[str, str]]


def build_local_tools(root: 'str | Path') -> 'tuple[LocalTool, ...]':
    project_root = Path(root).expanduser().resolve()

    async def list_dir(path: str = ".", recursive: bool = False) -> 'dict[str, LocalToolResultValue]':
        try:
            target = _resolve_path(project_root, path)
        except ValueError as error:
            return {"error": str(error)}
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

    async def read_file(path: str, max_chars: int = 6000) -> 'dict[str, LocalToolResultValue]':
        try:
            target = _resolve_path(project_root, path)
        except ValueError as error:
            return {"error": str(error)}
        if not target.is_file():
            return {"error": f"not a file: {path}"}
        content = await asyncio.to_thread(target.read_text, encoding="utf-8")
        limit = max(1, int(max_chars))
        _logger.info("local tool completed name=read_file path=%s", path)
        return {"path": target.relative_to(project_root).as_posix(), "content": content[:limit], "truncated": len(content) > limit}

    async def write_file(path: str, content: str) -> 'dict[str, LocalToolResultValue]':
        try:
            target = _resolve_path(project_root, path)
        except ValueError as error:
            return {"error": str(error)}
        target.parent.mkdir(parents=True, exist_ok=True)
        value = content
        await asyncio.to_thread(target.write_text, value, encoding="utf-8")
        size = len(value.encode("utf-8"))
        _logger.info("local tool completed name=write_file path=%s bytes=%s", path, size)
        return {"path": target.relative_to(project_root).as_posix(), "bytes": size}

    async def bash(command: str, timeout_ms: 'int | None' = None) -> 'dict[str, LocalToolResultValue]':
        if not command.strip():
            return {"error": "command must not be empty"}
        timeout_value = max(1, int(timeout_ms or 60_000))
        _logger.info("local tool started name=bash command=%s", command)
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(project_root),
            env={key: value for key, value in os.environ.items() if key in {"PATH", "HOME", "LANG", "LC_ALL"}},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_value / 1000)
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return {"exit_code": None, "error": f"timeout after {timeout_value}ms"}
        _logger.info("local tool completed name=bash exit_code=%s", process.returncode)
        return {"exit_code": process.returncode, "stdout": stdout.decode("utf-8", errors="replace")[-16_000:], "stderr": stderr.decode("utf-8", errors="replace")[-4_000:]}

    return cast(tuple[LocalTool, ...], (list_dir, read_file, write_file, bash))


def build_local_capabilities(root: 'str | Path') -> 'tuple[AgentCapability[None], ...]':
    from pydantic_ai_harness.filesystem import FileSystem
    from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS, Shell

    project_root = Path(root).expanduser().resolve()
    _logger.info("local Harness capabilities configured root=%s", project_root)
    return (
        FileSystem(root_dir=project_root),
        Shell(cwd=project_root, denied_env_patterns=LLM_API_KEY_ENV_PATTERNS),
    )


def build_local_tool_map(root: 'str | Path') -> 'Mapping[str, LocalTool]':
    tools = build_local_tools(root)
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


class LocalToolState:
    def __init__(self, root: 'str | Path | None' = None) -> None:
        self._states: dict[str, ToolState] = {}
        self._path = None if root is None else Path(root).expanduser().resolve() / ".linktools" / "tool-state.json"
        self._loaded = False
        self._lock = asyncio.Lock()

    async def get(self, operation_id: str) -> 'ToolState | None':
        await self._initialize()
        return self._states.get(operation_id)

    async def put(self, state: ToolState) -> ToolState:
        await self._initialize()
        async with self._lock:
            previous = self._states.get(state.operation_id)
            if previous is not None and previous != state:
                raise ValueError("tool operation state conflict")
            self._states[state.operation_id] = state
            self._persist()
        return state

    async def _initialize(self) -> None:
        if self._loaded:
            return
        async with self._lock:
            if self._loaded:
                return
            if self._path is not None and self._path.exists():
                raw = read_json(self._path)
                for operation_id, value in raw.items():
                    if not isinstance(value, dict):
                        raise ValueError("local tool state is invalid")
                    self._states[operation_id] = ToolState(
                        operation_id,
                        str(value.get("state", "")),
                        None if value.get("result_digest") is None else str(value["result_digest"]),
                    )
            self._loaded = True

    def _persist(self) -> None:
        if self._path is None:
            return
        payload = {
            operation_id: {"state": state.state, "result_digest": state.result_digest}
            for operation_id, state in sorted(self._states.items())
        }
        write_json_atomic(self._path, payload, fsync=True)


__all__ = ["LocalTool", "LocalToolState", "build_local_capabilities", "build_local_tool_map", "build_local_tools"]
