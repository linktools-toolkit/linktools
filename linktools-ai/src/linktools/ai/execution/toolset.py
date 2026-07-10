#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Builtin toolset wiring: FunctionToolset signatures forwarding to an ExecutionBackend."""

from dataclasses import dataclass
from typing import Any

from pydantic_ai.toolsets import FunctionToolset

from .protocols import ExecutionBackend


@dataclass(frozen=True, slots=True)
class BuiltinToolContext:
    backend: ExecutionBackend
    enabled_tools: "set[str]"


def build_builtin_toolset(context: BuiltinToolContext) -> FunctionToolset:
    toolset: FunctionToolset = FunctionToolset()
    backend = context.backend
    enabled_tools = context.enabled_tools

    if "file" in enabled_tools:
        async def list_dir(path: str = ".", recursive: bool = False) -> "dict[str, Any]":
            """List directory contents (relative paths resolve from runtime_dir)."""
            return await backend.list_dir(path, recursive)

        async def read_file(path: str, selectors: "list[str] | None" = None, max_chars: int = 6000) -> "dict[str, Any]":
            """Read one file. For JSON, pass `selectors` to fetch only selected fields."""
            return await backend.read_file(path, selectors, max_chars)

        async def write_file(path: str, content: Any = None, updates: "list[dict[str, Any]] | None" = None) -> "dict[str, Any]":
            """Write one file. String content writes text; object content writes JSON; `updates` patches JSON fields."""
            return await backend.write_file(path, content, updates)

        async def batch_files(operations: "list[dict[str, Any]]") -> "dict[str, Any]":
            """Run multiple file operations in one call. Each item uses `action` = read|write|update plus path/selectors/content/updates."""
            return await backend.batch_files(operations)

        async def apply_patch(diff: str) -> "dict[str, Any]":
            """Apply a unified diff (git-style `a/`/`b/` path prefixes accepted) to files under runtime_dir."""
            return await backend.apply_patch(diff)

        for fn in (list_dir, read_file, write_file, batch_files, apply_patch):
            toolset.add_function(fn)

    if "terminal" in enabled_tools:
        async def bash(command: str, timeout_ms: "int | None" = None) -> "dict[str, Any]":
            """Execute a shell command with cwd set to runtime_dir."""
            return await backend.run_bash(command, timeout_ms)

        toolset.add_function(bash)

    return toolset
