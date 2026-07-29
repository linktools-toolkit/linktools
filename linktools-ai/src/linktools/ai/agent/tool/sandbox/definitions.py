#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Builtin handler wiring forwarding to a Sandbox."""

from dataclasses import dataclass
from typing import Any

from ..models import ToolHandlerSet
from .protocols import Sandbox


@dataclass(frozen=True, slots=True)
class BuiltinToolContext:
    sandbox: Sandbox
    enabled_tools: "set[str]"


def build_builtin_toolset(context: BuiltinToolContext) -> ToolHandlerSet:
    toolset = ToolHandlerSet()
    sandbox = context.sandbox
    enabled = set(context.enabled_tools)
    # the superseded monolithic "file" grant maps to read + write.
    if "file" in enabled:
        enabled.update({"file-read", "file-write"})

    if "file-read" in enabled:

        async def list_dir(
            path: str = ".", recursive: bool = False
        ) -> "dict[str, Any]":
            """List directory contents (relative paths resolve from runtime_dir)."""
            return await sandbox.list_dir(path, recursive)

        async def read_file(
            path: str, selectors: "list[str] | None" = None, max_chars: int = 6000
        ) -> "dict[str, Any]":
            """Read one file. For JSON, pass `selectors` to fetch only selected fields."""
            return await sandbox.read_file(path, selectors, max_chars)

        for fn in (list_dir, read_file):
            toolset.add_function(fn)

    if "file-write" in enabled:

        async def write_file(
            path: str,
            content: Any = None,
            updates: "list[dict[str, Any]] | None" = None,
        ) -> "dict[str, Any]":
            """Write one file. String content writes text; object content writes JSON; `updates` patches JSON fields."""
            return await sandbox.write_file(path, content, updates)

        async def batch_files(operations: "list[dict[str, Any]]") -> "dict[str, Any]":
            """Run multiple file operations in one call. Each item uses `action` = read|write|update plus path/selectors/content/updates."""
            return await sandbox.batch_files(operations)

        async def apply_patch(diff: str) -> "dict[str, Any]":
            """Apply a unified diff (git-style `a/`/`b/` path prefixes accepted) to files under runtime_dir."""
            return await sandbox.apply_patch(diff)

        for fn in (write_file, batch_files, apply_patch):
            toolset.add_function(fn)

    if "terminal" in enabled:

        async def bash(
            command: str, timeout_ms: "int | None" = None
        ) -> "dict[str, Any]":
            """Execute a shell command with cwd set to runtime_dir."""
            return await sandbox.run_bash(command, timeout_ms)

        toolset.add_function(bash)

    return toolset
