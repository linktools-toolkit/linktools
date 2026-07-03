#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Builtin toolset wiring: FunctionToolset signatures forwarding to an ExecutionBackend."""

import time
from dataclasses import dataclass
from typing import Any

from pydantic_ai.toolsets import FunctionToolset, WrapperToolset

from .protocols import ExecutionBackend


@dataclass(frozen=True, slots=True)
class BuiltinToolContext:
    backend: ExecutionBackend
    enabled_tools: "set[str]"


class HookedBuiltinToolset(WrapperToolset):
    """Wraps the builtin `FunctionToolset` (file/terminal) to fire
    mcp_call_start/post_mcp_call events with `server="builtin"`, mirroring the
    attribution used by SkillCapability/SubagentCapability/HookedMCPCapability
    for their own tool categories.
    """

    def __init__(self, wrapped, kernel, trace_id: str, parent_call_id: "str | None"):
        super().__init__(wrapped)
        self._kernel = kernel
        self._trace_id = trace_id
        self._parent_call_id = parent_call_id

    async def call_tool(self, name, tool_args, ctx, tool):
        t = time.monotonic()
        success = True
        error: "str | None" = None
        result: Any = None
        if self._kernel:
            self._kernel.trigger(
                "mcp_call_start",
                trace_id=self._trace_id,
                server="builtin",
                tool_name=name,
                arguments=tool_args,
                call_id=ctx.tool_call_id,
                parent_call_id=self._parent_call_id,
            )
        try:
            result = await self.wrapped.call_tool(name, tool_args, ctx, tool)
            return result
        except Exception as exc:
            success = False
            error = str(exc)
            raise
        finally:
            if self._kernel:
                self._kernel.trigger(
                    "post_mcp_call",
                    trace_id=self._trace_id,
                    server="builtin",
                    tool_name=name,
                    duration_ms=round((time.monotonic() - t) * 1000, 2),
                    success=success,
                    data_gaps=[] if success else [f"builtin_tool_failed: {name}"],
                    result=result,
                    error=error,
                    call_id=ctx.tool_call_id,
                    parent_call_id=self._parent_call_id,
                    tool_use_id=ctx.tool_call_id or name,
                    source="builtin",
                )


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
