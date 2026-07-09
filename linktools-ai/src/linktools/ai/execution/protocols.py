#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ExecutionBackend Protocol: where file/terminal tool calls actually land.

`fork`/`apply_patch` are added by the swarm+fork plan; the rest of this
Protocol covers the five operations the builtin toolset already exposes.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ExecutionBackend(Protocol):

    async def list_dir(self, path: str = ".", recursive: bool = False) -> "dict[str, Any]": ...

    async def read_file(
        self, path: str, selectors: "list[str] | None" = None, max_chars: int = 6000,
    ) -> "dict[str, Any]": ...

    async def write_file(
        self, path: str, content: Any = None, updates: "list[dict[str, Any]] | None" = None,
    ) -> "dict[str, Any]": ...

    async def batch_files(self, operations: "list[dict[str, Any]]") -> "dict[str, Any]": ...

    async def run_bash(self, command: str, timeout_ms: "int | None" = None) -> "dict[str, Any]": ...

    async def apply_patch(self, diff: str) -> "dict[str, Any]": ...

    async def fork(self, branch_dir: "Any") -> "ExecutionBackend": ...

    async def terminate(self) -> None:
        """Best-effort: kill every live subprocess this backend started and wait
        for it to exit. Idempotent; safe to call after a normal run."""
        ...
