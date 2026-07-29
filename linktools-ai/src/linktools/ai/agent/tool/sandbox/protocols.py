#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sandbox Protocol: where file/terminal tool calls actually land.

`fork`/`apply_patch` are added by the swarm+fork work; the rest of this
Protocol covers the five operations the builtin toolset already exposes.
"""

from enum import Enum
from typing import Any, Protocol, runtime_checkable


class ExecutionIsolationLevel(str, Enum):
    """Trust boundary provided by an execution backend."""

    TRUSTED_LOCAL = "trusted_local"
    CONTAINER = "container"
    REMOTE_SANDBOX = "remote_sandbox"


@runtime_checkable
class Sandbox(Protocol):
    @property
    def isolation_level(self) -> ExecutionIsolationLevel: ...

    async def list_dir(
        self, path: str = ".", recursive: bool = False
    ) -> "dict[str, Any]": ...

    async def read_file(
        self,
        path: str,
        selectors: "list[str] | None" = None,
        max_chars: int = 6000,
    ) -> "dict[str, Any]": ...

    async def write_file(
        self,
        path: str,
        content: Any = None,
        updates: "list[dict[str, Any]] | None" = None,
    ) -> "dict[str, Any]": ...

    async def batch_files(
        self, operations: "list[dict[str, Any]]"
    ) -> "dict[str, Any]": ...

    async def run_bash(
        self, command: str, timeout_ms: "int | None" = None
    ) -> "dict[str, Any]": ...

    async def apply_patch(self, diff: str) -> "dict[str, Any]": ...

    async def fork(self, branch_dir: "Any") -> "Sandbox": ...

    async def terminate(self) -> None:
        """Best-effort: kill every live subprocess this backend started and wait
        for it to exit. Idempotent; safe to call after a normal run."""
        ...
