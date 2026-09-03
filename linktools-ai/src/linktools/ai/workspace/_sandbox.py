#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace filesystem and process execution boundary."""

from typing import Protocol

from ..errors import AIError, ErrorCode


class Sandbox(Protocol):
    async def open(self) -> "SandboxSession": ...


class SandboxSession(Protocol):
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


class DisabledSandbox:
    async def open(self) -> SandboxSession:
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)


__all__ = ["DisabledSandbox", "Sandbox", "SandboxSession"]
