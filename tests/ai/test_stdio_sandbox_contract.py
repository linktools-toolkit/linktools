#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stdio sandboxes preserve their session capability for typed callers."""

import asyncio
import sys
from pathlib import Path
from typing import get_type_hints

import pytest

from linktools.ai.workspace import (
    BubblewrapSandbox,
    LocalSandbox,
    StdioSandbox,
    StdioSandboxSession,
)


@pytest.mark.parametrize("backend", (StdioSandbox, LocalSandbox, BubblewrapSandbox))
def test_stdio_sandbox_open_declares_stdio_session_capability(
    backend: type[StdioSandbox],
) -> None:
    assert get_type_hints(backend.open)["return"] is StdioSandboxSession


async def _exchange(sandbox: StdioSandbox, root: Path) -> bytes:
    session: StdioSandboxSession = await sandbox.open(root=root)
    try:
        process = await session.open_stdio_process(
            sys.executable,
            ("-c", "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())"),
        )
        try:
            await process.write_stdin(b"stdio capability")
            await process.close_stdin()
            chunks: list[bytes] = []
            while chunk := await asyncio.wait_for(process.read_stdout(), 5):
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            await process.close()
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_stdio_sandbox_caller_can_use_process_without_session_narrowing(
    tmp_path: Path,
) -> None:
    sandbox: StdioSandbox = LocalSandbox()
    assert await _exchange(sandbox, tmp_path) == b"stdio capability"
