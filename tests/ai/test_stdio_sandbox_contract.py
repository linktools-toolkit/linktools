#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stdio sandboxes preserve their session capability for typed callers."""

import asyncio
import sys
from pathlib import Path
from typing import get_type_hints

import pytest

from linktools.ai.workspace import (
    BubblewrapWorkspaceBackend,
    LocalWorkspaceBackend,
    SupportsStdio,
)


@pytest.mark.parametrize("backend", (LocalWorkspaceBackend, BubblewrapWorkspaceBackend))
def test_workspace_backend_declares_stdio_process_capability(backend) -> None:
    from linktools.ai.workspace import WorkspaceStdioProcess
    assert get_type_hints(backend.open_stdio_process)["return"] is WorkspaceStdioProcess


async def _exchange(session: SupportsStdio) -> bytes:
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
    backend = LocalWorkspaceBackend(tmp_path)
    await backend.working_dir()
    assert await _exchange(backend) == b"stdio capability"
