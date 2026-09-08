#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-platform LocalSandbox acceptance coverage."""

import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from linktools.ai.workspace import LocalSandbox

pytestmark = pytest.mark.asyncio


def _python_command(code: str) -> str:
    arguments = [sys.executable, "-c", code]
    if sys.platform == "win32":
        return subprocess.list2cmdline(arguments)
    return shlex.join(arguments)


async def test_local_sandbox_runs_python_in_shared_workspace(tmp_path: Path) -> None:
    marker = tmp_path / "platform-marker.txt"
    command = _python_command(
        "from pathlib import Path; "
        "Path('platform-marker.txt').write_text('ok', encoding='utf-8')"
    )
    session = await LocalSandbox().open(root=tmp_path)
    try:
        result = await session.run_command(command, timeout_seconds=10)
    finally:
        await session.close()

    assert "exit_code: 0" in result
    assert marker.read_text(encoding="utf-8") == "ok"
