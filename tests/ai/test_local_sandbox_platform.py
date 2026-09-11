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


def _command(arguments: list[str]) -> str:
    if sys.platform == "win32":
        return subprocess.list2cmdline(arguments)
    return shlex.join(arguments)


async def test_local_sandbox_runs_python_in_shared_workspace(tmp_path: Path) -> None:
    marker = tmp_path / "platform-marker.txt"
    script = tmp_path / "platform-runner.py"
    script.write_text(
        "from pathlib import Path\n"
        "Path('platform-marker.txt').write_text('ok', encoding='utf-8')\n",
        encoding="utf-8",
    )
    command = _command([sys.executable, script.name])
    session = await LocalSandbox().open(root=tmp_path)
    try:
        result = await session.run_command(command, timeout_seconds=10)
    finally:
        await session.close()

    assert "exit_code: 0" in result
    assert marker.read_text(encoding="utf-8") == "ok"
