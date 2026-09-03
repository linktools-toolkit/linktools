#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Default local Sandbox delegation regressions."""

import shlex
import sys
from pathlib import Path

import pytest
from linktools.ai.capability import workspace_capabilities
from linktools.ai.workspace import Workspace
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS, Shell


def _context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
    )


@pytest.mark.asyncio
async def test_local_sandbox_preserves_harness_shell_dispatch(tmp_path: Path) -> None:
    context = _context()
    command = (
        f"{shlex.quote(sys.executable)} -c "
        f"{shlex.quote(\"print('x' * 60000)\")}"
    )
    args = {"command": command, "timeout_seconds": None}

    harness = Shell[None](
        cwd=tmp_path,
        denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
    ).get_toolset()
    harness_tools = await harness.get_tools(context)
    try:
        expected = await harness.call_tool(
            "run_command",
            args,
            context,
            harness_tools["run_command"],
        )
    finally:
        await harness.__aexit__(None, None, None)

    capability = workspace_capabilities(
        Workspace.load(tmp_path),
        ("run_command",),
    )[0]
    toolset = capability.toolset  # type: ignore[attr-defined]
    run_toolset = await toolset.for_run(context)  # type: ignore[arg-type]
    try:
        actual = await run_toolset.tools["run_command"].function(command)  # type: ignore[attr-defined]
    finally:
        await run_toolset.__aexit__(None, None, None)

    assert actual == expected
    assert isinstance(actual, str)
    assert len(actual) < 60_000
