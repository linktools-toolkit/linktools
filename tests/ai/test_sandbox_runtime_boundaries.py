#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sandbox lifecycle boundary regressions."""

from pathlib import Path

import pytest
from linktools.ai.capability import ShellCapability
from linktools.ai.core import PromptLimits
from linktools.ai.runtime._compaction import (
    CompactionCapability,
    CompactionPolicy,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai.capabilities import AbstractCapability


def _context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
    )


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run_command(self, command: str, *, timeout_seconds=None):
        self.calls.append(command)
        return "ok"

    async def start_command(self, command: str):
        raise NotImplementedError

    async def check_command(self, command_id: str):
        raise NotImplementedError

    async def stop_command(self, command_id: str):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_workspace_capability_adapts_the_caller_owned_session(
    tmp_path: Path,
) -> None:
    session = _FakeSession()
    toolset = ShellCapability().bind(("run_command",), backend=session).get_toolset()
    context = _context()
    tools = await toolset.get_tools(context)
    result = await toolset.call_tool("run_command", {"command": "echo ok"}, context, tools["run_command"])

    assert "ok" in result
    assert len(session.calls) == 1
    assert session.calls[0] == "echo ok"


def test_runtime_compaction_uses_current_file_deduplication() -> None:
    compaction = CompactionCapability(
        4096,
        limits=PromptLimits(),
        policy=CompactionPolicy(
            context_dedupe_by_tool={
                "read_file": "workspace_file_read_v1",
            },
        ),
        journal=None,
        observer=None,
        projection_sink=None,
    )

    assert isinstance(compaction._deduplicate, AbstractCapability)
    assert compaction._target_tokens == 4096
