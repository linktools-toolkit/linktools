#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plan-mode visibility for transient file reads."""

import pytest
from pydantic_ai.tools import ToolDefinition

from linktools.ai.runtime._agent_executor import _plan_mode_prepare


@pytest.mark.asyncio
async def test_plan_mode_keeps_workspace_file_reads_only() -> None:
    prepare = _plan_mode_prepare(
        business_descriptors={},
        business_plan_safe=frozenset(),
        plan_mode=True,
    )
    tools = [
        ToolDefinition(name="attach_files"),
        ToolDefinition(name="read_file"),
        ToolDefinition(name="write_file"),
    ]

    selected = await prepare(None, tools)  # type: ignore[arg-type]

    assert [tool.name for tool in selected] == ["attach_files", "read_file"]
