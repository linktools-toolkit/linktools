#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plan-mode visibility for transient file reads."""

import pytest
from pydantic_ai import Tool
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetTool
from pydantic_ai.usage import RunUsage

from linktools.ai.capability import tool_semantic_metadata
from linktools.ai.runtime._agent_executor import _plan_mode_prepare
from linktools.ai.runtime._compaction import RuntimeCompactionPolicy


@pytest.mark.asyncio
async def test_plan_mode_keeps_workspace_file_reads_only() -> None:
    prepare = _plan_mode_prepare(
        plan_mode=True,
    )
    tools = [
        ToolDefinition(
            name="attach_files",
            metadata=tool_semantic_metadata(
                effect="none",
                plan_safe=True,
                tool_class="filesystem.read",
                path_fields=["paths"],
            ),
        ),
        ToolDefinition(
            name="read_file",
            metadata=tool_semantic_metadata(
                effect="none",
                plan_safe=True,
                tool_class="filesystem.read",
                path_fields=["path"],
            ),
        ),
        ToolDefinition(
            name="write_file",
            metadata=tool_semantic_metadata(
                effect="non_replay_safe",
                tool_class="filesystem.write",
                path_fields=["path"],
            ),
        ),
    ]

    selected = await prepare(None, tools)  # type: ignore[arg-type]

    assert [tool.name for tool in selected] == ["attach_files", "read_file"]


@pytest.mark.asyncio
async def test_plan_mode_prepare_captures_wrapped_per_run_tool_semantics() -> None:
    async def control(_ctx: RunContext[None]) -> str:
        return "control"

    async def ordinary(_ctx: RunContext[None]) -> str:
        return "ordinary"

    toolset = FunctionToolset(
        [
            Tool(
                control,
                name="control",
                metadata=tool_semantic_metadata(
                    effect="none",
                    plan_safe=True,
                    compaction_keep_result=True,
                ),
            ),
            Tool(
                ordinary,
                name="ordinary",
                metadata=tool_semantic_metadata(
                    effect="non_replay_safe",
                ),
            ),
        ],
        id="test.tools",
    )

    class _PerRunToolset(AbstractToolset[None]):
        @property
        def id(self) -> str:
            return "test.per-run"

        async def get_tools(
            self,
            ctx: RunContext[None],
        ) -> dict[str, ToolsetTool[None]]:
            return await toolset.get_tools(ctx)

        async def call_tool(
            self,
            name: str,
            tool_args: dict[str, object],
            ctx: RunContext[None],
            tool: ToolsetTool[None],
        ) -> object:
            return await toolset.call_tool(name, tool_args, ctx, tool)

        async def for_run(
            self,
            _ctx: RunContext[None],
        ) -> AbstractToolset[None]:
            return toolset.filtered(
                lambda _filter_ctx, tool_def: tool_def.name == "control"
            )

    ctx = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
    )
    compaction_policy = RuntimeCompactionPolicy()
    prepare = _plan_mode_prepare(
        plan_mode=True,
        compaction_policy=compaction_policy,
    )

    run_toolset = await _PerRunToolset().for_run(ctx)
    tools = await run_toolset.get_tools(ctx)
    selected = await prepare(ctx, [tool.tool_def for tool in tools.values()])

    assert [tool.name for tool in selected] == ["control"]
    assert compaction_policy.keep_result_tools == {"control"}
