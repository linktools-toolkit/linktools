#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace model-correctable failures must remain known durable failures."""

from pathlib import Path
from typing import Any

import pytest
from linktools.ai.capability import workspace_capabilities
from linktools.ai.runtime._tool import ToolOperationDecision
from linktools.ai.runtime._tool_boundary import (
    ManagedToolDescriptor,
    RuntimeToolBoundaryToolset,
)
from linktools.ai.workspace import LocalSandbox, Workspace
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage


class _Bridge:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.decision = ToolOperationDecision("operation", "owner", 1, False)

    async def effective_args(self, ctx, call, tool_def, args):
        del ctx, call, tool_def
        return args

    async def begin(
        self,
        ctx: RunContext[None],
        call: ToolCallPart,
        tool_def: object,
        args: dict[str, Any],
        replay_safe: bool,
    ) -> ToolOperationDecision:
        del ctx, call, tool_def, args
        assert replay_safe is False
        self.calls.append("begin")
        return self.decision

    async def renew(self, decision: ToolOperationDecision) -> ToolOperationDecision:
        return decision

    async def complete(self, decision: ToolOperationDecision, result: Any) -> bool:
        del decision, result
        self.calls.append("complete")
        return False

    async def fail(self, decision: ToolOperationDecision, error: BaseException) -> bool:
        del decision, error
        self.calls.append("fail")
        return False

    async def unknown(
        self, decision: ToolOperationDecision, error: BaseException
    ) -> None:
        del decision, error
        self.calls.append("unknown")

    async def existing_call_ids(self, tool_call_ids: tuple[str, ...]) -> frozenset[str]:
        del tool_call_ids
        return frozenset()

    async def list_operations(self) -> tuple[object, ...]:
        return ()


@pytest.mark.asyncio
async def test_missing_write_parent_is_known_failure_not_effect_unknown(
    tmp_path: Path,
) -> None:
    workspace = Workspace.load(tmp_path, workspace_id="workspace")
    session = await LocalSandbox().open(root=workspace.root)
    try:
        workspace_capability = workspace_capabilities(
            workspace,
            ("write_file",),
            session=session,
        )[0]
        bridge = _Bridge()
        boundary = RuntimeToolBoundaryToolset(
            (workspace_capability.get_toolset(),),
            {
                "write_file": ManagedToolDescriptor(
                    effect_owner="tool_operation",
                    effect="non_replay_safe",
                    tool_class="filesystem.write",
                    workspace_path_fields=("path",),
                )
            },
            id="workspace-boundary",
            sandbox_session=session,
            tool_operations=bridge,  # type: ignore[arg-type]
        )
        context = RunContext(
            deps=None,
            model=TestModel(),
            usage=RunUsage(),
            run_id="run",
            tool_call_id="call",
        )
        args = {"path": "missing/report.txt", "content": "report"}
        with pytest.raises(ModelRetry):
            tools = await boundary.get_tools(context)
            await boundary.call_tool("write_file", args, context, tools["write_file"])

        assert bridge.calls == ["begin", "fail"]
        assert not (tmp_path / "missing").exists()
    finally:
        await session.close()
