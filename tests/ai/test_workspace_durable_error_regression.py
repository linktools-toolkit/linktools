#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace model-correctable failures must remain known durable failures."""

from pathlib import Path
from typing import Any

import pytest
from linktools.ai.capability import workspace_capabilities
from linktools.ai.runtime._capabilities import ToolOperationDecision, _RuntimeStepPersistence
from linktools.ai.runtime.state import StagingStepStore
from linktools.ai.workspace import LocalSandbox, Workspace
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage


class _Bridge:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.decision = ToolOperationDecision("operation", "owner", 1, False)

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

    async def unknown(self, decision: ToolOperationDecision, error: BaseException) -> None:
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
    workspace = Workspace.load(tmp_path)
    session = await LocalSandbox().open(root=workspace.root)
    try:
        workspace_capability = workspace_capabilities(
            workspace,
            ("write_file",),
            session=session,
        )[0]
        tool = workspace_capability.get_toolset().tools["write_file"]
        definition = ToolDefinition(
            name="write_file",
            capability_id="workspace-sandbox",
            metadata={"linktools.ai.workspace_tool_class": "filesystem.write"},
        )
        bridge = _Bridge()
        persistence = _RuntimeStepPersistence(
            store=StagingStepStore(),
            tool_operations=bridge,  # type: ignore[arg-type]
            agent_name="agent",
            run_id="run",
            trusted_tool_classes=(("write_file", "filesystem.write"),),
        )
        context = RunContext(
            deps=None,
            model=TestModel(),
            usage=RunUsage(),
            run_id="run",
        )
        args = {"path": "missing/report.txt", "content": "report"}
        call = ToolCallPart("write_file", args, tool_call_id="call")
        await persistence.before_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args=args,
        )

        async def handler(validated: dict[str, Any]) -> str:
            return await tool.function(**validated)

        with pytest.raises(ModelRetry):
            await persistence.wrap_tool_execute(
                context,
                call=call,
                tool_def=definition,
                args=args,
                handler=handler,
            )

        assert bridge.calls == ["begin", "fail"]
        assert not (tmp_path / "missing").exists()
    finally:
        await session.close()
