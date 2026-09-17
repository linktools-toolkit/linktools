#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""System instruction composition and refresh contracts."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from linktools.ai.capability import ToolCallRejected, workspace_capabilities
from linktools.ai.runtime._agent_executor import _CachedRepositoryInstructionBoundary
from linktools.ai.runtime._tool_boundary import RuntimeToolBoundaryToolset
from linktools.ai.workspace import LocalSandbox, Workspace
from pydantic_ai.messages import InstructionPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage


class _RefreshBoundary:
    def __init__(self) -> None:
        self.rendered = "root rules"
        self.active_checks = 0
        self.max_active_checks = 0
        self.applied: list[str] = []

    def render(self) -> str:
        return self.rendered

    async def check(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, Any],
        path_fields: tuple[str, ...],
    ) -> None:
        del tool_name, tool_call_id, path_fields
        self.active_checks += 1
        self.max_active_checks = max(self.max_active_checks, self.active_checks)
        try:
            await asyncio.sleep(0)
            path = arguments.get("path")
            if not isinstance(path, str):
                return
            self.applied.append(path)
            self.rendered = "root rules\n" + "\n".join(self.applied)
            raise ToolCallRejected("Repository instructions changed; reconsider the call")
        finally:
            self.active_checks -= 1


def _context() -> RunContext[object]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
    )


@pytest.mark.asyncio
async def test_runtime_tool_boundary_flattens_one_instruction_sequence() -> None:
    toolset = FunctionToolset(
        id="source",
        instructions=(
            InstructionPart(content="first", name="first", dynamic=False),
            "second",
        ),
    )
    boundary = RuntimeToolBoundaryToolset((toolset,), {}, id="boundary")

    instructions = await boundary.get_instructions(_context())

    assert instructions is not None
    assert [item.content for item in instructions if isinstance(item, InstructionPart)] == [
        "first",
        "second",
    ]
    assert all(
        isinstance(item, InstructionPart) and item.dynamic is False
        for item in instructions
    )


@pytest.mark.asyncio
async def test_workspace_guidance_is_static_when_workspace_tools_are_selected(
    tmp_path: Path,
) -> None:
    workspace = Workspace.load(tmp_path, workspace_id="workspace")
    sandbox = LocalSandbox()
    session = await sandbox.open(root=workspace.root)
    try:
        capability = workspace_capabilities(
            workspace,
            ("read_file",),
            session=session,
        )[0]
        instructions = await capability.get_toolset().get_instructions(_context())
    finally:
        await session.close()

    assert instructions is not None and len(instructions) == 1
    instruction = instructions[0]
    assert isinstance(instruction, InstructionPart)
    assert instruction.name == "workspace"
    assert instruction.dynamic is False
    assert "logical Workspace root" in instruction.content
    assert "not automatically authorized" in instruction.content


@pytest.mark.asyncio
async def test_repository_instruction_refresh_is_serialized_and_published_atomically() -> None:
    source = _RefreshBoundary()
    boundary = _CachedRepositoryInstructionBoundary(source)

    assert boundary.render() == "root rules"

    async def refresh(path: str) -> None:
        with pytest.raises(ToolCallRejected):
            await boundary.check(
                tool_name="read_file",
                tool_call_id=f"call-{path}",
                arguments={"path": path},
                path_fields=("path",),
            )

    await asyncio.gather(refresh("a"), refresh("b"))

    assert source.max_active_checks == 1
    assert source.applied == ["a", "b"]
    assert boundary.render() == "root rules\na\nb"
