#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""System instruction composition and refresh contracts."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from linktools.ai.capability import ToolCallRetry, workspace_capabilities
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.capabilities import AbstractCapability, Capability
from linktools.ai.runtime._repository_instructions import _RepositoryInstructionBoundary
from linktools.ai.runtime._tool_boundary import RuntimeToolBoundaryToolset
from linktools.ai.workspace import (
    LocalSandbox,
    RepositoryInstructionDocument,
    RepositoryInstructions,
    Workspace,
)
from pydantic_ai.messages import InstructionPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage


class _RefreshCoordinator:
    def __init__(self) -> None:
        self.active_checks = 0
        self.max_active_checks = 0
        self.paths: list[str] = []

    async def check_repository_instructions(
        self,
        *,
        execution: object,
        initial: RepositoryInstructions | None,
        overlay: RepositoryInstructions | None,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, object],
        path_fields: tuple[str, ...],
    ) -> tuple[RepositoryInstructions | None, bool]:
        del execution, initial, tool_name, tool_call_id, path_fields
        self.active_checks += 1
        self.max_active_checks = max(self.max_active_checks, self.active_checks)
        try:
            await asyncio.sleep(0)
            path = arguments.get("path")
            if not isinstance(path, str):
                return overlay, False
            self.paths.append(path)
            if path != "a" or overlay is not None:
                return overlay, False
            return (
                RepositoryInstructions(
                    (
                        RepositoryInstructionDocument(
                            "agents:pkg/AGENTS.md",
                            "pkg",
                            "package rules",
                        ),
                    )
                ),
                True,
            )
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
        instruction = capability.get_instructions()
        toolset_instructions = await capability.get_toolset().get_instructions(_context())
    finally:
        await session.close()

    assert isinstance(instruction, InstructionPart)
    assert instruction.name == "workspace"
    assert instruction.dynamic is False
    assert "logical Workspace root" in instruction.content
    assert "not automatically authorized" in instruction.content
    assert toolset_instructions is None


class _InstructionCapture(AbstractCapability[None]):
    def __init__(self) -> None:
        self.id = "instruction-capture"
        self.parts: tuple[InstructionPart, ...] = ()

    async def before_model_request(
        self,
        ctx: RunContext[None],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        del ctx
        self.parts = tuple(
            request_context.model_request_parameters.instruction_parts or ()
        )
        return request_context


@pytest.mark.asyncio
async def test_request_instruction_parts_follow_f0_f1_o_order() -> None:
    capture = _InstructionCapture()

    def repository_overlay(_: RunContext[None]) -> str:
        return "O: repository overlay"

    agent = PydanticAgent(
        TestModel(),
        system_prompt="standing system prompt",
        instructions=("F0: agent", repository_overlay),
    )
    await agent.run(
        "test",
        capabilities=(
            Capability(id="skill", instructions="F0: skill"),
            Capability(id="workspace", instructions="F0: workspace"),
            Capability(id="subagent", instructions="F1: subagent"),
            Capability(id="memory", instructions="F1: memory"),
            Capability(id="planning", instructions="F1: planning"),
            Capability(
                id="repository-initial",
                instructions="F1: repository initial",
            ),
            capture,
        ),
    )

    assert [part.content for part in capture.parts] == [
        "F0: agent",
        "F0: skill",
        "F0: workspace",
        "F1: subagent",
        "F1: memory",
        "F1: planning",
        "F1: repository initial",
        "O: repository overlay",
    ]
    assert [part.dynamic for part in capture.parts] == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]


@pytest.mark.asyncio
async def test_repository_partition_is_stable_and_fences_same_turn_siblings() -> None:
    initial = RepositoryInstructions(
        (
            RepositoryInstructionDocument(
                "agents:AGENTS.md",
                ".",
                "root rules",
            ),
        )
    )
    coordinator = _RefreshCoordinator()
    boundary = _RepositoryInstructionBoundary(
        coordinator,  # type: ignore[arg-type]
        SimpleNamespace(execution_id="execution", tenant_id="tenant"),  # type: ignore[arg-type]
        initial,
        None,
    )

    initial_text = boundary.render_initial()
    assert "Repository instructions are workspace guidance." in initial_text
    assert "root rules" in initial_text
    assert boundary.render_overlay() == ""

    async def refresh(path: str) -> None:
        with pytest.raises(ToolCallRetry):
            await boundary.check(
                tool_name="read_file",
                tool_call_id=f"call-{path}",
                arguments={"path": path},
                path_fields=("path",),
            )

    await asyncio.gather(refresh("a"), refresh("b"))

    assert coordinator.max_active_checks == 1
    assert coordinator.paths == ["a"]
    assert boundary.render_initial() == initial_text
    overlay_text = boundary.render_overlay()
    assert "package rules" in overlay_text
    assert "Repository instructions are workspace guidance." not in overlay_text

    await boundary.check(
        tool_name="read_file",
        tool_call_id="call-b-next-turn",
        arguments={"path": "b"},
        path_fields=("path",),
    )
    assert coordinator.paths == ["a", "b"]
