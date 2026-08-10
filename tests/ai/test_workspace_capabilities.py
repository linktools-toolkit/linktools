#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Workspace-local Agent Skills and sub-agent discovery."""

from pathlib import Path

import pytest
from linktools.ai.agent import (
    AgentCatalogSnapshot,
    AgentRunScope,
    compose_platform_capabilities,
)
from linktools.ai.capability import SkillCapability
from linktools.ai.workspace import build_workspace_capabilities
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence import InMemoryStepStore
from pydantic_ai_harness.subagents import SubAgents


def test_workspace_capabilities_load_project_skills(tmp_path: Path) -> None:
    skill_file = tmp_path / ".linktools" / "skills" / "review" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        "---\nname: review\ndescription: Review project changes.\n---\n\nReview the diff.\n",
        encoding="utf-8",
    )

    capabilities = build_workspace_capabilities(tmp_path)

    assert any(isinstance(capability, SkillCapability) for capability in capabilities)


@pytest.mark.asyncio
async def test_skill_prompt_and_tools_use_skill_semantics(tmp_path: Path) -> None:
    skill_file = tmp_path / ".linktools" / "skills" / "review" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        "---\nname: review\ndescription: Review project changes.\n---\n\nReview the diff.\n",
        encoding="utf-8",
    )

    capability = next(
        capability for capability in build_workspace_capabilities(tmp_path) if isinstance(capability, SkillCapability)
    )
    prompt: list[str | None] = []
    tool_names: list[str] = []

    async def model(messages, info):
        del messages
        prompt.append(info.instructions)
        tool_names.extend(tool.name for tool in info.function_tools)
        from pydantic_ai.messages import ModelResponse, TextPart

        return ModelResponse(parts=[TextPart(content="ok")])

    from pydantic_ai import Agent
    from pydantic_ai.models.function import FunctionModel

    await Agent(FunctionModel(model), capabilities=[capability]).run("review this change")

    assert prompt == [(
        "The following skills are available for this agent run.\n"
        "Use the `load_skill` tool to load the full instructions for a skill when it is relevant.\n"
        "- review: Review project changes."
    )]
    assert set(tool_names) == {"list_skills", "load_skill"}
    assert "capability" not in prompt[0].lower()


@pytest.mark.asyncio
async def test_platform_capabilities_load_project_subagents(tmp_path: Path) -> None:
    agent_file = tmp_path / ".linktools" / "agents" / "reviewer.md"
    agent_file.parent.mkdir(parents=True)
    agent_file.write_text(
        "---\nname: reviewer\ndescription: Review project changes.\n---\n\nReview the diff.\n",
        encoding="utf-8",
    )

    capabilities = await compose_platform_capabilities(
        AgentRunScope(
            root=tmp_path,
            agent_name="main",
            conversation_id=None,
            step_run_id="run",
            segment_sequence=1,
            memory_namespace=None,
            step_store=InMemoryStepStore(),
            inherited_capabilities=(),
            agent_catalog=AgentCatalogSnapshot(()),
            memory_store=None,
        ),
        model_factory=lambda value: TestModel(call_tools=[]),
        parent_model=TestModel(call_tools=[]),
    )

    assert any(isinstance(capability, SubAgents) for capability in capabilities)
