#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporary focused patcher for prompt instruction ordering."""

from pathlib import Path


def replace_once(text, old, new, label):
    if text.count(old) != 1:
        raise RuntimeError("unexpected %s count: %s" % (label, text.count(old)))
    return text.replace(old, new, 1)


workspace_path = Path("linktools-ai/src/linktools/ai/capability/_workspace.py")
workspace = workspace_path.read_text(encoding="utf-8")
workspace = replace_once(
    workspace,
    '''        super().__init__(
            id=_WORKSPACE_SANDBOX_CAPABILITY_ID,
            instructions=_WORKSPACE_INSTRUCTIONS,
        )
''',
    '''        super().__init__(id=_WORKSPACE_SANDBOX_CAPABILITY_ID)
''',
    "workspace toolset instructions",
)
workspace = replace_once(
    workspace,
    '''    def get_toolset(self) -> _WorkspaceSandboxToolset:
        return _WorkspaceSandboxToolset(
''',
    '''    def get_instructions(self) -> InstructionPart:
        return _WORKSPACE_INSTRUCTIONS

    def get_toolset(self) -> _WorkspaceSandboxToolset:
        return _WorkspaceSandboxToolset(
''',
    "workspace capability instructions",
)
workspace_path.write_text(workspace, encoding="utf-8")


executor_path = Path("linktools-ai/src/linktools/ai/runtime/_agent_executor.py")
executor = executor_path.read_text(encoding="utf-8")
executor = replace_once(
    executor,
    '''from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
''',
    '''from pydantic_ai.capabilities import (
    AbstractCapability,
    Capability,
    CapabilityOrdering,
''',
    "capability import",
)
executor = replace_once(
    executor,
    '''        capabilities.append(skill_capability)
    if scope.subagent_available and scope.binding.snapshot.subagents:
''',
    '''        capabilities.append(skill_capability)

    workspace_capability_values = workspace_capabilities(
        scope.context.workspace,
        workspace_names,
        session=scope.sandbox_session,
        vision=definition.model.vision,
    )
    if workspace_capability_values:
        workspace_guidance = workspace_capability_values[0].get_instructions()
        if workspace_guidance is not None:
            capabilities.append(
                Capability(
                    id="linktools.ai.workspace-guidance",
                    instructions=workspace_guidance,
                )
            )

    if scope.subagent_available and scope.binding.snapshot.subagents:
''',
    "workspace guidance placement",
)
executor = replace_once(
    executor,
    '''    raw_toolsets: list[AbstractToolset[AgentContext[object]]] = []
    workspace_toolsets = workspace_capabilities(
        scope.context.workspace,
        workspace_names,
        session=scope.sandbox_session,
        vision=definition.model.vision,
    )
    workspace_toolset_values = tuple(
        toolset
        for capability in workspace_toolsets
        if (toolset := capability.get_toolset()) is not None
    )
''',
    '''    raw_toolsets: list[AbstractToolset[AgentContext[object]]] = []
    workspace_toolset_values = tuple(
        toolset
        for capability in workspace_capability_values
        if (toolset := capability.get_toolset()) is not None
    )
''',
    "workspace toolset reuse",
)
executor = replace_once(
    executor,
    '''    capabilities.extend(
        cast("tuple[AbstractCapability[AgentContext[object]], ...]", platform)
    )

    business_output_type: object
''',
    '''    capabilities.extend(
        cast("tuple[AbstractCapability[AgentContext[object]], ...]", platform)
    )

    repository_initial_instructions = (
        repository_boundary.render_initial()
        if repository_boundary is not None
        else ""
        if scope.repository_instructions is None
        else scope.repository_instructions.render()
    )
    if repository_initial_instructions:
        capabilities.append(
            Capability(
                id="linktools.ai.repository-initial",
                instructions=InstructionPart(
                    content=repository_initial_instructions,
                    dynamic=False,
                ),
            )
        )

    business_output_type: object
''',
    "repository initial capability",
)
old_runtime = '''    runtime_instructions: list[Any] = []
    if base_instructions:
        runtime_instructions.append(base_instructions)
    if repository_boundary is None:
        repository_instructions = (
            ""
            if scope.repository_instructions is None
            else scope.repository_instructions.render()
        )
        if repository_instructions:
            runtime_instructions.append(
                InstructionPart(
                    content=repository_instructions,
                    name="repository-initial",
                    dynamic=False,
                )
            )
    else:
        initial_repository_instructions = repository_boundary.render_initial()
        if initial_repository_instructions:
            runtime_instructions.append(
                InstructionPart(
                    content=initial_repository_instructions,
                    name="repository-initial",
                    dynamic=False,
                )
            )

        def repository_overlay(
            _: PydanticRunContext[object],
        ) -> str:
            return repository_boundary.render_overlay()

        runtime_instructions.append(repository_overlay)
'''
new_runtime = '''    runtime_instructions: list[Any] = []
    if base_instructions:
        runtime_instructions.append(base_instructions)
    if repository_boundary is not None:

        def repository_overlay(
            _: PydanticRunContext[object],
        ) -> str:
            return repository_boundary.render_overlay()

        runtime_instructions.append(repository_overlay)
'''
executor = replace_once(executor, old_runtime, new_runtime, "runtime instruction partition")
executor_path.write_text(executor, encoding="utf-8")


test_path = Path("tests/ai/test_system_instruction_composition.py")
test = test_path.read_text(encoding="utf-8")
test = replace_once(
    test,
    '''import pytest
from linktools.ai.capability import ToolCallRetry, workspace_capabilities
''',
    '''import pytest
from linktools.ai.capability import ToolCallRetry, workspace_capabilities
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.capabilities import AbstractCapability, Capability
''',
    "test pydantic imports",
)
test = replace_once(
    test,
    '''from pydantic_ai.models.test import TestModel
''',
    '''from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.test import TestModel
''',
    "test model request import",
)
old_workspace_test = '''        capability = workspace_capabilities(
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
'''
new_workspace_test = '''        capability = workspace_capabilities(
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
'''
test = replace_once(test, old_workspace_test, new_workspace_test, "workspace guidance test")
insert_marker = '''

@pytest.mark.asyncio
async def test_repository_partition_is_stable_and_fences_same_turn_siblings() -> None:
'''
order_test = '''

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
'''
if test.count(insert_marker) != 1:
    raise RuntimeError("unexpected repository test marker count")
test = test.replace(insert_marker, order_test + insert_marker, 1)
test_path.write_text(test, encoding="utf-8")
