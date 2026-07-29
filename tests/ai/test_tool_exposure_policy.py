from dataclasses import dataclass

import pytest

from linktools.ai.agent.assembly import (
    AgentAssembler,
    AgentContribution,
    AgentFeatureContext,
    AgentFeatureRef,
    AgentFeatureRegistry,
)
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.agent.tool.execution.schema import JsonSchemaToolValidator
from linktools.ai.agent.tool.exposure import ToolAssembler, ToolExposurePolicy
from linktools.ai.agent.tool.models import (
    ToolCategory,
    ToolDefinition,
    ToolDescriptor,
    ToolSource,
)
from linktools.ai.governance.policy.rule import RiskLevel, SideEffectKind
from linktools.ai.model.policy import ModelPolicy


async def _handler(**arguments):
    return arguments


def _definition(ref: AgentFeatureRef, *, mutating: bool) -> ToolDefinition:
    return ToolDefinition(
        descriptor=ToolDescriptor(
            name="risky_call",
            source=ToolSource.MCP,
            category=(
                ToolCategory.NETWORK_WRITE
                if mutating
                else ToolCategory.NETWORK_READ
            ),
            risk=RiskLevel.HIGH,
            side_effect=(
                SideEffectKind.NAMESPACE_MUTATING
                if mutating
                else SideEffectKind.READ_ONLY
            ),
            feature=ref,
        ),
        handler=_handler,
        input_schema={"type": "object"},
    )


@dataclass
class _Provider:
    supported_kinds = ("mcp",)

    async def resolve(self, ref, context):
        return AgentContribution(tools=(_definition(ref, mutating=True),))


def _assembler(policy: ToolExposurePolicy) -> AgentAssembler:
    registry = AgentFeatureRegistry()
    registry.register(_Provider())
    registry.freeze()
    return AgentAssembler(
        registry=registry,
        tool_assembler=ToolAssembler(
            exposure=policy,
            schema_validator=JsonSchemaToolValidator(),
        ),
    )


def _spec() -> AgentSpec:
    return AgentSpec(
        id="a1",
        name="a1",
        model=ModelPolicy(primary="m"),
        instructions=PromptSpec(instructions="hi"),
        features=(AgentFeatureRef(name="x", kind="mcp"),),
    )


def _context() -> AgentFeatureContext:
    return AgentFeatureContext(
        agent_id="a1",
        execution_id="e1",
        root_execution_id="e1",
        parent_execution_id=None,
        session_id="s1",
        tenant_id="t1",
        user_id="u1",
        workspace=None,
        sandbox=None,
    )


def test_defaults_are_conservative():
    policy = ToolExposurePolicy()
    assert policy.expose_discovery_tools is True
    assert policy.expose_execution_tools is False
    assert policy.max_tools_total == 64
    assert policy.max_tools_per_feature == 16


@pytest.mark.asyncio
async def test_mutating_tool_hidden_by_default():
    assembly = await _assembler(ToolExposurePolicy()).assemble(_spec(), _context())
    assert assembly.tools == ()


@pytest.mark.asyncio
async def test_mutating_tool_exposed_when_enabled():
    assembly = await _assembler(
        ToolExposurePolicy(expose_execution_tools=True)
    ).assemble(_spec(), _context())
    assert [definition.descriptor.name for definition in assembly.tools] == [
        "risky_call"
    ]
