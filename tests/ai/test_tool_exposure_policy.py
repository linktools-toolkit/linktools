#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CapabilityToolExposurePolicy (contract): conservative defaults + immutability."""

from linktools.ai.capability.exposure import CapabilityToolExposurePolicy


def test_defaults_are_conservative():
    p = CapabilityToolExposurePolicy()
    assert p.expose_prompt_catalog is True
    assert p.expose_discovery_tools is True
    # Execution tools must NOT be on by default.
    assert p.expose_execution_tools is False
    assert p.max_tools_total == 64
    assert p.max_tools_per_capability == 16
    assert p.max_resources_per_list == 50
    assert p.max_read_bytes == 65536
    assert p.max_entrypoints_per_extension == 20
    assert p.allowed_entrypoint_kinds == ("agent",)
    assert p.require_explicit_entrypoint_allowlist is True


def test_policy_is_frozen():
    import pytest

    p = CapabilityToolExposurePolicy()
    with pytest.raises(Exception):
        p.expose_execution_tools = True  # type: ignore[misc]


def test_policy_overridable_via_constructor():
    p = CapabilityToolExposurePolicy(expose_execution_tools=True, max_tools_total=8)
    assert p.expose_execution_tools is True
    assert p.max_tools_total == 8


# --- is_descriptor_exposable: the centralized gate every Provider goes through ---


def _descriptor(category, mutating):
    from linktools.ai.tool.models import ToolDescriptor

    return ToolDescriptor(
        name="t",
        source="test",
        category=category,
        risk="low",
        mutating=mutating,
    )


def test_discovery_category_gated_by_expose_discovery_tools():
    from linktools.ai.capability.exposure import is_descriptor_exposable

    on = CapabilityToolExposurePolicy(expose_discovery_tools=True)
    off = CapabilityToolExposurePolicy(expose_discovery_tools=False)
    d = _descriptor("discovery", mutating=False)
    assert is_descriptor_exposable(d, on) is True
    assert is_descriptor_exposable(d, off) is False


def test_mutating_tool_gated_by_expose_execution_tools_regardless_of_category():
    from linktools.ai.capability.exposure import is_descriptor_exposable

    off = CapabilityToolExposurePolicy(expose_execution_tools=False)
    on = CapabilityToolExposurePolicy(expose_execution_tools=True)
    for category in (
        "terminal",
        "file-write",
        "subagent",
        "mcp-write",
        "extension-execute",
    ):
        d = _descriptor(category, mutating=True)
        assert is_descriptor_exposable(d, off) is False, category
        assert is_descriptor_exposable(d, on) is True, category


def test_non_discovery_non_mutating_tool_always_exposed():
    from linktools.ai.capability.exposure import is_descriptor_exposable

    off = CapabilityToolExposurePolicy(
        expose_discovery_tools=False, expose_execution_tools=False
    )
    d = _descriptor("file-read", mutating=False)
    assert is_descriptor_exposable(d, off) is True


# --- Resolver-level enforcement for providers other than Builtin (MCP, subagent) ---

import pytest  # noqa: E402

from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef  # noqa: E402
from linktools.ai.capability.provider import CapabilityContext  # noqa: E402
from linktools.ai.capability.resolver import CapabilityResolver  # noqa: E402
from linktools.ai.capability.models import CapabilityBundle  # noqa: E402
from linktools.ai.model.policy import ModelPolicy  # noqa: E402
from linktools.ai.tool.models import ToolDescriptor  # noqa: E402
from linktools.ai.tool.models import ToolContribution, declared_tool_definitions  # noqa: E402


class _FakeMutatingProvider:
    """Stands in for MCP/subagent-style providers: one mutating tool."""

    kind = "mcp"

    async def resolve(self, ref, context):
        async def handler(**kw):
            return "ran"

        from pydantic_ai.toolsets import FunctionToolset

        ts = FunctionToolset()
        ts.add_function(handler, name="risky_call")
        descriptor = ToolDescriptor(
            name="risky_call",
            source="mcp",
            category="mcp-write",
            risk="high",
            mutating=True,
        )
        return CapabilityBundle(
            tool_contributions=(
                ToolContribution(tools=declared_tool_definitions(ts, (descriptor,))),
            ),
        )


def _spec():
    return AgentSpec(
        id="a1",
        name="a1",
        model=ModelPolicy(primary="m"),
        instructions=PromptSpec(instructions="hi"),
        tools=(ToolRef(name="x", kind="mcp"),),
    )


@pytest.mark.asyncio
async def test_mutating_mcp_tool_hidden_by_default_policy():
    from linktools.ai.capability.exposure import CapabilityToolExposurePolicy as Policy

    asm = CapabilityResolver({"mcp": _FakeMutatingProvider()})
    ctx = CapabilityContext(agent_id="a1", exposure_policy=Policy())
    bundle = await asm.resolve(_spec(), ctx)
    assert bundle.tool_contributions == ()


@pytest.mark.asyncio
async def test_mutating_mcp_tool_exposed_when_execution_tools_allowed():
    from linktools.ai.capability.exposure import CapabilityToolExposurePolicy as Policy

    asm = CapabilityResolver({"mcp": _FakeMutatingProvider()})
    ctx = CapabilityContext(
        agent_id="a1", exposure_policy=Policy(expose_execution_tools=True)
    )
    bundle = await asm.resolve(_spec(), ctx)
    names = {md.descriptor.name for c in bundle.tool_contributions for md in c.tools}
    assert names == {"risky_call"}
