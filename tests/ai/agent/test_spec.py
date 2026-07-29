#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/agent/test_spec.py"""

from linktools.ai.agent.spec import AgentSpec, MiddlewareRef, PromptSpec
from linktools.ai.agent.assembly import AgentFeatureRef
from linktools.ai.model.policy import ModelPolicy


def test_agent_spec_construction():
    spec = AgentSpec(
        id="agent-1",
        name="security-agent",
        model=ModelPolicy(primary="gpt-4"),
        instructions=PromptSpec(instructions="You are a security analyst."),
        features=(
            AgentFeatureRef(kind="builtin", name="file"),
            AgentFeatureRef(kind="builtin", name="terminal"),
        ),
        middleware=(MiddlewareRef(name="budget", config={"budget_usd": 5.0}),),
    )
    assert spec.id == "agent-1"
    assert spec.model.primary == "gpt-4"
    assert len(spec.features) == 2
    assert spec.middleware[0].name == "budget"
    assert spec.output_schema is None
    assert dict(spec.metadata) == {}


def test_agent_spec_is_frozen():
    import pytest

    spec = AgentSpec(
        id="agent-1",
        name="a",
        model=ModelPolicy(primary="gpt-4"),
        instructions=PromptSpec(instructions="hi"),
    )
    with pytest.raises(Exception):
        spec.id = "agent-2"


def test_prompt_spec_defaults():
    prompt = PromptSpec(instructions="hi")
    assert dict(prompt.sections) == {}


def test_tool_ref_and_middleware_ref():
    assert AgentFeatureRef(kind="builtin", name="file").name == "file"
    ref = MiddlewareRef(name="budget")
    assert ref.name == "budget"
    assert dict(ref.config) == {}


def test_tool_ref_kind_and_config_defaults():
    assert dict(AgentFeatureRef(kind="builtin", name="file").config) == {}
    structured = AgentFeatureRef(name="sql", kind="skill", config={"limit": 5})
    assert structured.kind == "skill"
    assert structured.config == {"limit": 5}


def test_parse_agent_feature_refs_handles_kind_name_strings_and_struct():
    from linktools.ai.agent.assembly import parse_agent_feature_refs

    import pytest

    with pytest.raises(Exception):
        parse_agent_feature_refs(["file"])
    with pytest.raises(Exception):
        parse_agent_feature_refs(["skill:sql"])
    # structured mapping
    (struct,) = parse_agent_feature_refs([{"kind": "mcp", "name": "risk", "config": {"k": 1}}])
    assert struct.name == "risk" and struct.kind == "mcp" and struct.config == {"k": 1}
    with pytest.raises(Exception):
        parse_agent_feature_refs([{"name": "file"}])


def test_parse_agent_feature_refs_rejects_bad_shapes():
    import pytest
    from linktools.ai.errors import InvalidSpecError
    from linktools.ai.agent.assembly import parse_agent_feature_refs

    with pytest.raises((TypeError, ValueError)):
        parse_agent_feature_refs([":file"])  # empty kind
    with pytest.raises((TypeError, ValueError)):
        parse_agent_feature_refs([{"kind": "skill"}])  # missing name
    with pytest.raises((TypeError, ValueError)):
        parse_agent_feature_refs(42)
