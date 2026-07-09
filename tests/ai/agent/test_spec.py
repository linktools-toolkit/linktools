#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/agent/test_spec.py"""
from linktools.ai.agent.spec import AgentSpec, MiddlewareRef, PromptSpec, ToolRef
from linktools.ai.model.policy import ModelPolicy


def test_agent_spec_construction():
    spec = AgentSpec(
        id="agent-1", name="security-agent", model=ModelPolicy(primary="gpt-4"),
        instructions=PromptSpec(instructions="You are a security analyst."),
        tools=(ToolRef(name="file"), ToolRef(name="terminal")),
        middleware=(MiddlewareRef(name="budget", config={"budget_usd": 5.0}),),
    )
    assert spec.id == "agent-1"
    assert spec.model.primary == "gpt-4"
    assert len(spec.tools) == 2
    assert spec.middleware[0].name == "budget"
    assert spec.output_schema is None
    assert dict(spec.metadata) == {}


def test_agent_spec_is_frozen():
    import pytest
    spec = AgentSpec(
        id="agent-1", name="a", model=ModelPolicy(primary="gpt-4"),
        instructions=PromptSpec(instructions="hi"),
    )
    with pytest.raises(Exception):
        spec.id = "agent-2"


def test_prompt_spec_defaults():
    prompt = PromptSpec(instructions="hi")
    assert dict(prompt.sections) == {}


def test_tool_ref_and_middleware_ref():
    assert ToolRef(name="file").name == "file"
    ref = MiddlewareRef(name="budget")
    assert ref.name == "budget"
    assert dict(ref.config) == {}
