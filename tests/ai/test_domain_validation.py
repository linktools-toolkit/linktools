#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Domain-model invariants enforced at construction (WP-09). A custom provider
can build these directly, bypassing the registry parser, so each model validates
its own contract + deep-freezes its mappings."""

import math
from decimal import Decimal

import pytest

from linktools.ai.agent.spec import AgentSpec, MiddlewareRef, PromptSpec, ToolRef
from linktools.ai.capability.models import CapabilityBundle, CapabilityRef
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.tool.models import ToolDescriptor


# --- ModelPolicy -----------------------------------------------------------


def _policy(**kw) -> ModelPolicy:
    base = {"primary": "gpt"}
    base.update(kw)
    return ModelPolicy(**base)


def test_model_policy_valid_minimal():
    p = ModelPolicy(primary="gpt")
    assert p.primary == "gpt"


def test_model_policy_blank_primary_rejected():
    with pytest.raises(ValueError):
        ModelPolicy(primary="  ")


def test_model_policy_negative_max_retries_rejected():
    with pytest.raises(ValueError):
        _policy(max_retries=-1)


def test_model_policy_bool_max_retries_rejected():
    with pytest.raises(ValueError):
        _policy(max_retries=True)


@pytest.mark.parametrize("bad", [0, -1, math.nan, math.inf])
def test_model_policy_bad_timeout_rejected(bad):
    with pytest.raises(ValueError):
        _policy(timeout_seconds=bad)


def test_model_policy_non_decimal_budget_rejected():
    with pytest.raises(TypeError):
        _policy(budget=1.50)


def test_model_policy_negative_budget_rejected():
    with pytest.raises(ValueError):
        _policy(budget=Decimal("-1"))


# --- AgentSpec / PromptSpec / ToolRef / MiddlewareRef ----------------------


def _agent(**kw) -> AgentSpec:
    base = {
        "id": "a",
        "name": "a",
        "model": ModelPolicy(primary="gpt"),
        "instructions": PromptSpec(instructions="hi"),
    }
    base.update(kw)
    return AgentSpec(**base)


def test_agent_spec_valid_minimal():
    a = _agent()
    assert a.id == "a"


def test_agent_spec_blank_id_and_name_rejected():
    with pytest.raises(ValueError):
        _agent(id="  ")
    with pytest.raises(ValueError):
        _agent(name="")


def test_agent_spec_wrong_model_type_rejected():
    with pytest.raises(TypeError):
        _agent(model="gpt")


def test_agent_spec_wrong_instructions_type_rejected():
    with pytest.raises(TypeError):
        _agent(instructions="hi")


def test_agent_spec_tools_wrong_element_rejected():
    with pytest.raises(TypeError):
        _agent(tools=("not-a-toolref",))


def test_agent_spec_metadata_frozen_after_construction():
    src = {"k": "v"}
    a = _agent(metadata=src)
    src["k"] = "mutated"
    src["new"] = "x"
    assert a.metadata == {"k": "v"}  # source mutation does not leak in


def test_prompt_spec_sections_frozen():
    src = {"a": "b"}
    p = PromptSpec(instructions="hi", sections=src)
    src["a"] = "z"
    assert p.sections == {"a": "b"}


def test_tool_ref_blank_kind_and_name_rejected():
    with pytest.raises(ValueError):
        ToolRef(kind="  ", name="n")
    with pytest.raises(ValueError):
        ToolRef(kind="k", name="")


def test_tool_ref_config_frozen():
    src = {"x": 1}
    t = ToolRef(kind="k", name="n", config=src)
    src["x"] = 2
    assert t.config == {"x": 1}


def test_middleware_ref_blank_name_rejected():
    with pytest.raises(ValueError):
        MiddlewareRef(name="  ")


# --- ToolDescriptor --------------------------------------------------------


def _descriptor(**kw) -> ToolDescriptor:
    base = {
        "name": "t",
        "source": "test",
        "category": "misc",
        "risk": "low",
        "mutating": False,
    }
    base.update(kw)
    return ToolDescriptor(**base)


def test_tool_descriptor_valid():
    assert _descriptor().name == "t"


@pytest.mark.parametrize("field", ["name", "source", "category", "risk"])
def test_tool_descriptor_blank_field_rejected(field):
    with pytest.raises(ValueError):
        _descriptor(**{field: "  "})


def test_tool_descriptor_mutating_must_be_bool():
    with pytest.raises(TypeError):
        _descriptor(mutating="yes")


def test_tool_descriptor_metadata_frozen():
    src = {"k": 1}
    d = _descriptor(metadata=src)
    src["k"] = 2
    assert d.metadata == {"k": 1}


# --- CapabilityRef / CapabilityBundle --------------------------------------


def test_capability_ref_blank_kind_and_name_rejected():
    with pytest.raises(ValueError):
        CapabilityRef(kind="  ", name="n")
    with pytest.raises(ValueError):
        CapabilityRef(kind="k", name="")


def test_capability_ref_config_frozen():
    src = {"x": 1}
    r = CapabilityRef(kind="k", name="n", config=src)
    src["x"] = 2
    assert r.config == {"x": 1}


def test_capability_bundle_tool_contributions_must_be_tuple():
    with pytest.raises(TypeError):
        CapabilityBundle(tool_contributions=[1, 2])
