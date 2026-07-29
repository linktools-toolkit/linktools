#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JSON round-trip AgentSpecCodec: encode/decode must be lossless
for model policy, instructions + sections, tools, middleware, metadata, and the
structured output type (by registry ref). A structured output that is not
registered must fail rather than be restored via an empty create_model."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from linktools.ai.agent.codec import AgentSpecCodec, OutputTypeRegistry
from linktools.ai.agent.spec import AgentSpec, MiddlewareRef, PromptSpec
from linktools.ai.errors import InvalidSpecError
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.tool.models import ToolRef


def _spec(*, output_schema=None, tools=None) -> AgentSpec:
    return AgentSpec(
        id="agent-1",
        name="Agent One",
        model=ModelPolicy(primary="gpt-test", fallbacks=("gpt-fallback",)),
        instructions=PromptSpec(instructions="do the thing", sections={"safety": "be careful"}),
        tools=tools,
        middleware=(MiddlewareRef(name="logger", config={"level": "info"}),),
        output_schema=output_schema,
        metadata={"env": "test", "count": 3},
    )


def test_codec_round_trips_every_field() -> None:
    spec = _spec(tools=(ToolRef(kind="builtin", name="shell"),))
    codec = AgentSpecCodec()
    decoded = codec.decode(codec.encode(spec))
    assert decoded.id == spec.id
    assert decoded.name == spec.name
    assert decoded.model == spec.model
    assert decoded.instructions.instructions == spec.instructions.instructions
    assert dict(decoded.instructions.sections) == dict(spec.instructions.sections)
    assert decoded.tools is not None and len(decoded.tools) == 1
    assert decoded.tools[0].kind == "builtin" and decoded.tools[0].name == "shell"
    assert decoded.middleware == spec.middleware
    assert dict(decoded.metadata) == dict(spec.metadata)
    assert decoded.output_schema is None


def test_codec_round_trips_tools_three_state() -> None:
    codec = AgentSpecCodec()
    # None (unset) stays None; () (explicit empty) stays ().
    assert codec.decode(codec.encode(_spec(tools=None))).tools is None
    explicit_empty = codec.decode(codec.encode(_spec(tools=())))
    assert explicit_empty.tools == ()


def test_codec_round_trips_structured_output_via_registry() -> None:
    class Out(BaseModel):
        answer: str

    registry = OutputTypeRegistry()
    registry.register("out.v1", Out)
    codec = AgentSpecCodec(output_types=registry)
    spec = _spec(output_schema=Out)
    decoded = codec.decode(codec.encode(spec))
    assert decoded.output_schema is Out


def test_codec_rejects_unregistered_structured_output() -> None:
    class Out(BaseModel):
        answer: str

    codec = AgentSpecCodec(OutputTypeRegistry())  # Out is NOT registered
    with pytest.raises(InvalidSpecError):
        codec.encode(_spec(output_schema=Out))


def test_codec_rejects_unknown_output_ref_on_decode() -> None:
    codec = AgentSpecCodec(OutputTypeRegistry())
    encoded = {"schema": "agent-spec.v1", "id": "a", "name": "a",
               "model": {"primary": "m", "fallbacks": []},
               "instructions": {"instructions": "x", "sections": {}},
               "tools": None, "middleware": [], "output_ref": "missing.v1", "metadata": {}}
    with pytest.raises(InvalidSpecError):
        codec.decode(encoded)


def test_output_registry_rejects_conflicting_registration() -> None:
    class A(BaseModel):
        x: int

    class B(BaseModel):
        y: int

    registry = OutputTypeRegistry()
    registry.register("a.v1", A)
    # Same ref+type is idempotent.
    registry.register("a.v1", A)
    # Different type under the same ref, or same type under a different ref, conflicts.
    with pytest.raises(InvalidSpecError):
        registry.register("a.v1", B)
    with pytest.raises(InvalidSpecError):
        registry.register("a.v2", A)
