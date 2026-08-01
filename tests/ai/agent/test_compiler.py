#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentCompiler: resolve a ModelPolicy and build the underlying pydantic-ai Agent.

Focuses on the model-layer wiring the compiler owns -- in particular that
``ResolvedModel.output_retries`` reaches ``Agent(retries=...)`` (the
Agent-layer structured-output retry count, distinct from the HTTP-client
``request_retries``)."""

import asyncio

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.resolver import ModelResolver


def _fn(text: str = "ok"):
    def _f(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=text)])

    return _f


def _spec() -> AgentSpec:
    return AgentSpec(
        id="a",
        name="a",
        model=ModelPolicy(primary="m"),
        instructions=PromptSpec(instructions="hi"),
    )


def test_compile_wires_output_retries_into_agent():
    # resolved.output_retries flows into Agent(retries=...), which sets both the
    # output and tool-call retry ceilings.
    registry = ModelRegistry()
    registry.register("m", model=FunctionModel(_fn()), output_retries=3)
    compiler = AgentCompiler(model_resolver=ModelResolver(registry=registry))
    compiled = asyncio.run(compiler.compile(_spec()))
    pydantic_agent = compiled.pydantic_agent
    assert pydantic_agent._max_output_retries == 3
    assert pydantic_agent._max_tool_retries == 3


def test_compile_defaults_to_one_retry_when_unconfigured():
    # The default output_retries is 1 (pydantic-ai's own Agent default), so a
    # never-configured caller sees no behavior change.
    registry = ModelRegistry()
    registry.register("m", model=FunctionModel(_fn()))
    compiler = AgentCompiler(model_resolver=ModelResolver(registry=registry))
    compiled = asyncio.run(compiler.compile(_spec()))
    pydantic_agent = compiled.pydantic_agent
    assert pydantic_agent._max_output_retries == 1
    assert pydantic_agent._max_tool_retries == 1
