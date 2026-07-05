#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/agent/test_compiler.py"""
import pytest
from pydantic_ai import Agent as PydanticAgent

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.models import CompiledAgent
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.core.model_runtime import ModelRegistry, RuntimeModelConfig
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.router import ModelRouter


def _config(model_type: str) -> RuntimeModelConfig:
    return RuntimeModelConfig(
        model_type=model_type, protocol="openai", model="gpt-4", base_url="http://localhost",
        api_key="test-key", auth_token=None, timeout_seconds=30, raw={},
    )


@pytest.mark.asyncio
async def test_compile_produces_a_compiled_agent_with_no_runtime_state():
    registry = ModelRegistry()
    registry.register("test-model", config=_config("test-model"))
    router = ModelRouter(registry=registry)
    compiler = AgentCompiler(model_router=router)

    spec = AgentSpec(
        id="agent-1", name="test-agent", model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="You are a test agent."),
    )
    compiled = await compiler.compile(spec)

    assert isinstance(compiled, CompiledAgent)
    assert compiled.spec is spec
    assert isinstance(compiled.pydantic_agent, PydanticAgent)
    assert compiled.policy_capability.current_context is None
    assert not hasattr(compiled, "session")
    assert not hasattr(compiled, "workdir")


@pytest.mark.asyncio
async def test_compile_reuses_model_router_fallback():
    registry = ModelRegistry()
    registry.register("fallback-model", config=_config("fallback-model"))
    router = ModelRouter(registry=registry)
    compiler = AgentCompiler(model_router=router)

    spec = AgentSpec(
        id="agent-2", name="test-agent-2", model=ModelPolicy(primary="missing", fallbacks=("fallback-model",)),
        instructions=PromptSpec(instructions="hi"),
    )
    compiled = await compiler.compile(spec)
    assert compiled.model_bundle.config.model_type == "fallback-model"


@pytest.mark.asyncio
async def test_compile_wires_middleware_capability_when_pipeline_provided():
    from linktools.ai.middleware.capability import MiddlewareCapability
    from linktools.ai.middleware.pipeline import MiddlewarePipeline

    registry = ModelRegistry()
    registry.register("test-model", config=_config("test-model"))
    router = ModelRouter(registry=registry)
    pipeline = MiddlewarePipeline(middlewares=())
    compiler = AgentCompiler(model_router=router, middleware_pipeline=pipeline)
    spec = AgentSpec(
        id="agent-mw", name="mw-agent", model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
    )
    compiled = await compiler.compile(spec)
    assert isinstance(compiled.middleware_capability, MiddlewareCapability)
    assert compiled.middleware_capability.current_context is None
    # Both capabilities must end up on the pydantic-ai Agent. In pydantic-ai
    # 1.107 capabilities are nested under root_capability.capabilities (a list).
    capability_types = {type(c).__name__ for c in compiled.pydantic_agent.root_capability.capabilities}
    assert "PolicyCapability" in capability_types
    assert "MiddlewareCapability" in capability_types


@pytest.mark.asyncio
async def test_compile_leaves_middleware_capability_none_when_no_pipeline():
    registry = ModelRegistry()
    registry.register("test-model", config=_config("test-model"))
    router = ModelRouter(registry=registry)
    compiler = AgentCompiler(model_router=router)
    spec = AgentSpec(
        id="agent-nomw", name="nomw-agent", model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
    )
    compiled = await compiler.compile(spec)
    assert compiled.middleware_capability is None
