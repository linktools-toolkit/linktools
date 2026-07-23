#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/agent/test_compiler.py"""

import pytest
from pydantic_ai import Agent as PydanticAgent

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.dependencies import AgentDependencies
from linktools.ai.agent.models import CompiledAgent
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.model.registry import ModelRegistry, RuntimeModelConfig
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.resolver import ModelResolver
from linktools.ai.governance.policy.engine import PolicyEngine
from linktools.ai.tool.executor import GovernedToolInvoker


def _config(model_type: str) -> RuntimeModelConfig:
    return RuntimeModelConfig(
        model_type=model_type,
        protocol="openai",
        model="gpt-4",
        base_url="http://localhost",
        api_key="test-key",
        auth_token=None,
        timeout_seconds=30,
        raw={},
    )


@pytest.mark.asyncio
async def test_compile_produces_a_compiled_agent_with_no_runtime_state():
    registry = ModelRegistry()
    registry.register("test-model", config=_config("test-model"))
    router = ModelResolver(registry=registry)
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())), model_resolver=router
    )

    spec = AgentSpec(
        id="agent-1",
        name="test-agent",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="You are a test agent."),
    )
    compiled = await compiler.compile(spec)

    assert isinstance(compiled, CompiledAgent)
    assert compiled.spec is spec
    assert isinstance(compiled.pydantic_agent, PydanticAgent)
    # Capabilities carry no mutable per-Run state -- the
    # ToolContext arrives via pydantic-ai DI (ctx.deps.tool_context), so there
    # is no current_context field to assert is None. The deps_type the agent
    # was compiled with IS AgentDependencies (the gate the runner relies on).
    assert not hasattr(compiled.policy_capability, "current_context")
    assert compiled.pydantic_agent.deps_type is AgentDependencies
    assert not hasattr(compiled, "session")
    assert not hasattr(compiled, "workdir")


@pytest.mark.asyncio
async def test_compile_wires_spec_instructions_into_pydantic_agent():
    # PromptSpec.instructions is the agent's declared system prompt; it must
    # actually reach the underlying pydantic-ai Agent, not just round-trip
    # through AgentSpec/RunDefinitionSnapshot serialization. pydantic-ai has
    # no public getter for the configured static instructions, so this reads
    # the private `_instructions` list -- the only way to observe what was
    # passed to the constructor.
    registry = ModelRegistry()
    registry.register("test-model", config=_config("test-model"))
    router = ModelResolver(registry=registry)
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())), model_resolver=router
    )

    spec = AgentSpec(
        id="agent-1",
        name="test-agent",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="You are a very specific test agent."),
    )
    compiled = await compiler.compile(spec)

    assert compiled.pydantic_agent._instructions == [
        "You are a very specific test agent."
    ]


@pytest.mark.asyncio
async def test_compile_reuses_model_resolver_fallback():
    registry = ModelRegistry()
    registry.register("fallback-model", config=_config("fallback-model"))
    router = ModelResolver(registry=registry)
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())), model_resolver=router
    )

    spec = AgentSpec(
        id="agent-2",
        name="test-agent-2",
        model=ModelPolicy(primary="missing", fallbacks=("fallback-model",)),
        instructions=PromptSpec(instructions="hi"),
    )
    compiled = await compiler.compile(spec)
    # primary "missing" is unregistered, so resolve falls through to the
    # registered fallback. With request_retries at its default (0) the
    # config-backed fallback's model is reused as-is, so the compiled agent
    # carries exactly that model.
    assert compiled.model_bundle.model is registry.get("fallback-model").model


@pytest.mark.asyncio
async def test_compile_wires_middleware_capability_when_pipeline_provided():
    from linktools.ai.middleware.capability import MiddlewareCapability
    from linktools.ai.middleware.pipeline import MiddlewarePipeline

    registry = ModelRegistry()
    registry.register("test-model", config=_config("test-model"))
    router = ModelResolver(registry=registry)
    pipeline = MiddlewarePipeline(middlewares=())
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=router,
        middleware_pipeline=pipeline,
    )
    spec = AgentSpec(
        id="agent-mw",
        name="mw-agent",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
    )
    compiled = await compiler.compile(spec)
    assert isinstance(compiled.middleware_capability, MiddlewareCapability)
    assert not hasattr(compiled.middleware_capability, "current_context")
    # Both capabilities must end up on the pydantic-ai Agent. In pydantic-ai
    # 1.107 capabilities are nested under root_capability.capabilities (a list).
    capability_types = {
        type(c).__name__ for c in compiled.pydantic_agent.root_capability.capabilities
    }
    assert "PolicyCapability" in capability_types
    assert "MiddlewareCapability" in capability_types


@pytest.mark.asyncio
async def test_compile_leaves_middleware_capability_none_when_no_pipeline():
    registry = ModelRegistry()
    registry.register("test-model", config=_config("test-model"))
    router = ModelResolver(registry=registry)
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())), model_resolver=router
    )
    spec = AgentSpec(
        id="agent-nomw",
        name="nomw-agent",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
    )
    compiled = await compiler.compile(spec)
    assert compiled.middleware_capability is None


def test_compiler_requires_tool_executor():
    """AgentCompiler without an explicit GovernedToolInvoker fails loudly
    (no silent ALLOW-all fallback)."""
    from linktools.ai.errors import RuntimeInitializationError

    with pytest.raises(RuntimeInitializationError):
        AgentCompiler(
            model_resolver=ModelResolver(registry=ModelRegistry()), tool_executor=None
        )
