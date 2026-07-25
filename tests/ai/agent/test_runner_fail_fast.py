#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentEngine missing-dependency fail-fast (via the Store-free execute_pure).

A spec that needs tools must fail eagerly when the resolver or the managed
executor is missing -- before any capability resolution work. ``tools=()`` is
a model-only run and never raises.

FS-29: the engine's legacy ``run()`` (full Run-lifecycle) is gone; these
checks now drive ``execute_pure`` directly. The fail-fast ``RuntimeInitializationError``
is raised inside ``execute_pure`` BEFORE any model work, so the assertion is
unchanged -- and because execute_pure touches no Store, the test no longer
wires one up."""

import asyncio

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.engine import AgentEngine
from linktools.ai.agent.models import AgentCompleted, AgentInput
from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
from linktools.ai.capability.resolver import CapabilityResolver
from linktools.ai.errors import RuntimeInitializationError
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.resolver import ModelResolver
from linktools.ai.run.cancellation import CancellationToken
from linktools.ai.run.context import RunContext
from linktools.ai.run.live_events import NullRunLiveEventSink, NullSecurityEventSink
from linktools.ai.run.models import RunnableType
from linktools.ai.governance.policy.engine import PolicyEngine
from linktools.ai.tool.executor import GovernedToolInvoker


def _model_fn(messages, info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content='{"response": {"answer": 42}}')])


def _registry():
    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_model_fn))
    return registry


def _context() -> RunContext:
    return RunContext(
        run_id="run-1",
        root_run_id="run-1",
        parent_run_id=None,
        session_id="session-1",
        runnable_id="agent-1",
        runnable_type=RunnableType.AGENT,
        user_id=None,
        tenant_id=None,
        workspace=None,
    )


def _execute_pure(engine: AgentEngine, compiled, prompt: str = "hi"):
    """Drive execute_pure with the Store-free sinks execute_pure requires."""
    return asyncio.run(
        engine.execute_pure(
            compiled,
            AgentInput(prompt=prompt),
            _context(),
            cancellation=CancellationToken(),
            live_events=NullRunLiveEventSink(),
            security_events=NullSecurityEventSink(),
        )
    )


def _compiled_spec_with_tools():
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=_registry()),
    )
    spec = AgentSpec(
        id="agent-1",
        name="a",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
        tools=(ToolRef(kind="builtin", name="file"),),
    )
    return asyncio.run(compiler.compile(spec))


def _make_runner(*, capability_resolver, managed_tool_executor) -> AgentEngine:
    # FS-29: the engine accepts only the pure-execution dependencies it
    # actually uses -- no run/session/event/checkpoint Store, no
    # commit_coordinator. The fail-fast wiring under test lives entirely in
    # execute_pure's capability-resolution prelude.
    return AgentEngine(
        capability_resolver=capability_resolver,
        managed_tool_executor=managed_tool_executor,
    )


@pytest.mark.parametrize(
    "resolver,executor,match",
    [
        (None, None, "CapabilityResolver"),
        (CapabilityResolver({}), None, "GovernedToolInvoker"),
    ],
    ids=["no-resolver", "no-executor"],
)
def test_eager_fail_fast_when_tools_declared(tmp_path, resolver, executor, match):
    """Declaring tools with a missing resolver or executor must fail eagerly
    (before any capability resolution work)."""
    runner = _make_runner(
        capability_resolver=resolver,
        managed_tool_executor=executor,
    )
    compiled = _compiled_spec_with_tools()

    with pytest.raises(RuntimeInitializationError, match=match):
        _execute_pure(runner, compiled)


def test_empty_tools_never_raises_even_without_assembler_or_executor(tmp_path):
    """tools=() is a model-only run and does not require tool wiring."""
    runner = _make_runner(
        capability_resolver=None,
        managed_tool_executor=None,
    )
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=_registry()),
    )
    spec = AgentSpec(
        id="agent-1",
        name="a",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
        tools=(),
    )
    compiled = asyncio.run(compiler.compile(spec))
    outcome = _execute_pure(runner, compiled)
    assert isinstance(outcome, AgentCompleted)
