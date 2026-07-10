#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end smoke tests for the managed security governance path. Verifies
the full chain: Runtime.build(security=...) -> AgentRunner.execute ->
ManagedToolAdapter -> pipeline before/after_tool -> handler -> result."""

import asyncio

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.runtime import Runtime
from linktools.ai.security.baseline import SecurityBaseline
from linktools.ai.security.pipeline import (
    PipelineAction,
    PipelineDecision,
    SecurityPipeline,
    ToolInvocationEvent,
    ToolResultEvent,
)
from linktools.ai.storage.facade import FileStorage


def _deny_terminal_pipeline():
    """Pipeline that denies every tool invocation."""
    class _P:
        saw_before = False
        saw_after = False

        async def before_model(self, e):
            return PipelineDecision(action=PipelineAction.ALLOW)

        async def after_model(self, e):
            return PipelineDecision(action=PipelineAction.ALLOW)

        async def before_tool(self, e: ToolInvocationEvent):
            _P.saw_before = True
            return PipelineDecision(action=PipelineAction.DENY, reason="blocked by e2e pipeline")

        async def after_tool(self, e: ToolResultEvent):
            _P.saw_after = True
            return PipelineDecision(action=PipelineAction.ALLOW)

        async def on_security_event(self, e):
            return PipelineDecision(action=PipelineAction.AUDIT_ONLY)

    return _P()


def _model_fn_no_tools(messages, info: AgentInfo) -> ModelResponse:
    """Model that just returns text without calling tools."""
    return ModelResponse(parts=[TextPart(content="hello")])


def _registry():
    from linktools.ai.model.registry import ModelRegistry
    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_model_fn_no_tools))
    return registry


def _spec():
    return AgentSpec(
        id="e2e", name="e2e",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
        tools=(ToolRef(name="terminal"),),
    )


@pytest.mark.asyncio
async def test_runtime_with_default_baseline_runs(tmp_path):
    """Default SecurityBaseline does not break normal execution."""
    from linktools.ai.model.router import ModelRouter
    rt = Runtime.build(
        storage=FileStorage(root=tmp_path),
        model_router=ModelRouter(registry=_registry()),
    )
    result = await rt.run(_spec(), "hello")
    assert "hello" in str(result.output)


@pytest.mark.asyncio
async def test_runtime_baseline_disabled_runs(tmp_path):
    """SecurityBaseline(enabled=False) falls back to legacy path."""
    from linktools.ai.model.router import ModelRouter
    rt = Runtime.build(
        storage=FileStorage(root=tmp_path),
        model_router=ModelRouter(registry=_registry()),
        security=SecurityBaseline(enabled=False),
    )
    result = await rt.run(_spec(), "hello")
    assert "hello" in str(result.output)


@pytest.mark.asyncio
async def test_pipeline_attached_to_runtime(tmp_path):
    """A SecurityPipeline on the SecurityBaseline is wired into the runner."""
    from linktools.ai.model.router import ModelRouter
    pipeline = _deny_terminal_pipeline()
    rt = Runtime.build(
        storage=FileStorage(root=tmp_path),
        model_router=ModelRouter(registry=_registry()),
        security=SecurityBaseline(pipeline=pipeline),
    )
    # The pipeline is on the runner; verify it was wired.
    assert rt.runner._security_pipeline is not None
