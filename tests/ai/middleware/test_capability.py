#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The per-Run ToolContext reaches the
capability via pydantic-ai dependency injection (``deps=AgentDependencies(...)``
on ``agent.run()`` -> ``ctx.deps.tool_context`` inside hooks). No mutable
``current_context`` field is set on the capability."""

import asyncio

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.dependencies import AgentDependencies
from linktools.ai.middleware.base import Middleware
from linktools.ai.middleware.capability import build_middleware_capability
from linktools.ai.middleware.pipeline import MiddlewarePipeline
from linktools.ai.policy.engine import ToolContext


class _RecordingMiddleware(Middleware):
    def __init__(self, log: list) -> None:
        self.log = log

    async def before_model(self, context, request):
        self.log.append(("before_model", len(request.messages)))
        return request

    async def after_model(self, context, response):
        self.log.append(("after_model", type(response).__name__))
        return response

    async def before_tool(self, context, request):
        self.log.append(("before_tool", request.tool_name, dict(request.arguments)))
        return request

    async def after_tool(self, context, request, result):
        self.log.append(("after_tool", request.tool_name, result))
        return result


def _model_fn(messages, info: AgentInfo) -> ModelResponse:
    if len(messages) <= 1:
        return ModelResponse(
            parts=[ToolCallPart(tool_name="echo", args={"text": "hi"})]
        )
    return ModelResponse(parts=[TextPart(content="done")])


def _agent_with(capability) -> Agent:
    agent = Agent(
        FunctionModel(_model_fn),
        capabilities=[capability],
        deps_type=AgentDependencies,
    )

    @agent.tool_plain
    def echo(text: str) -> str:
        return f"echoed:{text}"

    return agent


def _deps(run_id: str = "run-1", session_id: str = "session-1") -> AgentDependencies:
    return AgentDependencies(
        tool_context=ToolContext(run_id=run_id, session_id=session_id)
    )


def test_middleware_capability_fires_all_four_hooks_in_order_with_deps():
    log: "list[tuple]" = []
    pipeline = MiddlewarePipeline(middlewares=(_RecordingMiddleware(log),))
    capability = build_middleware_capability(pipeline)
    agent = _agent_with(capability)

    async def _run():
        return await agent.run("say hi via the tool", deps=_deps())

    asyncio.run(_run())
    kinds = [entry[0] for entry in log]
    assert "before_model" in kinds
    assert "after_model" in kinds
    assert "before_tool" in kinds
    assert "after_tool" in kinds
    assert kinds.index("before_tool") < kinds.index("after_tool")
    before_tool_entry = next(e for e in log if e[0] == "before_tool")
    assert before_tool_entry[1] == "echo"
    assert before_tool_entry[2] == {"text": "hi"}


def test_middleware_capability_has_no_current_context_field():
    # The mutable ``current_context`` field is gone. A
    # fresh capability has no such attribute, and a Run driven purely via
    # deps= leaves it gone -- the concurrency-safety invariant.
    log: "list[tuple]" = []
    pipeline = MiddlewarePipeline(middlewares=(_RecordingMiddleware(log),))
    capability = build_middleware_capability(pipeline)
    assert not hasattr(capability, "current_context")
    agent = _agent_with(capability)

    async def _run():
        return await agent.run("say hi via the tool", deps=_deps())

    asyncio.run(_run())
    assert not hasattr(capability, "current_context")
    assert any(e[0] == "before_tool" for e in log)


def test_on_tool_execute_error_runs_pipeline_on_error_then_reraises():
    log: "list[str]" = []

    class _ErrMiddleware(Middleware):
        def __init__(self, log: list) -> None:
            self.log = log

        async def on_error(self, context, error):
            log.append(f"on_error:{type(error).__name__}")

    pipeline = MiddlewarePipeline(middlewares=(_ErrMiddleware(log),))
    capability = build_middleware_capability(pipeline)

    def _exploding_model_fn(messages, info: AgentInfo) -> ModelResponse:
        if len(messages) <= 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="boom", args={})])
        return ModelResponse(parts=[TextPart(content="done")])

    agent = Agent(
        FunctionModel(_exploding_model_fn),
        capabilities=[capability],
        deps_type=AgentDependencies,
    )

    @agent.tool_plain
    def boom() -> str:
        raise RuntimeError("tool blew up")

    async def _run():
        return await agent.run("call the failing tool", deps=_deps())

    try:
        asyncio.run(_run())
    except RuntimeError:
        pass
    assert "on_error:RuntimeError" in log
