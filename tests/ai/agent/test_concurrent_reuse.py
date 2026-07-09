#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 1 review-doc refactoring: concurrency-safety assertions.

CompiledAgent is compiled once and reused across many Runs. Under the old
``current_context`` pattern, the runner set a mutable field on
PolicyCapability / MiddlewareCapability around each ``agent.run()`` -- a data
race whenever two Runs shared one CompiledAgent. The refactor routes the
per-Run ToolContext through pydantic-ai dependency injection
(``deps=AgentDependencies(tool_context=...)`` -> ``ctx.deps.tool_context``),
so the capabilities carry no mutable per-Run state at all.

These tests assert the invariant both ways:

1. Sequential reuse -- two Runs back-to-back on the same CompiledAgent, each
   with a distinct run_id, each observes its OWN run_id (no leakage).

2. Concurrent reuse -- two ``pydantic_agent.run()`` calls driven in parallel
   via ``asyncio.gather`` on the SAME compiled agent, each with its own
   ``deps=``. A recording middleware forces the two calls to overlap inside
   the ``before_tool`` hook (the exact window where mutable shared state would
   race) and asserts each call still observes its OWN run_id. Under the old
   code the second ``current_context`` assignment would clobber the first and
   both calls would see the same run_id; under deps-based DI they cannot.
"""
import asyncio
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.dependencies import AgentDependencies
from linktools.ai.agent.runner import AgentRunner
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.middleware.base import Middleware
from linktools.ai.middleware.pipeline import MiddlewarePipeline
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.router import ModelRouter
from linktools.ai.policy.engine import ToolContext
from linktools.ai.run.context import RunContext as AiRunContext
from linktools.ai.run.models import RunInput, RunnableType
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.file.checkpoint import FileCheckpointStore
from linktools.ai.storage.file.event import FileEventStore
from linktools.ai.storage.file.run import FileRunStore
from linktools.ai.storage.file.session import FileSessionStore


class _ContextCapturingMiddleware(Middleware):
    """Records ``context.run_id`` (the per-Run ToolContext the capability
    threaded in via ``ctx.deps.tool_context``) every time ``before_tool`` fires.
    The list is appended in observation order; tests assert on its contents."""

    def __init__(self) -> None:
        self.observed: "list[str]" = []

    async def before_tool(self, context, request):
        self.observed.append(context.run_id)
        return request


def _tool_then_text_model_fn(messages, info: AgentInfo) -> ModelResponse:
    """Emit one ToolCallPart on the first turn, a final TextPart on the next so
    the run terminates. The tool call is what drives ``before_tool`` (and thus
    the recording middleware)."""
    if not any(getattr(p, "part_kind", None) == "tool-return" for m in messages for p in m.parts):
        return ModelResponse(parts=[ToolCallPart(tool_name="ping", args={})])
    return ModelResponse(parts=[TextPart(content="done")])


def _registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_tool_then_text_model_fn))
    return registry


async def _compiled(pipeline: MiddlewarePipeline):
    """Compile a CompiledAgent whose middleware pipeline captures the per-Run
    ToolContext. The compiled agent carries an extra ``ping`` tool so the model
    has something to call (driving the ``before_tool`` hook)."""
    compiler = AgentCompiler(
        model_router=ModelRouter(registry=_registry()),
        middleware_pipeline=pipeline,
    )
    spec = AgentSpec(
        id="reuse-agent", name="reuse",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
        output_schema=str,
    )
    compiled = await compiler.compile(spec)

    @compiled.pydantic_agent.tool_plain
    def ping() -> str:  # noqa: D401
        return "pong"

    return compiled


async def _seed_session(store, session_id) -> None:
    now = datetime.now(timezone.utc)
    await store.create(SessionRecord(
        id=session_id, parent_id=None, status=SessionStatus.ACTIVE,
        version=1, created_at=now, updated_at=now,
    ))


def _make_runner(tmp_path, pipeline) -> AgentRunner:
    return AgentRunner(
        run_store=FileRunStore(root=tmp_path / "runs"),
        session_store=FileSessionStore(root=tmp_path / "sessions"),
        event_store=FileEventStore(root=tmp_path / "events"),
        checkpoint_store=FileCheckpointStore(root=tmp_path / "checkpoints"),
        middleware_pipeline=pipeline,
    )


def _ai_run_context(run_id, session_id) -> AiRunContext:
    return AiRunContext(
        run_id=run_id, root_run_id=run_id, parent_run_id=None, session_id=session_id,
        runnable_id="reuse-agent", runnable_type=RunnableType.AGENT,
        user_id=None, tenant_id=None, workspace=None,
    )


@pytest.mark.asyncio
async def test_sequential_reuse_each_run_sees_own_tool_context(tmp_path):
    """Drive two Runs back-to-back on the SAME CompiledAgent via the runner,
    each with a distinct run_id. The recording middleware must observe each
    run's OWN run_id in order -- no leakage from run N into run N+1. (Under the
    old mutable current_context pattern this also happens to hold for strictly
    sequential runs because set/clear bracketed each call; the assertion is
    the more interesting concurrent case below, but the sequential baseline
    documents the expected behavior.)"""
    mw = _ContextCapturingMiddleware()
    pipeline = MiddlewarePipeline(middlewares=(mw,))
    compiled = await _compiled(pipeline)
    runner = _make_runner(tmp_path, pipeline)
    await _seed_session(runner._session_store, "session-seq")

    await runner.run(
        compiled, RunInput(prompt="ping"), _ai_run_context("run-A", "session-seq"))
    await runner.run(
        compiled, RunInput(prompt="ping"), _ai_run_context("run-B", "session-seq"))

    assert mw.observed == ["run-A", "run-B"], (
        f"expected each run to observe its own run_id in order, got {mw.observed!r}"
    )
    # The capabilities must remain field-for-field immutable across runs -- no
    # current_context was ever set (the refactor removed the field entirely).
    assert not hasattr(compiled.policy_capability, "current_context")
    assert not hasattr(compiled.middleware_capability, "current_context")


@pytest.mark.asyncio
async def test_concurrent_reuse_each_run_sees_own_tool_context_no_pollution(tmp_path):
    """Drive two ``pydantic_agent.run()`` calls in parallel via
    ``asyncio.gather`` on the SAME compiled agent, each with its own ``deps=``.
    A gate inside ``before_tool`` forces the two calls to overlap inside the
    hook -- the exact window where mutable shared state would race. Under the
    old ``current_context`` pattern the second set would clobber the first and
    both calls would observe the same run_id; under deps-based DI each call
    observes its OWN run_id.

    This is the concurrency-safety assertion the refactor delivers."""
    observed: "list[str]" = []
    # Gate that the FIRST arrival waits on; the SECOND arrival opens it. This
    # guarantees both calls are inside ``before_tool`` simultaneously before
    # either proceeds -- maximal race window.
    first_inside = asyncio.Event()
    second_inside = asyncio.Event()

    class _GatedMiddleware(Middleware):
        async def before_tool(self, context, request):
            observed.append(context.run_id)
            if not first_inside.is_set():
                # First arrival: signal we're inside, wait for the second call
                # to also enter before_tool (force the overlap), then proceed.
                first_inside.set()
                try:
                    await asyncio.wait_for(second_inside.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
            else:
                # Second arrival: signal we're inside too, releasing the first.
                second_inside.set()
            return request

    pipeline = MiddlewarePipeline(middlewares=(_GatedMiddleware(),))
    compiled = await _compiled(pipeline)

    async def _run(run_id: str):
        return await compiled.pydantic_agent.run(
            "ping",
            deps=AgentDependencies(
                tool_context=ToolContext(run_id=run_id, session_id="session-cc")),
        )

    # Drive both calls concurrently on the SAME compiled agent.
    await asyncio.gather(_run("run-A"), _run("run-B"))

    # Each call made exactly one tool call, so observed holds exactly two
    # entries -- one per call, each carrying that call's OWN run_id. Sorting
    # sidesteps the scheduler's arrival order.
    assert sorted(observed) == ["run-A", "run-B"], (
        f"expected each concurrent run to observe its own run_id (no pollution), "
        f"got {observed!r}"
    )
    assert len(observed) == 2, (
        f"expected exactly one before_tool firing per run (2 total), got {len(observed)}"
    )
