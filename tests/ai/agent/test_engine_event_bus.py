#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/agent/test_engine_event_bus.py

WP9 step 3 (event-bus increment): when an ``AgentEngine`` is constructed with
an ``event_bus``, every text/tool/paused event ``execute()`` yields is ALSO
published to that bus for the same ``run_id`` -- the seam a future
``run_stream()`` will consume from directly instead of iterating execute()'s
own generator. This module reuses the same (function, stream_function) model
fixtures as ``test_runner_stream.py`` to drive real text + tool events."""

import asyncio
from datetime import datetime, timezone

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.engine import AgentEngine
from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
from linktools.ai.capability.models import CapabilityBundle
from linktools.ai.capability.provider import CapabilityProvider
from linktools.ai.capability.resolver import CapabilityResolver
from linktools.ai.governance.policy.engine import PolicyEngine
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.resolver import ModelResolver
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.run.context import RunContext
from linktools.ai.run.events_bus import RunEventBus
from linktools.ai.run.models import RunInput, RunnableType
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.filesystem.approval import FilesystemApprovalStore
from linktools.ai.storage.filesystem.checkpoint import FilesystemCheckpointStore
from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator
from linktools.ai.storage.filesystem.event import FilesystemEventStore
from linktools.ai.storage.filesystem.run import FilesystemRunStore
from linktools.ai.storage.filesystem.session import FilesystemSessionStore
from linktools.ai.tool.executor import GovernedToolInvoker
from linktools.ai.tool.models import (
    ManagedToolDefinition,
    ToolContribution,
    ToolDescriptor,
)


class _EchoProvider(CapabilityProvider):
    supported_kinds = ("test",)

    async def resolve(self, ref, context):
        async def echo(text: str) -> str:
            return f"echoed: {text}"

        return CapabilityBundle(
            tool_contributions=(
                ToolContribution(
                    tools=(
                        ManagedToolDefinition(
                            descriptor=ToolDescriptor(
                                name="echo",
                                source="test",
                                category="discovery",
                                risk="low",
                                mutating=False,
                            ),
                            handler=echo,
                        ),
                    )
                ),
            )
        )


def _text_pair(text: str = "streamed-answer"):
    def _fn(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=text)])

    async def _stream_fn(messages, info: AgentInfo):
        yield text

    return _fn, _stream_fn


def _run_context(run_id="run-b1", session_id="session-b1") -> RunContext:
    return RunContext(
        run_id=run_id,
        root_run_id=run_id,
        parent_run_id=None,
        session_id=session_id,
        runnable_id="agent-1",
        runnable_type=RunnableType.AGENT,
        user_id=None,
        tenant_id=None,
        workspace=None,
    )


def _seed_session(store, session_id) -> None:
    now = datetime.now(timezone.utc)
    asyncio.run(
        store.create(
            SessionRecord(
                id=session_id,
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
    )


def _make_runner(tmp_path, event_bus=None):
    run_store = FilesystemRunStore(root=tmp_path / "runs")
    session_store = FilesystemSessionStore(root=tmp_path / "sessions")
    event_store = FilesystemEventStore(root=tmp_path / "events")
    checkpoint_store = FilesystemCheckpointStore(root=tmp_path / "checkpoints")
    return AgentEngine(
        run_store=run_store,
        session_store=session_store,
        event_store=event_store,
        capability_resolver=CapabilityResolver({"test": _EchoProvider()}),
        managed_tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        commit_coordinator=FilesystemRunCommitCoordinator(
            approval_store=FilesystemApprovalStore(root=tmp_path / "approvals"),
            checkpoint_store=checkpoint_store,
            run_store=run_store,
            session_store=session_store,
            event_store=event_store,
        ),
        event_bus=event_bus,
    )


def _compile(tmp_path, fn, stream_fn):
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(
            registry=_registry(fn, stream_fn)
        ),
    )
    return asyncio.run(
        compiler.compile(
            AgentSpec(
                id="agent-1",
                name="a",
                model=ModelPolicy(primary="test-model"),
                instructions=PromptSpec(instructions="hi"),
                output_schema=str,
                tools=(ToolRef(kind="test", name="echo"),),
            )
        )
    )


def _registry(fn, stream_fn) -> ModelRegistry:
    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(fn, stream_function=stream_fn))
    return registry


async def _drain(runner, compiled, context):
    events = []
    async for event in runner.run_stream(compiled, RunInput(prompt="hi"), context):
        events.append(event)
    return events


def test_run_stream_with_event_bus_wired_still_yields_text_events(tmp_path):
    """When an event_bus is wired, run_stream() drains it internally (a
    background execute_outcome() task publishes, this generator subscribes)
    instead of iterating execute()'s own generator directly -- the caller-
    visible event sequence must still carry the same text deltas."""
    bus = RunEventBus()
    fn, stream_fn = _text_pair(text="streamed-answer")
    compiled = _compile(tmp_path, fn, stream_fn)
    runner = _make_runner(tmp_path, event_bus=bus)
    _seed_session(runner._session_store, "session-b1")
    context = _run_context()

    events = asyncio.run(_drain(runner, compiled, context))
    text_events = [e for e in events if e["type"] == "text"]
    assert len(text_events) >= 1
    assert "streamed-answer" in "".join(e["text"] for e in text_events)
    # The bus's internal queue for this run is cleaned up once done.
    assert "run-b1" not in bus._queues


def test_no_event_bus_is_a_no_op(tmp_path):
    """Default (no event_bus) behaves exactly like before -- run_stream still
    yields events normally, nothing published anywhere."""
    fn, stream_fn = _text_pair(text="streamed-answer")
    compiled = _compile(tmp_path, fn, stream_fn)
    runner = _make_runner(tmp_path, event_bus=None)
    _seed_session(runner._session_store, "session-b1")

    events = asyncio.run(_drain(runner, compiled, _run_context()))
    assert any(e["type"] == "text" for e in events)


def _boom_pair():
    def _fn(messages, info: AgentInfo) -> ModelResponse:
        raise RuntimeError("model exploded")

    async def _stream_fn(messages, info: AgentInfo):
        raise RuntimeError("model exploded")
        yield  # pragma: no cover  -- makes this an async generator

    return _fn, _stream_fn


def test_run_stream_with_event_bus_yields_failed_event_instead_of_raising(tmp_path):
    """With an event_bus wired, run_stream() is rebuilt on execute_outcome()
    (spec 12.3's Outcome model) -- a model failure surfaces as a final
    {"type": "failed", ...} event instead of a raised exception (a deliberate,
    user-approved break from the no-bus fallback's raise-based contract)."""
    bus = RunEventBus()
    fn, stream_fn = _boom_pair()
    compiled = _compile(tmp_path, fn, stream_fn)
    runner = _make_runner(tmp_path, event_bus=bus)
    _seed_session(runner._session_store, "session-b1")

    events = asyncio.run(_drain(runner, compiled, _run_context(run_id="run-boom")))
    assert events[-1]["type"] == "failed"
    assert events[-1]["error_type"] == "RuntimeError"
    assert "model exploded" in events[-1]["message"]

    record = asyncio.run(runner._run_store.get("run-boom"))
    assert record.status.value == "failed"
