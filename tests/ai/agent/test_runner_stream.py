#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for AgentRunner.run_stream -- the streaming variant of run().

Drives pydantic-ai's ``agent.iter()`` graph and yields the dict-event shape the
CLI REPL consumes:
  - ``{"type": "text", "text": <delta>}``
  - ``{"type": "tool", "name": <tool>, "phase": "start"|"end", "ok": <bool|None>}``

FunctionModel is wired with BOTH ``function`` (used by the graph's node.run()
when iterating) and ``stream_function`` (used by ``node.stream(run.ctx)`` to
emit the incremental events). The two must agree so the graph and the stream
see the same model behavior."""

import asyncio
import json
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.models import CompiledAgent
from linktools.ai.agent.runner import AgentRunner
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.router import ModelRouter
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import RunInput, RunnableType, RunStatus
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.file.checkpoint import FileCheckpointStore
from linktools.ai.storage.file.event import FileEventStore
from linktools.ai.storage.file.run import FileRunStore
from linktools.ai.storage.file.session import FileSessionStore


# -- Model fixtures ---------------------------------------------------------


def _text_pair(text: str = "streamed-answer"):
    """A (function, stream_function) pair that emits a plain text response."""

    def _fn(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=text)])

    async def _stream_fn(messages, info: AgentInfo):
        yield text

    return _fn, _stream_fn


def _tool_then_text_pair(final_text: str = "final-answer", tool_name: str = "echo"):
    """A pair whose first turn emits a ToolCallPart and whose second turn (after
    the tool returns) emits text. The stream_function mirrors this with a
    DeltaToolCall on turn one and text on turn two."""

    def _has_result(messages) -> bool:
        return any(
            isinstance(p, ToolReturnPart)
            for m in messages
            for p in getattr(m, "parts", ()) or ()
        )

    def _fn(messages, info: AgentInfo) -> ModelResponse:
        if _has_result(messages):
            return ModelResponse(parts=[TextPart(content=final_text)])
        return ModelResponse(
            parts=[ToolCallPart(tool_name=tool_name, args={"text": "ping"})]
        )

    async def _stream_fn(messages, info: AgentInfo):
        if _has_result(messages):
            yield final_text
        else:
            yield {
                0: DeltaToolCall(name=tool_name, json_args=json.dumps({"text": "ping"}))
            }

    return _fn, _stream_fn


def _boom_pair():
    def _fn(messages, info: AgentInfo) -> ModelResponse:
        raise RuntimeError("model exploded")

    async def _stream_fn(messages, info: AgentInfo):
        raise RuntimeError("model exploded")
        yield  # pragma: no cover  -- makes this an async generator

    return _fn, _stream_fn


def _registry(fn, stream_fn) -> ModelRegistry:
    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(fn, stream_function=stream_fn))
    return registry


def _run_context(run_id="run-s1", session_id="session-s1") -> RunContext:
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


def _make_runner(tmp_path):
    return AgentRunner(
        run_store=FileRunStore(root=tmp_path / "runs"),
        session_store=FileSessionStore(root=tmp_path / "sessions"),
        event_store=FileEventStore(root=tmp_path / "events"),
        checkpoint_store=FileCheckpointStore(root=tmp_path / "checkpoints"),
    )


async def _collect(gen):
    out: "list[dict]" = []
    async for event in gen:
        out.append(event)
    return out


def _compile(
    tmp_path, fn, stream_fn, *, agent_id="agent-1", output_schema=str
) -> "tuple[AgentRunner, CompiledAgent]":
    compiler = AgentCompiler(
        model_router=ModelRouter(registry=_registry(fn, stream_fn))
    )
    compiled = asyncio.run(
        compiler.compile(
            AgentSpec(
                id=agent_id,
                name="a",
                model=ModelPolicy(primary="test-model"),
                instructions=PromptSpec(instructions="hi"),
                output_schema=output_schema,
            )
        )
    )
    runner = _make_runner(tmp_path)
    return runner, compiled


# -- Tests ------------------------------------------------------------------


def test_run_stream_yields_text_events_and_succeeds(tmp_path):
    fn, stream_fn = _text_pair(text="streamed-answer")
    runner, compiled = _compile(tmp_path, fn, stream_fn)
    _seed_session(runner._session_store, "session-s1")

    events = asyncio.run(
        _collect(
            runner.run_stream(
                compiled,
                RunInput(prompt="what is the answer?"),
                _run_context(),
            )
        )
    )

    # At least one text delta was yielded with the streamed content.
    text_events = [e for e in events if e["type"] == "text"]
    assert len(text_events) >= 1
    assert "streamed-answer" in "".join(e["text"] for e in text_events)

    async def _verify():
        run_record = await runner._run_store.get("run-s1")
        messages = await runner._session_store.list_messages("session-s1")
        events_list = await runner._event_store.list("run-s1")
        return run_record, messages, events_list

    run_record, messages, events_list = asyncio.run(_verify())

    # Driving RunRecord ended SUCCEEDED.
    assert run_record.status is RunStatus.SUCCEEDED
    # Session received the assistant message carrying the streamed text.
    assert any("streamed-answer" in str(m.content) for m in messages)
    # Event store carries both RunStarted and RunCompleted envelopes.
    payload_types = {type(e.payload).__name__ for e in events_list.items}
    assert "RunStarted" in payload_types
    assert "RunCompleted" in payload_types


def test_run_stream_yields_tool_and_text_events(tmp_path):
    fn, stream_fn = _tool_then_text_pair(final_text="final-answer", tool_name="echo")
    runner, compiled = _compile(tmp_path, fn, stream_fn)

    # Register the tool the model will call on the compiled pydantic-ai agent.
    @compiled.pydantic_agent.tool
    async def echo(ctx, text: str) -> str:
        return f"echoed: {text}"

    _seed_session(runner._session_store, "session-s1")

    events = asyncio.run(
        _collect(
            runner.run_stream(
                compiled,
                RunInput(prompt="call the tool"),
                _run_context(run_id="run-s2", session_id="session-s1"),
            )
        )
    )

    tool_events = [e for e in events if e["type"] == "tool"]
    text_events = [e for e in events if e["type"] == "text"]
    # Both a tool start AND a tool end were yielded.
    assert any(e["phase"] == "start" for e in tool_events), f"no tool start in {events}"
    assert any(e["phase"] == "end" for e in tool_events), f"no tool end in {events}"
    assert any(e["name"] == "echo" for e in tool_events)
    # And at least one text delta from the final turn.
    assert len(text_events) >= 1
    assert "final-answer" in "".join(e["text"] for e in text_events)

    run_record = asyncio.run(runner._run_store.get("run-s2"))
    assert run_record.status is RunStatus.SUCCEEDED


def test_run_stream_transitions_to_failed_on_model_error(tmp_path):
    fn, stream_fn = _boom_pair()
    runner, compiled = _compile(tmp_path, fn, stream_fn)
    _seed_session(runner._session_store, "session-s1")

    with pytest.raises(Exception):
        asyncio.run(
            _collect(
                runner.run_stream(
                    compiled,
                    RunInput(prompt="hi"),
                    _run_context(run_id="run-s3", session_id="session-s1"),
                )
            )
        )

    run_record = asyncio.run(runner._run_store.get("run-s3"))
    assert run_record.status is RunStatus.FAILED
