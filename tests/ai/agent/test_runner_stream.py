#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for AgentEngine.run_stream -- the streaming variant of run().

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
from linktools.ai.agent.engine import AgentEngine
from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
from linktools.ai.capability.resolver import CapabilityResolver
from linktools.ai.capability.models import CapabilityBundle
from linktools.ai.capability.provider import CapabilityProvider
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.resolver import ModelResolver
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import RunInput, RunnableType, RunStatus
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.filesystem.checkpoint import FilesystemCheckpointStore
from linktools.ai.storage.filesystem.event import FilesystemEventStore
from linktools.ai.storage.filesystem.run import FilesystemRunStore
from linktools.ai.storage.filesystem.session import FilesystemSessionStore
from linktools.ai.governance.policy.engine import PolicyEngine
from linktools.ai.tool.executor import GovernedToolInvoker
from linktools.ai.tool.models import (
    ManagedToolDefinition,
    ToolContribution,
    ToolDescriptor,
)


# -- Model fixtures ---------------------------------------------------------


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
    from linktools.ai.storage.filesystem.approval import FilesystemApprovalStore
    from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator

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
    )


async def _collect(gen):
    out: "list[dict]" = []
    async for event in gen:
        out.append(event)
    return out


def _compile(
    tmp_path, fn, stream_fn, *, agent_id="agent-1", output_schema=str
) -> "tuple[AgentEngine, CompiledAgent]":
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=_registry(fn, stream_fn)),
    )
    compiled = asyncio.run(
        compiler.compile(
            AgentSpec(
                id=agent_id,
                name="a",
                model=ModelPolicy(primary="test-model"),
                instructions=PromptSpec(instructions="hi"),
                output_schema=output_schema,
                tools=(ToolRef(kind="test", name="echo"),),
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


def test_run_stream_non_streaming_model_completes_via_result(tmp_path):
    """A model without a stream_function (pydantic-ai raises AssertionError)
    still completes via the final result -- the narrowed catch swallows only
    that signal, not real errors."""
    fn, _stream_fn = _text_pair(text="final-answer")
    # stream_fn=None -> FunctionModel raises AssertionError on node.stream().
    runner, compiled = _compile(tmp_path, fn, None)
    _seed_session(runner._session_store, "session-s1")

    events = asyncio.run(
        _collect(
            runner.run_stream(
                compiled,
                RunInput(prompt="what is the answer?"),
                _run_context(run_id="run-ns", session_id="session-s1"),
            )
        )
    )

    # No text deltas streamed (the model can't stream), but the run still
    # completes SUCCEEDED and the final result carries the answer.
    assert not any(e["type"] == "text" for e in events)

    async def _verify():
        return await runner._run_store.get("run-ns")

    run_record = asyncio.run(_verify())
    assert run_record.status is RunStatus.SUCCEEDED
    assert "final-answer" in str(run_record.result.output)


def test_run_stream_real_stream_error_fails_the_run(tmp_path):
    """A genuine stream error (NOT the non-streaming AssertionError signal) must
    propagate and FAIL the run -- this is the whole point of narrowing the catch
    away from ``except Exception``. ``_fn`` succeeds (the model call works) but
    the ``_stream_fn`` raises a real error mid-stream, which the narrowed
    ``except AssertionError`` deliberately does not swallow."""

    def _fn(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="final-answer")])

    async def _stream_fn(messages, info: AgentInfo):
        raise ValueError("stream broke")
        yield  # pragma: no cover  -- makes this an async generator

    runner, compiled = _compile(tmp_path, _fn, _stream_fn)
    _seed_session(runner._session_store, "session-s1")

    # The stream error propagates and ends the run FAILED.
    with pytest.raises(ValueError, match="stream broke"):
        asyncio.run(
            _collect(
                runner.run_stream(
                    compiled,
                    RunInput(prompt="what is the answer?"),
                    _run_context(run_id="run-err", session_id="session-s1"),
                )
            )
        )

    async def _verify():
        return await runner._run_store.get("run-err")

    run_record = asyncio.run(_verify())
    assert run_record.status is RunStatus.FAILED


def test_run_raises_run_invariant_error_when_no_result_persisted(tmp_path):
    """When the deferred success commit reports no persisted result (an
    invariant violation), run() raises RunInvariantError instead of
    fabricating an empty success. run() now reads the result from
    commit_coordinator.complete()'s return value (WP9 step 3's deferred-
    commit design), so stripping ``result`` from THAT response is what
    isolates the tail check -- run_store.get() is no longer consulted on
    this path."""
    import dataclasses

    from linktools.ai.errors import RunInvariantError

    fn, stream_fn = _text_pair(text="answer")
    runner, compiled = _compile(tmp_path, fn, stream_fn)
    _seed_session(runner._session_store, "session-s1")

    real_complete = runner._commit_coordinator.complete

    async def _stripped_complete(command):
        committed = await real_complete(command)
        return dataclasses.replace(committed, result=None)

    runner._commit_coordinator.complete = _stripped_complete  # type: ignore[assignment]

    with pytest.raises(RunInvariantError):
        asyncio.run(
            runner.run(
                compiled,
                RunInput(prompt="what is the answer?"),
                _run_context(run_id="run-inv", session_id="session-s1"),
            )
        )


def test_run_fail_transitions_when_complete_commit_raises(tmp_path):
    """When the complete-commit itself RAISES (after a successful model drain),
    the collector drives the fail-transition (run -> FAILED) via
    ``_fail_committed_run`` -- the case execute()'s own fail-closure cannot
    cover, since the generator already exited cleanly before the commit. The
    original error still propagates to the caller."""

    fn, stream_fn = _text_pair(text="answer")
    runner, compiled = _compile(tmp_path, fn, stream_fn)
    _seed_session(runner._session_store, "session-s1")

    async def _raising_complete(command):
        raise RuntimeError("complete commit broke")

    runner._commit_coordinator.complete = _raising_complete  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="complete commit broke"):
        asyncio.run(
            runner.run(
                compiled,
                RunInput(prompt="what is the answer?"),
                _run_context(run_id="run-cf", session_id="session-s1"),
            )
        )

    run_record = asyncio.run(runner._run_store.get("run-cf"))
    assert run_record.status is RunStatus.FAILED, (
        "a raising complete-commit must still transition the run to FAILED"
    )


def test_run_stream_yields_tool_and_text_events(tmp_path):
    fn, stream_fn = _tool_then_text_pair(final_text="final-answer", tool_name="echo")
    runner, compiled = _compile(tmp_path, fn, stream_fn)

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
