#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LocalRuntimeClient integration tests against a real Runtime.

Drives :class:`LocalRuntimeClient` (the in-process backend the console/TUI
use) through a Runtime built with a FunctionModel -- not a fake client. These
tests guard the event-contract translation (scalar run → text/tool/paused/
failed events) that the FakeRuntimeClient-only suite cannot reach."""

from pathlib import Path

import pytest

from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.cli.client import LocalRuntimeClient, RunRequest
from linktools.ai.cli.project import load_project
from linktools.ai.cli.runtime import build_cli_runtime
from linktools.ai.model.policy import ModelPolicy
from tests.ai.fakes.model import make_raising_router, make_router


def _spec() -> AgentSpec:
    # request_retries=None so a prebuilt FunctionModel manages its own retries
    # (an int would be rejected by the resolver for prebuilt models).
    return AgentSpec(
        id="default",
        name="default",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="You are a test assistant."),
    )


class _FixedAgentIndex:
    """An in-memory agent index that always returns the directly-built test
    spec. Avoids markdown-frontmatter parsing, whose request_retries default
    (0) is incompatible with prebuilt FunctionModel."""

    async def list_ids(self) -> "tuple[str, ...]":
        return ("default",)

    async def get(self, agent_id: str) -> AgentSpec:
        if agent_id != "default":
            raise KeyError(agent_id)
        return _spec()


async def _bundle(
    tmp_path: Path, *, resolver, live_events=None
) -> "LocalRuntimeClient":
    project = load_project(data_root=tmp_path / "data", start=tmp_path)
    client = LocalRuntimeClient(
        build_cli_runtime(
            project=project, model_resolver=resolver, live_events=live_events
        )
    )
    # Replace the filesystem agent index with the fixed in-memory one.
    object.__setattr__(client.bundle, "agents", _FixedAgentIndex())
    return client


@pytest.mark.asyncio
async def test_run_stream_emits_text_event_from_real_run(tmp_path: Path) -> None:
    client = await _bundle(tmp_path, resolver=make_router("hello world"))
    request = RunRequest(prompt="hi", session_id="main", run_id="run-1")
    events = [e async for e in client.run_stream(request)]
    kinds = [e.get("type") for e in events]
    # A completed run re-emits the model's text part as a text event.
    assert "text" in kinds
    text_event = next(e for e in events if e.get("type") == "text")
    assert "hello world" in text_event.get("text", "")


@pytest.mark.asyncio
async def test_run_stream_records_session_and_run(tmp_path: Path) -> None:
    client = await _bundle(tmp_path, resolver=make_router("hi"))
    _ = [
        e
        async for e in client.run_stream(
            RunRequest(prompt="hi", session_id="s1", run_id="r1")
        )
    ]
    sessions = await client.list_sessions()
    runs = await client.list_runs()
    assert any(getattr(s, "id", None) == "s1" for s in sessions)
    assert any(getattr(r, "id", None) == "r1" for r in runs)


@pytest.mark.asyncio
async def test_run_stream_surfaces_model_failure_as_failed_event(
    tmp_path: Path,
) -> None:
    client = await _bundle(tmp_path, resolver=make_raising_router(RuntimeError("boom")))
    events = [
        e
        async for e in client.run_stream(
            RunRequest(prompt="hi", session_id="s", run_id="r")
        )
    ]
    # The run failure is surfaced as a failed event, not raised.
    failed = [e for e in events if e.get("type") == "failed"]
    assert failed
    assert "boom" in failed[0].get("message", "")


@pytest.mark.asyncio
async def test_inspect_returns_tool_summary(tmp_path: Path) -> None:
    client = await _bundle(tmp_path, resolver=make_router("hi"))
    view = await client.inspect(None)
    assert view is not None
    assert getattr(view, "id", None) == "default"


@pytest.mark.asyncio
async def test_run_stream_does_not_duplicate_output(tmp_path: Path) -> None:
    # The model's text appears both as an interaction text part AND as the
    # run's final_output; the client must emit it exactly once (regression:
    # an earlier build emitted both, so every reply printed twice).
    client = await _bundle(tmp_path, resolver=make_router("hello world"))
    events = [
        e
        async for e in client.run_stream(
            RunRequest(prompt="hi", session_id="main", run_id="run-dedup")
        )
    ]
    text_events = [e for e in events if e.get("type") == "text"]
    assert len(text_events) == 1
    assert "hello world" in text_events[0]["text"]


@pytest.mark.asyncio
async def test_run_stream_streams_live_with_sink(tmp_path: Path) -> None:
    # With a StreamingRunLiveSink wired, run_stream takes the live path:
    # terminal classification is a 'completed' event (the no-sink replay path
    # emits no 'completed'). Live model text is published to the queue only
    # for streaming-capable models; FunctionModel cannot stream without a
    # stream_function (pydantic-ai raises), so this test asserts the path is
    # taken, not the live text deltas -- which real provider models
    # (OpenAI/GLM/Anthropic) emit via stream_text.
    from linktools.ai.execution.live_events import StreamingRunLiveSink

    sink = StreamingRunLiveSink()
    client = await _bundle(
        tmp_path, resolver=make_router("hello world"), live_events=sink
    )
    events = [
        e
        async for e in client.run_stream(
            RunRequest(prompt="hi", session_id="stream", run_id="run-stream")
        )
    ]
    kinds = [e.get("type") for e in events]
    # The sink path terminates with a 'completed' classification event.
    assert kinds[-1] == "completed"


@pytest.mark.asyncio
async def test_run_stream_no_sink_falls_back_to_trace_replay(tmp_path: Path) -> None:
    # A bundle built without a live sink (live_events=None) still streams the
    # recorded trace via the replay path -- same events, just not live.
    client = await _bundle(tmp_path, resolver=make_router("hi"))
    events = [
        e
        async for e in client.run_stream(
            RunRequest(prompt="hi", session_id="replay", run_id="run-replay")
        )
    ]
    assert any(e.get("type") == "text" for e in events)
