#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LocalRuntimeClient integration tests against a real Runtime.

Drives :class:`LocalRuntimeClient` (the in-process backend the console
use) through a Runtime built with a FunctionModel -- not a fake client. These
tests guard the event-contract translation (scalar run → text/tool/paused/
failed events) that isolated client doubles cannot reach."""

from pathlib import Path

import pytest

from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.agent.assembly.provider import AgentFeatureContext
from linktools.ai.cli.client import LocalRuntimeClient, RunRequest
from linktools.ai.cli.project import load_project
from linktools.ai.cli.runtime import build_cli_runtime, load_agent_spec
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
async def test_cli_default_agent_exposes_tools_skills_and_subagents(
    tmp_path: Path,
) -> None:
    client = await _bundle(tmp_path, resolver=make_router("hi"))
    spec = await load_agent_spec(client.bundle, None)

    assert {(ref.kind, ref.name) for ref in spec.features} == {
        ("builtin", "*"),
        ("skill", "*"),
        ("subagent", "*"),
    }
    assembly = await client.bundle.runtime.assembler.assemble(
        spec,
        AgentFeatureContext(
            agent_id=spec.id,
            execution_id="inspect",
            root_execution_id="inspect",
            parent_execution_id=None,
            session_id="inspect",
            tenant_id=client.principal.tenant_id,
            user_id=client.principal.user_id,
            workspace=None,
            sandbox=client.bundle.runtime.sandbox,
        ),
    )

    assert {tool.descriptor.name for tool in assembly.tools} == {
        "list_dir",
        "read_file",
        "write_file",
        "batch_files",
        "apply_patch",
        "bash",
        "list_skills",
        "read_skill",
        "call_subagent",
    }


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


@pytest.mark.asyncio
async def test_run_stream_second_turn_prompt_reaches_model(tmp_path: Path) -> None:
    # Regression: AgentEngine once called pydantic_agent.iter(message_history=...)
    # with no first positional user_prompt whenever message_history was
    # non-empty, silently dropping the new turn's prompt -- a session's second
    # (and every later) turn sent the model only stale history and no new
    # question. Assert the model actually receives the new prompt text
    # alongside history, not just history.
    from pydantic_ai.messages import ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from linktools.ai.model.registry import ModelRegistry
    from linktools.ai.model.resolver import ModelResolver

    captured: "list[list[object]]" = []

    def _fn(messages, info: AgentInfo) -> ModelResponse:
        captured.append(list(messages))
        return ModelResponse(parts=[TextPart(content="ok")])

    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_fn))
    resolver = ModelResolver(registry=registry)

    client = await _bundle(tmp_path, resolver=resolver)
    _ = [
        e
        async for e in client.run_stream(
            RunRequest(prompt="first turn", session_id="s", run_id="r1")
        )
    ]
    _ = [
        e
        async for e in client.run_stream(
            RunRequest(prompt="second turn text", session_id="s", run_id="r2")
        )
    ]
    assert len(captured) == 2
    second_call_messages = captured[1]
    user_texts = [
        part.content
        for message in second_call_messages
        for part in getattr(message, "parts", [])
        if isinstance(part, UserPromptPart)
    ]
    assert any("second turn text" in str(text) for text in user_texts)
