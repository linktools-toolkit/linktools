#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Runtime.run_stream -- the streaming variant of Runtime.run().

Compiles the spec, resolves (or creates) a Session, mints a RunContext, and
delegates to AgentEngine.run_stream, yielding the same dict-event shape."""

import asyncio
from datetime import datetime, timezone

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.router import ModelResolver
from linktools.ai.run.models import RunStatus
from linktools.ai.runtime import Runtime
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.facade import FilesystemStorage
from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator


def _text_pair(text: str = "hello from stream"):
    def _fn(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=text)])

    async def _stream_fn(messages, info: AgentInfo):
        yield text

    return _fn, _stream_fn


def _build_runtime(tmp_path):
    fn, stream_fn = _text_pair()
    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(fn, stream_function=stream_fn))
    storage = FilesystemStorage(root=tmp_path)
    runtime = Runtime.build(
        storage=storage,
        model_router=ModelResolver(registry=registry),
        commit_coordinator=FilesystemRunCommitCoordinator.from_storage(storage),
    )
    return runtime, storage


async def _collect(gen):
    out: "list[dict]" = []
    async for event in gen:
        out.append(event)
    return out


def test_runtime_run_stream_yields_text_events_and_completes(tmp_path):
    runtime, storage = _build_runtime(tmp_path)
    now = datetime.now(timezone.utc)

    async def _setup():
        await storage.sessions.create(
            SessionRecord(
                id="rt-stream-1",
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )

    asyncio.run(_setup())

    spec = AgentSpec(
        id="agent-rt-stream",
        name="rt-stream-agent",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
        output_schema=str,
    )

    events = asyncio.run(
        _collect(
            runtime.run_stream(
                spec,
                "hello",
                session_id="rt-stream-1",
                run_id="rt-run-stream-1",
            )
        )
    )

    text_events = [e for e in events if e["type"] == "text"]
    assert len(text_events) >= 1
    assert "hello from stream" in "".join(e["text"] for e in text_events)

    run = asyncio.run(storage.runs.get("rt-run-stream-1"))
    assert run is not None
    assert run.status is RunStatus.SUCCEEDED
