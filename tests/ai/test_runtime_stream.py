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
from linktools.ai.model.resolver import ModelResolver
from linktools.ai.run.models import RunStatus
from linktools.ai.runtime import Runtime, build_runtime
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.runtime.persistence.facade import FilesystemStorage
from linktools.ai.run.persistence.commit import FilesystemRunCommitCoordinator


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
    runtime = build_runtime(
        storage=storage,
        model_resolver=ModelResolver(registry=registry),
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


def test_runtime_run_stream_early_consumer_exit_does_not_orphan_engine_task(tmp_path):
    """A consumer that stops iterating early (then closes the generator) must
    not leave the background engine task pending. The Coordinator's streaming
    finally force-cancels AND awaits the engine task inside the finally; if it
    only set the cooperative flag (and awaited outside the finally), a task
    suspended in a bounded-queue publish would hang forever. Verified in the
    SAME loop: right after aclose, no non-current task may still be pending."""
    runtime, storage = _build_runtime(tmp_path)
    now = datetime.now(timezone.utc)

    async def _drive():
        await storage.sessions.create(
            SessionRecord(
                id="rt-stream-early",
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
        spec = AgentSpec(
            id="agent-rt-stream-early",
            name="rt-stream-agent",
            model=ModelPolicy(primary="test-model"),
            instructions=PromptSpec(instructions="hi"),
            output_schema=str,
        )
        gen = runtime.run_stream(
            spec, "go", session_id="rt-stream-early", run_id="rt-run-early"
        )
        # Take exactly one event, then abandon + close the generator.
        first = await gen.__anext__()
        await gen.aclose()
        # The engine task was force-cancelled by the finally; give it a moment
        # to finish cleaning up (heartbeat cancel/await, controller unregister).
        # A permanently-orphaned task (the bug this guards: one suspended in a
        # saturated bounded-queue publish with no cancel) would NEVER drain and
        # the loop would time out. The deadline is loose on purpose: under
        # suite-level load the engine task's cleanup (heartbeat cancel/await,
        # controller unregister) can take a few seconds, but a truly orphaned
        # task would still never finish within it.
        import time
        deadline = time.monotonic() + 10.0
        pending = ["initial"]
        while pending and time.monotonic() < deadline:
            await asyncio.sleep(0)
            pending = [
                t for t in asyncio.all_tasks() if t is not asyncio.current_task()
            ]
        assert pending == [], f"orphaned engine task still pending: {pending!r}"
        record = await storage.runs.get("rt-run-early")
        assert record is not None
        assert record.status in (
            RunStatus.CANCELLED,
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
        )
        return first

    assert asyncio.run(_drive()) is not None
