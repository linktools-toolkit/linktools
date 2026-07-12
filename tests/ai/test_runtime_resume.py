#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Runtime.resume -- the pause/approve/resume round trip (scenario).

Runtime.resume(run_id, spec) loads a paused run, deserializes its checkpoint's
message history, transitions WAITING_APPROVAL -> RUNNING, and re-enters
AgentRunner.run_stream with message_history=<deserialized messages>. The
ToolExecutor's resume gate (_already_approved) recognizes the now-APPROVED
request and lets the tool execute instead of re-raising RunPaused.

The full round trip at the Runtime level:

  1. ``runtime.run_stream(spec, "call risky", ...)`` -- the model emits a
     ToolCallPart for "risky"; the executor (pause_on_approval=True) raises
     RunPaused; run_stream checkpoints the partial message history and
     yields ``{"type": "paused", ...}``.
  2. ``storage.approvals.approve(approval_id, ...)`` -- the human approves.
  3. ``runtime.resume(run_id, spec)`` -- deserializes the checkpoint,
     transitions WAITING_APPROVAL -> RUNNING, re-enters run_stream with
     message_history. The graph resumes: the pending tool call executes
     (resume gate lets it through), the model emits "done", and the run
     SUCCEEDS.

All three phases run inside one ``asyncio.run`` so the FileApprovalStore's
``asyncio.Lock`` stays bound to one event loop (the lock is acquired during
both the pause create and the approve resolve)."""

import asyncio
import json
from datetime import datetime, timezone

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
from linktools.ai.capability.models import CapabilityBundle
from linktools.ai.capability.provider import CapabilityProvider
from linktools.ai.tool.models import (
    ManagedToolDefinition,
    ToolContribution,
    ToolDescriptor,
)
from linktools.ai.providers.bundle import ProviderBundle
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.errors import InvalidRunTransitionError, RunNotFoundError
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.router import ModelRouter
from linktools.ai.policy.approval import ApprovalRule
from linktools.ai.policy.engine import PolicyEngine
from linktools.ai.run.models import RunInput, RunnableType, RunRecord, RunStatus
from linktools.ai.runtime import Runtime
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.facade import FileStorage
from linktools.ai.tool.executor import ToolExecutor

TOOL_NAME = "risky"


class _RiskyProvider(CapabilityProvider):
    supported_kinds = ("test",)

    async def resolve(self, ref, context):
        async def risky(x: int) -> int:
            return x * 2

        return CapabilityBundle(
            tool_contributions=(
                ToolContribution(
                    tools=(
                        ManagedToolDefinition(
                            descriptor=ToolDescriptor(
                                name=TOOL_NAME,
                                source="test",
                                category="discovery",
                                risk="high",
                                mutating=False,
                            ),
                            handler=risky,
                        ),
                    )
                ),
            )
        )


# -- Model fixtures ---------------------------------------------------------


def _model_fn(messages, info: AgentInfo) -> ModelResponse:
    """Turn 1: emit a ToolCallPart for the risky tool. Turn 2 (after the tool
    has returned): emit TextPart("done"). The turn is determined by whether
    the history already contains a ToolReturnPart for risky."""
    for m in messages:
        parts = getattr(m, "parts", None) or []
        for p in parts:
            if (
                isinstance(p, ToolReturnPart)
                and getattr(p, "tool_name", None) == TOOL_NAME
            ):
                return ModelResponse(parts=[TextPart(content="done")])
    return ModelResponse(parts=[ToolCallPart(tool_name=TOOL_NAME, args={"x": 1})])


async def _stream_fn(messages, info: AgentInfo):
    for m in messages:
        parts = getattr(m, "parts", None) or []
        for p in parts:
            if (
                isinstance(p, ToolReturnPart)
                and getattr(p, "tool_name", None) == TOOL_NAME
            ):
                yield "done"
                return
    yield {0: DeltaToolCall(name=TOOL_NAME, json_args=json.dumps({"x": 1}))}


def _registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(
        "test-model",
        model=FunctionModel(_model_fn, stream_function=_stream_fn),
    )
    return registry


def _spec() -> AgentSpec:
    return AgentSpec(
        id="agent-resume",
        name="resume-agent",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
        output_schema=str,
        tools=(ToolRef(kind="test", name=TOOL_NAME),),
    )


def _build_runtime(tmp_path) -> "tuple[Runtime, FileStorage]":
    storage = FileStorage(root=tmp_path)
    executor = ToolExecutor(
        policy=PolicyEngine(rules=(ApprovalRule(require_for=frozenset({TOOL_NAME})),)),
        approval_store=storage.approvals,
        pause_on_approval=True,
    )
    runtime = Runtime.build(
        storage=storage,
        model_router=ModelRouter(registry=_registry()),
        tool_executor=executor,
        providers=ProviderBundle(capabilities=(_RiskyProvider(),)),
    )
    return runtime, storage


async def _collect(gen) -> "list[dict]":
    out: "list[dict]" = []
    async for event in gen:
        out.append(event)
    return out


# -- Tests ------------------------------------------------------------------


def test_resume_round_trip_pause_approve_resume_succeeds(tmp_path):
    """Full pause -> approve -> resume -> SUCCEEDED round trip at the Runtime
    level. All three phases run inside one event loop so the
    FileApprovalStore's asyncio.Lock stays bound to one loop."""

    async def _drive():
        runtime, storage = _build_runtime(tmp_path)
        spec = _spec()

        # Create a session.
        now = datetime.now(timezone.utc)
        await storage.sessions.create(
            SessionRecord(
                id="session-r1",
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )

        # run_stream pauses on the risky tool.
        pause_events = await _collect(
            runtime.run_stream(
                spec,
                "call risky",
                session_id="session-r1",
                run_id="run-r1",
            )
        )
        paused = next(e for e in pause_events if e["type"] == "paused")
        approval_id = paused["approval_id"]
        assert paused["run_id"] == "run-r1"

        # Run is WAITING_APPROVAL (NOT FAILED).
        record = await storage.runs.get("run-r1")
        assert record.status is RunStatus.WAITING_APPROVAL

        # human approves the pending request.
        await storage.approvals.approve(
            approval_id,
            expected_version=1,
            resolved_by="test",
        )

        # resume re-enters run_stream with message_history.
        resume_events = await _collect(runtime.resume("run-r1", spec))

        # Final record: SUCCEEDED.
        final = await storage.runs.get("run-r1")
        return resume_events, final

    resume_events, final_record = asyncio.run(_drive())

    # "resumed" signal is yielded first.
    assert resume_events[0]["type"] == "resumed"
    assert resume_events[0]["run_id"] == "run-r1"

    # Tool executed (end event with ok=True).
    tool_ends = [
        e for e in resume_events if e.get("type") == "tool" and e.get("phase") == "end"
    ]
    assert any(e["name"] == TOOL_NAME and e["ok"] is True for e in tool_ends), (
        f"expected tool end event for {TOOL_NAME} with ok=True, got {resume_events}"
    )

    # Text event present.
    text_events = [e for e in resume_events if e.get("type") == "text"]
    assert any("done" in e["text"] for e in text_events), (
        f"expected text event 'done', got {resume_events}"
    )

    # Run transitioned to SUCCEEDED.
    assert final_record.status is RunStatus.SUCCEEDED


def test_resume_run_not_found_raises(tmp_path):
    """Resume on a non-existent run raises RunNotFoundError."""
    runtime, _ = _build_runtime(tmp_path)
    spec = _spec()

    async def _drive():
        try:
            async for _ in runtime.resume("nonexistent", spec):
                pass
            return None
        except RunNotFoundError:
            return True

    assert asyncio.run(_drive()) is True


def test_resume_not_waiting_approval_raises(tmp_path):
    """Resume on a run that is SUCCEEDED (not WAITING_APPROVAL) raises
    InvalidRunTransitionError."""
    runtime, storage = _build_runtime(tmp_path)
    spec = _spec()

    async def _seed_and_resume():
        now = datetime.now(timezone.utc)
        await storage.runs.create(
            RunRecord(
                id="run-done",
                root_run_id="run-done",
                parent_run_id=None,
                session_id="session-x",
                runnable_id="agent-resume",
                runnable_type=RunnableType.AGENT,
                status=RunStatus.SUCCEEDED,
                input=RunInput(prompt=""),
                result=None,
                error=None,
                version=1,
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
        try:
            async for _ in runtime.resume("run-done", spec):
                pass
            return None
        except InvalidRunTransitionError:
            return True

    assert asyncio.run(_seed_and_resume()) is True


def test_resume_no_checkpoint_raises(tmp_path):
    """Resume on a WAITING_APPROVAL run without a checkpoint raises
    RunNotFoundError."""
    runtime, storage = _build_runtime(tmp_path)
    spec = _spec()

    async def _seed_and_resume():
        now = datetime.now(timezone.utc)
        await storage.runs.create(
            RunRecord(
                id="run-paused-nockpt",
                root_run_id="run-paused-nockpt",
                parent_run_id=None,
                session_id="session-y",
                runnable_id="agent-resume",
                runnable_type=RunnableType.AGENT,
                status=RunStatus.WAITING_APPROVAL,
                input=RunInput(prompt=""),
                result=None,
                error=None,
                version=1,
                created_at=now,
                started_at=now,
                finished_at=None,
            )
        )
        try:
            async for _ in runtime.resume("run-paused-nockpt", spec):
                pass
            return None
        except RunNotFoundError:
            return True

    assert asyncio.run(_seed_and_resume()) is True
