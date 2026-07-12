#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for AgentRunner.run_stream pause path (scenario).

When ``ToolExecutor.pause_on_approval=True`` and policy returns
REQUIRE_APPROVAL, the executor raises ``RunPaused``. PolicyCapability lets it
propagate (it's a ``RunError``, not a ``ToolError``) out of pydantic-ai's
tool-execution stack to ``AgentRunner.run_stream``, which must:

  1. Catch ``RunPaused`` INSIDE the ``async with agent.iter(prompt) as run:``
     block (so ``run`` is bound and ``run.all_messages()`` works).
  2. Save a real checkpoint (``serialize_messages`` of the partial history).
  3. Transition the Run ``RUNNING -> WAITING_APPROVAL``.
  4. Yield ``{"type": "paused", "run_id": ..., "approval_id": ...}``.
  5. ``return`` (stop the generator cleanly, do NOT re-raise).

The model emits a ``ToolCallPart`` for a tool whose ``ApprovalRule`` returns
REQUIRE_APPROVAL; pydantic-ai tries to execute the tool;
``PolicyCapability.before_tool_execute`` calls ``executor.check`` which raises
``RunPaused``; that propagates out of the ``async for node in run:`` loop into
the new ``except RunPaused`` handler."""

import asyncio
import json
from datetime import datetime, timezone

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from linktools.ai.agent.checkpoint import deserialize_messages
from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.models import CompiledAgent
from linktools.ai.agent.runner import AgentRunner
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.router import ModelRouter
from linktools.ai.policy.approval import ApprovalRule
from linktools.ai.policy.engine import PolicyEngine
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import RunInput, RunnableType, RunStatus
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.file.approval import FileApprovalStore
from linktools.ai.storage.file.checkpoint import FileCheckpointStore
from linktools.ai.storage.file.event import FileEventStore
from linktools.ai.storage.file.run import FileRunStore
from linktools.ai.storage.file.session import FileSessionStore
from linktools.ai.tool.executor import ToolExecutor

TOOL_NAME = "risky"


# -- Model fixtures ---------------------------------------------------------


def _model_fn(messages, info: AgentInfo) -> ModelResponse:
    """Always emit a ToolCallPart for the risky tool -- the executor will raise
    RunPaused before the tool returns, so the model is only called once."""
    return ModelResponse(parts=[ToolCallPart(tool_name=TOOL_NAME, args={"x": 1})])


async def _stream_fn(messages, info: AgentInfo):
    yield {0: DeltaToolCall(name=TOOL_NAME, json_args=json.dumps({"x": 1}))}


def _registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(
        "test-model", model=FunctionModel(_model_fn, stream_function=_stream_fn)
    )
    return registry


def _run_context(run_id="run-p1", session_id="session-p1") -> RunContext:
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


def _make_runner(tmp_path, *, approval_store=None) -> AgentRunner:
    return AgentRunner(
        run_store=FileRunStore(root=tmp_path / "runs"),
        session_store=FileSessionStore(root=tmp_path / "sessions"),
        event_store=FileEventStore(root=tmp_path / "events"),
        checkpoint_store=FileCheckpointStore(root=tmp_path / "checkpoints"),
        approval_store=approval_store,
    )


def _compile(
    tmp_path, *, agent_id="agent-1"
) -> "tuple[AgentRunner, CompiledAgent, FileApprovalStore]":
    approval_store = FileApprovalStore(root=tmp_path / "approvals")
    executor = ToolExecutor(
        policy=PolicyEngine(rules=(ApprovalRule(require_for=frozenset({TOOL_NAME})),)),
        approval_store=approval_store,
        pause_on_approval=True,
    )
    compiler = AgentCompiler(
        model_router=ModelRouter(registry=_registry()),
        tool_executor=executor,
    )
    compiled = asyncio.run(
        compiler.compile(
            AgentSpec(
                id=agent_id,
                name="a",
                model=ModelPolicy(primary="test-model"),
                instructions=PromptSpec(instructions="hi"),
                output_schema=str,
            )
        )
    )

    # Register a real pydantic-ai tool whose name matches the ApprovalRule.
    @compiled.pydantic_agent.tool
    async def risky(ctx, x: int) -> int:  # noqa: ANN001
        return x * 2

    # P0-6/G1: the runner (not the executor) persists the ApprovalRequest now
    # -- it must share the SAME approval_store instance the test asserts against.
    runner = _make_runner(tmp_path, approval_store=approval_store)
    return runner, compiled, approval_store


async def _collect(gen) -> "list[dict]":
    out: "list[dict]" = []
    async for event in gen:
        out.append(event)
    return out


# -- Tests ------------------------------------------------------------------


def test_run_stream_catches_run_paused_and_yields_paused_event(tmp_path):
    runner, compiled, approval_store = _compile(tmp_path)
    _seed_session(runner._session_store, "session-p1")

    events = asyncio.run(
        _collect(
            runner.run_stream(
                compiled,
                RunInput(prompt="call the risky tool"),
                _run_context(),
            )
        )
    )

    paused_events = [e for e in events if e["type"] == "paused"]
    assert len(paused_events) == 1, f"expected one paused event, got {events}"
    paused = paused_events[0]
    assert paused["run_id"] == "run-p1"
    approval_id = paused["approval_id"]
    assert approval_id

    # Run transitioned to WAITING_APPROVAL (NOT FAILED).
    run_record = asyncio.run(runner._run_store.get("run-p1"))
    assert run_record.status is RunStatus.WAITING_APPROVAL

    # ApprovalStore has a PENDING request with the matching approval_id.
    approval = asyncio.run(approval_store.get(approval_id))
    assert approval is not None
    assert approval.status.value == "pending"
    assert approval.tool_name == TOOL_NAME


def test_run_stream_pause_saves_real_checkpoint_with_model_response(tmp_path):
    runner, compiled, approval_store = _compile(tmp_path)
    _seed_session(runner._session_store, "session-p1")

    events = asyncio.run(
        _collect(
            runner.run_stream(
                compiled,
                RunInput(prompt="call the risky tool"),
                _run_context(run_id="run-p2", session_id="session-p1"),
            )
        )
    )

    paused = next(e for e in events if e["type"] == "paused")
    approval_id = paused["approval_id"]

    checkpoint = asyncio.run(runner._checkpoint_store.latest("run-p2"))
    assert checkpoint is not None, "no checkpoint saved for paused run"
    # Payload is non-empty real serialization (not the empty-bytes placeholder).
    assert checkpoint.payload != b""
    messages = deserialize_messages(checkpoint.payload)
    assert len(messages) > 0
    # The ModelResponse carrying the ToolCallPart is in the history.
    model_responses = [m for m in messages if isinstance(m, ModelResponse)]
    assert len(model_responses) >= 1
    has_risky_call = any(
        any(getattr(p, "tool_name", None) == TOOL_NAME for p in m.parts)
        for m in model_responses
    )
    assert has_risky_call, "no ToolCallPart for risky in checkpoint messages"
    # Checkpoint metadata carries the approval_id linking it to the pause UI.
    assert checkpoint.metadata.get("approval_id") == approval_id


def test_run_stream_pause_does_not_yield_terminal_events(tmp_path):
    """The paused path yields only the paused event -- no completed/failed."""
    runner, compiled, _ = _compile(tmp_path)
    _seed_session(runner._session_store, "session-p1")

    events = asyncio.run(
        _collect(
            runner.run_stream(
                compiled,
                RunInput(prompt="call the risky tool"),
                _run_context(run_id="run-p3", session_id="session-p1"),
            )
        )
    )

    types = [e["type"] for e in events]
    assert "paused" in types
    assert "completed" not in types
    assert "failed" not in types
