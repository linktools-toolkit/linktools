#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for AgentRunner.run pause path (Task 7).

In the non-streaming path, the unified ``execute()`` lifecycle must catch
``RunPaused`` BEFORE the generic ``except Exception`` handler, transition the
Run to ``WAITING_APPROVAL``, emit a ``RunPaused`` event, save a real
checkpoint, and yield a pause event that ``run()`` re-raises as ``RunPaused``
(caller gets the signal). Phase 2A: ``execute()`` is the single lifecycle for
both ``run()`` and ``run_stream()``.

If ``RunPaused`` were caught by the generic ``except Exception`` handler, the
Run would transition to ``FAILED`` instead of ``WAITING_APPROVAL`` -- this is
the bug the test guards against."""
import asyncio
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.models import CompiledAgent
from linktools.ai.agent.runner import AgentRunner
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.errors import RunPaused
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
    """Always emit a ToolCallPart for the risky tool."""
    return ModelResponse(parts=[ToolCallPart(tool_name=TOOL_NAME, args={"x": 1})])


def _registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_model_fn))
    return registry


def _run_context(run_id="run-n1", session_id="session-n1") -> RunContext:
    return RunContext(
        run_id=run_id, root_run_id=run_id, parent_run_id=None, session_id=session_id,
        runnable_id="agent-1", runnable_type=RunnableType.AGENT,
        user_id=None, tenant_id=None, workspace=None,
    )


def _seed_session(store, session_id) -> None:
    now = datetime.now(timezone.utc)
    asyncio.run(store.create(SessionRecord(
        id=session_id, parent_id=None, status=SessionStatus.ACTIVE,
        version=1, created_at=now, updated_at=now,
    )))


def _make_runner(tmp_path) -> AgentRunner:
    return AgentRunner(
        run_store=FileRunStore(root=tmp_path / "runs"),
        session_store=FileSessionStore(root=tmp_path / "sessions"),
        event_store=FileEventStore(root=tmp_path / "events"),
        checkpoint_store=FileCheckpointStore(root=tmp_path / "checkpoints"),
    )


def _compile(tmp_path, *, agent_id="agent-1") -> "tuple[AgentRunner, CompiledAgent, FileApprovalStore]":
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
    compiled = asyncio.run(compiler.compile(AgentSpec(
        id=agent_id, name="a", model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"), output_schema=str,
    )))
    @compiled.pydantic_agent.tool
    async def risky(ctx, x: int) -> int:  # noqa: ANN001
        return x * 2
    runner = _make_runner(tmp_path)
    return runner, compiled, approval_store


# -- Tests ------------------------------------------------------------------


def test_run_catches_run_paused_transitions_to_waiting_and_reraises(tmp_path):
    """``run()`` raises ``RunPaused`` AND the Run is ``WAITING_APPROVAL`` (not
    ``FAILED`` -- the latter would mean the generic ``except Exception``
    swallowed the pause signal)."""
    runner, compiled, _ = _compile(tmp_path)
    _seed_session(runner._session_store, "session-n1")

    with pytest.raises(RunPaused):
        asyncio.run(runner.run(
            compiled, RunInput(prompt="call the risky tool"), _run_context(),
        ))

    run_record = asyncio.run(runner._run_store.get("run-n1"))
    assert run_record.status is RunStatus.WAITING_APPROVAL


def test_run_pause_carries_approval_id_on_exception(tmp_path):
    """The re-raised ``RunPaused`` carries ``approval_id`` so the caller can
    direct the human to the right approval request."""
    runner, compiled, approval_store = _compile(tmp_path)
    _seed_session(runner._session_store, "session-n1")

    raised: "RunPaused | None" = None
    try:
        asyncio.run(runner.run(
            compiled, RunInput(prompt="call the risky tool"),
            _run_context(run_id="run-n2", session_id="session-n1"),
        ))
    except RunPaused as exc:
        raised = exc

    assert raised is not None
    assert raised.run_id == "run-n2"
    assert raised.approval_id
    # The PENDING approval exists with the matching id.
    approval = asyncio.run(approval_store.get(raised.approval_id))
    assert approval is not None
    assert approval.status.value == "pending"


def test_run_pause_emits_run_paused_event(tmp_path):
    """The pause path emits a ``RunPaused`` event payload (best-effort audit)."""
    runner, compiled, _ = _compile(tmp_path)
    _seed_session(runner._session_store, "session-n1")

    with pytest.raises(RunPaused):
        asyncio.run(runner.run(
            compiled, RunInput(prompt="call the risky tool"),
            _run_context(run_id="run-n3", session_id="session-n1"),
        ))

    events = asyncio.run(runner._event_store.list("run-n3"))
    payload_types = {type(e.payload).__name__ for e in events.items}
    assert "RunPaused" in payload_types
    assert "RunFailed" not in payload_types
