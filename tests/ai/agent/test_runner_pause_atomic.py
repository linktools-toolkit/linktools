#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Approval-pause atomicity.

When ``AgentRunner`` is wired with a ``uow_factory`` (SqlAlchemy mode), the
``RunPaused`` handler wraps checkpoint-save + Run-transition(WAITING_APPROVAL)
+ event-append in ONE UnitOfWork so they commit/rollback together. File mode
(``uow_factory=None``) keeps the non-atomic best-effort shape (contract).

These tests prove the contrast:

1. SqlAlchemy happy path: all three pause writes commit through one UoW.
2. SqlAlchemy rollback: a failure in the event append rolls back the
   checkpoint AND the WAITING_APPROVAL transition (atomicity guarantee).
3. File mode: a failure in the (best-effort) event append does NOT roll back
   the checkpoint or the transition -- the non-atomic shape contract documents.
"""
import asyncio
import json
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.models import CompiledAgent
from linktools.ai.agent.runner import AgentRunner
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.events.payloads import RunPaused as RunPausedPayload
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.router import ModelRouter
from linktools.ai.policy.approval import ApprovalRule
from linktools.ai.policy.engine import PolicyEngine
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import RunInput, RunnableType, RunStatus
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage import SqlAlchemyStorage
from linktools.ai.storage.sqlalchemy.event import SqlAlchemyEventStore
from linktools.ai.storage.sqlalchemy.models import Base
from linktools.ai.storage.file.approval import FileApprovalStore
from linktools.ai.storage.file.checkpoint import FileCheckpointStore
from linktools.ai.storage.file.event import FileEventStore
from linktools.ai.storage.file.run import FileRunStore
from linktools.ai.storage.file.session import FileSessionStore
from linktools.ai.tool.executor import ToolExecutor

TOOL_NAME = "risky"


# -- Model fixtures ---------------------------------------------------------


def _model_fn(messages, info: AgentInfo) -> ModelResponse:
    """Always emit a ToolCallPart for the risky tool -- the executor raises
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


def _run_context(run_id, session_id) -> RunContext:
    return RunContext(
        run_id=run_id, root_run_id=run_id, parent_run_id=None, session_id=session_id,
        runnable_id="agent-1", runnable_type=RunnableType.AGENT,
        user_id=None, tenant_id=None, workspace=None,
    )


def _sqlalchemy_storage(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/pause.db")

    async def _create():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()

    asyncio.run(_create())
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/pause.db")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return SqlAlchemyStorage(session_factory=session_factory)


def _seed_session(storage, session_id) -> None:
    now = datetime.now(timezone.utc)
    asyncio.run(storage.sessions.create(SessionRecord(
        id=session_id, parent_id=None, status=SessionStatus.ACTIVE,
        version=1, created_at=now, updated_at=now,
    )))


def _compile_with_storage(storage) -> "tuple[CompiledAgent, ToolExecutor]":
    """Build a pause-enabled ToolExecutor + compiled agent bound to the given
    storage's approval store. The ApprovalRule forces REQUIRE_APPROVAL for the
    risky tool, so the executor raises RunPaused before the tool runs."""
    executor = ToolExecutor(
        policy=PolicyEngine(rules=(ApprovalRule(require_for=frozenset({TOOL_NAME})),)),
        approval_store=storage.approvals,
        pause_on_approval=True,
    )
    compiler = AgentCompiler(
        model_router=ModelRouter(registry=_registry()),
        tool_executor=executor,
    )
    compiled = asyncio.run(compiler.compile(AgentSpec(
        id="agent-1", name="a", model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"), output_schema=str,
    )))

    @compiled.pydantic_agent.tool
    async def risky(ctx, x: int) -> int:  # noqa: ANN001
        return x * 2

    return compiled, executor


def _sqla_runner(storage) -> AgentRunner:
    """AgentRunner wired with SqlAlchemy stores + the UoW factory -- the
    Phase-2D atomic configuration."""
    return AgentRunner(
        run_store=storage.runs,
        session_store=storage.sessions,
        event_store=storage.events,
        checkpoint_store=storage.checkpoints,
        uow_factory=storage.transaction,
    )


async def _collect(gen) -> "list[dict]":
    out: "list[dict]" = []
    async for event in gen:
        out.append(event)
    return out


# -- Tests: SqlAlchemy atomic happy path ------------------------------------


def test_sqla_pause_writes_all_three_operations_atomically_on_success(tmp_path):
    """SqlAlchemy happy path: checkpoint + WAITING_APPROVAL transition +
    RunPaused event all commit through one UoW when the pause completes."""
    storage = _sqlalchemy_storage(tmp_path)
    _seed_session(storage, "session-a1")
    compiled, _ = _compile_with_storage(storage)
    runner = _sqla_runner(storage)

    events = asyncio.run(_collect(runner.run_stream(
        compiled, RunInput(prompt="call the risky tool"),
        _run_context("run-a1", "session-a1"),
    )))
    paused = [e for e in events if e["type"] == "paused"]
    assert len(paused) == 1, f"expected one paused event, got {events}"

    # All three writes committed through the UoW.
    run_record = asyncio.run(storage.runs.get("run-a1"))
    assert run_record.status is RunStatus.WAITING_APPROVAL

    checkpoint = asyncio.run(storage.checkpoints.latest("run-a1"))
    assert checkpoint is not None, "checkpoint not committed by UoW"
    assert checkpoint.payload != b""

    event_page = asyncio.run(storage.events.list("run-a1"))
    payload_types = {type(e.payload).__name__ for e in event_page.items}
    assert "RunPaused" in payload_types
    assert "RunFailed" not in payload_types
    assert "ApprovalRequested" in payload_types


def test_sqla_pause_persists_approval_request_atomically(tmp_path):
    """P0-6/G1 (review3 contract): the ApprovalRequest itself now commits through
    the SAME UoW as checkpoint/transition/event -- ToolExecutor no longer
    persists it directly. Verifies the approval is actually queryable through
    storage.approvals after the pause completes."""
    storage = _sqlalchemy_storage(tmp_path)
    _seed_session(storage, "session-a2")
    compiled, _ = _compile_with_storage(storage)
    runner = _sqla_runner(storage)

    events = asyncio.run(_collect(runner.run_stream(
        compiled, RunInput(prompt="call the risky tool"),
        _run_context("run-a2", "session-a2"),
    )))
    paused = [e for e in events if e["type"] == "paused"]
    assert len(paused) == 1
    approval_id = paused[0]["approval_id"]

    approval = asyncio.run(storage.approvals.get(approval_id))
    assert approval is not None, "ApprovalRequest was not persisted by the UoW"
    assert approval.run_id == "run-a2"
    assert approval.tool_name == TOOL_NAME
    from linktools.ai.agent.approval import ApprovalStatus
    assert approval.status is ApprovalStatus.PENDING


def test_sqla_pause_dedups_repeated_tool_call_id_to_one_pending_approval(tmp_path):
    """G2: pausing twice for the SAME (run_id, tool_call_id) -- e.g. a retried
    lifecycle re-entering the pause path for the same tool call -- must reuse
    the existing PENDING approval rather than creating a second one."""
    storage = _sqlalchemy_storage(tmp_path)
    _seed_session(storage, "session-a3")

    from linktools.ai.agent.approval import build_approval_request
    # Simulate: an approval already exists for this (run_id, tool_call_id)
    # PRIOR to the pause handler running (e.g. a previous partial attempt).
    pre_existing = build_approval_request(
        run_id="run-a3", tool_call_id="tc-fixed", tool_name=TOOL_NAME,
        reason="prior", arguments={"x": 1}, approval_id="approval-fixed",
    )
    asyncio.run(storage.approvals.create(pre_existing))

    result = asyncio.run(storage.approvals.create_or_get_pending(
        run_id="run-a3", tool_call_id="tc-fixed", tool_name=TOOL_NAME,
        reason="new-reason", arguments={"x": 1}, approval_id="approval-different",
    ))
    assert result.id == "approval-fixed", "dedup must return the EXISTING request, not create a new one"
    all_for_run = asyncio.run(storage.approvals.list_for_run("run-a3"))
    assert len(all_for_run) == 1, "a second PENDING approval must not have been created"


def test_sqla_create_or_get_pending_conflicts_on_different_arguments(tmp_path):
    """scenario (contract): the SAME dedupe key (run_id, tool_call_id) reused
    with DIFFERENT tool_name/arguments is a conflict, not a replay -- it must
    raise rather than silently handing back the first call's request."""
    from linktools.ai.agent.approval import build_approval_request
    from linktools.ai.errors import ApprovalConflictError

    storage = _sqlalchemy_storage(tmp_path)
    _seed_session(storage, "session-a4")
    pre_existing = build_approval_request(
        run_id="run-a4", tool_call_id="tc-fixed", tool_name=TOOL_NAME,
        reason="prior", arguments={"x": 1}, approval_id="approval-fixed",
    )
    asyncio.run(storage.approvals.create(pre_existing))

    with pytest.raises(ApprovalConflictError):
        asyncio.run(storage.approvals.create_or_get_pending(
            run_id="run-a4", tool_call_id="tc-fixed", tool_name=TOOL_NAME,
            reason="new-reason", arguments={"x": 2}, approval_id="approval-different",
        ))


def test_sqla_create_or_get_pending_concurrent_calls_create_exactly_one_row(tmp_path):
    """scenario (contract): the ai_approvals.uq_approval_run_tool_call UNIQUE
    constraint is the real backstop -- N concurrent create_or_get_pending
    calls for the SAME (run_id, tool_call_id) must all resolve to the SAME
    persisted row, not each create their own."""
    storage = _sqlalchemy_storage(tmp_path)
    _seed_session(storage, "session-a5")

    async def _attempt(i: int):
        return await storage.approvals.create_or_get_pending(
            run_id="run-a5", tool_call_id="tc-shared", tool_name=TOOL_NAME,
            reason="r", arguments={"x": 1}, approval_id=f"approval-{i}",
        )

    async def _run_all():
        return await asyncio.gather(*(_attempt(i) for i in range(10)))

    results = asyncio.run(_run_all())
    ids = {r.id for r in results}
    assert len(ids) == 1, f"expected exactly one winning approval id, got {ids}"
    all_for_run = asyncio.run(storage.approvals.list_for_run("run-a5"))
    assert len(all_for_run) == 1, "concurrent create_or_get_pending must create exactly one row"


# -- Tests: SqlAlchemy atomic rollback --------------------------------------


def test_sqla_pause_rolls_back_checkpoint_and_transition_when_event_append_fails(
    tmp_path, monkeypatch,
):
    """Atomicity guarantee: when the event append fails INSIDE the UoW, the
    checkpoint-save and WAITING_APPROVAL transition roll back too -- the run
    ends up FAILED (via the outer generic-except handler), with NO checkpoint
    and NO RunPaused event persisted. This is the contract contract: any one of
    the three operations failing undoes all of them.

    The failure is injected on the RunPaused payload append ONLY, so the
    RunStarted append (earlier in execute()) and the RunFailed append (the
    outer generic-except handler) both still succeed -- isolating the
    rollback to the pause-path UoW."""
    storage = _sqlalchemy_storage(tmp_path)
    _seed_session(storage, "session-r1")
    compiled, _ = _compile_with_storage(storage)
    runner = _sqla_runner(storage)

    original_append = SqlAlchemyEventStore.append

    async def _failing_append(self, *, payload, **kwargs):
        if isinstance(payload, RunPausedPayload):
            raise RuntimeError("simulated event append failure")
        return await original_append(self, payload=payload, **kwargs)

    monkeypatch.setattr(SqlAlchemyEventStore, "append", _failing_append)

    # The UoW failure surfaces out of run_stream. pydantic-ai's iter()
    # TaskGroup stores the original RunPaused and may re-raise it during
    # __aexit__ cleanup (masking the inner RuntimeError), so the exact
    # exception type is an implementation detail -- what matters for contract
    # is the resulting STATE (asserted below), not the exception type.
    raised: "BaseException | None" = None
    try:
        asyncio.run(_collect(runner.run_stream(
            compiled, RunInput(prompt="call the risky tool"),
            _run_context("run-r1", "session-r1"),
        )))
    except BaseException as exc:  # noqa: BLE001
        raised = exc
    assert raised is not None, "UoW failure should have surfaced"

    # Atomicity: checkpoint + WAITING_APPROVAL were rolled back. The outer
    # generic-except handler then transitioned the run to FAILED using the
    # pre-pause version (the WAITING_APPROVAL bump did NOT survive).
    run_record = asyncio.run(storage.runs.get("run-r1"))
    assert run_record is not None
    assert run_record.status is RunStatus.FAILED, (
        "WAITING_APPROVAL should have rolled back; run should be FAILED"
    )

    checkpoint = asyncio.run(storage.checkpoints.latest("run-r1"))
    assert checkpoint is None, "checkpoint leaked past UoW rollback"

    event_page = asyncio.run(storage.events.list("run-r1"))
    payload_types = {type(e.payload).__name__ for e in event_page.items}
    assert "RunPaused" not in payload_types, (
        "RunPaused event leaked past UoW rollback"
    )
    assert "ApprovalRequested" not in payload_types, (
        "ApprovalRequested event leaked past UoW rollback"
    )
    # The outer handler's RunFailed append (not a RunPaused payload) succeeded.
    assert "RunFailed" in payload_types

    # P0-6/G1: the ApprovalRequest write shares the SAME UoW, so it must also
    # roll back -- no orphaned PENDING approval left behind after the run
    # ended up FAILED.
    pending = asyncio.run(storage.approvals.list_pending("run-r1"))
    assert pending == (), "ApprovalRequest leaked past UoW rollback"


# -- Tests: File mode stays non-atomic (contract) ------------------------------


def test_file_pause_does_not_rollback_when_event_append_fails(tmp_path):
    """File mode (``uow_factory=None``): cross-store transactions are
    unavailable, so a failure in the best-effort RunPaused event append does
    NOT undo the checkpoint or the WAITING_APPROVAL transition. This is the
    non-atomic shape contract documents -- the inverse of the SqlAlchemy test
    above, and the reason the File path keeps its best-effort try/except."""
    approval_store = FileApprovalStore(root=tmp_path / "approvals")

    class _FailingOnRunPausedEvents:
        """File EventStore wrapper: passes every append through except the
        RunPaused payload, which raises. Mirrors the SqlAlchemy rollback
        test's isolation so the two modes are directly comparable."""

        def __init__(self, inner):
            self._inner = inner

        async def append(self, *, payload, **kwargs):
            if isinstance(payload, RunPausedPayload):
                raise RuntimeError("simulated event append failure")
            return await self._inner.append(payload=payload, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

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
        id="agent-1", name="a", model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"), output_schema=str,
    )))

    @compiled.pydantic_agent.tool
    async def risky(ctx, x: int) -> int:  # noqa: ANN001
        return x * 2

    runner = AgentRunner(
        run_store=FileRunStore(root=tmp_path / "runs"),
        session_store=FileSessionStore(root=tmp_path / "sessions"),
        event_store=_FailingOnRunPausedEvents(FileEventStore(root=tmp_path / "events")),
        checkpoint_store=FileCheckpointStore(root=tmp_path / "checkpoints"),
        # uow_factory deliberately omitted -> File mode non-atomic path.
    )
    now = datetime.now(timezone.utc)
    asyncio.run(runner._session_store.create(SessionRecord(
        id="session-f1", parent_id=None, status=SessionStatus.ACTIVE,
        version=1, created_at=now, updated_at=now,
    )))

    events = asyncio.run(_collect(runner.run_stream(
        compiled, RunInput(prompt="call the risky tool"),
        _run_context("run-f1", "session-f1"),
    )))
    assert any(e["type"] == "paused" for e in events), (
        "File pause should still yield paused despite best-effort event failure"
    )

    # Non-atomic: checkpoint + transition persisted; only the event is missing.
    run_record = asyncio.run(runner._run_store.get("run-f1"))
    assert run_record.status is RunStatus.WAITING_APPROVAL, (
        "File mode transition should persist (non-atomic)"
    )
    checkpoint = asyncio.run(runner._checkpoint_store.latest("run-f1"))
    assert checkpoint is not None, "File mode checkpoint should persist (non-atomic)"

    event_page = asyncio.run(runner._event_store.list("run-f1"))
    payload_types = {type(e.payload).__name__ for e in event_page.items}
    assert "RunPaused" not in payload_types, (
        "best-effort RunPaused append should have failed silently"
    )
