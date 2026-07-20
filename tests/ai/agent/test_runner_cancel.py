#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentEngine CancelledError handling. When the caller cancels the
asyncio.Task driving runner.run(), CancelledError surfaces at the current await
point inside the lifecycle try/except. The handler transitions the RunRecord to
CANCELLED (best-effort) and re-raises so the asyncio machinery observes the
cancellation. This file drives that path end-to-end: a blocking middleware
gives the test a deterministic await point to cancel against."""

import asyncio
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.runner import AgentEngine
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.middleware.base import Middleware
from linktools.ai.middleware.pipeline import MiddlewarePipeline
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.router import ModelRouter
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import RunInput, RunnableType, RunStatus
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.filesystem.checkpoint import FilesystemCheckpointStore
from linktools.ai.storage.filesystem.event import FilesystemEventStore
from linktools.ai.storage.filesystem.run import FilesystemRunStore
from linktools.ai.storage.filesystem.session import FilesystemSessionStore
from linktools.ai.governance.policy.engine import PolicyEngine
from linktools.ai.tool.executor import GovernedToolInvoker


def _model_fn(messages, info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content='{"response": {"answer": 42}}')])


def _registry():
    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_model_fn))
    return registry


def _run_context(run_id="run-cancel-1", session_id="session-cancel-1") -> RunContext:
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


def _make_runner(tmp_path, pipeline=None) -> AgentEngine:
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
        checkpoint_store=checkpoint_store,
        middleware_pipeline=pipeline,
        commit_coordinator=FilesystemRunCommitCoordinator(
            approval_store=FilesystemApprovalStore(root=tmp_path / "approvals"),
            checkpoint_store=checkpoint_store,
            run_store=run_store,
            session_store=session_store,
            event_store=event_store,
        ),
    )


async def _seed_session(store, session_id) -> None:
    now = datetime.now(timezone.utc)
    await store.create(
        SessionRecord(
            id=session_id,
            parent_id=None,
            status=SessionStatus.ACTIVE,
            version=1,
            created_at=now,
            updated_at=now,
        )
    )


# 4. A run whose driving asyncio.Task is cancelled mid-execution transitions to
#    CANCELLED and re-raises CancelledError.


def test_run_cancelled_mid_lifecycle_transitions_to_cancelled(tmp_path):
    async def _run():
        # The blocking middleware gives the test a deterministic await point:
        # before_run sets `ready` then awaits `block.wait()` forever. The test
        # waits for `ready`, cancels the task, and asserts CancelledError +
        # CANCELLED status. Events are created inside the running loop so they
        # bind to the correct loop.
        ready = asyncio.Event()
        block = asyncio.Event()

        class _BlockingMiddleware(Middleware):
            async def before_run(self, context) -> None:
                ready.set()
                await block.wait()  # never set -- test cancels the task

            async def after_run(self, context, result):
                return result

            async def on_error(self, context, error):
                pass

        pipeline = MiddlewarePipeline(middlewares=(_BlockingMiddleware(),))
        runner = _make_runner(tmp_path, pipeline=pipeline)
        await _seed_session(runner._session_store, "session-cancel-1")

        compiler = AgentCompiler(
            tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
            model_router=ModelRouter(registry=_registry()),
        )
        compiled = await compiler.compile(
            AgentSpec(
                id="agent-1",
                name="a",
                model=ModelPolicy(primary="test-model"),
                instructions=PromptSpec(instructions="hi"),
            )
        )

        task = asyncio.create_task(
            runner.run(compiled, RunInput(prompt="x"), _run_context())
        )
        # Wait until the task has entered the lifecycle try block (PENDING ->
        # RUNNING transition + RunStarted event already done; before_run is
        # the next await, inside the try). Then cancel.
        await ready.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        record = await runner._run_store.get("run-cancel-1")
        return record

    record = asyncio.run(_run())
    assert record is not None
    assert record.status is RunStatus.CANCELLED
