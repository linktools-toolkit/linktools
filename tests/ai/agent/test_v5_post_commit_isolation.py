#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""BUG-04 (v5 guide §11): post-commit observation hooks (after_run, metrics,
on_error) must be isolated from the run's terminal state.

- after_run runs BEFORE the commit, so its failure takes the normal FAILED
  path instead of corrupting an already-SUCCEEDED run.
- success metrics are best-effort: a metrics failure never flips a committed
  run to FAILED or changes the caller's result.
- on_error failure never replaces the original exception the caller needs.
- a run that already reached a commit point (SUCCEEDED / WAITING_APPROVAL)
  never gets a contradictory second terminal transition or RunFailed event.

Each test drives a real AgentEngine with a recording commit coordinator and
scripted middleware/metrics. The after_run and on_error tests fail under the
old order (after_run after complete; on_error replacing the original error)."""

import asyncio
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.runner import AgentEngine
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.router import ModelRouter
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import RunInput, RunnableType, RunStatus
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.filesystem.approval import FilesystemApprovalStore
from linktools.ai.storage.filesystem.checkpoint import FilesystemCheckpointStore
from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator
from linktools.ai.storage.filesystem.event import FilesystemEventStore
from linktools.ai.storage.filesystem.run import FilesystemRunStore
from linktools.ai.storage.filesystem.session import FilesystemSessionStore
from linktools.ai.governance.policy.engine import PolicyEngine
from linktools.ai.tool.executor import GovernedToolInvoker


def _model_fn_ok():
    def _fn(messages, info: AgentInfo) -> ModelResponse:  # noqa: ARG001
        return ModelResponse(parts=[TextPart(content='{"response": {"answer": 42}}')])

    return _fn


def _registry(model_fn):
    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(model_fn))
    return registry


def _run_context(run_id="run-1", session_id="session-1") -> RunContext:
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


class _RecordingCoordinator:
    """Wraps the real coordinator; records whether complete() was called."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.complete_calls = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def complete(self, command):
        self.complete_calls += 1
        return await self._inner.complete(command)


class _ScriptedPipeline:
    def __init__(self, *, after_run_exc=None, on_error_exc=None) -> None:
        self.after_run_exc = after_run_exc
        self.on_error_exc = on_error_exc
        self.on_error_calls = 0

    async def run_before_run(self, context):  # noqa: ARG002
        return None

    async def run_after_run(self, context, result):  # noqa: ARG002
        if self.after_run_exc is not None:
            raise self.after_run_exc

    async def run_on_error(self, context, exc):  # noqa: ARG002
        self.on_error_calls += 1
        if self.on_error_exc is not None:
            raise self.on_error_exc


class _FailingMetrics:
    def __init__(self, exc) -> None:
        self._exc = exc

    def counter(self, name, attributes=None):  # noqa: ARG002
        raise self._exc

    def histogram(self, name, value=None, attributes=None):  # noqa: ARG002
        raise self._exc


def _build(tmp_path, *, pipeline=None, metrics=None):
    run_store = FilesystemRunStore(root=tmp_path / "runs")
    session_store = FilesystemSessionStore(root=tmp_path / "sessions")
    event_store = FilesystemEventStore(root=tmp_path / "events")
    checkpoint_store = FilesystemCheckpointStore(root=tmp_path / "checkpoints")
    coordinator = _RecordingCoordinator(
        FilesystemRunCommitCoordinator(
            approval_store=FilesystemApprovalStore(root=tmp_path / "approvals"),
            checkpoint_store=checkpoint_store,
            run_store=run_store,
            session_store=session_store,
            event_store=event_store,
        )
    )
    runner = AgentEngine(
        run_store=run_store,
        session_store=session_store,
        event_store=event_store,
        checkpoint_store=checkpoint_store,
        middleware_pipeline=pipeline,
        metrics=metrics,
        commit_coordinator=coordinator,
    )
    return runner, coordinator, run_store, event_store, session_store


def _seed(session_store, session_id="session-1") -> None:
    now = datetime.now(timezone.utc)
    asyncio.run(
        session_store.create(
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


def _compiled(model_fn):
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_router=ModelRouter(registry=_registry(model_fn)),
    )
    return asyncio.run(
        compiler.compile(
            __import__(
                "linktools.ai.agent.spec", fromlist=["AgentSpec", "PromptSpec"]
            ).AgentSpec(
                id="agent-1",
                name="a",
                model=ModelPolicy(primary="test-model"),
                instructions=__import__(
                    "linktools.ai.agent.spec", fromlist=["PromptSpec"]
                ).PromptSpec(instructions="hi"),
                output_schema=str,
            )
        )
    )


def _payload_types(event_store, run_id) -> set:
    page = asyncio.run(event_store.list(run_id, after_sequence=0, limit=10000))
    return {type(e.payload).__name__ for e in page.items}


def test_after_run_failure_runs_failed_without_committing(tmp_path):
    pipeline = _ScriptedPipeline(after_run_exc=RuntimeError("after_run boom"))
    runner, coordinator, run_store, event_store, session_store = _build(
        tmp_path, pipeline=pipeline
    )
    _seed(session_store)

    with pytest.raises(RuntimeError, match="after_run boom"):
        asyncio.run(
            runner.run(_compiled(_model_fn_ok()), RunInput(prompt="go"), _run_context())
        )

    run = asyncio.run(run_store.get("run-1"))
    assert run.status is RunStatus.FAILED
    # after_run runs BEFORE complete -> the run is not committed.
    assert coordinator.complete_calls == 0
    types = _payload_types(event_store, "run-1")
    assert "RunCompleted" not in types
    assert "RunFailed" in types


def test_success_metrics_failure_keeps_run_succeeded(tmp_path):
    metrics = _FailingMetrics(RuntimeError("metrics boom"))
    runner, _coord, run_store, event_store, session_store = _build(
        tmp_path, metrics=metrics
    )
    _seed(session_store)

    result = asyncio.run(
        runner.run(_compiled(_model_fn_ok()), RunInput(prompt="go"), _run_context())
    )

    # The caller still gets the result; the run stays SUCCEEDED.
    assert result is not None
    run = asyncio.run(run_store.get("run-1"))
    assert run.status is RunStatus.SUCCEEDED
    types = _payload_types(event_store, "run-1")
    assert "RunCompleted" in types
    assert "RunFailed" not in types


def test_on_error_failure_preserves_original_exception(tmp_path):
    def _boom(messages, info: AgentInfo):  # noqa: ARG001
        raise ValueError("original model failure")

    pipeline = _ScriptedPipeline(on_error_exc=RuntimeError("on_error boom"))
    runner, _coord, run_store, _event_store, session_store = _build(
        tmp_path, pipeline=pipeline
    )
    _seed(session_store)

    with pytest.raises(ValueError, match="original model failure"):
        asyncio.run(runner.run(_compiled(_boom), RunInput(prompt="go"), _run_context()))

    run = asyncio.run(run_store.get("run-1"))
    assert run.status is RunStatus.FAILED
    assert pipeline.on_error_calls == 1
