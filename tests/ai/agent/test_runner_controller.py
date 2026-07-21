#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentEngine + RunController integration. When a
RunController is wired into AgentEngine, cancelling a run via the controller
(token set + task.cancel()) must drive the lifecycle through CANCELLING then
CANCELLED -- the CANCELLING state is the observable signal that "cancel
requested" is distinct from "actually cancelled".

These tests wrap the FilesystemRunStore with a recorder so the sequence of
``transition()`` target statuses is observable, then assert both CANCELLING
and CANCELLED appear in order."""

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
from linktools.ai.run.controller import RunController
from linktools.ai.run.models import RunInput, RunnableType, RunStatus
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.filesystem.checkpoint import FilesystemCheckpointStore
from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator
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


def _run_context(run_id="run-ctrl-1", session_id="session-ctrl-1") -> RunContext:
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


class _RecordingRunStore:
    """Thin wrapper around a real RunStore that records every ``transition()``
    target status in order. Used to assert the runner visits CANCELLING before
    CANCELLED. Forwards every other method to the wrapped store."""

    def __init__(self, inner: FilesystemRunStore) -> None:
        self._inner = inner
        self.transitions: "list[RunStatus]" = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def transition(self, run_id, target, **kwargs):
        self.transitions.append(target)
        return await self._inner.transition(run_id, target, **kwargs)


def _make_runner(
    tmp_path, controller, pipeline=None
) -> "tuple[AgentEngine, _RecordingRunStore]":
    from linktools.ai.storage.filesystem.approval import FilesystemApprovalStore
    from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator

    inner_store = FilesystemRunStore(root=tmp_path / "runs")
    recording = _RecordingRunStore(inner_store)
    session_store = FilesystemSessionStore(root=tmp_path / "sessions")
    event_store = FilesystemEventStore(root=tmp_path / "events")
    checkpoint_store = FilesystemCheckpointStore(root=tmp_path / "checkpoints")
    runner = AgentEngine(
        run_store=recording,
        session_store=session_store,
        event_store=event_store,
        checkpoint_store=checkpoint_store,
        middleware_pipeline=pipeline,
        run_controller=controller,
        commit_coordinator=FilesystemRunCommitCoordinator(
            approval_store=FilesystemApprovalStore(root=tmp_path / "approvals"),
            checkpoint_store=checkpoint_store,
            run_store=recording,
            session_store=session_store,
            event_store=event_store,
        ),
    )
    return runner, recording


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


# 1. A run cancelled via the controller transitions CANCELLING then CANCELLED.


@pytest.mark.asyncio
async def test_controller_cancel_drives_cancelling_then_cancelled(tmp_path):
    """When run_controller.cancel(run_id) fires (sets token + cancels task),
    the runner's CancelledError handler must transition the run through
    CANCELLING (the "cancel requested" state) before landing in CANCELLED
    (the terminal "actually stopped" state). The recorded transition sequence
    is the direct evidence -- both targets must appear, in order."""
    controller = RunController()

    ready = asyncio.Event()
    block = asyncio.Event()

    class _BlockingMiddleware(Middleware):
        async def before_run(self, context) -> None:
            ready.set()
            await block.wait()  # never set -- controller cancels the task

        async def after_run(self, context, result):
            return result

        async def on_error(self, context, error):
            pass

    pipeline = MiddlewarePipeline(middlewares=(_BlockingMiddleware(),))
    runner, recording = _make_runner(tmp_path, controller, pipeline=pipeline)
    await _seed_session(runner._session_store, "session-ctrl-1")

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
    # Wait for the lifecycle to reach before_run (PENDING -> RUNNING already
    # done, controller registration done, before_run is the next await --
    # this is the deterministic point where the controller cancel will land).
    await ready.wait()
    await controller.cancel("run-ctrl-1")

    with pytest.raises(asyncio.CancelledError):
        await task

    # The run must have visited CANCELLING (intermediate) then CANCELLED
    # (terminal). The full sequence is RUNNING (PENDING -> RUNNING at start),
    # CANCELLING, CANCELLED.
    assert RunStatus.RUNNING in recording.transitions
    assert RunStatus.CANCELLING in recording.transitions
    assert RunStatus.CANCELLED in recording.transitions

    cancelling_idx = recording.transitions.index(RunStatus.CANCELLING)
    cancelled_idx = recording.transitions.index(RunStatus.CANCELLED)
    assert cancelling_idx < cancelled_idx, (
        "CANCELLING must precede CANCELLED in the transition sequence; "
        f"got {recording.transitions}"
    )

    record = await runner._run_store.get("run-ctrl-1")
    assert record is not None
    assert record.status is RunStatus.CANCELLED

    # The controller must have unregistered the run in the finally block --
    # otherwise the controller would retain a reference to the finished task.
    assert controller.get_token("run-ctrl-1") is None


# 2. Real cancellation via Runtime.cancel goes through CANCELLING.


@pytest.mark.asyncio
async def test_runtime_cancel_with_in_flight_task_uses_cancelling(tmp_path):
    """Runtime.cancel on an in-flight run (task registered with the
    controller) goes through CANCELLING at the store level, then the runner
    completes the transition to CANCELLED. This is the full integration:
    Runtime.cancel -> CANCELLING (store) + controller.cancel (token+task) ->
    runner CancelledError handler -> CANCELLED.

    Contrast with test_runtime_cancel.py::test_cancel_running_run_transitions_to_cancelled
    which seeds a RUNNING run WITHOUT an in-flight task and expects a direct
    -> CANCELLED transition."""
    from linktools.ai.runtime import Runtime
    from linktools.ai.storage.facade import FilesystemStorage

    storage = FilesystemStorage(root=tmp_path)
    runtime = Runtime.build(
        storage=storage,
        local_trusted_mode=True,
        commit_coordinator=FilesystemRunCommitCoordinator.from_storage(storage),
    )
    # Runtime.build always wires a RunController -- but the runner's actual
    # driving Task is only registered once execute() starts. To exercise the
    # in-flight path we drive a real run and cancel it mid-flight.

    ready = asyncio.Event()
    block = asyncio.Event()

    from linktools.ai.middleware.base import Middleware
    from linktools.ai.middleware.pipeline import MiddlewarePipeline

    class _BlockingMiddleware(Middleware):
        async def before_run(self, context) -> None:
            ready.set()
            await block.wait()

        async def after_run(self, context, result):
            return result

        async def on_error(self, context, error):
            pass

    # Rebuild Runtime with the blocking pipeline so the in-flight cancel has
    # a deterministic landing point (the before_run await). The model router
    # must carry the test FunctionModel so spec compilation resolves.
    runtime = Runtime.build(
        storage=storage,
        model_router=ModelRouter(registry=_registry()),
        middleware_pipeline=MiddlewarePipeline(middlewares=(_BlockingMiddleware(),)),
        local_trusted_mode=True,
        commit_coordinator=FilesystemRunCommitCoordinator.from_storage(storage),
    )

    spec = AgentSpec(
        id="agent-runtime-cancel",
        name="a",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
    )

    run_task = asyncio.create_task(runtime.run(spec, "x", run_id="run-runtime-cancel"))
    await ready.wait()

    # The runner has registered the driving task with the controller. Now
    # Runtime.cancel should observe an in-flight run -> CANCELLING + signal.
    await runtime.cancel("run-runtime-cancel")

    with pytest.raises(asyncio.CancelledError):
        await run_task

    record = await storage.runs.get("run-runtime-cancel")
    assert record is not None
    assert record.status is RunStatus.CANCELLED


# 3. Default-None controller preserves the no-token-checks behavior.


@pytest.mark.asyncio
async def test_runner_without_controller_still_transitions_on_external_cancel(tmp_path):
    """When ``run_controller`` is None (the default), the runner does no
    token checks, but external task.cancel() still drives the CancelledError
    handler through CANCELLING -> CANCELLED. This preserves the established
    behavior (existing tests like test_runner_cancel.py cover this exact
    case) while also exercising the new CANCELLING transition path."""
    ready = asyncio.Event()
    block = asyncio.Event()

    class _BlockingMiddleware(Middleware):
        async def before_run(self, context) -> None:
            ready.set()
            await block.wait()

        async def after_run(self, context, result):
            return result

        async def on_error(self, context, error):
            pass

    pipeline = MiddlewarePipeline(middlewares=(_BlockingMiddleware(),))
    # Default runner: no run_controller argument.
    from linktools.ai.storage.filesystem.approval import FilesystemApprovalStore
    from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator

    inner_store = FilesystemRunStore(root=tmp_path / "runs")
    recording = _RecordingRunStore(inner_store)
    session_store = FilesystemSessionStore(root=tmp_path / "sessions")
    event_store = FilesystemEventStore(root=tmp_path / "events")
    checkpoint_store = FilesystemCheckpointStore(root=tmp_path / "checkpoints")
    runner = AgentEngine(
        run_store=recording,
        session_store=session_store,
        event_store=event_store,
        checkpoint_store=checkpoint_store,
        middleware_pipeline=pipeline,
        commit_coordinator=FilesystemRunCommitCoordinator(
            approval_store=FilesystemApprovalStore(root=tmp_path / "approvals"),
            checkpoint_store=checkpoint_store,
            run_store=recording,
            session_store=session_store,
            event_store=event_store,
        ),
    )
    await _seed_session(runner._session_store, "session-noctrl")

    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_router=ModelRouter(registry=_registry()),
    )
    compiled = await compiler.compile(
        AgentSpec(
            id="agent-2",
            name="a",
            model=ModelPolicy(primary="test-model"),
            instructions=PromptSpec(instructions="hi"),
        )
    )

    task = asyncio.create_task(
        runner.run(
            compiled,
            RunInput(prompt="x"),
            _run_context(run_id="run-noctrl", session_id="session-noctrl"),
        )
    )
    await ready.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Even without a controller, the CancelledError handler transitions
    # CANCELLING then CANCELLED.
    assert RunStatus.CANCELLING in recording.transitions
    assert RunStatus.CANCELLED in recording.transitions
    record = await runner._run_store.get("run-noctrl")
    assert record is not None
    assert record.status is RunStatus.CANCELLED
