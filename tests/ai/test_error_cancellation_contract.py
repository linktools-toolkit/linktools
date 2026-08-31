#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cancellation and capability error-boundary regressions."""

import asyncio
from types import SimpleNamespace
from typing import get_args, get_origin, get_type_hints

import pytest
from linktools.ai.asset._sql import SqlAssetBackend
from linktools.ai.capability import SubagentDelegate, materialize_mcp_servers
from linktools.ai.core import ExecutionStatus, Principal, ResourceKind, ResourceRef
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._evaluation import DefaultEvaluationService
from linktools.ai.runtime._execution import DefaultExecutionService
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime._planner import _AgentTaskNodeHandler
from linktools.ai.runtime._session import DefaultSessionService
from linktools.ai.runtime._subagent import SubagentDispatcher
from linktools.ai.spec import MCPServerSpec
from linktools.ai.task._local import LocalTaskGraphLauncher, _InflightNode
from linktools.ai.task._service_impl import DefaultTaskService
from linktools.ai.workspace import trusted_workspace_principal


def test_subagent_delegate_contract_requires_mapping_result() -> None:
    return_type = get_type_hints(SubagentDelegate.__call__)["return"]
    args = get_args(return_type)
    assert get_origin(return_type) is dict
    assert len(args) == 2
    assert args[0] is str


@pytest.mark.asyncio
async def test_mcp_materialization_rejects_unselected_server(tmp_path) -> None:
    principal = Principal("principal", "tenant")
    execution = ResourceRef(ResourceKind.EXECUTION, "execution", "tenant")

    with pytest.raises(AIError) as error:
        await materialize_mcp_servers(
            (MCPServerSpec("server", "echo"),),
            (),
            principal=principal,
            execution=execution,
            execution_root=str(tmp_path),
        )

    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
async def test_asset_sql_apply_preserves_cancellation(monkeypatch) -> None:
    async def cancelled(self, changes, expected_revision):
        del self, changes, expected_revision
        raise asyncio.CancelledError

    monkeypatch.setattr(SqlAssetBackend, "_apply_once_transaction", cancelled)
    backend = object.__new__(SqlAssetBackend)

    with pytest.raises(asyncio.CancelledError):
        await backend._apply_once((), None)


@pytest.mark.asyncio
async def test_execution_handoff_cleanup_restores_state_before_cancellation() -> None:
    service = object.__new__(DefaultExecutionService)
    service._handoff_condition = asyncio.Condition()

    async def cancelled(execution_id: str, *, tenant_id: str) -> None:
        del execution_id, tenant_id
        raise asyncio.CancelledError

    service._release_terminal = cancelled
    state = SimpleNamespace(release_in_progress=True, release_requested=False)

    with pytest.raises(asyncio.CancelledError):
        await service._run_handoff_cleanup("execution", "tenant", state)

    assert state.release_in_progress is False
    assert state.release_requested is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("service_type", "consumer_name", "resource_id"),
    (
        (DefaultEvaluationService, "_evaluation_consumer", "evaluation"),
        (DefaultSessionService, "_session_consumer", "session"),
        (DefaultTaskService, "_graph_consumer", "graph"),
    ),
)
async def test_transient_handoff_cleanup_preserves_cancellation(
    service_type,
    consumer_name: str,
    resource_id: str,
) -> None:
    service = object.__new__(service_type)
    service._handoff_condition = asyncio.Condition()
    service._handoff_states = {}

    async def cancelled(*args, **kwargs) -> None:
        del args, kwargs
        raise asyncio.CancelledError

    service._release_terminal = cancelled
    key = ("tenant", resource_id)

    with pytest.raises(asyncio.CancelledError):
        async with getattr(service, consumer_name)(resource_id, "tenant"):
            service._handoff_states[key].release_requested = True

    state = service._handoff_states[key]
    assert state.release_in_progress is False
    assert state.release_requested is True


@pytest.mark.asyncio
async def test_task_runner_cancellation_does_not_business_cancel_running_execution() -> None:
    class Execution:
        def __init__(self) -> None:
            self.wait_started = asyncio.Event()
            self.wait_cancelled = asyncio.Event()
            self.cancel_called = asyncio.Event()

        async def run(self, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace(execution_id="execution")

        async def wait(self, *args, **kwargs):
            del args, kwargs
            self.wait_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.wait_cancelled.set()
                raise

        async def cancel(self, *args, **kwargs):
            del args, kwargs
            self.cancel_called.set()
            return SimpleNamespace(cancelled=True)

    class Control:
        def __init__(self) -> None:
            self.bound: list[str] = []

        async def bind_execution(self, execution_id: str) -> None:
            self.bound.append(execution_id)

    execution = Execution()
    handler = _AgentTaskNodeHandler(execution, object(), object())
    handler._prepare_request = lambda *args, **kwargs: ("binding", object())
    control = Control()
    task = asyncio.create_task(
        handler.run_node(
            SimpleNamespace(node_id="node"),
            graph_id="graph",
            principal=trusted_workspace_principal("tenant"),
            dependencies={},
            control=control,
        )
    )
    await execution.wait_started.wait()
    assert control.bound == ["execution"]
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(execution.wait_cancelled.wait(), 1)
    await asyncio.sleep(0)
    assert not execution.cancel_called.is_set()
    assert handler.background_failure is None


@pytest.mark.asyncio
async def test_task_runner_binds_execution_that_finishes_launch_after_caller_cancel() -> None:
    class Execution:
        def __init__(self) -> None:
            self.launch_started = asyncio.Event()
            self.release_launch = asyncio.Event()
            self.cancel_called = asyncio.Event()

        async def run(self, *args, **kwargs):
            del args, kwargs
            self.launch_started.set()
            await self.release_launch.wait()
            return SimpleNamespace(execution_id="execution")

        async def wait(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("wait must not start after caller cancellation")

        async def cancel(self, *args, **kwargs):
            del args, kwargs
            self.cancel_called.set()
            return SimpleNamespace(cancelled=True)

    class Control:
        def __init__(self) -> None:
            self.bound = asyncio.Event()
            self.execution_id: str | None = None

        async def bind_execution(self, execution_id: str) -> None:
            self.execution_id = execution_id
            self.bound.set()

    execution = Execution()
    handler = _AgentTaskNodeHandler(execution, object(), object())
    handler._prepare_request = lambda *args, **kwargs: ("binding", object())
    control = Control()
    task = asyncio.create_task(
        handler.run_node(
            SimpleNamespace(node_id="node"),
            graph_id="graph",
            principal=trusted_workspace_principal("tenant"),
            dependencies={},
            control=control,
        )
    )
    await execution.launch_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert handler.pending_background_tasks
    execution.release_launch.set()
    await asyncio.wait_for(control.bound.wait(), 1)
    pending = handler.pending_background_tasks
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    await asyncio.sleep(0)
    assert control.execution_id == "execution"
    assert not execution.cancel_called.is_set()
    assert handler.pending_background_tasks == ()
    assert handler.background_failure is None


@pytest.mark.asyncio
async def test_task_runner_start_unknown_after_caller_cancel_blocks_shutdown() -> None:
    class Execution:
        def __init__(self) -> None:
            self.launch_started = asyncio.Event()
            self.release_launch = asyncio.Event()

        async def run(self, *args, **kwargs):
            del args, kwargs
            self.launch_started.set()
            await self.release_launch.wait()
            raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)

        async def wait(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("wait must not start after unknown launch")

    class Control:
        async def bind_execution(self, execution_id: str) -> None:
            del execution_id
            raise AssertionError("unknown launch must not bind execution")

    execution = Execution()
    handler = _AgentTaskNodeHandler(execution, object(), object())
    handler._prepare_request = lambda *args, **kwargs: ("binding", object())
    task = asyncio.create_task(
        handler.run_node(
            SimpleNamespace(node_id="node"),
            graph_id="graph",
            principal=trusted_workspace_principal("tenant"),
            dependencies={},
            control=Control(),
        )
    )
    await execution.launch_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    execution.release_launch.set()
    pending = handler.pending_background_tasks
    assert pending
    await asyncio.gather(*pending, return_exceptions=True)
    await asyncio.sleep(0)
    assert handler.pending_background_tasks == ()
    failure = handler.background_failure
    assert failure is not None
    assert failure.code is ErrorCode.STORAGE_RECOVERY_REQUIRED

    launcher = object.__new__(LocalTaskGraphLauncher)
    launcher._accepting = True
    launcher._graphs = {}
    launcher._lock = asyncio.Lock()
    launcher._runner = handler
    with pytest.raises(AIError) as shutdown_error:
        await launcher.shutdown()
    assert shutdown_error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED


@pytest.mark.asyncio
async def test_task_scheduler_arm_cancellation_detaches_pending_launcher() -> None:
    class Launcher:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def start(self, launch: object) -> object:
            del launch
            self.started.set()
            await self.release.wait()
            return object()

    launcher = Launcher()
    service = DefaultTaskService(SimpleNamespace(), SimpleNamespace(), launcher)
    launch = SimpleNamespace(
        principal=trusted_workspace_principal("tenant"),
        graph=SimpleNamespace(graph_id="graph"),
    )
    task = asyncio.create_task(service._arm_graph(launch))
    await launcher.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    pending = tuple(service._detached_finalizers)
    assert pending
    launcher.release.set()
    await asyncio.gather(*pending, return_exceptions=True)
    await asyncio.sleep(0)
    assert service._detached_finalizers == set()


@pytest.mark.asyncio
async def test_task_inflight_cleanup_detaches_cancellation_resistant_node() -> None:
    launcher = object.__new__(LocalTaskGraphLauncher)
    launcher._detached_tasks = set()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    async def node() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()

    task = asyncio.create_task(node())
    await asyncio.sleep(0)
    inflight = {"node": _InflightNode(task, SimpleNamespace())}

    await launcher._cancel_inflight(inflight)
    await cancelled.wait()

    assert inflight == {}
    pending = tuple(launcher._detached_tasks)
    assert pending
    release.set()
    await asyncio.gather(*pending, return_exceptions=True)
    await asyncio.sleep(0)
    assert launcher._detached_tasks == set()


@pytest.mark.asyncio
async def test_task_shutdown_allows_one_turn_for_cancelled_detached_cleanup() -> None:
    launcher = object.__new__(LocalTaskGraphLauncher)
    launcher._accepting = True
    launcher._graphs = {}
    launcher._wait_observations = {}
    launcher._detached_tasks = set()
    launcher._runner = SimpleNamespace()
    started = asyncio.Event()

    async def cleanup() -> None:
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(cleanup())
    await started.wait()
    task.cancel()
    launcher._detach(task, "test cleanup")

    await launcher.shutdown()

    assert task.done()
    await asyncio.sleep(0)
    assert launcher._detached_tasks == set()


@pytest.mark.asyncio
async def test_task_shutdown_still_rejects_cancellation_resistant_detached_cleanup() -> None:
    launcher = object.__new__(LocalTaskGraphLauncher)
    launcher._accepting = True
    launcher._graphs = {}
    launcher._wait_observations = {}
    launcher._detached_tasks = set()
    launcher._runner = SimpleNamespace()
    started = asyncio.Event()
    release = asyncio.Event()

    async def cleanup() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()

    task = asyncio.create_task(cleanup())
    await started.wait()
    task.cancel()
    launcher._detach(task, "test stuck cleanup")
    try:
        with pytest.raises(AIError) as error:
            await launcher.shutdown()
        assert error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        assert error.value.safe_details["phase"] == "task_graph_shutdown"
    finally:
        release.set()
        await task
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_task_heartbeat_loss_detaches_cancellation_resistant_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Repository:
        async def claim(self, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace(owner="owner", fence=1, lease_expires_at=None)

        async def list_nodes(self, *args, **kwargs):
            del args, kwargs
            return ()

    class Runner:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.release = asyncio.Event()

        async def run(self, *args, **kwargs):
            del args, kwargs
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                await self.release.wait()
                raise

        async def cancel(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("heartbeat cleanup must cancel the owned runner task")

    runner = Runner()

    async def lost_heartbeat(self, state, tenant_id):
        del self, state, tenant_id
        await runner.started.wait()
        raise AIError(ErrorCode.TASK_FENCE_STALE)

    monkeypatch.setattr(LocalTaskGraphLauncher, "_heartbeat", lost_heartbeat)
    launcher = object.__new__(LocalTaskGraphLauncher)
    launcher._repository = Repository()
    launcher._runner = runner
    launcher._owner = "owner"
    launcher._detached_tasks = set()

    request = SimpleNamespace(
        principal=trusted_workspace_principal("tenant"),
        graph=SimpleNamespace(graph_id="graph"),
    )
    node = SimpleNamespace(node_id="node", dependencies=())
    inflight = {"node": SimpleNamespace(lease=None)}
    run = SimpleNamespace(
        closed=False,
        activity=asyncio.Event(),
        activity_generation=0,
    )
    task = asyncio.create_task(launcher._run_node(request, node, inflight, run))
    await runner.started.wait()

    try:
        await asyncio.wait_for(asyncio.shield(task), 0.2)
    finally:
        runner.release.set()
        if not task.done():
            await task

    assert runner.cancelled.is_set()
    pending = tuple(launcher._detached_tasks)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.sleep(0)
    assert launcher._detached_tasks == set()


@pytest.mark.asyncio
async def test_subagent_child_cleanup_failure_does_not_replace_cancellation() -> None:
    class Execution:
        def __init__(self) -> None:
            self.cancel_started = asyncio.Event()
            self.finish_cancel = asyncio.Event()
            self.status = ExecutionStatus.STARTED

        async def inspect(self, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace(status=self.status)

        async def cancel(self, *args, **kwargs):
            del args, kwargs
            self.cancel_started.set()
            await self.finish_cancel.wait()
            raise RuntimeError("cleanup failed")

    execution = Execution()
    dispatcher = object.__new__(SubagentDispatcher)
    dispatcher._execution = execution
    dispatcher._detached_tasks = set()
    dispatcher._background_failures = {}
    task = asyncio.create_task(
        dispatcher.cancel_child(
            "execution",
            parent_execution_id="parent",
            principal=trusted_workspace_principal("tenant"),
        )
    )
    await execution.cancel_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    pending = dispatcher.pending_background_tasks
    assert pending
    execution.finish_cancel.set()
    await asyncio.gather(*pending, return_exceptions=True)
    await asyncio.sleep(0)
    assert dispatcher.pending_background_tasks == ()
    failure = dispatcher.background_failure
    assert failure is not None
    assert failure.code is ErrorCode.STORAGE_RECOVERY_REQUIRED

    backend = object.__new__(LocalExecutionBackend)
    backend._accepting = True
    backend._tasks = {}
    backend._executor = SimpleNamespace(pending_background_tasks=())
    backend._subagent_dispatcher = dispatcher
    backend._checkpoint_tasks = set()
    backend._execution_durable_tasks = {}
    backend._captured_usage = {}
    backend._terminal_events = {}
    backend._worker_failures = {}
    backend._worker_cancel_requests = set()
    backend._pending_audit_events = {}
    backend._pending_audit_locks = {}
    backend._approval_pause_segments = {}
    backend._segment_only_worker_exits = set()
    backend._repository_instruction_provenance = {}
    with pytest.raises(AIError) as close_error:
        await backend.close()
    assert close_error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    assert close_error.value.safe_details["background_failures"] == 1

    execution.status = ExecutionStatus.CANCELLED
    await dispatcher.cancel_child(
        "execution",
        parent_execution_id="parent",
        principal=trusted_workspace_principal("tenant"),
    )
    assert dispatcher.background_failure is None
    await backend.close()
