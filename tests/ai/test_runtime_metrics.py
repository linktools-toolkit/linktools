#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime automatic Metrics producer and fail-open regressions."""

import asyncio
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from linktools.ai.core import ExecutionStatus, JsonValue, Page, Principal, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.observe import (
    MetricQuery,
    MetricWindow,
    Metrics,
    Observation,
)
from linktools.ai.observe._memory import InMemoryMetricStore
from linktools.ai.runtime import Runtime
from linktools.ai.runtime import _metrics as runtime_metrics
from linktools.ai.runtime._metric_capability import _RuntimeMetricCapability
from linktools.ai.spec import AgentSpec, AgentSpecCodec
from linktools.ai.task import (
    LocalTaskGraphLauncher,
    TaskEvent,
    TaskEventType,
    TaskGraph,
    TaskGraphLaunch,
    TaskGraphLimits,
    TaskGraphView,
    TaskLease,
    TaskNode,
    TaskNodeRunResult,
    TaskNodeView,
)
from linktools.ai.task._metrics import record_task_event
from linktools.ai.workspace import Workspace
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition


class _TextModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:test"
    fingerprint = "d" * 64
    semantic_payload: dict[str, JsonValue] = {"provider": "test", "model": "test"}

    def materialize(self) -> TestModel:
        return TestModel(custom_output_text="ok")


class _TextModels:
    def snapshot(self) -> "_TextModels":
        return self

    def resolve(self, route_id: str) -> _TextModelBinding:
        if route_id != "default":
            raise AssertionError(route_id)
        return _TextModelBinding()

    def restore(
        self,
        payload: Mapping[str, JsonValue],
        *,
        route_id: str | None = None,
    ) -> _TextModelBinding:
        if (
            route_id not in {None, "default"}
            or dict(payload) != _TextModelBinding.semantic_payload
        ):
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return _TextModelBinding()


class _CaptureRecorder:
    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def try_record(self, observation: Observation) -> bool:
        self.observations.append(observation)
        return True


class _FailingMetricStore:
    async def put_definition(self, namespace: str, definition: object) -> None:
        del namespace, definition
        raise AssertionError("automatic producers do not define metrics")

    async def get_definition(
        self,
        namespace: str,
        name: str,
        revision: int | None,
    ) -> None:
        del namespace, name, revision
        raise AssertionError("automatic producers do not read definitions")

    async def put_observations(
        self,
        namespace: str,
        observations: tuple[Observation, ...],
    ) -> None:
        del namespace, observations
        raise RuntimeError("metrics backend unavailable")

    async def scan_observations(
        self,
        *args: object,
        **kwargs: object,
    ) -> tuple[Observation, ...]:
        del args, kwargs
        return ()

    async def prune(self, namespace: str, *, before: datetime) -> int:
        del namespace, before
        return 0


class _BlockingMetricStore(_FailingMetricStore):
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.batches: list[tuple[Observation, ...]] = []

    async def put_observations(
        self,
        namespace: str,
        observations: tuple[Observation, ...],
    ) -> None:
        del namespace
        self.entered.set()
        await self.release.wait()
        self.batches.append(observations)


class _CommitUnknownOnceMetricStore(_FailingMetricStore):
    def __init__(self) -> None:
        self.batches: list[tuple[Observation, ...]] = []

    async def put_observations(
        self,
        namespace: str,
        observations: tuple[Observation, ...],
    ) -> None:
        del namespace
        self.batches.append(observations)
        if len(self.batches) == 1:
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)


def _write_default_agent(root: Path) -> None:
    path = root / ".linktools" / "agents" / "default"
    path.parent.mkdir(parents=True)
    path.write_bytes(
        AgentSpecCodec().encode(
            AgentSpec("default", model="default", allow_tools=())
        )
    )


def _observation(observation_id: str) -> Observation:
    return Observation(
        version=1,
        observation_id=observation_id,
        kind="test.runtime.metric",
        occurred_at=datetime.now(timezone.utc),
        source_namespace="workspace",
        tenant_id="default",
        status="SUCCEEDED",
        error_code=None,
        correlation={},
        dimensions={},
        measurements=(),
    )


@pytest.mark.asyncio
async def test_runtime_projects_model_agent_and_execution_metrics(tmp_path: Path) -> None:
    _write_default_agent(tmp_path)
    store = InMemoryMetricStore()
    metrics = Metrics.from_store(store, namespace="runtime-metrics")
    start = datetime.now(timezone.utc) - timedelta(seconds=1)
    secret = "TOP_SECRET_PROMPT_VALUE"

    async with Runtime.open(
        Workspace.load(tmp_path),
        models=_TextModels(),  # type: ignore[arg-type]
        metrics=metrics,
    ) as runtime:
        result = await runtime.agent("default").run(secret, timeout_seconds=10)
        assert result.status is ExecutionStatus.SUCCEEDED

    end = datetime.now(timezone.utc) + timedelta(seconds=1)
    window = MetricWindow.between(start, end)
    for metric_name in (
        "linktools.model.request.count",
        "linktools.agent.run.count",
        "linktools.execution.count",
    ):
        query = await metrics.query(MetricQuery(metric_name, window))
        assert len(query.points) == 1
        assert query.points[0].value == 1
        assert query.points[0].sample_count == 1

    observations = []
    for kind in (
        "linktools.model.request",
        "linktools.agent.run",
        "linktools.execution.terminal",
    ):
        observations.extend(
            await store.scan_observations(
                "runtime-metrics",
                kind=kind,
                start=start,
                end=end,
                limit=100,
            )
        )
    assert len(observations) == 3
    assert secret not in repr(observations)


@pytest.mark.asyncio
async def test_runtime_metrics_backend_failure_does_not_change_execution_result(
    tmp_path: Path,
) -> None:
    _write_default_agent(tmp_path)
    metrics = Metrics.from_store(_FailingMetricStore(), namespace="runtime-fail-open")

    async with Runtime.open(
        Workspace.load(tmp_path),
        models=_TextModels(),  # type: ignore[arg-type]
        metrics=metrics,
    ) as runtime:
        result = await runtime.agent("default").run("hello", timeout_seconds=10)
        assert result.status is ExecutionStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_runtime_metric_buffer_is_bounded_and_close_is_fail_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_metrics, "_QUEUE_CAPACITY", 1)
    store = _BlockingMetricStore()
    buffer = runtime_metrics._RuntimeMetricBuffer(
        Metrics.from_store(store, namespace="buffer")
    )

    assert buffer.try_record(_observation("first")) is True
    await asyncio.wait_for(store.entered.wait(), timeout=1)
    assert buffer.try_record(_observation("second")) is True
    assert buffer.try_record(_observation("third")) is False

    store.release.set()
    await buffer.close()
    assert buffer.try_record(_observation("after-close")) is False
    assert sum(len(batch) for batch in store.batches) == 2


@pytest.mark.asyncio
async def test_runtime_metric_buffer_retries_commit_unknown_with_same_observation() -> None:
    store = _CommitUnknownOnceMetricStore()
    buffer = runtime_metrics._RuntimeMetricBuffer(
        Metrics.from_store(store, namespace="buffer-retry")
    )
    observation = _observation("retry-stable")

    assert buffer.try_record(observation) is True
    await buffer.close()

    assert len(store.batches) == 2
    assert store.batches[0] == store.batches[1] == (observation,)


@pytest.mark.asyncio
async def test_model_and_tool_producers_do_not_capture_payload_or_exception_text() -> None:
    recorder = _CaptureRecorder()
    capability = _RuntimeMetricCapability(
        recorder,
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        session_id="session",
        step_run_id="run",
        agent_id="agent",
        provider="provider",
        model_identity="provider:model",
        route_id="default",
    )
    secret = "DO_NOT_PERSIST_THIS_SECRET"

    async def model_handler(_request: object) -> object:
        raise RuntimeError(f"provider failed with {secret}")

    with pytest.raises(RuntimeError):
        await capability.wrap_model_request(  # type: ignore[arg-type]
            None,
            request_context=object(),
            handler=model_handler,  # type: ignore[arg-type]
        )

    call = ToolCallPart(
        "dangerous_tool",
        {"token": secret},
        tool_call_id="call-1",
    )

    async def tool_handler(_args: dict[str, Any]) -> object:
        raise RuntimeError(f"tool failed with {secret}")

    with pytest.raises(RuntimeError):
        await capability.wrap_tool_execute(  # type: ignore[arg-type]
            None,
            call=call,
            tool_def=ToolDefinition(name="dangerous_tool"),
            args={"token": secret},
            handler=tool_handler,
        )

    assert [item.kind for item in recorder.observations] == [
        "linktools.model.request",
        "linktools.tool.execution",
    ]
    assert recorder.observations[0].error_code == ErrorCode.MODEL_API_ERROR.value
    assert (
        recorder.observations[1].error_code
        == ErrorCode.TOOL_EXECUTION_FAILED.value
    )
    assert secret not in repr(recorder.observations)


def test_task_attempt_observation_replay_uses_durable_event_identity_and_time() -> None:
    occurred_at = datetime(2026, 9, 5, 2, 0, tzinfo=timezone.utc)
    event = TaskEvent(
        version=1,
        graph_id="graph",
        sequence=3,
        event_type=TaskEventType.NODE_CHANGED,
        occurred_at=occurred_at,
        status=TaskStatus.SUCCEEDED,
        previous_status=TaskStatus.RUNNING,
        node_id="node",
        fence=7,
        execution_id="execution",
        result_digest="a" * 64,
    )
    recorder = _CaptureRecorder()
    nodes = {"node": TaskNode("node", input={"type": "agent"})}

    for _ in range(2):
        record_task_event(
            recorder,
            source_namespace="workspace",
            tenant_id="tenant",
            event=event,
            nodes=nodes,
        )

    assert len(recorder.observations) == 2
    first, second = recorder.observations
    assert first == second
    assert first.occurred_at == occurred_at
    assert first.correlation == {
        "graph_id": "graph",
        "node_id": "node",
        "fence": 7,
        "execution_id": "execution",
    }
    assert first.dimensions == {"task_type": "agent"}


class _CommitUnknownTaskRepository:
    def __init__(self, node: TaskNode) -> None:
        self.node = node
        self.status = TaskStatus.READY
        self.fence = 0
        self.complete_calls = 0
        self.list_event_calls = 0
        self.terminal_time = datetime(2026, 9, 5, 3, 0, tzinfo=timezone.utc)
        self.events = [
            TaskEvent(
                version=1,
                graph_id="graph",
                sequence=1,
                event_type=TaskEventType.GRAPH_ADMITTED,
                occurred_at=self.terminal_time - timedelta(seconds=3),
                status=TaskStatus.PENDING,
            )
        ]

    async def reconcile_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        assert graph_id == "graph"
        assert tenant_id == "tenant"
        graph_status = (
            TaskStatus.SUCCEEDED
            if self.status is TaskStatus.SUCCEEDED
            else TaskStatus.PENDING
        )
        return TaskGraphView("graph", graph_status, (self.node,))

    async def get_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        return await self.reconcile_graph(graph_id, tenant_id=tenant_id)

    async def list_nodes(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> tuple[TaskNodeView, ...]:
        assert graph_id == "graph"
        assert tenant_id == "tenant"
        return (
            TaskNodeView(
                "graph",
                "node",
                (),
                self.status,
                None,
                self.fence,
                None,
                "a" * 64 if self.status is TaskStatus.SUCCEEDED else None,
                None,
                None,
                "execution" if self.status is TaskStatus.SUCCEEDED else None,
            ),
        )

    async def get_results(
        self,
        graph_id: str,
        node_ids: tuple[str, ...],
        *,
        tenant_id: str,
    ) -> dict[str, object]:
        assert graph_id == "graph"
        assert node_ids == ("node",)
        assert tenant_id == "tenant"
        return {}

    async def list_events(
        self,
        graph_id: str,
        *,
        tenant_id: str,
        after_sequence: int,
        limit: int,
    ) -> Page[TaskEvent]:
        assert graph_id == "graph"
        assert tenant_id == "tenant"
        assert limit == 1000
        self.list_event_calls += 1
        items = tuple(
            event for event in self.events if event.sequence > after_sequence
        )[:limit]
        return Page(items)

    async def claim(
        self,
        graph_id: str,
        node_id: str,
        *,
        tenant_id: str,
        owner: str,
        lease_seconds: int,
    ) -> TaskLease:
        assert graph_id == "graph"
        assert node_id == "node"
        assert tenant_id == "tenant"
        assert lease_seconds > 0
        self.status = TaskStatus.RUNNING
        self.fence = 1
        self.events.append(
            TaskEvent(
                version=1,
                graph_id="graph",
                sequence=2,
                event_type=TaskEventType.NODE_CHANGED,
                occurred_at=self.terminal_time - timedelta(seconds=2),
                status=TaskStatus.RUNNING,
                previous_status=TaskStatus.READY,
                node_id="node",
                owner=owner,
                fence=1,
            )
        )
        return TaskLease(
            "graph",
            "node",
            "tenant",
            owner,
            1,
            datetime.now(timezone.utc) + timedelta(minutes=1),
        )

    async def renew(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        lease_seconds: int,
    ) -> TaskLease:
        assert tenant_id == "tenant"
        assert lease_seconds > 0
        return lease

    async def bind_execution(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str,
    ) -> TaskNodeView:
        raise AssertionError("runner does not bind execution before completion")

    async def complete(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str | None,
        result_digest: str,
        result_payload: object = None,
    ) -> None:
        assert lease.fence == 1
        assert tenant_id == "tenant"
        assert execution_id == "execution"
        assert result_digest == "a" * 64
        assert result_payload is None
        self.complete_calls += 1
        self.status = TaskStatus.SUCCEEDED
        self.events.extend(
            (
                TaskEvent(
                    version=1,
                    graph_id="graph",
                    sequence=3,
                    event_type=TaskEventType.NODE_CHANGED,
                    occurred_at=self.terminal_time,
                    status=TaskStatus.SUCCEEDED,
                    previous_status=TaskStatus.RUNNING,
                    node_id="node",
                    fence=1,
                    execution_id="execution",
                    result_digest="a" * 64,
                ),
                TaskEvent(
                    version=1,
                    graph_id="graph",
                    sequence=4,
                    event_type=TaskEventType.GRAPH_CHANGED,
                    occurred_at=self.terminal_time + timedelta(seconds=1),
                    status=TaskStatus.SUCCEEDED,
                    previous_status=TaskStatus.PENDING,
                ),
            )
        )
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

    async def fail(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("runner succeeds")

    async def cancel_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        raise AssertionError("graph is not cancelled")


class _SuccessfulTaskRunner:
    async def run(self, *args: object, **kwargs: object) -> TaskNodeRunResult:
        del args, kwargs
        return TaskNodeRunResult("a" * 64, execution_id="execution")

    async def cancel(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("successful graph does not cancel node effects")


@pytest.mark.asyncio
async def test_task_commit_unknown_readback_projects_durable_terminal_event() -> None:
    node = TaskNode("node", input={"type": "agent"})
    repository = _CommitUnknownTaskRepository(node)
    recorder = _CaptureRecorder()
    launcher = LocalTaskGraphLauncher(
        repository,  # type: ignore[arg-type]
        _SuccessfulTaskRunner(),  # type: ignore[arg-type]
        owner="worker",
        metric_recorder=recorder,
        metric_source_namespace="workspace",
    )
    launch = TaskGraphLaunch(
        TaskGraph("graph", (node,)),
        Principal("owner", "tenant"),
        TaskGraphLimits(),
    )

    await launcher.start(launch)
    for _ in range(100):
        if not launcher.owns_graph("graph", tenant_id="tenant"):
            break
        await asyncio.sleep(0)
    await launcher.shutdown()

    assert repository.complete_calls == 1
    assert repository.list_event_calls == 1
    attempts = [
        observation
        for observation in recorder.observations
        if observation.kind == "linktools.task.node.attempt"
    ]
    graph_terminals = [
        observation
        for observation in recorder.observations
        if observation.kind == "linktools.task.graph.terminal"
    ]
    assert len(attempts) == 1
    assert attempts[0].occurred_at == repository.terminal_time
    assert attempts[0].correlation == {
        "graph_id": "graph",
        "node_id": "node",
        "fence": 1,
        "execution_id": "execution",
    }
    assert len(graph_terminals) == 1
    assert graph_terminals[0].status == TaskStatus.SUCCEEDED.value
