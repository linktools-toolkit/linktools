#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime automatic Metrics producer and fail-open regressions."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from linktools.ai.capability import RuntimeFeatures
from linktools.ai.core import ExecutionEventType, ExecutionStatus, JsonValue, Page, Principal, TaskStatus, UsageMetrics
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.observe import MetricQuery, MetricWindow, Metrics, Observation
from linktools.ai.observe._memory import InMemoryMetricStore
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime import _metrics as runtime_metrics
from linktools.ai.runtime._metric_capability import ModelObservationCapability
from linktools.ai.runtime._agent_executor import AgentExecutor, AgentExecutionResult
from linktools.ai.runtime.state._runtime_commands import RuntimeStateCommands
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime.service_api import CancelExecutionRequest
from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.task import (
    LocalTaskGraphLauncher,
    TaskEvent,
    TaskEventType,
    TaskGraph,
    TaskGraphLaunch,
    TaskGraphLimits,
    TaskGraphState,
    TaskGraphView,
    TaskLease,
    Task,
    TaskNode,
    TaskNodeContext,
    TaskNodeRunResult,
    TaskNodeView,
)
from linktools.ai.task._metrics import _TaskMetricProjector
from pydantic_ai.models.test import TestModel
from pydantic_ai import RunContext
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.usage import RunUsage


from linktools.ai.runtime import MetricsCapability, TraceCapability


class _TextModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:test"
    contract: dict[str, JsonValue] = {"provider": "test", "model": "test"}

    def materialize(self) -> TestModel:
        return TestModel(custom_output_text="ok")


class _TextModels:
    def capture(self) -> _TextModels:
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
            or dict(payload) != _TextModelBinding.contract
        ):
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return _TextModelBinding()


class _CaptureRecorder:
    def __init__(self) -> None:
        self.observations: list[Observation] = []
        self.terminal_recorded = asyncio.Event()

    async def try_record(self, observation: Observation) -> bool:
        self.observations.append(observation)
        if observation.kind == "linktools.task.graph.terminal":
            self.terminal_recorded.set()
        return True


class _FailingMetricStore:
    async def put_definition(self, namespace: str, definition: object) -> object:
        del namespace, definition
        raise AssertionError("automatic producers do not define metrics")

    async def get_definition(
        self,
        namespace: str,
        name: str,
        revision: int,
    ) -> None:
        del namespace, name, revision
        raise AssertionError("automatic producers do not read definitions")

    async def latest_definition(self, namespace: str, name: str) -> None:
        del namespace, name
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
        namespace: str,
        kind: str,
        start: datetime,
        end: datetime,
        *,
        cursor: str | None,
        limit: int,
    ) -> Page[Observation]:
        del namespace, kind, start, end, cursor, limit
        return Page(())

    async def prune_observations(self, namespace: str, *, before: datetime) -> int:
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

    async def get_observation(
        self,
        namespace: str,
        observation_id: str,
    ) -> None:
        del namespace, observation_id
        return None


def _agent_group() -> RuntimeFeatures[object]:
    group = RuntimeFeatures[object]("application")
    group.capability(MetricsCapability(), scope="runtime")
    group.agent("default", model="default", allow_tools=())
    return group


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
    store = InMemoryMetricStore()
    metrics = Metrics.from_store(store, namespace="runtime-metrics")
    start = datetime.now(timezone.utc) - timedelta(seconds=1)
    secret = "TOP_SECRET_PROMPT_VALUE"

    async with Runtime.open(
        "default",
        models=_TextModels(),  # type: ignore[arg-type]
        storage=RuntimeStorage.in_memory(),
        runtime_features=_agent_group(),
        metrics=metrics,
    ) as runtime:
        result = (await runtime.agents.get("default").run(secret, timeout_seconds=10)).result
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

    observations: list[Observation] = []
    for kind in (
        "linktools.model.request",
        "linktools.agent.run",
        "linktools.execution.terminal",
    ):
        page = await store.scan_observations(
            "runtime-metrics",
            kind=kind,
            start=start,
            end=end,
            cursor=None,
            limit=100,
        )
        assert page.next_cursor is None
        observations.extend(page.items)
    assert len(observations) == 3
    assert secret not in repr(observations)


@pytest.mark.asyncio
async def test_runtime_metrics_backend_failure_does_not_change_execution_result(
    tmp_path: Path,
) -> None:
    metrics = Metrics.from_store(_FailingMetricStore(), namespace="runtime-fail-open")  # type: ignore[arg-type]

    async with Runtime.open(
        "default",
        models=_TextModels(),  # type: ignore[arg-type]
        storage=RuntimeStorage.in_memory(),
        runtime_features=_agent_group(),
        metrics=metrics,
    ) as runtime:
        result = (await runtime.agents.get("default").run("hello", timeout_seconds=10)).result
        assert result.status is ExecutionStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_isolated_task_failure_keeps_results_events_and_metrics_consistent() -> None:
    store = InMemoryMetricStore()
    metrics = Metrics.from_store(store, namespace="task-isolation-metrics")
    start = datetime.now(timezone.utc) - timedelta(seconds=1)

    async def fail(_context: TaskNodeContext[None]) -> JsonValue:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

    async def succeed(_context: TaskNodeContext[None]) -> JsonValue:
        return {"value": "ok"}

    failed_task = Task(
        "example.metrics-isolated-failure",
        fail,
        effect_policy="none",
    )
    successful_task = Task(
        "example.metrics-isolated-success",
        succeed,
        effect_policy="none",
    )
    graph = TaskGraph(
        "isolated-task-failure",
        (
            TaskNode("failed", task=failed_task, failure_policy="isolate"),
            TaskNode("succeeded", task=successful_task),
        ),
    )
    observed_events: list[TaskEvent] = []

    async with Runtime.open(
        "task-isolation-metrics",
        models=_TextModels(),  # type: ignore[arg-type]
        storage=RuntimeStorage.in_memory(),
        metrics=metrics,
    ) as runtime:
        run = await runtime.tasks.bind(failed_task, successful_task).start(
            graph,
            idempotency_key="isolated-task-failure-0001",
        )
        result = (await run.wait(timeout_seconds=10)).result
        state = await run.state()

        async def observe(event: object) -> None:
            payload = getattr(event, "event", None)
            if isinstance(payload, TaskEvent):
                observed_events.append(payload)

        replayed = await run.replay(observe)
        assert await run.result("succeeded") == {"value": "ok"}
        results = await run.results(limit=10, include_content=True)

    assert result.status is TaskStatus.SUCCEEDED
    assert replayed.status is TaskStatus.SUCCEEDED
    assert state.status is TaskStatus.SUCCEEDED
    result_statuses = {node.node_id: node.status for node in result.node_states}
    assert result_statuses == {
        "failed": TaskStatus.FAILED,
        "succeeded": TaskStatus.SUCCEEDED,
    }
    assert results.next_cursor is None
    paged_results = {item.node_id: item for item in results.items}
    assert paged_results["failed"].status is TaskStatus.FAILED
    assert paged_results["failed"].error_code == ErrorCode.REQUEST_FIELD_INVALID.value
    assert paged_results["failed"].output is None
    assert paged_results["succeeded"].status is TaskStatus.SUCCEEDED
    assert paged_results["succeeded"].output == {"value": "ok"}
    assert any(
        event.node_id == "failed" and event.status is TaskStatus.FAILED
        for event in observed_events
    )
    assert observed_events[-1].event_type is TaskEventType.GRAPH_CHANGED
    assert observed_events[-1].status is TaskStatus.SUCCEEDED

    end = datetime.now(timezone.utc) + timedelta(seconds=1)
    graph_metrics = await store.scan_observations(
        "task-isolation-metrics",
        kind="linktools.task.graph.terminal",
        start=start,
        end=end,
        cursor=None,
        limit=10,
    )
    failed_attempts = await store.scan_observations(
        "task-isolation-metrics",
        kind="linktools.task.node.attempt",
        start=start,
        end=end,
        cursor=None,
        limit=10,
    )
    assert len(graph_metrics.items) == 1
    assert graph_metrics.items[0].status == TaskStatus.SUCCEEDED.value
    attempts_by_node = {
        observation.correlation["linktools.node_id"]: observation
        for observation in failed_attempts.items
    }
    assert set(attempts_by_node) == {"failed", "succeeded"}
    assert attempts_by_node["failed"].status == TaskStatus.FAILED.value
    assert attempts_by_node["succeeded"].status == TaskStatus.SUCCEEDED.value


@pytest.mark.asyncio
async def test_runtime_metric_buffer_is_bounded_and_close_is_fail_open() -> None:
    store = _BlockingMetricStore()
    buffer = runtime_metrics._RuntimeMetricRecorder(
        Metrics.from_store(store, namespace="buffer", cache=InMemoryCacheBackend(max_bytes=1024 * 1024, max_size=2)).recorder()  # type: ignore[arg-type]
    )

    assert await buffer.try_record(_observation("first")) is True
    await buffer.flush(timeout_seconds=0)
    await asyncio.wait_for(store.entered.wait(), timeout=1)
    assert await buffer.try_record(_observation("second")) is True
    assert await buffer.try_record(_observation("third")) is False

    store.release.set()
    await buffer.close()
    assert await buffer.try_record(_observation("after-close")) is False
    assert sum(len(batch) for batch in store.batches) == 2


@pytest.mark.asyncio
async def test_runtime_metric_buffer_commit_unknown_uses_facade_exact_replay() -> None:
    store = _CommitUnknownOnceMetricStore()
    buffer = runtime_metrics._RuntimeMetricRecorder(
        Metrics.from_store(store, namespace="buffer-retry", cache=InMemoryCacheBackend(max_bytes=1024 * 1024)).recorder()  # type: ignore[arg-type]
    )
    observation = _observation("retry-stable")

    assert await buffer.try_record(observation) is True
    await buffer.close()

    assert len(store.batches) == 2
    assert store.batches[0] is store.batches[1]
    assert store.batches[0] == (observation,)


@pytest.mark.asyncio
async def test_model_metric_does_not_capture_prompt_or_exception_text() -> None:
    recorder = _CaptureRecorder()
    capability = ModelObservationCapability(
        recorder,
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        session_id="session",
        agent_run_id="run",
        agent_id="agent",
    )
    secret = "DO_NOT_PERSIST_THIS_SECRET"

    async def model_handler(_request: object) -> object:
        raise RuntimeError(f"provider failed with {secret}")

    model = TestModel()
    context = RunContext(
        deps=type("Deps", (), {"correlation": {}})(),
        model=model,
        usage=RunUsage(),
        run_id="run",
        run_step=1,
    )
    request_context = ModelRequestContext(
        model=model,
        messages=[],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )

    with pytest.raises(RuntimeError):
        await capability.wrap_model_request(  # type: ignore[arg-type]
            context,
            request_context=request_context,
            handler=model_handler,  # type: ignore[arg-type]
        )

    assert len(recorder.observations) == 1
    observation = recorder.observations[0]
    assert observation.kind == "linktools.model.request"
    assert observation.error_code == ErrorCode.INTERNAL_ERROR.value
    assert secret not in repr(observation)


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
                event_seq=1,
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
            else self.status
            if self.status is TaskStatus.RUNNING
            else TaskStatus.PENDING
        )
        return TaskGraphView("graph", graph_status, (self.node,))

    async def scheduler_state(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphState:
        view = await self.reconcile_graph(graph_id, tenant_id=tenant_id)
        return TaskGraphState(
            view.graph_id,
            view.status,
            view.nodes,
            await self.list_nodes(graph_id, tenant_id=tenant_id),
        )

    async def graph_state(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphState:
        return await self.scheduler_state(graph_id, tenant_id=tenant_id)

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
        after_event_seq: int,
        limit: int,
    ) -> Page[TaskEvent]:
        assert graph_id == "graph"
        assert tenant_id == "tenant"
        assert limit in {1, 1000}
        self.list_event_calls += 1
        selected = tuple(
            event for event in self.events if event.event_seq > after_event_seq
        )
        items = selected[:limit]
        return Page(items, "more" if len(selected) > limit else None)

    async def latest_event(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskEvent | None:
        assert graph_id == "graph"
        assert tenant_id == "tenant"
        return self.events[-1] if self.events else None

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
                event_seq=2,
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

    async def handoff_execution(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str,
    ) -> TaskNodeView:
        del lease, tenant_id, execution_id
        raise AssertionError("runner does not hand off execution before completion")

    async def complete(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str | None,
        result_digest: str,
        result_payload: object = None,
        graph_id: str | None = None,
        node_id: str | None = None,
        expanded_nodes: tuple[TaskNode, ...] = (),
        expected_fence: int | None = None,
    ) -> None:
        del graph_id, node_id, expanded_nodes
        assert lease.fence == 1
        assert expected_fence is None
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
                    event_seq=3,
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
                    event_seq=4,
                    event_type=TaskEventType.GRAPH_CHANGED,
                    occurred_at=self.terminal_time + timedelta(seconds=1),
                    status=TaskStatus.SUCCEEDED,
                    previous_status=TaskStatus.PENDING,
                ),
            )
        )
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

    async def fail(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("runner succeeds")

    async def cancel_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        del graph_id, tenant_id
        raise AssertionError("graph is not cancelled")


class _SuccessfulTaskRunner:
    async def run(self, *args: object, **kwargs: object) -> TaskNodeRunResult:
        del args, kwargs
        return TaskNodeRunResult("a" * 64, execution_id="execution")

    async def cancel(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("successful graph does not cancel node effects")


@pytest.mark.asyncio
async def test_task_commit_unknown_readback_projects_durable_terminal_history() -> None:
    node = TaskNode(
        "node",
        input={"task_id": "agent", "task_revision": 1},
    )
    repository = _CommitUnknownTaskRepository(node)
    recorder = _CaptureRecorder()
    projector = _TaskMetricProjector(
        repository,  # type: ignore[arg-type]
        recorder,
        source_namespace="workspace",
    )
    launcher = LocalTaskGraphLauncher(
        repository,  # type: ignore[arg-type]
        _SuccessfulTaskRunner(),  # type: ignore[arg-type]
        owner="worker",
    )
    launcher.bind_metric_projector(projector)
    launch = TaskGraphLaunch(
        "graph",
        Principal("owner", "tenant"),
        TaskGraphLimits(),
    )

    await launcher.start(launch)
    await asyncio.wait_for(recorder.terminal_recorded.wait(), timeout=1)
    for _ in range(100):
        if not launcher.owns_graph("graph", tenant_id="tenant"):
            break
        await asyncio.sleep(0)
    await launcher.shutdown()
    await projector.close()

    assert repository.complete_calls == 1
    assert repository.list_event_calls == 1
    attempts = tuple(
        observation
        for observation in recorder.observations
        if observation.kind == "linktools.task.node.attempt"
    )
    graph_terminals = tuple(
        observation
        for observation in recorder.observations
        if observation.kind == "linktools.task.graph.terminal"
    )
    assert len(attempts) == 1
    assert attempts[0].occurred_at == repository.terminal_time
    assert dict(attempts[0].correlation) == {
        "linktools.attempt_index": 1,
        "linktools.execution_id": "execution",
        "linktools.fence": 1,
        "linktools.graph_id": "graph",
        "linktools.node_id": "node",
    }
    assert {item.name: item.value for item in attempts[0].measurements} == {
        "latency_ns": 2_000_000_000,
        "retry_count": 0,
    }
    assert len(graph_terminals) == 1
    assert graph_terminals[0].status == TaskStatus.SUCCEEDED.value
    assert {item.name: item.value for item in graph_terminals[0].measurements} == {
        "latency_ns": 4_000_000_000,
        "retry_count": 0,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", (False, True))
async def test_agent_metric_cancellation_follows_usage_accounting(failure: bool) -> None:
    entered = asyncio.Event()
    captured_usage: list[UsageMetrics] = []
    observations: list[Observation] = []

    class BlockingRecorder:
        async def try_record(self, observation: Observation) -> bool:
            observations.append(observation)
            entered.set()
            await asyncio.Event().wait()
            return True

    executor = object.__new__(AgentExecutor)
    executor._metrics = BlockingRecorder()
    usage = UsageMetrics(model_requests=1, input_tokens=4, output_tokens=2)
    original_error = AIError(ErrorCode.MODEL_UNAVAILABLE)

    async def execute(scope: object, *, run_usage: RunUsage, usage_limits: object) -> AgentExecutionResult:
        if failure:
            run_usage.requests = 1
            raise original_error
        return AgentExecutionResult("run", "done", [], usage)

    executor._execute_with_sandbox = execute
    scope = SimpleNamespace(
        binding=SimpleNamespace(compiled_agent=SimpleNamespace(
            spec=SimpleNamespace(id="agent", usage_limits=None),
        )),
        context=SimpleNamespace(
            namespace="workspace", execution_id="execution", session_id=None,
            principal=Principal("owner", "tenant"),
        ),
        agent_run_id="run", usage_sink=captured_usage.append,
    )
    task = asyncio.create_task(executor.execute(scope))
    await asyncio.wait_for(entered.wait(), timeout=1)
    assert len(captured_usage) == 1
    assert captured_usage[0].model_requests == 1
    assert observations[0].status == ("FAILED" if failure else "SUCCEEDED")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(captured_usage) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("commit_unknown", (False, True))
async def test_terminal_metric_cancellation_preserves_published_success_and_history(
    monkeypatch: pytest.MonkeyPatch,
    commit_unknown: bool,
) -> None:
    original_commit = RuntimeStateCommands.commit_terminal_checkpoint
    committed_unknown = False

    async def commit(self: RuntimeStateCommands, *args: object, **kwargs: object) -> object:
        nonlocal committed_unknown
        result = await original_commit(self, *args, **kwargs)
        if commit_unknown and not committed_unknown:
            committed_unknown = True
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        return result

    monkeypatch.setattr(RuntimeStateCommands, "commit_terminal_checkpoint", commit)
    class BlockingTerminalStore(InMemoryMetricStore):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.producer: asyncio.Task[object] | None = None

        async def put_observations(self, namespace: str, observations: tuple[Observation, ...]) -> None:
            if any(item.kind == "linktools.execution.terminal" for item in observations):
                self.producer = asyncio.current_task()
                self.entered.set()
                await asyncio.Event().wait()
            await super().put_observations(namespace, observations)

    store = BlockingTerminalStore()
    metrics = Metrics.from_store(store, namespace="terminal-order")
    features = _agent_group()
    features.capability(TraceCapability(), scope="runtime")
    async with Runtime.open(
        "terminal-order", models=_TextModels(), storage=RuntimeStorage.in_memory(),
        runtime_features=features, metrics=metrics,
    ) as runtime:
        execution = await runtime.agents.get("default").start("hello")
        try:
            await asyncio.wait_for(store.entered.wait(), timeout=2)
            assert store.producer is not None
            assert not store.producer.done()
            view = await asyncio.wait_for(runtime.executions.inspect(
                execution.execution_id, principal=runtime.default_principal,
            ), timeout=2)
            assert view.status is ExecutionStatus.SUCCEEDED
            assert committed_unknown is commit_unknown
            history = await asyncio.wait_for(execution.model_interactions(include_content=True), timeout=2)
            assert len(history.items) == 1
            assert history.items[0].status == "SUCCEEDED"
            events = await asyncio.wait_for(execution.list_events(), timeout=2)
            assert events.items[-1].event_type == ExecutionEventType.EXECUTION_SUCCEEDED.value
            store.producer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(store.producer, timeout=2)
            result = await execution.wait(timeout_seconds=2)
            assert result.result.status is ExecutionStatus.SUCCEEDED
            after = await runtime.executions.result(
                execution.execution_id, principal=runtime.default_principal,
            )
            assert after.status is ExecutionStatus.SUCCEEDED
            assert await execution.model_interactions(include_content=True) == history
        finally:
            if store.producer is not None and not store.producer.done():
                store.producer.cancel()
                await asyncio.gather(store.producer, return_exceptions=True)


@pytest.mark.asyncio
async def test_pending_cancel_delivers_terminal_metric_in_caller_after_admission_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = asyncio.Event()
    release_prepare = asyncio.Event()
    execution_ids: list[str] = []
    aborted: list[str] = []
    original_prepare = LocalExecutionBackend.prepare_start
    original_abort = LocalExecutionBackend.abort_start

    async def prepare(self: LocalExecutionBackend, request: object, execution: object, identity: object) -> object:
        execution_ids.append(execution.execution_id)
        prepared.set()
        await release_prepare.wait()
        return await original_prepare(self, request, execution, identity)

    async def abort(self: LocalExecutionBackend, execution: object) -> None:
        await original_abort(self, execution)
        aborted.append(execution.execution_id)

    monkeypatch.setattr(LocalExecutionBackend, "prepare_start", prepare)
    monkeypatch.setattr(LocalExecutionBackend, "abort_start", abort)

    class Store(InMemoryMetricStore):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.attempts: list[Observation] = []
            self.caller: asyncio.Task[object] | None = None

        async def put_observations(self, namespace: str, observations: tuple[Observation, ...]) -> None:
            self.attempts.extend(observations)
            self.caller = asyncio.current_task()
            self.entered.set()
            await self.release.wait()
            await super().put_observations(namespace, observations)

    store = Store()
    async with Runtime.open(
        "pending-cancel-metric", models=_TextModels(), storage=RuntimeStorage.in_memory(),
        runtime_features=_agent_group(), metrics=Metrics.from_store(store),
    ) as runtime:
        starter = asyncio.create_task(runtime.agents.get("default").start("hello"))
        cancellation: asyncio.Task[object] | None = None
        try:
            await asyncio.wait_for(prepared.wait(), timeout=2)
            execution_id = execution_ids[0]
            request = CancelExecutionRequest(runtime.default_principal, "cancel-pending")
            cancellation = asyncio.create_task(runtime.executions.cancel(execution_id, request))
            await asyncio.wait_for(store.entered.wait(), timeout=2)
            assert store.caller is cancellation
            assert aborted == [execution_id]
            assert len(store.attempts) == 1
            assert store.attempts[0].status == "CANCELLED"
            cancellation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancellation
            terminal = await runtime.executions.result(execution_id, principal=runtime.default_principal)
            assert terminal.status is ExecutionStatus.CANCELLED
            repeated = await runtime.executions.cancel(execution_id, request)
            assert repeated.cancelled
            assert len(store.attempts) == 1
        finally:
            starter.cancel()
            if cancellation is not None:
                cancellation.cancel()
            store.release.set()
            await asyncio.gather(starter, *(() if cancellation is None else (cancellation,)), return_exceptions=True)
