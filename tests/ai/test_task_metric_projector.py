#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable Task event history is the only authority for Task metrics."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from linktools.ai.core import (
    OperationStatus,
    Page,
    Principal,
    TaskStatus,
    TenantAuthorizationPolicy,
    canonical_sha256,
    idempotency_key_digest,
)
from linktools.ai.observe import Observation
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.task import (
    CancelGraphRequest,
    DefaultTaskGraphService,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphRequest,
    TaskGraphState,
    TaskGraphSubmission,
    TaskNode,
)
from linktools.ai.task import _metrics as task_metrics
from linktools.ai.task._event import TaskEvent, TaskEventType
from linktools.ai.task._metrics import _TaskMetricProjector

pytestmark = pytest.mark.asyncio


class _Repository:
    def __init__(self, events: tuple[TaskEvent, ...]) -> None:
        self.events = events
        self.list_calls = 0

    async def list_events(
        self,
        graph_id: str,
        *,
        tenant_id: str,
        after_event_seq: int,
        limit: int,
    ) -> Page[TaskEvent]:
        del tenant_id
        self.list_calls += 1
        selected = tuple(
            event
            for event in self.events
            if event.graph_id == graph_id and event.event_seq > after_event_seq
        )
        page = selected[:limit]
        return Page(page, "more" if len(selected) > limit else None)

    async def latest_event(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskEvent | None:
        del tenant_id
        selected = tuple(event for event in self.events if event.graph_id == graph_id)
        return selected[-1] if selected else None


class _Recorder:
    def __init__(self, *, fail_first: bool = False) -> None:
        self.observations: list[Observation] = []
        self.calls = 0
        self.fail_first = fail_first

    async def try_record(self, observation: Observation) -> bool:
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("synthetic metric failure")
        self.observations.append(observation)
        return True


def _values(observation: Observation) -> dict[str, int | float]:
    return {item.name: item.value for item in observation.measurements}


def _admitted(graph_id: str, at: datetime, *, terminal: bool = False) -> TaskEvent:
    return TaskEvent(
        1,
        graph_id,
        1,
        TaskEventType.GRAPH_ADMITTED,
        at,
        TaskStatus.SUCCEEDED if terminal else TaskStatus.PENDING,
    )


def _running(
    graph_id: str,
    sequence: int,
    at: datetime,
    *,
    fence: int,
    previous_status: TaskStatus = TaskStatus.READY,
) -> TaskEvent:
    return TaskEvent(
        1,
        graph_id,
        sequence,
        TaskEventType.NODE_CHANGED,
        at,
        TaskStatus.RUNNING,
        previous_status=previous_status,
        node_id="node",
        owner="owner",
        fence=fence,
    )


def _waiting(
    graph_id: str,
    sequence: int,
    at: datetime,
    *,
    fence: int,
    execution_id: str,
) -> TaskEvent:
    return TaskEvent(
        1,
        graph_id,
        sequence,
        TaskEventType.NODE_CHANGED,
        at,
        TaskStatus.WAITING,
        previous_status=TaskStatus.RUNNING,
        node_id="node",
        fence=fence,
        execution_id=execution_id,
    )


def _succeeded(
    graph_id: str,
    sequence: int,
    at: datetime,
    *,
    fence: int,
    execution_id: str | None,
    previous_status: TaskStatus = TaskStatus.RUNNING,
) -> TaskEvent:
    return TaskEvent(
        1,
        graph_id,
        sequence,
        TaskEventType.NODE_CHANGED,
        at,
        TaskStatus.SUCCEEDED,
        previous_status=previous_status,
        node_id="node",
        fence=fence,
        execution_id=execution_id,
        result_digest=canonical_sha256({"result": sequence}),
    )


def _graph_terminal(
    graph_id: str,
    sequence: int,
    at: datetime,
    status: TaskStatus = TaskStatus.SUCCEEDED,
) -> TaskEvent:
    return TaskEvent(
        1,
        graph_id,
        sequence,
        TaskEventType.GRAPH_CHANGED,
        at,
        status,
        previous_status=TaskStatus.PENDING,
    )


async def _project(events: tuple[TaskEvent, ...], recorder: _Recorder) -> None:
    projector = _TaskMetricProjector(
        _Repository(events),
        recorder,
        source_namespace="workspace",
    )
    await projector._project(events[0].graph_id, tenant_id="tenant")


async def test_empty_graph_records_zero_terminal_latency() -> None:
    now = datetime(2026, 9, 5, tzinfo=timezone.utc)
    event = _admitted("empty", now, terminal=True)
    recorder = _Recorder()

    await _project((event,), recorder)

    assert len(recorder.observations) == 1
    observation = recorder.observations[0]
    assert observation.kind == "linktools.task.graph.terminal"
    assert observation.status == "SUCCEEDED"
    assert _values(observation) == {"latency_ns": 0, "retry_count": 0}


async def test_node_attempt_uses_first_running_event_for_same_fence() -> None:
    start = datetime(2026, 9, 5, tzinfo=timezone.utc)
    events = (
        _admitted("graph", start),
        _running("graph", 2, start + timedelta(seconds=1), fence=1),
        _waiting(
            "graph",
            3,
            start + timedelta(seconds=3),
            fence=1,
            execution_id="execution",
        ),
        _succeeded(
            "graph",
            4,
            start + timedelta(seconds=6),
            fence=1,
            execution_id="execution",
            previous_status=TaskStatus.WAITING,
        ),
        _graph_terminal("graph", 5, start + timedelta(seconds=7)),
    )
    recorder = _Recorder()

    await _project(events, recorder)

    attempt = next(
        observation
        for observation in recorder.observations
        if observation.kind == "linktools.task.node.attempt"
    )
    assert _values(attempt) == {
        "latency_ns": 5_000_000_000,
        "retry_count": 0,
    }
    assert dict(attempt.correlation) == {
        "linktools.attempt_index": 1,
        "linktools.execution_id": "execution",
        "linktools.fence": 1,
        "linktools.graph_id": "graph",
        "linktools.node_id": "node",
    }


async def test_unmatched_old_fence_is_not_paired_with_new_attempt() -> None:
    start = datetime(2026, 9, 5, tzinfo=timezone.utc)
    events = (
        _admitted("reclaim", start),
        _running("reclaim", 2, start + timedelta(seconds=1), fence=1),
        _running(
            "reclaim",
            3,
            start + timedelta(seconds=4),
            fence=2,
            previous_status=TaskStatus.RUNNING,
        ),
        _succeeded(
            "reclaim",
            4,
            start + timedelta(seconds=9),
            fence=2,
            execution_id="execution-2",
        ),
        _graph_terminal("reclaim", 5, start + timedelta(seconds=10)),
    )
    recorder = _Recorder()

    await _project(events, recorder)

    attempts = tuple(
        observation
        for observation in recorder.observations
        if observation.kind == "linktools.task.node.attempt"
    )
    assert len(attempts) == 1
    assert attempts[0].correlation["linktools.fence"] == 2
    assert attempts[0].correlation["linktools.attempt_index"] == 2
    assert _values(attempts[0]) == {
        "latency_ns": 5_000_000_000,
        "retry_count": 1,
    }


async def test_one_rejected_task_fact_does_not_abort_later_attempts() -> None:
    start = datetime(2026, 9, 5, tzinfo=timezone.utc)
    events = (
        _admitted("fail-open", start),
        _running("fail-open", 2, start + timedelta(seconds=1), fence=1),
        _succeeded(
            "fail-open",
            3,
            start + timedelta(seconds=2),
            fence=1,
            execution_id=None,
        ),
        _graph_terminal("fail-open", 4, start + timedelta(seconds=3)),
    )
    recorder = _Recorder(fail_first=True)

    await _project(events, recorder)

    assert recorder.calls == 2
    assert [observation.kind for observation in recorder.observations] == [
        "linktools.task.node.attempt"
    ]


async def test_trigger_deduplicates_successful_projection() -> None:
    now = datetime(2026, 9, 5, tzinfo=timezone.utc)
    repository = _Repository(
        (
            _admitted("dedup", now),
            _graph_terminal("dedup", 2, now + timedelta(seconds=1)),
        )
    )
    recorder = _Recorder()
    projector = _TaskMetricProjector(repository, recorder, source_namespace="workspace")
    await projector.trigger("dedup", tenant_id="tenant")
    first_calls = repository.list_calls
    assert first_calls == 1
    await projector.trigger("dedup", tenant_id="tenant")
    assert repository.list_calls == first_calls
    await projector.close()


class _BlockingRecorder(_Recorder):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.attempted: list[Observation] = []
        self.producer_tasks: list[asyncio.Task[object] | None] = []

    async def try_record(self, observation: Observation) -> bool:
        self.attempted.append(observation)
        self.producer_tasks.append(asyncio.current_task())
        self.started.set()
        await self.release.wait()
        return await super().try_record(observation)


def _terminal_repository(graph_id: str) -> _Repository:
    now = datetime(2026, 9, 5, tzinfo=timezone.utc)
    return _Repository((
        _admitted(graph_id, now),
        _graph_terminal(graph_id, 2, now + timedelta(seconds=1)),
    ))


async def test_trigger_waits_for_recorder_in_the_producing_task() -> None:
    recorder = _BlockingRecorder()
    projector = _TaskMetricProjector(
        _terminal_repository("direct"), recorder, source_namespace="workspace",
    )
    producer = asyncio.create_task(projector.trigger("direct", tenant_id="tenant"))
    try:
        await asyncio.wait_for(recorder.started.wait(), 1)
        assert not producer.done()
        assert recorder.producer_tasks == [producer]
        recorder.release.set()
        await producer
        assert len(recorder.observations) == 1
    finally:
        recorder.release.set()
        await asyncio.gather(producer, return_exceptions=True)
        await projector.close()


async def test_duplicate_trigger_waiter_does_not_cancel_active_projection() -> None:
    recorder = _BlockingRecorder()
    projector = _TaskMetricProjector(
        _terminal_repository("concurrent"), recorder, source_namespace="workspace",
    )
    producer = asyncio.create_task(projector.trigger("concurrent", tenant_id="tenant"))
    await asyncio.wait_for(recorder.started.wait(), 1)
    duplicate = asyncio.create_task(projector.trigger("concurrent", tenant_id="tenant"))
    try:
        await asyncio.sleep(0)
        assert not duplicate.done()
        duplicate.cancel()
        with pytest.raises(asyncio.CancelledError):
            await duplicate
        assert not producer.done()
        recorder.release.set()
        await producer
        await projector.trigger("concurrent", tenant_id="tenant")
        assert len(recorder.observations) == 1
    finally:
        recorder.release.set()
        await asyncio.gather(producer, duplicate, return_exceptions=True)
        await projector.close()


async def test_cancelled_projection_can_retry_with_the_same_stable_identity() -> None:
    recorder = _BlockingRecorder()
    projector = _TaskMetricProjector(
        _terminal_repository("retry"), recorder, source_namespace="workspace",
    )
    producer = asyncio.create_task(projector.trigger("retry", tenant_id="tenant"))
    await asyncio.wait_for(recorder.started.wait(), 1)
    producer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await producer
    assert not recorder.observations
    recorder.release.set()
    await projector.trigger("retry", tenant_id="tenant")
    assert len(recorder.observations) == 1
    assert recorder.attempted[0] == recorder.attempted[1]
    await projector.close()


async def test_fact_collection_budget_does_not_cancel_direct_metric_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(task_metrics, "_PROJECTION_TIMEOUT_SECONDS", 0.01)
    recorder = _BlockingRecorder()
    projector = _TaskMetricProjector(
        _terminal_repository("slow-write"), recorder, source_namespace="workspace",
    )
    producer = asyncio.create_task(projector.trigger("slow-write", tenant_id="tenant"))
    try:
        await asyncio.wait_for(recorder.started.wait(), 1)
        await asyncio.sleep(0.03)
        assert not producer.done()
        recorder.release.set()
        await producer
        assert len(recorder.observations) == 1
    finally:
        recorder.release.set()
        await asyncio.gather(producer, return_exceptions=True)
        await projector.close()


async def test_fact_collection_timeout_records_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(task_metrics, "_PROJECTION_TIMEOUT_SECONDS", 0.01)

    class BlockedRepository(_Repository):
        async def latest_event(self, graph_id: str, *, tenant_id: str) -> TaskEvent | None:
            await asyncio.Event().wait()
            return None

    recorder = _Recorder()
    projector = _TaskMetricProjector(
        BlockedRepository(()), recorder, source_namespace="workspace",
    )
    await asyncio.wait_for(projector.trigger("blocked", tenant_id="tenant"), 1)
    assert recorder.observations == []
    await projector.close()


async def test_invalid_late_history_cannot_partially_record_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(task_metrics, "_EVENT_PAGE_SIZE", 2)
    now = datetime(2026, 9, 5, tzinfo=timezone.utc)
    repository = _Repository((
        _admitted("invalid", now),
        _running("invalid", 2, now + timedelta(seconds=1), fence=1),
        _graph_terminal("invalid", 4, now + timedelta(seconds=3)),
    ))
    recorder = _Recorder()
    projector = _TaskMetricProjector(repository, recorder, source_namespace="workspace")
    await projector.trigger("invalid", tenant_id="tenant")
    assert repository.list_calls == 2
    assert recorder.observations == []
    await projector.close()


class _DependencyRelease:
    def __init__(self) -> None:
        self.released: set[str] = set()

    async def release_graph_dependencies(
        self, state: TaskGraphState, *, tenant_id: str,
    ) -> None:
        assert tenant_id == "tenant"
        assert state.status is TaskStatus.CANCELLED
        self.released.add(state.graph_id)


@pytest.mark.parametrize("via_submission", [False, True])
async def test_metric_cancellation_preserves_cancel_ledger_and_dependency_handoff(
    via_submission: bool,
) -> None:
    storage = RuntimeStorage.in_memory()
    await storage.initialize(namespace="task-metric-cancel", tenant_id="tenant")
    recorder = _BlockingRecorder()
    dependencies = _DependencyRelease()
    service = DefaultTaskGraphService(
        storage.task,
        TenantAuthorizationPolicy("tenant"),
        preflight=dependencies,  # type: ignore[arg-type]
        metric_recorder=recorder,
        metric_source_namespace="workspace",
    )
    principal = Principal("owner", "tenant")
    graph = TaskGraph("cancel-metric", (TaskNode("node"),))
    request = TaskGraphRequest(graph, principal, "submit")
    admission = TaskGraphAdmission.from_request(request)
    await storage.task.admissions.admit(admission, graph)
    if via_submission:
        submission = TaskGraphSubmission(storage.task.admissions.namespace, admission, graph)
        operation = service.cancel_submission(
            submission.ref, principal=principal, idempotency_key="cancel",
        )
    else:
        operation = service.cancel(graph.graph_id, CancelGraphRequest(principal, "cancel"))
    producer = asyncio.create_task(operation)
    try:
        await asyncio.wait_for(recorder.started.wait(), 1)
        assert recorder.producer_tasks == [producer]
        assert dependencies.released == {graph.graph_id}
        ledger = await storage.task.operations.get(
            idempotency_key_digest("cancel"), tenant_id="tenant",
        )
        assert ledger is not None
        assert ledger.status is OperationStatus.SUCCEEDED
        producer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await producer
        await service.preflight_close()
        durable = await storage.task.tasks.graph_state(graph.graph_id, tenant_id="tenant")
        assert durable is not None
        assert durable.status is TaskStatus.CANCELLED
        recorder.release.set()
        await service.cancel(graph.graph_id, CancelGraphRequest(principal, "cancel"))
        assert recorder.attempted[0] == recorder.attempted[1]
        assert len(recorder.observations) == 1
    finally:
        recorder.release.set()
        await asyncio.gather(producer, return_exceptions=True)
        await service.drain_owned_finalizers()
        await service.drain_metric_projector()
        await storage.close()


async def test_detached_cancel_cleanup_does_not_take_over_metric_delivery() -> None:
    class BlockedDependencies(_DependencyRelease):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def release_graph_dependencies(
            self, state: TaskGraphState, *, tenant_id: str,
        ) -> None:
            self.started.set()
            await self.release.wait()
            await super().release_graph_dependencies(state, tenant_id=tenant_id)

    storage = RuntimeStorage.in_memory()
    await storage.initialize(namespace="task-metric-detached", tenant_id="tenant")
    recorder = _BlockingRecorder()
    dependencies = BlockedDependencies()
    service = DefaultTaskGraphService(
        storage.task,
        TenantAuthorizationPolicy("tenant"),
        preflight=dependencies,  # type: ignore[arg-type]
        metric_recorder=recorder,
        metric_source_namespace="workspace",
    )
    principal = Principal("owner", "tenant")
    graph = TaskGraph("detached", (TaskNode("node"),))
    request = TaskGraphRequest(graph, principal, "submit")
    await storage.task.admissions.admit(TaskGraphAdmission.from_request(request), graph)
    producer = asyncio.create_task(
        service.cancel(graph.graph_id, CancelGraphRequest(principal, "cancel")),
    )
    try:
        await asyncio.wait_for(dependencies.started.wait(), 1)
        producer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await producer
        dependencies.release.set()
        await asyncio.wait_for(service.drain_owned_finalizers(), 1)
        await service.preflight_close()
        assert dependencies.released == {graph.graph_id}
        assert not recorder.started.is_set()
        ledger = await storage.task.operations.get(
            idempotency_key_digest("cancel"), tenant_id="tenant",
        )
        assert ledger is not None
        assert ledger.status is OperationStatus.SUCCEEDED
    finally:
        dependencies.release.set()
        recorder.release.set()
        await asyncio.gather(producer, return_exceptions=True)
        await service.drain_owned_finalizers()
        await service.drain_metric_projector()
        await storage.close()


async def test_terminal_wait_deadline_excludes_metric_io_and_cancellation_preserves_truth() -> None:
    storage = RuntimeStorage.in_memory()
    await storage.initialize(namespace="task-metric-wait", tenant_id="tenant")
    recorder = _BlockingRecorder()
    dependencies = _DependencyRelease()
    service = DefaultTaskGraphService(
        storage.task,
        TenantAuthorizationPolicy("tenant"),
        preflight=dependencies,  # type: ignore[arg-type]
        metric_recorder=recorder,
        metric_source_namespace="workspace",
    )
    principal = Principal("owner", "tenant")
    graph = TaskGraph("wait-metric", (TaskNode("node"),))
    request = TaskGraphRequest(graph, principal, "submit")
    await storage.task.admissions.admit(TaskGraphAdmission.from_request(request), graph)
    await storage.task.tasks.cancel_graph(graph.graph_id, tenant_id="tenant")
    producer = asyncio.create_task(service.wait(
        graph.graph_id, principal=principal, timeout_seconds=0.01,
    ))
    try:
        await asyncio.wait_for(recorder.started.wait(), 1)
        assert dependencies.released == {graph.graph_id}
        assert recorder.producer_tasks == [producer]
        await asyncio.sleep(0.03)
        assert not producer.done()
        producer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await producer
        durable = await storage.task.tasks.graph_state(graph.graph_id, tenant_id="tenant")
        assert durable is not None
        assert durable.status is TaskStatus.CANCELLED
        recorder.release.set()
        result = await service.wait(graph.graph_id, principal=principal, timeout_seconds=1)
        assert result.status is TaskStatus.CANCELLED
        assert recorder.attempted[0] == recorder.attempted[1]
        assert len(recorder.observations) == 1
    finally:
        recorder.release.set()
        await asyncio.gather(producer, return_exceptions=True)
        await service.drain_metric_projector()
        await storage.close()
