#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import linktools.ai.runtime._event as event_module
from linktools.ai.agent import AgentBindingContract
from linktools.ai.agent._output import bind_output
from linktools.ai.core import (
    ExecutionDeltaType,
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    Page,
    Principal,
    StopReason,
    TaskStatus,
    UsageMetrics,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeState, TaskGraphRun, TaskGraphRunEvent
from linktools.ai.runtime._event import (
    DefaultEventService,
    ExecutionDelta,
    LiveExecutionEventBroker,
    _LiveEvent,
)
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime._watch_cursor import decode_graph_watch_cursor
from linktools.ai.runtime.service_api import ExecutionEvent, ExecutionTreeEvent
from linktools.ai.runtime.state._commands import RuntimeStateCommands
from linktools.ai.runtime.state._contracts import (
    ExecutionCancelRequestCommit,
    ExecutionEventAppend,
    ExecutionRecord,
    ExecutionTerminalCommit,
    ExecutionTerminalCommitResult,
    ResultRecord,
)
from linktools.ai.spec import AgentSpec
from linktools.ai.task import (
    TaskEvent,
    TaskEventType,
    TaskGraphResult,
    TaskGraphSnapshot,
    TaskNode,
    TaskNodeView,
)
from ._runtime_test_helpers import execution_owner_fields


def _binding_contract() -> AgentBindingContract:
    output = bind_output()
    return AgentBindingContract(
        agent_spec=AgentSpec("default", model_route="default"),
        model_contract={"route_id": "default", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
    )


def _execution(
    *,
    status: ExecutionStatus = ExecutionStatus.STARTED,
    revision: int = 0,
    event_sequence: int = 0,
) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    return ExecutionRecord(
        execution_id="execution",
        session_id=None,
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=status,
        revision=revision,
        event_sequence=event_sequence,
        agent_run_sequence=0,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        mode="run",
        planning=False,
        thinking=False,
        binding=_binding_contract(),
        **execution_owner_fields(),
    )


class _AllowAll:
    async def authorize(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


class _ExecutionReader:
    def __init__(self, execution: ExecutionRecord) -> None:
        self.execution = execution

    async def get_header(self, execution_id: str, *, tenant_id: str) -> object:
        del execution_id, tenant_id
        return object()

    async def get(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord:
        del execution_id, tenant_id
        return self.execution


class _EventReader:
    def __init__(self, pages: dict[int, tuple[ExecutionEvent, ...]]) -> None:
        self.pages = pages
        self.started: asyncio.Event | None = None
        self.release: asyncio.Event | None = None

    async def list(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        after_sequence: int,
        limit: int,
    ) -> Page[ExecutionEvent]:
        del execution_id, tenant_id, limit
        if self.started is not None and after_sequence == 0:
            self.started.set()
            assert self.release is not None
            await self.release.wait()
        return Page(self.pages.get(after_sequence, ()), None)


def _service(
    execution: ExecutionRecord,
    events: _EventReader,
    broker: LiveExecutionEventBroker,
) -> DefaultEventService:
    return DefaultEventService(
        _ExecutionReader(execution),
        events,
        _AllowAll(),
        lambda execution_id, tenant_id: None,
        broker,
    )


@pytest.mark.asyncio
async def test_unconfirmed_terminal_event_converges_to_durable_tail() -> None:
    broker = LiveExecutionEventBroker()
    broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 0)
    broker.publish_event(
        "execution",
        ExecutionEventType.EXECUTION_SUCCEEDED,
        {},
        durable_sequence=None,
    )
    broker.complete("execution")
    service = _service(
        _execution(status=ExecutionStatus.SUCCEEDED, event_sequence=1),
        _EventReader(
            {0: (ExecutionEvent("execution", 1, "EXECUTION_SUCCEEDED", {}),)}
        ),
        broker,
    )

    streamed = [
        event
        async for event in service.stream(
            "execution",
            principal=Principal("owner", "tenant"),
        )
    ]

    assert len(streamed) == 1
    assert streamed[0].durable_sequence is None
    assert streamed[0].event_type == ExecutionEventType.EXECUTION_SUCCEEDED


@pytest.mark.asyncio
async def test_unconfirmed_completion_rejects_missing_durable_tail() -> None:
    broker = LiveExecutionEventBroker()
    broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 0)
    broker.publish_event(
        "execution",
        ExecutionEventType.EXECUTION_SUCCEEDED,
        {},
        durable_sequence=None,
    )
    broker.complete("execution")
    service = _service(
        _execution(status=ExecutionStatus.SUCCEEDED, event_sequence=1),
        _EventReader({}),
        broker,
    )

    with pytest.raises(AIError) as raised:
        async for _ in service.stream(
            "execution",
            principal=Principal("owner", "tenant"),
        ):
            pass

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_graph_wait_terminal_result_waits_for_observer_completion() -> None:
    principal = Principal("owner", "tenant")
    broker = LiveExecutionEventBroker()
    broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 0)
    executions = _ExecutionReader(_execution(event_sequence=1))
    events = _EventReader(
        {0: (ExecutionEvent("execution", 1, "EXECUTION_STARTED", {}),)}
    )
    execution_service = DefaultEventService(
        executions,
        events,
        _AllowAll(),
        lambda execution_id, tenant_id: None,
        broker,
    )
    waiter_release = asyncio.Event()
    graph_terminal_release = asyncio.Event()
    observed_live = asyncio.Event()
    observed_terminal = asyncio.Event()
    now = datetime.now(timezone.utc)
    terminal_graph_event = TaskEvent(
        1,
        "graph",
        2,
        TaskEventType.GRAPH_CHANGED,
        now,
        TaskStatus.SUCCEEDED,
        previous_status=TaskStatus.RUNNING,
    )

    class GraphService:
        async def snapshot(
            self,
            graph_id: str,
            *,
            principal: Principal,
        ) -> TaskGraphSnapshot:
            del principal
            state = TaskNodeView(
                graph_id,
                "node",
                (),
                TaskStatus.WAITING,
                None,
                1,
                None,
                None,
                None,
                None,
                "execution",
            )
            return TaskGraphSnapshot(
                graph_id,
                TaskStatus.RUNNING,
                (TaskNode("node"),),
                (state,),
                1,
            )

        async def wait(
            self,
            graph_id: str,
            *,
            principal: Principal,
            timeout_seconds: float | None = None,
        ) -> TaskGraphResult:
            del principal, timeout_seconds
            await waiter_release.wait()
            return TaskGraphResult(graph_id, TaskStatus.SUCCEEDED)

        async def stream_events(
            self,
            graph_id: str,
            *,
            principal: Principal,
            after_sequence: int = 0,
        ) -> AsyncIterator[TaskEvent]:
            del principal, after_sequence
            yield TaskEvent(
                1,
                graph_id,
                1,
                TaskEventType.NODE_CHANGED,
                now,
                TaskStatus.WAITING,
                previous_status=TaskStatus.RUNNING,
                node_id="node",
                fence=1,
                execution_id="execution",
            )
            await graph_terminal_release.wait()
            yield terminal_graph_event

    async def watch_tree(
        execution_id: str,
        *,
        principal: Principal,
        after_sequences: Mapping[str, int] | None = None,
        include_content: bool = False,
    ) -> AsyncIterator[ExecutionTreeEvent]:
        del after_sequences, include_content
        async for event in execution_service.stream(execution_id, principal=principal):
            yield ExecutionTreeEvent(
                execution_id,
                "agent",
                ExecutionLineageKind.RUN,
                None,
                execution_id,
                None,
                0,
                event,
            )

    run = TaskGraphRun(
        SimpleNamespace(namespace="watch-test", graph=GraphService()),
        "graph",
        principal,
        watch_tree,
    )
    observed: list[TaskGraphRunEvent] = []

    async def observer(event: TaskGraphRunEvent) -> None:
        observed.append(event)
        if (
            isinstance(event.event, ExecutionTreeEvent)
            and event.event.event.durable_sequence is None
        ):
            observed_live.set()
        if event.event == terminal_graph_event:
            observed_terminal.set()

    broker.publish_event(
        "execution",
        "EXECUTION_STARTED",
        {},
        durable_sequence=1,
    )
    broker.publish_event(
        "execution",
        ExecutionEventType.EXECUTION_SUCCEEDED,
        {},
        durable_sequence=None,
    )
    waiting = asyncio.create_task(run.wait(observer=observer))
    await asyncio.wait_for(observed_live.wait(), 1)

    executions.execution = _execution(
        status=ExecutionStatus.SUCCEEDED,
        event_sequence=2,
    )
    events.pages[1] = (
        ExecutionEvent("execution", 2, "EXECUTION_SUCCEEDED", {}),
    )
    waiter_release.set()
    await asyncio.sleep(0)
    assert not waiting.done()

    broker.complete("execution")
    graph_terminal_release.set()
    result = await asyncio.wait_for(waiting, 1)

    assert result.status is TaskStatus.SUCCEEDED
    assert observed_terminal.is_set()
    assert observed[-1].event == terminal_graph_event
    assert observed[-1].cursor is not None


@pytest.mark.asyncio
async def test_live_semantic_events_keep_agent_source_order() -> None:
    broker = LiveExecutionEventBroker()
    broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 1)
    live = broker.claim_local_producer("execution")
    assert live is not None
    broker.publish(ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "before"))
    broker.publish_event(
        "execution",
        ExecutionEventType.TOOL_CALL_STARTED,
        {"call_id": "call", "tool_name": "tool"},
        durable_sequence=None,
    )
    broker.publish(ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "after"))
    broker.confirm_events("execution", first_sequence=2, count=1)
    broker.publish_event(
        "execution",
        ExecutionEventType.EXECUTION_SUCCEEDED,
        {"agent_run_id": "run"},
        durable_sequence=3,
    )
    broker.complete("execution")
    items = [item async for item in live]
    assert isinstance(items[0], ExecutionDelta) and items[0].content == "before"
    assert isinstance(items[1], _LiveEvent)
    assert items[1].event_type == ExecutionEventType.TOOL_CALL_STARTED
    assert items[1].durable_sequence == 2
    assert isinstance(items[2], ExecutionDelta) and items[2].content == "after"
    assert isinstance(items[3], _LiveEvent)
    assert items[3].event_type == ExecutionEventType.EXECUTION_SUCCEEDED
    assert items[3].durable_sequence == 3


@pytest.mark.asyncio
async def test_live_durable_terminal_publication_is_idempotent() -> None:
    broker = LiveExecutionEventBroker()
    broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 0)
    live = broker.claim_local_producer("execution")
    assert live is not None
    payload = {"agent_run_id": "run"}

    broker.publish_event(
        "execution",
        ExecutionEventType.EXECUTION_SUCCEEDED,
        payload,
        durable_sequence=1,
    )
    broker.publish_event(
        "execution",
        ExecutionEventType.EXECUTION_SUCCEEDED,
        payload,
        durable_sequence=1,
    )
    with pytest.raises(AIError) as error:
        broker.publish_event(
            "execution",
            ExecutionEventType.EXECUTION_FAILED,
            {"error_code": "internal_error"},
            durable_sequence=1,
        )
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

    broker.complete("execution")
    items = [item async for item in live]
    assert len(items) == 1


@pytest.mark.asyncio
async def test_public_stream_emits_pending_semantic_event_immediately() -> None:
    broker = LiveExecutionEventBroker()
    broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 0)
    service = _service(_execution(), _EventReader({}), broker)
    principal = Principal("user", "tenant", "user")
    iterator = service.stream(
        "execution",
        principal=principal,
    ).__aiter__()

    pending = asyncio.create_task(anext(iterator))
    await asyncio.sleep(0)
    broker.publish_event(
        "execution",
        ExecutionEventType.TOOL_CALL_STARTED,
        {"call_id": "call", "tool_name": "tool"},
        durable_sequence=None,
    )
    event = await asyncio.wait_for(pending, timeout=1.0)
    assert event.durable_sequence is None
    assert event.event_type == ExecutionEventType.TOOL_CALL_STARTED

    await iterator.aclose()


@pytest.mark.asyncio
async def test_live_overflow_skips_semantic_events_already_emitted_ephemerally() -> None:
    broker = LiveExecutionEventBroker()
    broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 0)
    events = _EventReader({})
    service = _service(
        _execution(status=ExecutionStatus.SUCCEEDED, revision=260, event_sequence=260),
        events,
        broker,
    )
    principal = Principal("user", "tenant", "user")
    iterator = service.stream(
        "execution",
        principal=principal,
    ).__aiter__()

    first_task = asyncio.create_task(anext(iterator))
    await asyncio.sleep(0)
    first_payload = {"call_id": "call-1", "tool_name": "tool"}
    broker.publish_event(
        "execution",
        ExecutionEventType.TOOL_CALL_STARTED,
        first_payload,
        durable_sequence=None,
    )
    first = await asyncio.wait_for(first_task, timeout=1.0)
    assert first.durable_sequence is None
    assert first.payload == first_payload

    broker.confirm_events("execution", first_sequence=1, count=1)
    durable = tuple(
        ExecutionEvent(
            "execution",
            sequence,
            ExecutionEventType.EXECUTION_SUCCEEDED
            if sequence == 260
            else ExecutionEventType.TOOL_CALL_STARTED,
            {}
            if sequence == 260
            else {"call_id": f"call-{sequence}", "tool_name": "tool"},
        )
        for sequence in range(1, 261)
    )
    events.pages[0] = durable
    for event in durable[1:]:
        broker.publish_event(
            event.execution_id,
            event.event_type,
            event.payload,
            durable_sequence=event.sequence,
        )
    broker.complete("execution")

    remaining = [item async for item in iterator]
    assert [item.durable_sequence for item in remaining] == list(range(2, 261))
    assert remaining[-1].event_type == ExecutionEventType.EXECUTION_SUCCEEDED


@pytest.mark.asyncio
async def test_live_broker_durable_overflow_switches_to_repository_replay() -> None:
    broker = LiveExecutionEventBroker()
    broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 0)
    for sequence in range(1, 300):
        broker.publish_event(
            "execution",
            ExecutionEventType.TOOL_CALL_STARTED,
            {"call_id": f"call-{sequence}", "tool_name": "tool"},
            durable_sequence=sequence,
        )

    assert "execution" in broker._replay_required
    assert "execution" not in broker._buffers


@pytest.mark.asyncio
async def test_live_broker_bounds_slow_subscribers_and_pending_events() -> None:
    broker = LiveExecutionEventBroker()
    broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 0)
    first = broker.claim_local_producer("execution")
    second = broker.subscribe("execution")
    assert first is not None

    for index in range(300):
        broker.publish_event(
            "execution",
            ExecutionEventType.TOOL_CALL_STARTED,
            {"call_id": f"call-{index}", "tool_name": "tool"},
            durable_sequence=None,
        )

    assert first.replay_required
    assert second.replay_required
    assert len(first._queue) == 1
    assert len(second._queue) == 1
    assert broker._pending_event_counts["execution"] == 300

    await broker.wait_for_activity("execution")
    broker.publish(
        ExecutionDelta(
            "execution",
            ExecutionDeltaType.ASSISTANT_TEXT_DELTA,
            "discarded-after-replay",
        )
    )
    assert not broker._activity["execution"].is_set()
    broker.publish_event(
        "execution",
        ExecutionEventType.TOOL_CALL_STARTED,
        {"call_id": "late-call", "tool_name": "tool"},
        durable_sequence=None,
    )
    assert not broker._activity["execution"].is_set()

    broker.confirm_events("execution", first_sequence=1, count=301)
    assert broker._activity["execution"].is_set()
    assert "execution" not in broker._pending_event_counts
    await first.close()
    await second.close()


@pytest.mark.asyncio
async def test_live_overflow_replays_all_durable_events_without_loss() -> None:
    broker = LiveExecutionEventBroker()
    broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 0)
    durable = tuple(
        ExecutionEvent(
            "execution",
            sequence,
            ExecutionEventType.EXECUTION_SUCCEEDED
            if sequence == 300
            else ExecutionEventType.TOOL_CALL_STARTED,
            {}
            if sequence == 300
            else {"call_id": f"call-{sequence}", "tool_name": "tool"},
        )
        for sequence in range(1, 301)
    )
    for event in durable:
        broker.publish_event(
            event.execution_id,
            event.event_type,
            event.payload,
            durable_sequence=event.sequence,
        )
    broker.complete("execution")

    service = _service(
        _execution(
            status=ExecutionStatus.SUCCEEDED,
            revision=300,
            event_sequence=300,
        ),
        _EventReader({0: durable}),
        broker,
    )
    principal = Principal("user", "tenant", "user")
    streamed = [
        item
        async for item in service.stream(
            "execution",
            principal=principal,
        )
    ]

    assert [item.durable_sequence for item in streamed] == list(range(1, 301))
    assert streamed[-1].event_type == ExecutionEventType.EXECUTION_SUCCEEDED


@pytest.mark.asyncio
async def test_cancel_batches_pending_audit_in_one_filesystem_mutation(tmp_path: Path) -> None:
    state = RuntimeState.filesystem(tmp_path / "runtime")
    await state.initialize(namespace="stream-order", tenant_id="tenant")
    try:
        now = datetime.now(timezone.utc)
        execution = _execution()
        await state.execution.executions.create(execution)
        generation = next((tmp_path / "runtime" / "execution").rglob("generation"))
        before = int(generation.read_text(encoding="utf-8"))
        pending = (
            ExecutionEventAppend(ExecutionEventType.ASSISTANT_PART_COMPLETED, {"part": "text"}),
            ExecutionEventAppend(
                ExecutionEventType.TOOL_CALL_STARTED,
                {"call_id": "call", "tool_name": "tool"},
            ),
        )
        committed = await state.execution.executions.request_cancel(
            ExecutionCancelRequestCommit("execution", 0, 0, "cancel-op", now),
            pending_events=pending,
        )
        after = int(generation.read_text(encoding="utf-8"))
        assert after == before + 1
        assert committed.status is ExecutionStatus.CANCELLING
        assert committed.revision == 3
        assert committed.event_sequence == 3
        page = await state.execution.events.list(
            "execution",
            tenant_id="tenant",
            after_sequence=0,
            limit=10,
        )
        assert [event.sequence for event in page.items] == [1, 2, 3]
        assert [event.event_type for event in page.items] == [
            ExecutionEventType.ASSISTANT_PART_COMPLETED,
            ExecutionEventType.TOOL_CALL_STARTED,
            ExecutionEventType.CANCEL_REQUESTED,
        ]
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_cancel_terminal_race_is_conflict_not_integrity() -> None:
    terminal = _execution(status=ExecutionStatus.SUCCEEDED, revision=1, event_sequence=1)

    class _ExecutionRepo:
        tenant_id = "tenant"

        async def request_cancel(
            self,
            commit: ExecutionCancelRequestCommit,
            *,
            pending_events: tuple[ExecutionEventAppend, ...] = (),
        ) -> ExecutionRecord:
            del commit, pending_events
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        async def get(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord:
            del execution_id, tenant_id
            return terminal

    class _Events:
        async def list(
            self,
            execution_id: str,
            *,
            tenant_id: str,
            after_sequence: int,
            limit: int,
        ) -> Page[object]:
            del execution_id, tenant_id, after_sequence, limit
            event = SimpleNamespace(
                sequence=1,
                event_type=ExecutionEventType.EXECUTION_SUCCEEDED,
                payload={},
            )
            return Page((event,), None)

    commands = RuntimeStateCommands(
        _ExecutionRepo(),
        namespace="stream-order",
        events=_Events(),
        background_tasks=set(),
    )
    with pytest.raises(AIError) as caught:
        await commands.commit_cancel_checkpoint(
            ExecutionCancelRequestCommit(
                "execution",
                0,
                0,
                "cancel-op",
                datetime.now(timezone.utc),
            ),
            expected_status=ExecutionStatus.STARTED,
        )
    assert caught.value.code is ErrorCode.STORAGE_CONFLICT


@pytest.mark.asyncio
async def test_cancel_local_bookkeeping_survives_caller_cancellation() -> None:
    broker = LiveExecutionEventBroker()
    broker.register_local_producer("execution", 0)
    broker.publish_event(
        "execution",
        ExecutionEventType.TOOL_CALL_STARTED,
        {"call_id": "call", "tool_name": "tool"},
        durable_sequence=None,
    )
    pending = ExecutionEventAppend(
        ExecutionEventType.TOOL_CALL_STARTED,
        {"call_id": "call", "tool_name": "tool"},
    )
    committed = _execution(status=ExecutionStatus.CANCELLING, revision=2, event_sequence=2)

    class _Commands:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def commit_cancel_checkpoint(
            self,
            commit: ExecutionCancelRequestCommit,
            *,
            expected_status: ExecutionStatus,
            audit_events: tuple[ExecutionEventAppend, ...],
            background_tasks: set[asyncio.Task[object]] | None = None,
        ) -> ExecutionRecord:
            del commit, expected_status, background_tasks
            assert audit_events == (pending,)
            self.started.set()
            await self.release.wait()
            return committed

    commands = _Commands()
    backend = object.__new__(LocalExecutionBackend)
    backend._tenant_id = "tenant"
    backend._pending_audit_events = {"execution": [pending]}
    backend._pending_audit_locks = {}
    backend._checkpoint_tasks = set()
    backend._execution_durable_tasks = {}
    backend._execution = SimpleNamespace(executions=_ExecutionReader(_execution()))
    backend._live_broker = broker
    backend._runtime_commands = commands
    backend._metric_recorder = None
    commit = ExecutionCancelRequestCommit(
        "execution",
        0,
        0,
        "cancel-op",
        datetime.now(timezone.utc),
    )
    task = asyncio.create_task(
        backend.commit_cancel_checkpoint(commit, expected_status=ExecutionStatus.STARTED)
    )
    await commands.started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    owners = tuple(backend._checkpoint_tasks)
    assert len(owners) == 1
    assert "execution" in backend._pending_audit_events
    commands.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.gather(*owners)
    assert not backend._checkpoint_tasks
    assert "execution" not in backend._pending_audit_events
    live = broker.subscribe("execution")
    first = await live.__anext__()
    second = await live.__anext__()
    await live.close()
    assert isinstance(first, _LiveEvent) and first.durable_sequence == 1
    assert isinstance(second, _LiveEvent)
    assert second.event_type == ExecutionEventType.CANCEL_REQUESTED
    assert second.durable_sequence == 2


@pytest.mark.asyncio
async def test_terminal_local_bookkeeping_survives_caller_cancellation() -> None:
    broker = LiveExecutionEventBroker()
    broker.register_local_producer("execution", 0)
    broker.publish_event(
        "execution",
        ExecutionEventType.ASSISTANT_PART_COMPLETED,
        {"part": "text"},
        durable_sequence=None,
    )
    pending = ExecutionEventAppend(ExecutionEventType.ASSISTANT_PART_COMPLETED, {"part": "text"})
    current = _execution()
    terminal = replace(
        current,
        status=ExecutionStatus.FAILED,
        revision=2,
        event_sequence=2,
        error_code=ErrorCode.EXECUTION_FAILED.value,
        safe_error_details={},
    )
    result = ResultRecord(
        output=None,
        stop_reason=StopReason.ERROR,
        usage=UsageMetrics(),
        created_at=datetime.now(timezone.utc),
    )
    commit = ExecutionTerminalCommit(
        0,
        0,
        terminal,
        result,
        ExecutionEventType.EXECUTION_FAILED,
        {
            "error_code": ErrorCode.EXECUTION_FAILED.value,
            "safe_error_details": {},
        },
    )
    committed = ExecutionTerminalCommitResult(terminal, result)

    class _Commands:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def commit_terminal_checkpoint(
            self,
            actual: ExecutionTerminalCommit,
            *,
            session_id: str | None,
            audit_events: tuple[ExecutionEventAppend, ...],
            background_tasks: set[asyncio.Task[object]] | None = None,
        ) -> ExecutionTerminalCommitResult:
            del session_id, background_tasks
            assert actual is commit
            assert audit_events == (pending,)
            self.started.set()
            await self.release.wait()
            return committed

    commands = _Commands()
    backend = object.__new__(LocalExecutionBackend)
    backend._tenant_id = "tenant"
    backend._pending_audit_events = {"execution": [pending]}
    backend._pending_audit_locks = {}
    backend._checkpoint_tasks = set()
    backend._execution_durable_tasks = {}
    backend._live_broker = broker
    backend._runtime_commands = commands
    backend._terminal_events = {}
    backend._metric_recorder = None
    live = broker.subscribe("execution")
    task = asyncio.create_task(backend.commit_terminal_checkpoint(commit, session_id=None))
    await commands.started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    owners = tuple(backend._checkpoint_tasks)
    assert len(owners) == 1
    assert "execution" in backend._pending_audit_events
    commands.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.gather(*owners)
    assert not backend._checkpoint_tasks
    assert "execution" not in backend._pending_audit_events
    first = await live.__anext__()
    second = await live.__anext__()
    await live.close()
    assert isinstance(first, _LiveEvent) and first.durable_sequence == 1
    assert isinstance(second, _LiveEvent)
    assert second.event_type == ExecutionEventType.EXECUTION_FAILED
    assert second.durable_sequence == 2


@pytest.mark.asyncio
async def test_second_subscriber_pins_buffer_during_durable_prefix() -> None:
    broker = LiveExecutionEventBroker()
    broker.register_local_producer("execution", 1)
    base = ExecutionEvent("execution", 1, ExecutionEventType.EXECUTION_STARTED, {})
    events = _EventReader({0: (base,)})
    events.started = asyncio.Event()
    events.release = asyncio.Event()
    service = _service(
        _execution(status=ExecutionStatus.SUCCEEDED, revision=2, event_sequence=2),
        events,
        broker,
    )
    principal = Principal("user", "tenant", "user")

    async def collect() -> list[object]:
        return [item async for item in service.stream("execution", principal=principal)]

    task = asyncio.create_task(collect())
    await events.started.wait()
    broker.publish(ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "live"))
    broker.publish_event(
        "execution",
        ExecutionEventType.EXECUTION_SUCCEEDED,
        {},
        durable_sequence=2,
    )
    broker.complete("execution")
    events.release.set()
    streamed = await task
    assert streamed[0].durable_sequence == 1
    assert streamed[1].durable_sequence is None
    assert streamed[1].payload["text"] == "live"
    assert streamed[2].durable_sequence == 2
    assert streamed[2].event_type == ExecutionEventType.EXECUTION_SUCCEEDED


@pytest.mark.asyncio
async def test_local_completion_without_terminal_is_integrity_error() -> None:
    broker = LiveExecutionEventBroker()
    broker.register_local_producer("execution", 0)
    service = _service(_execution(), _EventReader({}), broker)
    principal = Principal("user", "tenant", "user")

    async def collect() -> list[object]:
        return [item async for item in service.stream("execution", principal=principal)]

    task = asyncio.create_task(collect())
    for _ in range(20):
        if broker._subscriptions.get("execution"):
            break
        await asyncio.sleep(0)
    broker.complete("execution")
    with pytest.raises(AIError) as caught:
        await task
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_durable_stream_stops_when_terminal_cursor_already_seen() -> None:
    broker = LiveExecutionEventBroker()
    service = _service(
        _execution(status=ExecutionStatus.SUCCEEDED, revision=2, event_sequence=2),
        _EventReader({}),
        broker,
    )
    principal = Principal("user", "tenant", "user")
    streamed = [
        item
        async for item in service.stream(
            "execution",
            principal=principal,
            after_sequence=2,
        )
    ]
    assert streamed == []


@pytest.mark.asyncio
async def test_local_stream_skips_already_seen_terminal_and_stops() -> None:
    broker = LiveExecutionEventBroker()
    broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 0)
    broker.publish_event(
        "execution",
        ExecutionEventType.EXECUTION_SUCCEEDED,
        {},
        durable_sequence=1,
    )
    broker.complete("execution")
    service = _service(
        _execution(status=ExecutionStatus.SUCCEEDED, revision=1, event_sequence=1),
        _EventReader({}),
        broker,
    )
    principal = Principal("user", "tenant", "user")
    streamed = [
        item
        async for item in service.stream(
            "execution",
            principal=principal,
            after_sequence=1,
        )
    ]
    assert streamed == []
    assert not broker.is_local_producer("execution")


@pytest.mark.asyncio
async def test_fast_completed_prepared_abort_releases_live_state() -> None:
    broker = LiveExecutionEventBroker()
    broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 0)
    broker.publish(
        ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "buffered")
    )
    broker.complete("execution")
    assert broker.is_local_producer("execution")
    broker.abandon_prepared_local_producer("execution")
    assert not broker.is_local_producer("execution")
    assert "execution" not in broker._buffers
    assert "execution" not in broker._completed
    assert "execution" not in broker._prepared


@pytest.mark.asyncio
async def test_reconnect_aligns_to_confirmed_cursor_before_replaying_deltas() -> None:
    broker = LiveExecutionEventBroker()
    broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 1)
    broker.publish(
        ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "before")
    )
    broker.publish_event(
        "execution",
        ExecutionEventType.TOOL_CALL_STARTED,
        {"call_id": "call", "tool_name": "tool"},
        durable_sequence=None,
    )
    broker.publish(
        ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "after")
    )
    service = _service(_execution(revision=1, event_sequence=1), _EventReader({}), broker)
    principal = Principal("user", "tenant", "user")
    iterator = service.stream(
        "execution",
        principal=principal,
        after_sequence=2,
    ).__aiter__()
    first_task = asyncio.create_task(anext(iterator))
    await asyncio.sleep(0)
    assert not first_task.done()

    broker.confirm_events("execution", first_sequence=2, count=1)
    first = await asyncio.wait_for(first_task, timeout=1.0)
    assert first.durable_sequence is None
    assert first.event_type is ExecutionDeltaType.ASSISTANT_TEXT_DELTA
    assert first.payload["text"] == "after"

    broker.publish_event(
        "execution",
        ExecutionEventType.EXECUTION_SUCCEEDED,
        {},
        durable_sequence=3,
    )
    broker.complete("execution")
    terminal = await asyncio.wait_for(anext(iterator), timeout=1.0)
    assert terminal.durable_sequence == 3
    assert terminal.event_type == ExecutionEventType.EXECUTION_SUCCEEDED
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)


@pytest.mark.asyncio
async def test_reconnect_after_terminal_cursor_does_not_replay_deltas() -> None:
    broker = LiveExecutionEventBroker()
    broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 1)
    broker.publish(
        ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "before")
    )
    broker.publish_event(
        "execution",
        ExecutionEventType.TOOL_CALL_STARTED,
        {"call_id": "call", "tool_name": "tool"},
        durable_sequence=2,
    )
    broker.publish(
        ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "after")
    )
    broker.publish_event(
        "execution",
        ExecutionEventType.EXECUTION_SUCCEEDED,
        {},
        durable_sequence=3,
    )
    broker.complete("execution")
    service = _service(
        _execution(status=ExecutionStatus.SUCCEEDED, revision=3, event_sequence=3),
        _EventReader({}),
        broker,
    )
    principal = Principal("user", "tenant", "user")
    streamed = [
        item
        async for item in service.stream(
            "execution",
            principal=principal,
            after_sequence=3,
        )
    ]
    assert streamed == []


@pytest.mark.asyncio
async def test_durable_stream_rejects_sequence_gap() -> None:
    broker = LiveExecutionEventBroker()
    gap = ExecutionEvent(
        "execution",
        2,
        ExecutionEventType.EXECUTION_SUCCEEDED,
        {},
    )
    service = _service(
        _execution(status=ExecutionStatus.SUCCEEDED, revision=2, event_sequence=2),
        _EventReader({0: (gap,)}),
        broker,
    )
    principal = Principal("user", "tenant", "user")
    with pytest.raises(AIError) as caught:
        _ = [item async for item in service.stream("execution", principal=principal)]
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_concurrent_cancel_winner_is_conflict_not_integrity() -> None:
    winner = _execution(
        status=ExecutionStatus.CANCELLING,
        revision=1,
        event_sequence=1,
    )

    class _ExecutionRepo:
        tenant_id = "tenant"

        async def request_cancel(
            self,
            commit: ExecutionCancelRequestCommit,
            *,
            pending_events: tuple[ExecutionEventAppend, ...] = (),
        ) -> ExecutionRecord:
            del commit, pending_events
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        async def get(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord:
            del execution_id, tenant_id
            return winner

    class _Events:
        async def list(
            self,
            execution_id: str,
            *,
            tenant_id: str,
            after_sequence: int,
            limit: int,
        ) -> Page[ExecutionEvent]:
            del execution_id, tenant_id, after_sequence, limit
            return Page(
                (
                    ExecutionEvent(
                        "execution",
                        1,
                        ExecutionEventType.CANCEL_REQUESTED,
                        {"operation_id": "other-cancel"},
                    ),
                ),
                None,
            )

    commands = RuntimeStateCommands(
        _ExecutionRepo(),
        namespace="stream-order",
        events=_Events(),
        background_tasks=set(),
    )
    with pytest.raises(AIError) as caught:
        await commands.commit_cancel_checkpoint(
            ExecutionCancelRequestCommit(
                "execution",
                0,
                0,
                "our-cancel",
                datetime.now(timezone.utc),
            ),
            expected_status=ExecutionStatus.STARTED,
        )
    assert caught.value.code is ErrorCode.STORAGE_CONFLICT


@pytest.mark.asyncio
async def test_revision_only_cancel_race_is_conflict_not_integrity() -> None:
    advanced = _execution(
        status=ExecutionStatus.STARTED,
        revision=1,
        event_sequence=0,
    )

    class _ExecutionRepo:
        tenant_id = "tenant"

        async def request_cancel(
            self,
            commit: ExecutionCancelRequestCommit,
            *,
            pending_events: tuple[ExecutionEventAppend, ...] = (),
        ) -> ExecutionRecord:
            del commit, pending_events
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        async def get(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord:
            del execution_id, tenant_id
            return advanced

    class _Events:
        async def list(
            self,
            execution_id: str,
            *,
            tenant_id: str,
            after_sequence: int,
            limit: int,
        ) -> Page[ExecutionEvent]:
            del execution_id, tenant_id, after_sequence, limit
            return Page((), None)

    commands = RuntimeStateCommands(
        _ExecutionRepo(),
        namespace="stream-order",
        events=_Events(),
        background_tasks=set(),
    )
    with pytest.raises(AIError) as caught:
        await commands.commit_cancel_checkpoint(
            ExecutionCancelRequestCommit(
                "execution",
                0,
                0,
                "cancel-op",
                datetime.now(timezone.utc),
            ),
            expected_status=ExecutionStatus.STARTED,
        )
    assert caught.value.code is ErrorCode.STORAGE_CONFLICT


@pytest.mark.asyncio
async def test_cancel_readback_accepts_own_suffix_after_revision_only_advance() -> None:
    advanced = _execution(
        status=ExecutionStatus.CANCELLING,
        revision=2,
        event_sequence=1,
    )

    class _ExecutionRepo:
        tenant_id = "tenant"

        async def request_cancel(
            self,
            commit: ExecutionCancelRequestCommit,
            *,
            pending_events: tuple[ExecutionEventAppend, ...] = (),
        ) -> ExecutionRecord:
            del commit, pending_events
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

        async def get(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord:
            del execution_id, tenant_id
            return advanced

    class _Events:
        async def list(
            self,
            execution_id: str,
            *,
            tenant_id: str,
            after_sequence: int,
            limit: int,
        ) -> Page[ExecutionEvent]:
            del execution_id, tenant_id, after_sequence, limit
            return Page(
                (
                    ExecutionEvent(
                        "execution",
                        1,
                        ExecutionEventType.CANCEL_REQUESTED,
                        {"operation_id": "cancel-op"},
                    ),
                ),
                None,
            )

    commands = RuntimeStateCommands(
        _ExecutionRepo(),
        namespace="stream-order",
        events=_Events(),
        background_tasks=set(),
    )
    committed = await commands.commit_cancel_checkpoint(
        ExecutionCancelRequestCommit(
            "execution",
            0,
            0,
            "cancel-op",
            datetime.now(timezone.utc),
        ),
        expected_status=ExecutionStatus.STARTED,
    )
    assert committed is advanced


@pytest.mark.asyncio
async def test_local_replay_timeout_keeps_waiting_on_python310_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = LiveExecutionEventBroker()
    broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 0)
    broker.publish_event(
        "execution",
        ExecutionEventType.TOOL_CALL_STARTED,
        {"call_id": "call", "tool_name": "tool"},
        durable_sequence=None,
    )
    broker.publish_event(
        "execution",
        ExecutionEventType.EXECUTION_SUCCEEDED,
        {},
        durable_sequence=2,
    )
    service = _service(
        _execution(status=ExecutionStatus.SUCCEEDED, revision=2, event_sequence=2),
        _EventReader({}),
        broker,
    )
    principal = Principal("user", "tenant", "user")
    calls = 0

    async def wait_for(awaitable: object, timeout: float) -> None:
        nonlocal calls
        del timeout
        close = getattr(awaitable, "close", None)
        if close is not None:
            close()
        calls += 1
        if calls == 1:
            raise asyncio.TimeoutError
        broker.confirm_events("execution", first_sequence=1, count=1)

    monkeypatch.setattr(event_module.asyncio, "wait_for", wait_for)
    streamed = [
        item
        async for item in service.stream(
            "execution",
            principal=principal,
            after_sequence=1,
        )
    ]

    assert calls == 2
    assert len(streamed) == 1
    assert streamed[0].durable_sequence == 2
    assert streamed[0].event_type == ExecutionEventType.EXECUTION_SUCCEEDED
    broker.complete("execution")


@pytest.mark.asyncio
async def test_remote_durable_polling_does_not_retain_broker_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = LiveExecutionEventBroker()
    terminal = ExecutionEvent(
        "execution",
        1,
        ExecutionEventType.EXECUTION_SUCCEEDED,
        {},
    )

    class _PollingEvents:
        def __init__(self) -> None:
            self.calls = 0

        async def list(
            self,
            execution_id: str,
            *,
            tenant_id: str,
            after_sequence: int,
            limit: int,
        ) -> Page[ExecutionEvent]:
            del execution_id, tenant_id, after_sequence, limit
            self.calls += 1
            return Page((), None) if self.calls == 1 else Page((terminal,), None)

    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(event_module.asyncio, "sleep", sleep)
    events = _PollingEvents()
    service = DefaultEventService(
        _ExecutionReader(_execution()),
        events,  # type: ignore[arg-type]
        _AllowAll(),
        lambda execution_id, tenant_id: None,
        broker,
    )
    principal = Principal("user", "tenant", "user")

    streamed = [
        item
        async for item in service.stream(
            "execution",
            principal=principal,
        )
    ]

    assert [item.durable_sequence for item in streamed] == [1]
    assert delays == [1.0]
    assert broker._activity == {}
