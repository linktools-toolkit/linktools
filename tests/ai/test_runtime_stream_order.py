
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._output import bind_output
from linktools.ai.core import (
    ExecutionDeltaType,
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    Page,
    Principal,
    StopReason,
    UsageMetrics,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.spec import AgentSpec
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime._event import (
    DefaultEventService,
    ExecutionDelta,
    LiveExecutionEventBroker,
    _LiveEvent,
)
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime.service_api import ExecutionEvent
from linktools.ai.runtime.state._commands import RuntimeStateCommands
from linktools.ai.runtime.state._contracts import (
    ExecutionCancelRequestCommit,
    ExecutionEventAppend,
    ExecutionRecord,
    ExecutionTerminalCommit,
    ExecutionTerminalCommitResult,
    ResultRecord,
)


def _binding_snapshot() -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("default", 1, "default"),
        agent_digest="b" * 64,
        output_type_module=output.value_type.__module__,
        output_type_qualname=output.value_type.__qualname__,
        output_schema_id=output.schema_id,
        output_schema_revision=output.schema_revision,
        output_schema_fingerprint=output.schema_fingerprint,
        local_runtime_capability_descriptors=(),
        binding_digest="a" * 64,
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
        tenant_id="tenant",
        session_id=None,
        binding_digest="a" * 64,
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
        planning=False,
        thinking=False,
        binding=_binding_snapshot(),
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
async def test_live_semantic_events_keep_agent_source_order() -> None:
    broker = LiveExecutionEventBroker()
    lease = broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 1)
    live = broker.claim_local_producer(lease)
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
        {"run_id": "run"},
        durable_sequence=3,
    )
    broker.complete("execution")
    items = [item async for item in live]
    assert isinstance(items[0], ExecutionDelta) and items[0].content == "before"
    assert isinstance(items[1], _LiveEvent)
    assert items[1].event_type is ExecutionEventType.TOOL_CALL_STARTED
    assert items[1].durable_sequence == 2
    assert isinstance(items[2], ExecutionDelta) and items[2].content == "after"
    assert isinstance(items[3], _LiveEvent)
    assert items[3].event_type is ExecutionEventType.EXECUTION_SUCCEEDED
    assert items[3].durable_sequence == 3


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
            ExecutionCancelRequestCommit("execution", "tenant", 0, 0, "cancel-op", now),
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

    commands = RuntimeStateCommands(_ExecutionRepo(), namespace="stream-order", events=_Events())
    with pytest.raises(AIError) as caught:
        await commands.commit_cancel_checkpoint(
            ExecutionCancelRequestCommit(
                "execution",
                "tenant",
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
        ) -> ExecutionRecord:
            del commit, expected_status
            assert audit_events == (pending,)
            self.started.set()
            await self.release.wait()
            return committed

    commands = _Commands()
    backend = object.__new__(LocalExecutionBackend)
    backend._pending_audit_events = {"execution": [pending]}
    backend._pending_audit_locks = {}
    backend._live_broker = broker
    backend._runtime_commands = commands
    commit = ExecutionCancelRequestCommit(
        "execution",
        "tenant",
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
    commands.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "execution" not in backend._pending_audit_events
    live = broker.subscribe("execution")
    first = await live.__anext__()
    second = await live.__anext__()
    await live.close()
    assert isinstance(first, _LiveEvent) and first.durable_sequence == 1
    assert isinstance(second, _LiveEvent)
    assert second.event_type is ExecutionEventType.CANCEL_REQUESTED
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
    terminal = replace(current, status=ExecutionStatus.FAILED, revision=2, event_sequence=2)
    result = ResultRecord(
        "execution",
        "tenant",
        None,
        None,
        None,
        None,
        StopReason.ERROR,
        UsageMetrics(),
        datetime.now(timezone.utc),
    )
    commit = ExecutionTerminalCommit(
        0,
        0,
        terminal,
        result,
        ExecutionEventType.EXECUTION_FAILED,
        {"error_code": ErrorCode.EXECUTION_FAILED.value},
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
        ) -> ExecutionTerminalCommitResult:
            del session_id
            assert actual is commit
            assert audit_events == (pending,)
            self.started.set()
            await self.release.wait()
            return committed

    commands = _Commands()
    backend = object.__new__(LocalExecutionBackend)
    backend._pending_audit_events = {"execution": [pending]}
    backend._pending_audit_locks = {}
    backend._live_broker = broker
    backend._runtime_commands = commands
    backend._terminal_events = {}
    task = asyncio.create_task(backend.commit_terminal_checkpoint(commit, session_id=None))
    await commands.started.wait()
    task.cancel()
    commands.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "execution" not in backend._pending_audit_events
    live = broker.subscribe("execution")
    first = await live.__anext__()
    second = await live.__anext__()
    await live.close()
    assert isinstance(first, _LiveEvent) and first.durable_sequence == 1
    assert isinstance(second, _LiveEvent)
    assert second.event_type is ExecutionEventType.EXECUTION_FAILED
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
    assert streamed[2].event_type is ExecutionEventType.EXECUTION_SUCCEEDED


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
    lease = broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 0)
    live = broker.claim_local_producer(lease)
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
        async for item in service._stream_claimed(
            lease,
            principal=principal,
            after_sequence=1,
        )
    ]
    assert streamed == []
    assert live is lease.subscription


@pytest.mark.asyncio
async def test_fast_completed_prepared_abort_releases_live_state() -> None:
    broker = LiveExecutionEventBroker()
    lease = broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 0)
    broker.publish(
        ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "buffered")
    )
    broker.complete("execution")
    assert broker.is_local_producer("execution")
    broker.abort_local_producer(lease)
    assert not broker.is_local_producer("execution")
    assert "execution" not in broker._buffers
    assert "execution" not in broker._completed
    assert "execution" not in broker._prepared


@pytest.mark.asyncio
async def test_reconnect_aligns_to_confirmed_cursor_before_replaying_deltas() -> None:
    broker = LiveExecutionEventBroker()
    lease = broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 1)
    broker.claim_local_producer(lease)
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
    iterator = service._stream_claimed(
        lease,
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
    assert terminal.event_type is ExecutionEventType.EXECUTION_SUCCEEDED
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)


@pytest.mark.asyncio
async def test_reconnect_after_terminal_cursor_does_not_replay_deltas() -> None:
    broker = LiveExecutionEventBroker()
    lease = broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 1)
    broker.claim_local_producer(lease)
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
        async for item in service._stream_claimed(
            lease,
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

    commands = RuntimeStateCommands(_ExecutionRepo(), namespace="stream-order", events=_Events())
    with pytest.raises(AIError) as caught:
        await commands.commit_cancel_checkpoint(
            ExecutionCancelRequestCommit(
                "execution",
                "tenant",
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

    commands = RuntimeStateCommands(_ExecutionRepo(), namespace="stream-order", events=_Events())
    with pytest.raises(AIError) as caught:
        await commands.commit_cancel_checkpoint(
            ExecutionCancelRequestCommit(
                "execution",
                "tenant",
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

    commands = RuntimeStateCommands(_ExecutionRepo(), namespace="stream-order", events=_Events())
    committed = await commands.commit_cancel_checkpoint(
        ExecutionCancelRequestCommit(
            "execution",
            "tenant",
            0,
            0,
            "cancel-op",
            datetime.now(timezone.utc),
        ),
        expected_status=ExecutionStatus.STARTED,
    )
    assert committed is advanced
