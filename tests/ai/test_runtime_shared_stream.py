#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional transport joins real execution owners without changing durable truth."""

from ._runtime_test_helpers import workspace_features

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from linktools.ai.core import CommitPolicy
from pydantic_ai.models.function import FunctionModel

from linktools.ai.capability import ExecutionCapability
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.observe import MetricQuery, MetricWindow, Metrics

from linktools.ai.runtime import Runtime, RuntimeStorage, RuntimeDomain
from linktools.ai.storage import InMemoryCacheBackend, InMemoryStreamBackend
from ._runtime_test_helpers import _Models


class _CountingStreamBackend(InMemoryStreamBackend):
    def __init__(self):
        super().__init__(max_bytes=1024 * 1024)
        self.open_keys = []
        self.read_entered = asyncio.Event()

    async def open_stream(self, key, producer):
        self.open_keys.append(key)
        return await super().open_stream(key, producer)

    async def read_stream(self, key, after, *, limit, wait_seconds):
        self.read_entered.set()
        return await super().read_stream(key, after, limit=limit, wait_seconds=wait_seconds)


class _CombinedCache:
    def __init__(self) -> None:
        content = InMemoryCacheBackend(max_bytes=1024 * 1024)
        streams = _CountingStreamBackend()
        self.get = content.get
        self.put = content.put
        self.delete = content.delete
        self.append = content.append
        self.read_batch = content.read_batch
        self.ack = content.ack
        self.open_stream = streams.open_stream
        self.read_stream = streams.read_stream
        self.open_keys = streams.open_keys
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_two_runtimes_observe_preview_before_completion(tmp_path) -> None:
    release = asyncio.Event()
    entered = asyncio.Event()

    async def model(messages, info):
        entered.set()
        yield "hello "
        await release.wait()
        yield "world"

    group = RuntimeFeatures[None]("shared-preview")
    group.agent("default", model="default", allow_tools=())
    from .test_runtime_content_cache_integration import CountingObjects, _put

    backend = _CombinedCache()
    group.capability(ExecutionCapability(coordination=backend), scope="runtime")
    metrics = Metrics.in_memory(commit=CommitPolicy.batch(batch_size=128, flush_interval=60.0, buffer=backend))
    start = datetime.now(timezone.utc) - timedelta(seconds=1)

    async def assert_persisted_metrics(expected: int) -> None:
        window = MetricWindow.between(start, datetime.now(timezone.utc) + timedelta(seconds=1))
        for name in (
            "linktools.model.request.count", "linktools.agent.run.count", "linktools.execution.count",
        ):
            result = await metrics.query(MetricQuery(name, window))
            assert sum(point.value for point in result.points) == expected

    objects = CountingObjects()
    reference = await _put(objects, b"shared cached body")
    models = _Models(FunctionModel(stream_function=model))
    path = tmp_path / "runtime.db"
    owner_storage = RuntimeStorage.sqlite(path, object_store=objects, cache=backend)
    reader_storage = RuntimeStorage.sqlite(path, object_store=objects, cache=backend)
    async with Runtime.open(
        "shared-preview", metrics=metrics, models=models, features=group,
        storage=owner_storage,
    ) as producer:
        async with Runtime.open(
            "shared-preview", metrics=metrics, models=models, features=group,
            storage=reader_storage,
        ) as reader:
            assert await producer._state.read_object(RuntimeDomain.EXECUTION, reference) == b"shared cached body"
            assert await reader._state.read_object(RuntimeDomain.EXECUTION, reference) == b"shared cached body"
            assert objects.reads == 1
            execution = await producer.agents.get().start("question")
            await asyncio.wait_for(entered.wait(), 5)
            remote = await reader.executions.get(execution.execution_id)
            observed = remote.watch(include_content=True)
            try:
                async def first_text():
                    async for item in observed:
                        if item.event.event_type == "ASSISTANT_TEXT_DELTA":
                            return item
                    raise AssertionError("no remote preview")
                first = await asyncio.wait_for(first_text(), 5)
                assert first.event.payload["text"] == "hello "
                assert first.cursor
                assert not release.is_set()
                release.set()
                remaining = [event async for event in observed]
                assert any(item.event.event_type == "ASSISTANT_TEXT_DELTA" for item in remaining)
                assert (await execution.wait()).result.output == {"text": "hello world"}
                assert (await remote.wait()).result.output == {"text": "hello world"}
            finally:
                release.set()
                await observed.aclose()
            await assert_persisted_metrics(0)
            await producer.close()
            await assert_persisted_metrics(1)
            assert not backend.closed
            assert await reader._state.read_object(RuntimeDomain.EXECUTION, reference) == b"shared cached body"
            second = await reader.agents.get().run("another question", timeout_seconds=5)
            assert second.result.output == {"text": "hello world"}
            await assert_persisted_metrics(1)
    await assert_persisted_metrics(2)
    assert backend.open_keys
    assert not backend.closed
    await backend.put("host-owned", b"after close")
    assert await backend.get("host-owned") == b"after close"


@pytest.mark.asyncio
async def test_remote_cursor_resumes_preview_without_cancelling_producer(tmp_path) -> None:
    release = asyncio.Event()

    async def model(messages, info):
        yield "first "
        await release.wait()
        yield "last"

    group = RuntimeFeatures[None]("shared-reconnect")
    group.agent("default", model="default", allow_tools=())
    backend = _CountingStreamBackend()
    group.capability(ExecutionCapability(coordination=backend), scope="runtime")
    models = _Models(FunctionModel(stream_function=model))
    path = tmp_path / "state.db"
    async with Runtime.open("reconnect", models=models, features=group,
                            storage=RuntimeStorage.sqlite(path)) as owner:
        async with Runtime.open("reconnect", models=models, features=group,
                                storage=RuntimeStorage.sqlite(path)) as other:
            execution = await owner.agents.get().start("question")
            remote = await other.executions.get(execution.execution_id)
            stream = remote.watch(include_content=True)
            cursor = None
            try:
                async def initial():
                    async for event in stream:
                        if event.event.event_type == "ASSISTANT_TEXT_DELTA":
                            return event
                    raise AssertionError("missing preview")
                first = await asyncio.wait_for(initial(), 5)
                assert first.event.payload["text"] == "first "
                cursor = first.cursor
            finally:
                await stream.aclose()
            assert not release.is_set()
            async with Runtime.open("reconnect", models=models, features=group,
                                    storage=RuntimeStorage.sqlite(path)) as reconnected:
                remote = await reconnected.executions.get(execution.execution_id)
                resumed = remote.watch(include_content=True, cursor=cursor)
                backend.read_entered.clear()
                consume = asyncio.create_task(_collect(resumed))
                try:
                    await asyncio.wait_for(backend.read_entered.wait(), 5)
                    release.set()
                    seen = await asyncio.wait_for(consume, 5)
                    text = "".join(item.event.payload["text"] for item in seen
                                   if item.event.event_type == "ASSISTANT_TEXT_DELTA")
                    assert text == "last"
                    assert (await execution.wait()).result.output == {"text": "first last"}
                finally:
                    release.set()
                    await resumed.aclose()



async def _collect(stream):
    return [item async for item in stream]


def _event_service(backend, *, durable_count=3):
    from types import SimpleNamespace
    from linktools.ai.core import ExecutionStatus, Page
    from linktools.ai.runtime._event import DefaultEventService, LiveExecutionEventBroker
    from linktools.ai.runtime._shared_stream import SharedStreamPublisher
    from linktools.ai.runtime.service_api import ExecutionEvent

    class Executions:
        reads = 0
        head = SimpleNamespace(producer_generation=1, producer_claim_id="claim", state=SimpleNamespace(value="open"))

        async def get_history_head(self, *args, **kwargs):
            self.reads += 1
            return self.head

        async def get(self, *args, **kwargs):
            return SimpleNamespace(status=ExecutionStatus.SUCCEEDED, event_seq=durable_count)

    class Events:
        reads = 0

        async def list(self, execution_id, *, after_event_seq, limit, **kwargs):
            self.reads += 1
            return Page(tuple(ExecutionEvent(execution_id, value, "COMMITTED", {})
                              for value in range(after_event_seq + 1, min(durable_count, after_event_seq + limit) + 1)), None)

    executions, events = Executions(), Events()
    shared = SharedStreamPublisher(backend, "scope", "tenant")
    service = DefaultEventService(executions, events, None, lambda *args, **kwargs: None,
                                  LiveExecutionEventBroker(shared=shared))
    return service, executions, events


def _principal():
    from linktools.ai.core import Principal
    return Principal(principal_id="reader", tenant_id="tenant")


@pytest.mark.asyncio
async def test_commit_marker_cursor_only_advances_after_entire_durable_range() -> None:
    from linktools.ai.core import canonical_json_bytes
    from linktools.ai.runtime._shared_stream import stream_key, shared_stream_positions
    backend = InMemoryStreamBackend(max_bytes=10000)
    writer = await backend.open_stream(stream_key("scope", "tenant", "execution", 1, "claim"), b"owner")
    await writer.append(1, canonical_json_bytes({"kind": "event", "durable": 3}))
    service, _, _ = _event_service(backend)
    first = service._stream_shared("execution", principal=_principal(), after_event_seq=0)
    item = await anext(first)
    assert item.durable_seq == 1 and item.position[1] == 0
    await first.aclose()
    with shared_stream_positions({"execution": item.position}):
        resumed = service._stream_shared("execution", principal=_principal(), after_event_seq=1)
        second = await anext(resumed)
        third = await anext(resumed)
        assert (second.durable_seq, third.durable_seq) == (2, 3)
        assert second.position[1] == 0 and third.position[1] == 1
        await resumed.aclose()


@pytest.mark.asyncio
async def test_durable_only_cursor_does_not_replay_already_committed_preview() -> None:
    from linktools.ai.core import canonical_json_bytes
    from linktools.ai.runtime._shared_stream import stream_key
    backend = InMemoryStreamBackend(max_bytes=10000)
    writer = await backend.open_stream(stream_key("scope", "tenant", "execution", 1, "claim"), b"owner")
    await writer.append(1, canonical_json_bytes({"kind": "delta", "type": "ASSISTANT_TEXT_DELTA", "text": "old"}))
    service, _, _ = _event_service(backend)
    values = await _collect(service._stream_shared("execution", principal=_principal(), after_event_seq=3))
    assert [item.event_type for item in values] == ["STREAM_RESET"]


@pytest.mark.asyncio
async def test_missing_stream_replays_durable_tail_after_bounded_open_grace() -> None:
    backend = InMemoryStreamBackend(max_bytes=0)
    service, _, events = _event_service(backend)
    values = await asyncio.wait_for(_collect(service._stream_shared(
        "execution", principal=_principal(), after_event_seq=0,
    )), 2)
    assert [item.durable_seq for item in values] == [None, 1, 2, 3]
    assert values[0].event_type == "STREAM_RESET" and events.reads > 0


@pytest.mark.asyncio
async def test_buffered_batch_cannot_postpone_terminal_authority_check() -> None:
    from types import SimpleNamespace
    from linktools.ai.core import canonical_json_bytes
    from linktools.ai.runtime._shared_stream import stream_key
    backend = InMemoryStreamBackend(max_bytes=10000)
    writer = await backend.open_stream(stream_key("scope", "tenant", "execution", 1, "claim"), b"owner")
    for sequence in (1, 2):
        await writer.append(sequence, canonical_json_bytes({"kind": "delta", "type": "ASSISTANT_TEXT_DELTA", "text": str(sequence)}))
    service, executions, _ = _event_service(backend, durable_count=0)
    stream = service._stream_shared("execution", principal=_principal(), after_event_seq=0)
    assert (await anext(stream)).payload["text"] == "1"
    executions.head.state = SimpleNamespace(value="sealed")
    await asyncio.sleep(1.01)
    assert (await anext(stream)).event_type == "STREAM_RESET"
    assert executions.reads == 2
    await stream.aclose()


@pytest.mark.asyncio
async def test_unknown_open_not_retried_and_completed_owners_are_released() -> None:
    from linktools.ai.runtime._shared_stream import SharedStreamPublisher

    class BrokenBackend:
        opens = 0

        async def open_stream(self, key, producer):
            self.opens += 1
            raise OSError("unknown open outcome")

    backend = BrokenBackend()
    publisher = SharedStreamPublisher(backend, "scope", "tenant")
    publisher.admit("execution", 1, "claim")
    first = publisher.producers["execution"].task
    await first
    await asyncio.sleep(0)
    publisher.admit("execution", 1, "claim")
    assert backend.opens == 1
    publisher.complete("execution")
    await publisher.close()
    assert not publisher.producers and not publisher.owned and not publisher.generations


@pytest.mark.asyncio
async def test_default_watch_closes_across_next_tasks() -> None:
    from types import SimpleNamespace
    from linktools.ai.core import ExecutionLineageKind
    from linktools.ai.runtime._agent import Execution
    from linktools.ai.runtime.service_api import ExecutionStreamEvent, ExecutionTreeEvent

    async def source():
        for sequence in (1, 2):
            yield ExecutionTreeEvent("execution", None, ExecutionLineageKind.RUN, None,
                                     "execution", None, 0, ExecutionStreamEvent(
                                         "execution", sequence, "MODEL_REQUEST_STARTED", {}))

    handle = Execution(SimpleNamespace(namespace="scope"), "execution", _principal(), None)
    stream = handle._watch_with_cursor(source(), None, False, None)
    assert (await asyncio.create_task(anext(stream))).event.durable_seq == 1
    assert (await asyncio.create_task(anext(stream))).event.durable_seq == 2
    await asyncio.create_task(stream.aclose())


@pytest.mark.asyncio
async def test_adjacent_preview_chunks_coalesce_and_handoff_releases_owner() -> None:
    from linktools.ai.runtime._shared_stream import SharedStreamPublisher, decode_stream_record, stream_key
    backend = InMemoryStreamBackend(max_bytes=10000)
    publisher = SharedStreamPublisher(backend, "scope", "tenant")
    publisher.admit("execution", 1, "claim")
    for _ in range(257):
        publisher.delta("execution", "ASSISTANT_TEXT_DELTA", "x", False)
    publisher.complete("execution")
    task = publisher.producers["execution"].task
    await task
    await asyncio.sleep(0)
    batch = await backend.read_stream(stream_key("scope", "tenant", "execution", 1, "claim"), 0,
                               limit=1000, wait_seconds=0)
    records = [decode_stream_record(payload) for _, payload in batch.records]
    assert all(record["kind"] != "gap" for record in records)
    assert "".join(record["text"] for record in records if record["kind"] == "delta") == "x" * 257
    publisher.admit("revoked", 1, "other")
    task = publisher.producers["revoked"].task
    publisher.handoff("revoked")
    await task
    await asyncio.sleep(0)
    assert not publisher.generations and not publisher.producers and not publisher.owned
    await publisher.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("record", [
    {"kind": "delta", "type": "EXECUTION_SUCCEEDED", "text": "forged"},
    {"kind": "delta", "type": "ASSISTANT_TEXT_DELTA", "text": "bad", "truncated": "false"},
    {"kind": "event", "durable": True},
])
async def test_corrupt_transport_frame_resets_to_authoritative_events(record) -> None:
    from linktools.ai.core import canonical_json_bytes
    from linktools.ai.runtime._shared_stream import stream_key
    backend = InMemoryStreamBackend(max_bytes=10000)
    writer = await backend.open_stream(stream_key("scope", "tenant", "execution", 1, "claim"), b"owner")
    await writer.append(1, canonical_json_bytes(record))
    service, _, _ = _event_service(backend)
    values = await _collect(service._stream_shared("execution", principal=_principal(), after_event_seq=0))
    assert [item.durable_seq for item in values] == [None, 1, 2, 3]
    assert values[0].event_type == "STREAM_RESET"


@pytest.mark.asyncio
async def test_missing_marker_tail_stops_later_preview_in_same_batch() -> None:
    from linktools.ai.core import canonical_json_bytes
    from linktools.ai.runtime._shared_stream import stream_key
    backend = InMemoryStreamBackend(max_bytes=10000)
    writer = await backend.open_stream(stream_key("scope", "tenant", "execution", 1, "claim"), b"owner")
    await writer.append(1, canonical_json_bytes({"kind": "event", "durable": 4}))
    await writer.append(2, canonical_json_bytes({"kind": "delta", "type": "ASSISTANT_TEXT_DELTA", "text": "stale"}))
    service, _, _ = _event_service(backend)
    values = await _collect(service._stream_shared("execution", principal=_principal(), after_event_seq=0))
    assert [item.durable_seq for item in values] == [1, 2, 3, None]
    assert values[-1].event_type == "STREAM_RESET"


@pytest.mark.asyncio
async def test_local_semantic_overflow_seals_shared_preview_with_gap() -> None:
    from linktools.ai.runtime._event import LiveExecutionEventBroker
    from linktools.ai.runtime._shared_stream import SharedStreamPublisher, decode_stream_record, stream_key
    backend = InMemoryStreamBackend(max_bytes=10000)
    publisher = SharedStreamPublisher(backend, "scope", "tenant")
    broker = LiveExecutionEventBroker(max_bytes=1, shared=publisher)
    broker.register_local_producer("execution", 0)
    publisher.admit("execution", 1, "claim")
    task = publisher.producers["execution"].task
    broker.publish_event("execution", "MODEL_REQUEST_STARTED", {"large": "payload"}, durable_seq=None)
    publisher.delta("execution", "ASSISTANT_TEXT_DELTA", "must not continue", False)
    await task
    await publisher.close()
    batch = await backend.read_stream(stream_key("scope", "tenant", "execution", 1, "claim"), 0,
                               limit=100, wait_seconds=0)
    assert [decode_stream_record(payload)["kind"] for _, payload in batch.records] == ["gap"]
    assert batch.sealed


@pytest.mark.asyncio
async def test_uncooperative_open_remains_owned_until_settled(monkeypatch) -> None:
    import linktools.ai.runtime._shared_stream as module
    from linktools.ai.errors import AIError, ErrorCode
    monkeypatch.setattr(module, "_IO_TIMEOUT", .015)

    class Backend:
        def __init__(self):
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.closed = False

        async def open_stream(self, *args):
            self.entered.set()
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    pass
            raise OSError("late open outcome")

        async def close(self):
            self.closed = True

    backend = Backend()
    publisher = module.SharedStreamPublisher(backend, "scope", "tenant")
    publisher.admit("execution", 1, "claim")
    await backend.entered.wait()
    try:
        with pytest.raises(AIError) as failure:
            await publisher.close()
        assert failure.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        assert publisher.owned and not backend.closed
    finally:
        backend.release.set()
        for _ in range(5):
            await asyncio.sleep(0)
        await publisher.close()
    assert not publisher.owned and not publisher.producers and not backend.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [1, 50])
async def test_preview_chunk_count_does_not_scale_authority_reads(chunks) -> None:
    from linktools.ai.core import canonical_json_bytes
    from linktools.ai.runtime._shared_stream import stream_key
    backend = InMemoryStreamBackend(max_bytes=10000)
    writer = await backend.open_stream(stream_key("scope", "tenant", "execution", 1, "claim"), b"owner")
    for sequence in range(1, chunks + 1):
        await writer.append(sequence, canonical_json_bytes({
            "kind": "delta", "type": "ASSISTANT_TEXT_DELTA", "text": "x",
        }))
    service, executions, events = _event_service(backend, durable_count=0)
    stream = service._stream_shared("execution", principal=_principal(), after_event_seq=0)
    try:
        for _ in range(chunks):
            assert (await anext(stream)).payload["text"] == "x"
        assert executions.reads == 1
        assert events.reads == 0
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_waiting_resume_opens_each_durable_admission_once(tmp_path) -> None:
    import json
    from datetime import datetime, timezone
    from linktools.ai.core import ApprovalDecision, ApprovalStatus, ExecutionStatus
    from linktools.ai.workspace import Workspace, WorkspacePolicy, ToolPermissionPolicy
    from ._runtime_test_helpers import _wait_for_committed
    from .test_runtime_approval_composition import _ToolModels

    workspace = Workspace.load(tmp_path, policy=WorkspacePolicy(
        tool_permissions=ToolPermissionPolicy(default="ask"),
    ))
    application = workspace_features("application", workspace=workspace)
    application.agent("default", model="default", allow_tools=("read_file",))
    backend = _CountingStreamBackend()
    application.capability(ExecutionCapability(coordination=backend), scope="runtime")
    path = tmp_path / "resume.db"
    state = RuntimeStorage.sqlite(path)
    async with Runtime.open("resume-preview", models=_ToolModels(), storage=state,
                            features=application) as owner:
        execution = await owner.agents.get().start("read a file")
        tenant = owner.tenant_id
        await _wait_for_committed(
            lambda: owner._state.execution.executions.get(execution.execution_id, tenant_id=tenant),
            lambda record: record is not None and record.status is ExecutionStatus.WAITING_DEFERRED,
        )
        worker = owner._execution_service.runtime_backend()._tasks.get(execution.execution_id)
        if worker is not None:
            await asyncio.wait_for(asyncio.shield(worker), 10)
        approvals = await owner._state.recovery.approvals.list_pending(execution.execution_id, tenant_id=tenant)
        await owner._state.recovery.approvals.decide(
            approvals[0].approval_id, tenant_id=tenant, expected_status=ApprovalStatus.PENDING,
            idempotency_key_digest="a" * 64, decision=ApprovalDecision.DENY,
            principal_id=owner.default_principal.principal_id, decision_digest="b" * 64,
            decided_at=datetime.now(timezone.utc),
        )
    assert len(backend.open_keys) == 1
    async with Runtime.open("resume-preview", models=_ToolModels(),
                            storage=RuntimeStorage.sqlite(path),
                            features=application) as resumed:
        result = await resumed.executions.wait(execution.execution_id,
                                               principal=resumed.default_principal, timeout_seconds=10)
        assert result.result.status is ExecutionStatus.SUCCEEDED
        from linktools.ai.runtime._watch_cursor import encode_execution_watch_cursor
        old_cursor = encode_execution_watch_cursor(
            "resume-preview", tenant, execution.execution_id, include_content=True,
            event_seqs={}, preview_positions={execution.execution_id: (backend.open_keys[0].hex(), 0)},
        )
        remote = await resumed.executions.get(execution.execution_id)
        replay = await asyncio.wait_for(_collect(remote.watch(include_content=True, cursor=old_cursor)), 5)
        assert replay[0].event.event_type == "STREAM_RESET"
        assert replay[-1].event.event_type == "EXECUTION_SUCCEEDED"
    assert len(backend.open_keys) == len(set(backend.open_keys)) == 2
    initial, restored = map(json.loads, backend.open_keys)
    assert initial[:4] == restored[:4]
    assert restored[4] > initial[4] and initial[5] != restored[5]


async def _marker_batch_stream(records, *, durable_count=3):
    from linktools.ai.core import canonical_json_bytes
    from linktools.ai.runtime._shared_stream import stream_key

    backend = InMemoryStreamBackend(max_bytes=10000)
    writer = await backend.open_stream(stream_key("scope", "tenant", "execution", 1, "claim"), b"owner")
    for sequence, record in enumerate(records, 1):
        await writer.append(sequence, canonical_json_bytes(record))
    service, executions, events = _event_service(backend, durable_count=durable_count)
    stream = service._stream_shared("execution", principal=_principal(), after_event_seq=0)
    return stream, executions, events


@pytest.mark.asyncio
@pytest.mark.parametrize("targets,positions", [
    ([1, 2, 3], [1, 2, 3]),
    ([1, 3], [1, 1, 2]),
    ([1, 1, 3], [1, 2, 3]),
    ([2, 1, 3], [0, 1, 3]),
])
async def test_marker_batch_reuses_page_without_changing_cursor_positions(
    targets: list[int], positions: list[int],
) -> None:
    stream, executions, events = await _marker_batch_stream(
        [{"kind": "event", "durable": target} for target in targets],
    )
    try:
        values = [await anext(stream) for _ in range(3)]
        assert [value.durable_seq for value in values] == [1, 2, 3]
        assert [value.position[1] for value in values] == positions
        assert events.reads == 1 and executions.reads == 1
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_marker_batch_does_not_prefetch_across_preview_delta() -> None:
    stream, _, events = await _marker_batch_stream([
        {"kind": "event", "durable": 1},
        {"kind": "delta", "type": "ASSISTANT_TEXT_DELTA", "text": "preview"},
        {"kind": "event", "durable": 3},
    ])
    try:
        assert (await anext(stream)).durable_seq == 1
        assert events.reads == 1
        assert (await anext(stream)).payload["text"] == "preview"
        assert (await anext(stream)).durable_seq == 2
        assert (await anext(stream)).durable_seq == 3
        assert events.reads == 2
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_marker_batch_corrupt_tail_preserves_valid_prefix() -> None:
    stream, _, events = await _marker_batch_stream([
        {"kind": "event", "durable": 1},
        {"kind": "event", "durable": True},
        {"kind": "event", "durable": 3},
    ])
    try:
        assert (await anext(stream)).durable_seq == 1
        assert events.reads == 1
        assert (await anext(stream)).event_type == "STREAM_RESET"
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_marker_batch_prefetched_tail_respects_authority_deadline() -> None:
    from types import SimpleNamespace

    stream, executions, events = await _marker_batch_stream([
        {"kind": "event", "durable": target} for target in (1, 2, 3)
    ])
    try:
        assert (await anext(stream)).durable_seq == 1
        assert events.reads == 1
        executions.head = SimpleNamespace(
            producer_generation=1, producer_claim_id="claim", state=SimpleNamespace(value="sealed"),
        )
        await asyncio.sleep(1.05)
        assert (await anext(stream)).event_type == "STREAM_RESET"
        assert executions.reads == 2
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_marker_batch_prefetch_remains_bounded_to_durable_page_size() -> None:
    stream, _, events = await _marker_batch_stream([
        {"kind": "event", "durable": target} for target in (1, 201, 401)
    ], durable_count=401)
    original_list = events.list
    limits = []

    async def record_list(*args, **kwargs):
        limits.append(kwargs["limit"])
        return await original_list(*args, **kwargs)

    events.list = record_list
    try:
        values = [await anext(stream) for _ in range(401)]
        assert [value.durable_seq for value in values] == list(range(1, 402))
        assert [value.position[1] for value in values] == [1] * 200 + [2] * 200 + [3]
        assert events.reads == 3 and limits == [200, 200, 1]
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_marker_batch_terminal_does_not_deliver_prefetched_suffix() -> None:
    from linktools.ai.core import Page
    from linktools.ai.runtime.service_api import ExecutionEvent

    stream, _, events = await _marker_batch_stream([
        {"kind": "event", "durable": target} for target in (1, 2, 3)
    ])
    original_list = events.list

    async def terminal_list(*args, **kwargs):
        page = await original_list(*args, **kwargs)
        return Page(tuple(
            ExecutionEvent(value.execution_id, value.event_seq,
                           "EXECUTION_SUCCEEDED" if value.event_seq == 2 else value.event_type,
                           value.payload)
            for value in page.items
        ), page.next_cursor)

    events.list = terminal_list
    try:
        values = [value async for value in stream]
        assert [value.durable_seq for value in values] == [1, 2]
        assert [value.position[1] for value in values] == [1, 2]
        assert values[-1].event_type == "EXECUTION_SUCCEEDED"
        assert events.reads == 1
    finally:
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("retention", ("volatile", "transient"))
async def test_nondurable_runtime_keeps_combined_cache_without_opening_shared_stream(retention: str) -> None:
    from pydantic_ai.models.test import TestModel
    from linktools.ai.runtime import RuntimeStoragePlan
    from linktools.ai.storage import StorageRoute

    cache = _CombinedCache()
    storage = (
        RuntimeStorage.in_memory(cache=cache)
        if retention == "volatile"
        else RuntimeStorage.from_plan(
            RuntimeStoragePlan(execution=StorageRoute.transient(cache=cache)),
        )
    )
    group = RuntimeFeatures[None]("local-preview")
    group.capability(ExecutionCapability(coordination=cache), scope="runtime")
    group.agent("default", model="default", allow_tools=())
    async with Runtime.open(
        "local-preview", storage=storage, models=_Models(TestModel()), features=group,
    ) as runtime:
        execution = await runtime.agents.get().run("hello", timeout_seconds=5)
        assert execution.result is not None
        assert storage.plan.route(RuntimeDomain.EXECUTION).cache is cache
    assert cache.open_keys == []
    assert not cache.closed


@pytest.mark.asyncio
async def test_first_admission_attach_rejects_changed_producer_identity() -> None:
    from types import SimpleNamespace
    from linktools.ai.core import canonical_json_bytes
    from linktools.ai.runtime._shared_stream import stream_key

    backend = InMemoryStreamBackend(max_bytes=10000)
    writer = await backend.open_stream(stream_key("scope", "tenant", "execution", 7, "replacement"), b"owner")
    await writer.append(1, canonical_json_bytes({
        "kind": "delta", "type": "ASSISTANT_TEXT_DELTA", "text": "wrong admission",
    }))
    service, executions, _ = _event_service(backend)
    executions.head = SimpleNamespace(
        producer_generation=7, producer_claim_id="replacement", state=SimpleNamespace(value="open"),
    )
    stream = service._stream_shared(
        "execution", principal=_principal(), after_event_seq=1, initial_admission=(4, "first"),
    )
    try:
        items = [item async for item in stream]
        assert items[0].event_type == "STREAM_RESET"
        assert [item.durable_seq for item in items[1:]] == [2, 3]
        assert all(item.event_type != "ASSISTANT_TEXT_DELTA" for item in items)
    finally:
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("local_takeover", (False, True))
async def test_early_shared_watch_does_not_attach_to_later_attempt(local_takeover: bool) -> None:
    from types import SimpleNamespace
    from linktools.ai.core import ExecutionStatus

    backend = _CountingStreamBackend()
    service, executions, _ = _event_service(backend, durable_count=0)
    if local_takeover:
        service._live.is_local_producer = lambda execution_id: True
    reads = 0

    async def get(*args, **kwargs):
        nonlocal reads
        reads += 1
        return SimpleNamespace(
            status=ExecutionStatus.STARTED if reads == 1 else ExecutionStatus.SUCCEEDED,
            event_seq=0, agent_run_seq=2,
        )

    async def read(callback):
        return await callback(None)

    executions.get = get
    executions.state_store = SimpleNamespace(read=read)
    executions.head = SimpleNamespace(
        producer_generation=7, producer_claim_id="later", state=SimpleNamespace(value="open"),
    )
    stream = service._stream_durable(
        "execution", tenant_id="tenant", after_event_seq=0, principal=_principal(), initial_shared=True,
    )
    try:
        assert await asyncio.wait_for(_collect(stream), 5) == []
        assert not backend.read_entered.is_set()
    finally:
        await stream.aclose()
