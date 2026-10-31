#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real Trace payload admission, fixed-prefix source publication and cleanup."""

import asyncio

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence._projection_payload import decode_projection_payload
from linktools.ai.persistence._step_contracts import StepEvent
from linktools.ai.persistence._transcript_staging import StagedTranscript
from linktools.ai.storage import InMemoryCacheBackend

from .test_observation_coalescing import _observe, _setup
from .test_projection_payload import _capture


async def _wait(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)
    await asyncio.wait_for(wait(), 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["scheduled", "explicit", "terminal"])
async def test_trace_buffer_contains_self_sufficient_payload_and_no_early_materialization(tmp_path, monkeypatch, boundary):
    state = await _setup(tmp_path, "sqlite")
    try:
        store = state.run_store
        staging, captured = _capture(store._staging)
        staging.stage_transcript("run", StagedTranscript(captured.local_messages))
        owner = store.trace_persistence
        owner.dirty.add("run")
        owner.configure("run", buffer=owner._caches["run"], batch_size=128, flush_interval=60)
        archive = store.read_store(PersistenceDomain.TRACE)
        prepared = 0
        original = archive.prepare_interactions

        async def prepare(*args, **kwargs):
            nonlocal prepared
            prepared += 1
            return await original(*args, **kwargs)

        monkeypatch.setattr(archive, "prepare_interactions", prepare)
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1, deferred=True)
        buffer = owner._buffers["run"]
        queued = await buffer.backend.read_batch(buffer.id, limit=10)
        assert len(queued) == 1 and isinstance(queued[0][2], bytes)
        decoded = decode_projection_payload(queued[0][2])
        assert decoded.payloads and decoded.interactions[0].client_requests[0].body == b"request-body"
        assert prepared == 0
        assert await archive.list_events(agent_run_id="run") == []
        assert not store._durability_flights
        # Keep only live-owner identity/frontiers: source materialization must read Redis-style bytes.
        staging._payloads.clear()
        staging._transcripts.clear()
        if boundary == "scheduled":
            await owner._automatic_flush(buffer)
        elif boundary == "explicit":
            await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        else:
            plan = await store.prepare_execution_terminal_seal(
                execution_id="execution", agent_run_ids=("run",), binding_digest="fixture",
            )
            await store.discard_execution_terminal_seal(plan)
        assert buffer.error is None and prepared >= 1
        observed = await archive.list_model_interactions(agent_run_id="run")
        request, response, envelope, clients, partials = await archive.resolve_model_interaction(
            observed[0], include_client_requests=True,
        )
        assert request[0].parts[0].content == "input"
        assert response[0].parts[0].content == "output"
        assert envelope == b'{"request":true}'
        assert clients[0].response.body == b"response-body"
        assert partials[0][0].parts[0].content == "output"
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_trace_publish_uses_admitted_prefix_not_waiting_tail_or_mutable_mirror(tmp_path, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()

    class Buffer(InMemoryCacheBackend):
        appends = 0

        async def append(self, *args, **kwargs):
            self.appends += 1
            if self.appends == 2:
                entered.set()
                await release.wait()
            return await super().append(*args, **kwargs)

    backend = Buffer(max_bytes=65536)
    state = await _setup(tmp_path, "sqlite", cache=backend)
    producer = None
    try:
        store = state.run_store
        owner = store.trace_persistence
        owner.configure("run", buffer=backend, batch_size=128, flush_interval=60)
        first = StepEvent("run", "TOOL_CALL_STARTED", 1, metadata={"value": "admitted"})
        await store.append_event(first)
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1, deferred=True)
        buffer = owner._buffers["run"]
        selected, publish = asyncio.Event(), asyncio.Event()
        original = owner._publish

        async def held(capture, **kwargs):
            selected.set()
            await publish.wait()
            return await original(capture, **kwargs)

        monkeypatch.setattr(owner, "_publish", held)
        consumer = asyncio.create_task(owner._automatic_flush(buffer))
        await selected.wait()
        first.metadata["value"] = "mutated after admission"
        producer = asyncio.create_task(_observe(store, 2))
        await entered.wait()
        publish.set()
        await consumer
        archive = store.read_store(PersistenceDomain.TRACE)
        observed = await archive.list_events(agent_run_id="run")
        assert len(observed) == 1 and observed[0].metadata == {"value": "admitted"}
        assert not producer.done()
        release.set()
        await producer
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert len(await archive.list_events(agent_run_id="run")) == 2
    finally:
        release.set()
        if producer is not None:
            await asyncio.gather(producer, return_exceptions=True)
        await state.close()


@pytest.mark.asyncio
async def test_trace_capacity_rejection_is_sticky_and_never_falls_back_to_source(tmp_path):
    backend = InMemoryCacheBackend(max_bytes=0)
    state = await _setup(tmp_path, "memory", cache=backend)
    with pytest.raises(AIError) as caught:
        await _observe(state.run_store, 1)
    assert caught.value.code is ErrorCode.STORAGE_UNAVAILABLE
    assert await state.run_store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id="run") == []
    assert backend._pending_count == 0
    with pytest.raises(AIError) as close:
        await state.close()
    assert close.value is caught.value


@pytest.mark.asyncio
@pytest.mark.parametrize("readback", [True, False])
async def test_trace_unknown_append_retains_identity_and_requires_readback(tmp_path, readback):
    class Buffer(InMemoryCacheBackend):
        appends = 0

        async def append(self, *args, **kwargs):
            self.appends += 1
            await super().append(*args, **kwargs)
            raise RuntimeError("append reply lost")

        async def read_batch(self, *args, **kwargs):
            if not readback:
                raise RuntimeError("readback unavailable")
            return await super().read_batch(*args, **kwargs)

    backend = Buffer(max_bytes=65536)
    state = await _setup(tmp_path, "memory", cache=backend)
    store = state.run_store
    if readback:
        await _observe(store, 1)
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert len(await store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id="run")) == 1
        await state.close()
    else:
        with pytest.raises(RuntimeError, match="append reply lost"):
            await _observe(store, 1)
        candidate = store.trace_persistence._buffers["run"].uncertain
        assert candidate is not None and isinstance(candidate[2], bytes)
        with pytest.raises(RuntimeError, match="append reply lost"):
            await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert store.trace_persistence._buffers["run"].uncertain is candidate
        assert await store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id="run") == []
        with pytest.raises(RuntimeError, match="append reply lost"):
            await state.close()
    assert backend.appends == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["short", "bytes", "order"])
async def test_trace_missing_or_corrupt_prefix_never_uses_ram_fallback(tmp_path, damage):
    class Buffer(InMemoryCacheBackend):
        broken = False
        acknowledgements = 0

        async def read_batch(self, *args, **kwargs):
            queued = await super().read_batch(*args, **kwargs)
            if not self.broken:
                return queued
            if damage == "short":
                return queued[:-1]
            if damage == "order":
                return tuple(reversed(queued))
            return ((queued[0][0], queued[0][1], b"corrupt"), *queued[1:])

        async def ack(self, *args, **kwargs):
            self.acknowledgements += 1
            await super().ack(*args, **kwargs)

    backend = Buffer(max_bytes=65536)
    state = await _setup(tmp_path, "memory", cache=backend)
    await _observe(state.run_store, 1)
    await _observe(state.run_store, 2)
    backend.broken = True
    with pytest.raises(AIError) as caught:
        await state.run_store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert backend.acknowledgements == 0
    assert await state.run_store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id="run") == []
    with pytest.raises(AIError) as close:
        await state.close()
    assert close.value is caught.value


@pytest.mark.asyncio
async def test_trace_ack_failure_releases_source_and_retries_cleanup_without_read_or_replay(tmp_path):
    class Buffer(InMemoryCacheBackend):
        broken = True
        reads = 0

        async def read_batch(self, *args, **kwargs):
            self.reads += 1
            return await super().read_batch(*args, **kwargs)

        async def ack(self, *args, **kwargs):
            if self.broken:
                raise RuntimeError("ack unavailable")
            await super().ack(*args, **kwargs)

    backend = Buffer(max_bytes=65536)
    state = await _setup(tmp_path, "sqlite", cache=backend)
    store = state.run_store
    await _observe(store, 1)
    await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
    before = await state.execution.executions.get_history_head("execution", tenant_id="tenant")
    reads = backend.reads
    assert not store.trace_persistence._buffers and store.trace_persistence._cleanup
    await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
    assert await state.execution.executions.get_history_head("execution", tenant_id="tenant") == before
    assert backend.reads == reads
    with pytest.raises(RuntimeError, match="ack unavailable"):
        await state.close()
    backend.broken = False
    await state.close()
    assert backend._pending_count == 0 and backend.reads == reads


@pytest.mark.asyncio
async def test_source_confirmation_does_not_wait_for_ack_and_old_ack_keeps_new_tail(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    class Buffer(InMemoryCacheBackend):
        acknowledgements = 0

        async def ack(self, scope, ids):
            self.acknowledgements += 1
            if self.acknowledgements == 1:
                entered.set()
                await release.wait()
            await super().ack(scope, ids)

    backend = Buffer(max_bytes=65536)
    state = await _setup(tmp_path, "memory", cache=backend)
    try:
        store = state.run_store
        owner = store.trace_persistence
        owner.configure("run", buffer=backend, batch_size=1, flush_interval=60)
        await _observe(store, 1)
        await asyncio.wait_for(entered.wait(), 2)
        assert "run" not in owner.dirty
        first = owner._buffers["run"]
        first_ids = tuple(entry.id for entry in first.entries)
        await asyncio.wait_for(_observe(store, 2), 1)
        await _wait(lambda: "run" not in owner.dirty)
        assert [event.tool_call_id for event in await store.read_store(PersistenceDomain.TRACE).list_events(
            agent_run_id="run",
        )] == ["1", "2"]
        assert tuple(entry.id for entry in first.entries[:len(first_ids)]) == first_ids
        assert len(first.entries) == 2
        # Business/source boundary is already complete, independent of slow Redis cleanup.
        await asyncio.wait_for(store.flush_execution_projection(
            "run", execution_id="execution", producer_generation=1,
        ), 1)
        assert not release.is_set()
        release.set()
        await state.close()
        assert backend._pending_count == 0
    finally:
        release.set()
        await state.close()


@pytest.mark.asyncio
async def test_cancelled_admission_is_owned_until_exact_payload_outcome(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    class Buffer(InMemoryCacheBackend):
        async def append(self, *args, **kwargs):
            accepted = await super().append(*args, **kwargs)
            entered.set()
            await release.wait()
            return accepted

    backend = Buffer(max_bytes=65536)
    state = await _setup(tmp_path, "memory", cache=backend)
    producer = asyncio.create_task(_observe(state.run_store, 1))
    try:
        await entered.wait()
        producer.cancel()
        await asyncio.sleep(0)
        producer.cancel()
        await asyncio.sleep(0)
        assert not producer.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await producer
        owner = state.run_store.trace_persistence
        assert len(owner._buffers["run"].pending) == 1
        await state.run_store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert len(await state.run_store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id="run")) == 1
    finally:
        release.set()
        await asyncio.gather(producer, return_exceptions=True)
        await state.close()


@pytest.mark.asyncio
async def test_later_snapshot_cannot_replace_an_already_admitted_event(tmp_path):
    state = await _setup(tmp_path, "memory")
    store = state.run_store
    owner = store.trace_persistence
    owner.configure("run", buffer=owner._caches["run"], batch_size=128, flush_interval=60)
    event = StepEvent("run", "TOOL_CALL_STARTED", 1, metadata={"value": "admitted"})
    await store.append_event(event)
    await store.flush_execution_projection("run", execution_id="execution", producer_generation=1, deferred=True)
    event.metadata["value"] = "changed before next capture"
    await _observe(store, 2)
    with pytest.raises(AIError) as caught:
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert await store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id="run") == []
    buffer = owner._buffers["run"]
    assert len(buffer.pending) == 2
    assert len(await buffer.backend.read_batch(buffer.id, limit=2)) == 2
    with pytest.raises(AIError):
        await state.close()


@pytest.mark.asyncio
async def test_concurrent_boundaries_do_not_retire_a_replacement_run_buffer(tmp_path, monkeypatch):
    state = await _setup(tmp_path, "memory")
    store = state.run_store
    owner = store.trace_persistence
    owner.configure("run", buffer=owner._caches["run"], batch_size=128, flush_interval=60)
    entered, release, first_finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = owner._stop_locked
    calls = 0

    async def held(run_id, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await release.wait()
        elif calls == 2:
            # The first boundary has retired its scope before this one can select an owner.
            await first_finished.wait()
        await original(run_id, **kwargs)

    monkeypatch.setattr(owner, "_stop_locked", held)
    first = second = None
    try:
        await _observe(store, 1)
        old = owner._buffers["run"]
        first = asyncio.create_task(store.flush_execution_projection("run", execution_id="execution", producer_generation=1))
        await entered.wait()
        second = asyncio.create_task(store.flush_execution_projection("run", execution_id="execution", producer_generation=1))
        await asyncio.sleep(0)
        release.set()
        await first
        await _observe(store, 2)
        replacement = owner._buffers["run"]
        assert replacement is not old
        first_finished.set()
        await second
        assert [event.tool_call_id for event in await store.read_store(PersistenceDomain.TRACE).list_events(
            agent_run_id="run",
        )] == ["1", "2"]
        assert not owner._buffers
    finally:
        release.set()
        first_finished.set()
        await asyncio.gather(*(task for task in (first, second) if task is not None), return_exceptions=True)
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("admitted", [False, True])
async def test_known_rejected_tail_can_seal_only_confirmed_source_prefix(tmp_path, admitted):
    class Buffer(InMemoryCacheBackend):
        rejecting = False

        async def append(self, *args, **kwargs):
            if self.rejecting:
                return False
            return await super().append(*args, **kwargs)

    backend = Buffer(max_bytes=65536)
    state = await _setup(tmp_path, "sqlite", cache=backend)
    source_group = state.execution.executions.state_store.storage_group
    try:
        store = state.run_store
        owner = store.trace_persistence
        owner.configure("run", buffer=backend, batch_size=128, flush_interval=60)
        if admitted:
            await _observe(store, 1)
        backend.rejecting = True
        with pytest.raises(AIError) as rejected:
            await _observe(store, 2)
        assert not await store.settle_trace_buffer_failure(
            "run", execution_id="execution", producer_generation=1,
            error=AIError(ErrorCode.STORAGE_UNAVAILABLE),
        )
        assert await store.settle_trace_buffer_failure(
            "run", execution_id="execution", producer_generation=1, error=rejected.value,
        )
        assert await store.settle_trace_buffer_failure(
            "run", execution_id="execution", producer_generation=1, error=rejected.value,
        )
        with pytest.raises(AIError) as flush_error:
            await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert flush_error.value is rejected.value
        archive = store.read_store(PersistenceDomain.TRACE)
        assert len(await archive.list_events(agent_run_id="run")) == int(admitted)
        plan = await store.prepare_execution_terminal_seal(
            execution_id="execution", agent_run_ids=("run",), binding_digest="fixture",
        )
        assert plan.seal_tokens and len(plan.projections) == int(admitted)
        if admitted:
            assert plan.projections[0].events == ()
            assert plan.projections[0].target_event_offset == 1
        await store.discard_execution_terminal_seal(plan)
        assert not store._terminal_seals and not store._durability_flights
        # The rejected mutable tail remains local until canonical FAILED commits.
        assert "run" in owner.dirty
        assert len(await archive.list_events(agent_run_id="run")) == int(admitted)
    finally:
        try:
            with pytest.raises(AIError, match="Trace payload buffer rejected admission"):
                await state.close()
        finally:
            # Keep unresolved payload ownership; release only the fixture's source.
            await source_group.close()
            await source_group.context.close()
            assert source_group.context.closed


@pytest.mark.asyncio
async def test_unknown_append_is_not_a_definitely_rejected_terminal_tail(tmp_path):
    class Buffer(InMemoryCacheBackend):
        async def append(self, *args, **kwargs):
            await super().append(*args, **kwargs)
            raise RuntimeError("reply lost")

        async def read_batch(self, *args, **kwargs):
            raise RuntimeError("buffer unavailable")

    backend = Buffer(max_bytes=65536)
    state = await _setup(tmp_path, "sqlite", cache=backend)
    source_group = state.execution.executions.state_store.storage_group
    try:
        store = state.run_store
        owner = store.trace_persistence
        owner.configure("run", buffer=backend, batch_size=128, flush_interval=60)
        with pytest.raises(RuntimeError) as unknown:
            await _observe(store, 1)
        buffer = owner._buffers["run"]
        candidate = buffer.uncertain
        assert candidate is not None
        assert not await store.settle_trace_buffer_failure(
            "run", execution_id="execution", producer_generation=1, error=unknown.value,
        )
        assert buffer.uncertain is candidate and not owner.failed_admission_settled("run")
        assert await store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id="run") == []
        with pytest.raises(RuntimeError, match="reply lost"):
            await state.close()
    finally:
        await source_group.close()
        await source_group.context.close()
        assert source_group.context.closed


@pytest.mark.asyncio
async def test_rejected_trace_tail_cannot_reenter_through_checkpoint_relocation(tmp_path):
    class Buffer(InMemoryCacheBackend):
        async def append(self, *args, **kwargs):
            return False

    backend = Buffer(max_bytes=65536)
    state = await _setup(tmp_path, "sqlite", cache=backend)
    source_group = state.execution.executions.state_store.storage_group
    try:
        store = state.run_store
        staging, captured = _capture(store._staging)
        staging.stage_transcript("run", StagedTranscript(captured.local_messages))
        owner = store.trace_persistence
        owner.mark_dirty("run")
        owner.configure("run", buffer=backend, batch_size=128, flush_interval=60)
        with pytest.raises(AIError) as rejected:
            await store.flush_execution_projection("run", execution_id="execution", producer_generation=1, deferred=True)
        assert await store.settle_trace_buffer_failure(
            "run", execution_id="execution", producer_generation=1, error=rejected.value,
        )
        assert staging.latest_checkpoint_local("run", include_interrupted=True) is not None
        assert await store.get_agent_run(agent_run_id="run") is None
        assert await store.latest_checkpoint(agent_run_id="run", include_interrupted=True) is None
        checkpoint_owner = store.checkpoint_persistence
        assert await checkpoint_owner.terminal_staged_interactions(agent_run_id="run") == ()
        await checkpoint_owner.materialize_recovery_checkpoint(agent_run_id="run", require_complete=False)
        assert await store.read_store(PersistenceDomain.CHECKPOINT).get_agent_run(agent_run_id="run") is None
        plan = await store.prepare_execution_terminal_seal(
            execution_id="execution", agent_run_ids=(), binding_digest="fixture",
        )
        assert plan.projections == () and plan.seal_tokens
        await store.discard_execution_terminal_seal(plan)
    finally:
        try:
            with pytest.raises(AIError, match="Trace payload buffer rejected admission"):
                await state.close()
        finally:
            # Keep unresolved payload ownership; release only the fixture's source.
            await source_group.close()
            await source_group.context.close()
            assert source_group.context.closed


@pytest.mark.asyncio
async def test_rejected_tail_does_not_override_unknown_source_prefix(tmp_path, monkeypatch):
    class Buffer(InMemoryCacheBackend):
        rejecting = False

        async def append(self, *args, **kwargs):
            if self.rejecting:
                return False
            return await super().append(*args, **kwargs)

    backend = Buffer(max_bytes=65536)
    state = await _setup(tmp_path, "sqlite", cache=backend)
    source_group = state.execution.executions.state_store.storage_group
    try:
        store = state.run_store
        owner = store.trace_persistence
        owner.configure("run", buffer=backend, batch_size=128, flush_interval=60)
        await _observe(store, 1)
        backend.rejecting = True
        with pytest.raises(AIError) as rejected:
            await _observe(store, 2)
        archive = store.read_store(PersistenceDomain.TRACE)
        original_head = archive.execution_history_head_record
        attempted = False

        async def write_failed(*args, **kwargs):
            nonlocal attempted
            attempted = True
            raise RuntimeError("write outcome unavailable")

        async def readback_failed(*args, **kwargs):
            if attempted:
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
            return await original_head(*args, **kwargs)

        monkeypatch.setattr(archive, "sync_prepared_projection", write_failed)
        monkeypatch.setattr(archive, "execution_history_head_record", readback_failed)
        with pytest.raises(AIError) as unknown:
            await store.settle_trace_buffer_failure(
                "run", execution_id="execution", producer_generation=1, error=rejected.value,
            )
        assert unknown.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        assert not store.trace_buffer_failure_settled("run", execution_id="execution", producer_generation=1)
        flight = store._durability_flights["run"]
        buffer = owner._buffers["run"]
        assert buffer.pending and buffer.source_error is unknown.value
        with pytest.raises(AIError) as close_error:
            await state.close()
        assert close_error.value is unknown.value and store._durability_flights["run"] is flight
    finally:
        await source_group.close()
        await source_group.context.close()
        assert source_group.context.closed
