#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Private terminal Trace payload preparation before public policy composition."""

import asyncio
from dataclasses import replace

import pytest

from linktools.ai.core import ExecutionEventType, ExecutionStatus, StopReason, UsageMetrics
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence._runtime_commands import StateCommands
from linktools.ai.persistence._transcript_staging import StagedTranscript
from linktools.ai.persistence.contracts import ExecutionTerminalCommit, ExecutionTerminalCommitResult, ResultRecord
from linktools.ai.storage import InMemoryCacheBackend, StoredPayload

from .test_observation_coalescing import _observe, _setup
from .test_projection_payload import _capture


async def _terminal_command(state, status=ExecutionStatus.FAILED):
    repository = state.execution.executions
    current = await repository.get("execution", tenant_id="tenant")
    error_code = (ErrorCode.EXECUTION_FAILED.value if status is ExecutionStatus.FAILED else
                  ErrorCode.EXECUTION_CANCELLED.value if status is ExecutionStatus.CANCELLED else None)
    output = StoredPayload.inline_json({"text": "done"}) if status is ExecutionStatus.SUCCEEDED else None
    reason = (StopReason.END_TURN if status is ExecutionStatus.SUCCEEDED else
              StopReason.CANCELLED if status is ExecutionStatus.CANCELLED else StopReason.ERROR)
    event = {ExecutionStatus.SUCCEEDED: ExecutionEventType.EXECUTION_SUCCEEDED,
             ExecutionStatus.FAILED: ExecutionEventType.EXECUTION_FAILED,
             ExecutionStatus.CANCELLED: ExecutionEventType.EXECUTION_CANCELLED}[status]
    commit = ExecutionTerminalCommit(
        current.revision, current.event_seq,
        replace(current, status=status, error_code=error_code),
        ResultRecord(output, reason, UsageMetrics(), current.updated_at), event,
        {"error_code": error_code, "safe_error_details": {}},
        producer_generation=1,
    )
    commands = StateCommands(
        repository, namespace="observation-batches", events=state.execution.events,
        execution_run_store=state.run_store.read_store(PersistenceDomain.TRACE), background_tasks=set(),
    )
    return commands, commit, current


async def _commit_terminal(state, plan, status=ExecutionStatus.FAILED):
    commands, commit, current = await _terminal_command(state, status)
    return await commands.commit_terminal_checkpoint(
        commit, expected_execution=current,
        execution_run=plan.projections[0].run if plan.projections else None, execution_projections=plan.projections,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED])
async def test_terminal_payload_prepare_reads_buffer_without_early_source_or_ack(tmp_path, status):
    backend = InMemoryCacheBackend(max_bytes=1024 * 1024)
    state = await _setup(tmp_path, "sqlite", cache=backend)
    try:
        store = state.run_store
        staging, captured = _capture(store._staging)
        staging.stage_transcript("run", StagedTranscript(captured.local_messages))
        owner = store.trace_persistence
        owner.mark_dirty("run")
        owner.configure("run", buffer=backend, terminal=True, batch_size=1, flush_interval=.001)
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1, deferred=True)
        buffer = owner._buffers["run"]
        queued = await backend.read_batch(buffer.id, limit=100)
        assert queued and isinstance(queued[0][2], bytes)
        await asyncio.sleep(.01)
        archive = store.read_store(PersistenceDomain.TRACE)
        assert await archive.get_agent_run(agent_run_id="run") is None
        assert buffer.task is None and not store._durability_flights
        staging._payloads.clear()
        staging._transcripts.clear()
        plan = await store.prepare_execution_terminal_seal(
            execution_id="execution", agent_run_ids=("run",), binding_digest="fixture",
        )
        assert await archive.get_agent_run(agent_run_id="run") is None
        assert await backend.read_batch(buffer.id, limit=100) == queued
        with pytest.raises(AIError) as unresolved:
            await asyncio.wait_for(owner.settle_buffers(), 2)
        assert unresolved.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        assert await backend.read_batch(buffer.id, limit=100) == queued
        committed = await _commit_terminal(state, plan, status)
        assert committed.execution.status is status
        assert await backend.read_batch(buffer.id, limit=100) == queued
        await store.finalize_execution_terminal_seal(plan)
        await owner.settle_buffers()
        assert await backend.read_batch(buffer.id, limit=100) == ()
        interactions = await archive.list_model_interactions(agent_run_id="run")
        assert len(interactions) == 1
        request, response, envelope, clients, partials = await archive.resolve_model_interaction(
            interactions[0], include_client_requests=True,
        )
        assert request[0].parts[0].content == "input" and response[0].parts[0].content == "output"
        assert envelope and clients[0].body == b"request-body" and partials[0]
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_terminal_payload_known_rollback_retries_same_ids_and_frozen_bytes(tmp_path, monkeypatch):
    backend = InMemoryCacheBackend(max_bytes=65536)
    state = await _setup(tmp_path, "sqlite", cache=backend)
    try:
        store = state.run_store
        owner = store.trace_persistence
        owner.configure("run", buffer=backend, terminal=True)
        await _observe(store, 1)
        buffer = owner._buffers["run"]
        queued = await backend.read_batch(buffer.id, limit=100)
        plan = await store.prepare_execution_terminal_seal(
            execution_id="execution", agent_run_ids=("run",), binding_digest="fixture",
        )
        repository = state.execution.executions
        original = repository.commit_terminal_in_transaction

        async def reject(*args, **kwargs):
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        monkeypatch.setattr(repository, "commit_terminal_in_transaction", reject)
        with pytest.raises(AIError):
            await _commit_terminal(state, plan)
        await store.discard_execution_terminal_seal(plan)
        assert await backend.read_batch(buffer.id, limit=100) == queued
        assert await store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id="run") == []
        with pytest.raises(AIError) as flush_error:
            await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert flush_error.value.code is ErrorCode.STORAGE_CONFLICT
        with pytest.raises(AIError) as close_error:
            await owner.settle_buffers()
        assert close_error.value.code is ErrorCode.STORAGE_CONFLICT
        assert await backend.read_batch(buffer.id, limit=100) == queued
        # A failed transaction is retried from the sealed buffer prefix, not live staging.
        store._staging._events["run"][0].metadata["unadmitted"] = "later"
        retry = await store.prepare_execution_terminal_seal(
            execution_id="execution", agent_run_ids=("run",), binding_digest="fixture",
        )
        assert retry.projections[0].events[0].metadata == {}
        monkeypatch.setattr(repository, "commit_terminal_in_transaction", original)
        await _commit_terminal(state, retry)
        await store.finalize_execution_terminal_seal(retry)
        await owner.settle_buffers()
        assert await backend.read_batch(buffer.id, limit=100) == ()
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("rejected_tail", [False, True])
async def test_terminal_payload_ack_failure_does_not_reopen_canonical_terminal(tmp_path, rejected_tail):
    class Buffer(InMemoryCacheBackend):
        unavailable = True
        rejecting = False

        async def append(self, *args, **kwargs):
            if self.rejecting:
                return False
            return await super().append(*args, **kwargs)

        async def ack(self, *args, **kwargs):
            if self.unavailable:
                raise RuntimeError("ack unavailable")
            return await super().ack(*args, **kwargs)

    backend = Buffer(max_bytes=65536)
    state = await _setup(tmp_path, "sqlite", cache=backend)
    try:
        store = state.run_store
        owner = store.trace_persistence
        owner.configure("run", buffer=backend, terminal=True)
        await _observe(store, 1)
        if rejected_tail:
            backend.rejecting = True
            with pytest.raises(AIError) as rejected:
                await _observe(store, 2)
            assert await store.settle_trace_buffer_failure(
                "run", execution_id="execution", producer_generation=1, error=rejected.value,
            )
        buffer = owner._buffers["run"]
        plan = await store.prepare_execution_terminal_seal(
            execution_id="execution", agent_run_ids=("run",), binding_digest="fixture",
        )
        result = await _commit_terminal(state, plan)
        await store.finalize_execution_terminal_seal(plan)
        assert result.execution.status is ExecutionStatus.FAILED
        with pytest.raises(RuntimeError, match="ack unavailable"):
            await owner.settle_buffers()
        source = await state.execution.executions.get("execution", tenant_id="tenant")
        assert source.status is ExecutionStatus.FAILED
        assert await backend.read_batch(buffer.id, limit=100)
        backend.unavailable = False
        await owner.settle_buffers()
        assert await backend.read_batch(buffer.id, limit=100) == ()
        assert len(await store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id="run")) == 1
    finally:
        backend.unavailable = False
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("rejected_tail", [False, True])
async def test_terminal_payload_unknown_source_keeps_same_plan_until_exact_readback(tmp_path, monkeypatch, rejected_tail):
    class Buffer(InMemoryCacheBackend):
        rejecting = False

        async def append(self, *args, **kwargs):
            if self.rejecting:
                return False
            return await super().append(*args, **kwargs)

    backend = Buffer(max_bytes=65536)
    state = await _setup(tmp_path, "sqlite", cache=backend)
    try:
        store = state.run_store
        owner = store.trace_persistence
        owner.configure("run", buffer=backend, terminal=True)
        await _observe(store, 1)
        if rejected_tail:
            backend.rejecting = True
            with pytest.raises(AIError) as rejected:
                await _observe(store, 2)
            assert await store.settle_trace_buffer_failure(
                "run", execution_id="execution", producer_generation=1, error=rejected.value,
            )
        buffer = owner._buffers["run"]
        queued = await backend.read_batch(buffer.id, limit=100)
        plan = await store.prepare_execution_terminal_seal(
            execution_id="execution", agent_run_ids=("run",), binding_digest="fixture",
        )
        commands, commit, current = await _terminal_command(state)
        group = state.execution.executions.state_store.storage_group
        mutate, get = group.mutate, state.execution.executions.get
        lost = False

        async def lose_reply(stores, callback):
            nonlocal lost
            result = await mutate(stores, callback)
            if isinstance(result, ExecutionTerminalCommitResult):
                lost = True
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return result

        async def unavailable(*args, **kwargs):
            if lost:
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
            return await get(*args, **kwargs)

        monkeypatch.setattr(group, "mutate", lose_reply)
        monkeypatch.setattr(state.execution.executions, "get", unavailable)
        with pytest.raises(AIError) as unknown:
            await commands.commit_terminal_checkpoint(
                commit, expected_execution=current,
                execution_run=plan.projections[0].run if plan.projections else None, execution_projections=plan.projections,
            )
        assert unknown.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        assert await backend.read_batch(buffer.id, limit=100) == queued
        with pytest.raises(AIError) as close_error:
            await asyncio.wait_for(owner.settle_buffers(), 2)
        assert close_error.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        monkeypatch.setattr(group, "mutate", mutate)
        monkeypatch.setattr(state.execution.executions, "get", get)
        confirmed = await commands.commit_terminal_checkpoint(
            commit, expected_execution=current,
            execution_run=plan.projections[0].run if plan.projections else None, execution_projections=plan.projections,
        )
        assert confirmed.execution.status is ExecutionStatus.FAILED
        await store.reconcile_execution_terminal_seal(plan)
        await owner.settle_buffers()
        assert await backend.read_batch(buffer.id, limit=100) == ()
        assert len(await store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id="run")) == 1
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_terminal_payload_freeze_blocks_flush_before_source_flight_is_installed(tmp_path):
    backend = InMemoryCacheBackend(max_bytes=65536)
    state = await _setup(tmp_path, "sqlite", cache=backend)
    try:
        store = state.run_store
        owner = store.trace_persistence
        owner.configure("run", buffer=backend, terminal=True)
        await _observe(store, 1)
        buffer = owner._buffers["run"]
        frozen = await owner.prepare_terminal_payload("run", execution_id="execution", producer_generation=1)
        assert frozen is not None and not store._durability_flights
        queued = await backend.read_batch(buffer.id, limit=100)
        with pytest.raises(AIError) as ordinary:
            await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert ordinary.value.code is ErrorCode.STORAGE_CONFLICT
        assert await store.read_store(PersistenceDomain.TRACE).get_agent_run(agent_run_id="run") is None
        assert await backend.read_batch(buffer.id, limit=100) == queued
        plan = await store.prepare_execution_terminal_seal(
            execution_id="execution", agent_run_ids=("run",), binding_digest="fixture",
        )
        await _commit_terminal(state, plan)
        await store.finalize_execution_terminal_seal(plan)
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("admitted", [False, True])
async def test_terminal_payload_rejection_joins_only_accepted_prefix_with_failed_control(tmp_path, admitted):
    class Buffer(InMemoryCacheBackend):
        rejecting = False

        async def append(self, *args, **kwargs):
            if self.rejecting:
                return False
            return await super().append(*args, **kwargs)

    backend = Buffer(max_bytes=65536)
    state = await _setup(tmp_path, "sqlite", cache=backend)
    try:
        store = state.run_store
        owner = store.trace_persistence
        owner.configure("run", buffer=backend, terminal=True)
        if admitted:
            await _observe(store, 1)
        backend.rejecting = True
        with pytest.raises(AIError) as rejected:
            await _observe(store, 2)
        buffer = owner._buffers["run"]
        queued = await backend.read_batch(buffer.id, limit=100)
        assert await store.settle_trace_buffer_failure(
            "run", execution_id="execution", producer_generation=1, error=rejected.value,
        )
        assert await store.read_store(PersistenceDomain.TRACE).get_agent_run(agent_run_id="run") is None
        assert await backend.read_batch(buffer.id, limit=100) == queued
        with pytest.raises(AIError) as unsettled:
            await owner.settle_buffers()
        assert unsettled.value is rejected.value
        plan = await store.prepare_execution_terminal_seal(
            execution_id="execution", agent_run_ids=(), binding_digest="fixture",
        )
        assert len(plan.projections) == int(admitted)
        await _commit_terminal(state, plan)
        await store.finalize_execution_terminal_seal(plan)
        await owner.settle_buffers()
        source = await state.execution.executions.get("execution", tenant_id="tenant")
        assert source.status is ExecutionStatus.FAILED
        assert await backend.read_batch(buffer.id, limit=100) == ()
        assert len(await store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id="run")) == int(admitted)
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("read_failure", ["unavailable", "short"])
async def test_terminal_rejection_requires_readback_of_every_accepted_payload(tmp_path, read_failure):
    class Buffer(InMemoryCacheBackend):
        rejecting = False
        failed_read = False

        async def append(self, *args, **kwargs):
            if self.rejecting:
                return False
            return await super().append(*args, **kwargs)

        async def read_batch(self, *args, **kwargs):
            if self.failed_read:
                if read_failure == "short":
                    return ()
                raise RuntimeError("buffer unavailable")
            return await super().read_batch(*args, **kwargs)

    backend = Buffer(max_bytes=65536)
    state = await _setup(tmp_path, "sqlite", cache=backend)
    try:
        store = state.run_store
        owner = store.trace_persistence
        owner.configure("run", buffer=backend, terminal=True)
        await _observe(store, 1)
        backend.rejecting = True
        with pytest.raises(AIError) as rejected:
            await _observe(store, 2)
        buffer = owner._buffers["run"]
        queued = await backend.read_batch(buffer.id, limit=100)
        backend.failed_read = True
        with pytest.raises((AIError, RuntimeError)):
            await store.settle_trace_buffer_failure(
                "run", execution_id="execution", producer_generation=1, error=rejected.value,
            )
        assert not store.trace_buffer_failure_settled("run", execution_id="execution", producer_generation=1)
        assert buffer.terminal_ids is None and buffer.error is rejected.value
        assert await store.read_store(PersistenceDomain.TRACE).get_agent_run(agent_run_id="run") is None
        backend.failed_read = False
        assert await backend.read_batch(buffer.id, limit=100) == queued
        assert await store.settle_trace_buffer_failure(
            "run", execution_id="execution", producer_generation=1, error=rejected.value,
        )
        plan = await store.prepare_execution_terminal_seal(
            execution_id="execution", agent_run_ids=("run",), binding_digest="fixture",
        )
        await _commit_terminal(state, plan)
        await store.finalize_execution_terminal_seal(plan)
        await owner.settle_buffers()
        assert await backend.read_batch(buffer.id, limit=100) == ()
    finally:
        await state.close()
