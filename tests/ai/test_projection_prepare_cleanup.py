#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Precommit history read failures leave projections retryable or revocable."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence._model_interaction import (
    StagedContextProjection,
    StagedModelInteraction,
)
from linktools.ai.persistence._step_contracts import StepEvent
from linktools.ai.storage import InMemoryCacheBackend

from .test_observation_coalescing import _setup

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
@pytest.mark.parametrize("buffered", (False, True))
@pytest.mark.parametrize("recovery", ("retry", "revoke"))
async def test_failed_initial_history_read_allows_projection_recovery(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch, buffered: bool, recovery: str,
) -> None:
    backend = InMemoryCacheBackend(max_bytes=65536) if buffered else None
    state = await _setup(tmp_path, "memory", cache=backend)
    store = state.run_store
    archive = store.read_store(PersistenceDomain.TRACE)
    original_head = archive.execution_history_head_record
    source_error = AIError(ErrorCode.STORAGE_UNAVAILABLE, "history read unavailable before commit")
    attempted = asyncio.Event()
    failed_flights = []
    revoked = False
    recovery_task = None

    async def fail_first_head(*args, **kwargs):
        if not attempted.is_set():
            failed_flights.append(store._durability_flights["run"])
            attempted.set()
            raise source_error
        return await original_head(*args, **kwargs)

    monkeypatch.setattr(archive, "execution_history_head_record", fail_first_head)
    try:
        body = b'{"request":true}'
        digest, _size = store.intern_payload("run", body)
        now = datetime.now(timezone.utc)
        store.stage_model_interaction(StagedModelInteraction(
            "run", 1, 1, "agent", None, {}, StagedContextProjection(()), digest, None,
            "FAILED", "MODEL_API_ERROR", 1, None, now, now,
        ))
        await store.append_event(StepEvent("run", "MODEL_REQUEST_FAILED", 1), execution_id="execution")
        if buffered:
            store.trace_persistence.configure("run", buffer=backend, batch_size=1, flush_interval=60)
            await store.flush_execution_projection(
                "run", execution_id="execution", producer_generation=1, deferred=True,
            )
            await asyncio.wait_for(attempted.wait(), 2)
            pending = store.trace_persistence._buffers["run"].task
            if pending is not None:
                await asyncio.wait_for(asyncio.shield(pending), 2)
            assert store.trace_persistence._buffers["run"].source_error is source_error
        else:
            with pytest.raises(AIError) as raised:
                await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
            assert raised.value is source_error

        assert await archive.list_events(agent_run_id="run") == []
        assert store.staged_payload("run", digest) == body
        assert "run" in store.trace_persistence.dirty
        flight = failed_flights[0]
        print({"buffered": buffered, "recovery": recovery, "source_events": 0,
               "retained_payloads": len(store._staging._payloads.get("run", {})),
               "completion_done": flight.completion.done()})

        if recovery == "revoke":
            repository = state.execution.executions
            current = await repository.get("execution", tenant_id="tenant")
            assert current is not None
            successor = await repository.compare_and_swap(
                "execution", tenant_id="tenant", expected_revision=current.revision,
                next_record=replace(current, revision=current.revision + 1),
            )
            await repository.state_store.mutate(
                lambda transaction: repository.admit_history_producer_in_transaction(transaction, successor)
            )
            revoked = True
            recovery_task = asyncio.create_task(store.discard_revoked_producer_staging(
                "run", execution_id="execution", producer_generation=1, settle_cleanup=True,
            ))
            completed, _pending = await asyncio.wait((recovery_task,), timeout=2)
            assert recovery_task in completed, "revoked projection cleanup did not finish"
            await recovery_task
            revoked = False
            assert await archive.list_events(agent_run_id="run") == []
            assert await archive.list_model_interactions(agent_run_id="run") == []
        else:
            recovery_task = asyncio.create_task(store.flush_execution_projection(
                "run", execution_id="execution", producer_generation=1,
            ))
            completed, _pending = await asyncio.wait((recovery_task,), timeout=2)
            assert recovery_task in completed, "same-projection retry did not finish"
            await recovery_task
            await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
            events = await archive.list_events(agent_run_id="run")
            interactions = await archive.list_model_interactions(agent_run_id="run")
            assert len(events) == len(interactions) == 1
            _request, _response, envelope, _clients, _partials = await archive.resolve_model_interaction(interactions[0])
            assert envelope == body
            await store.release_execution_staging("execution")

        await asyncio.wait_for(store.wait_projection_flight("run"), 2)
        assert flight.completion.done()
        assert not store._durability_flights
        assert not store._staging._payloads
        assert "run" not in store.trace_persistence.dirty
        if backend is not None:
            assert backend._pending_count == 0
        await asyncio.wait_for(state.close(), 2)
        assert all(task.done() for task in store._background_tasks)
    finally:
        monkeypatch.setattr(archive, "execution_history_head_record", original_head)
        if recovery_task is not None and not recovery_task.done():
            recovery_task.cancel()
        # The injected read fault precedes source submission, so an orphan
        # fixture flight can be abandoned before joining its owned task.
        if "run" in store._durability_flights:
            await store.abandon_execution_projection(store._durability_flights["run"])
        if recovery_task is not None:
            await asyncio.gather(recovery_task, return_exceptions=True)
        if revoked:
            await store.discard_revoked_producer_staging(
                "run", execution_id="execution", producer_generation=1, settle_cleanup=True,
            )
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("read_fails", (False, True))
async def test_caller_cancellation_settles_initial_history_read_before_returning(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch, read_fails: bool,
) -> None:
    state = await _setup(tmp_path, "memory", cache=None)
    store = state.run_store
    archive = store.read_store(PersistenceDomain.TRACE)
    original_head = archive.execution_history_head_record
    entered, release = asyncio.Event(), asyncio.Event()
    source_error = AIError(ErrorCode.STORAGE_UNAVAILABLE, "initial history read failed")
    task = None

    async def held_head(*args, **kwargs):
        if not entered.is_set():
            entered.set()
            await release.wait()
            if read_fails:
                raise source_error
        return await original_head(*args, **kwargs)

    monkeypatch.setattr(archive, "execution_history_head_record", held_head)
    try:
        await store.append_event(StepEvent("run", "MODEL_REQUEST_FAILED", 1), execution_id="execution")
        task = asyncio.create_task(store.flush_execution_projection(
            "run", execution_id="execution", producer_generation=1,
        ))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(AIError if read_fails else asyncio.CancelledError) as raised:
            await task
        if read_fails:
            assert raised.value is source_error
            assert await archive.list_events(agent_run_id="run") == []
            await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert len(await archive.list_events(agent_run_id="run")) == 1
        await store.wait_projection_flight("run")
        assert not store._durability_flights
        await store.release_execution_staging("execution")
        await state.close()
    finally:
        release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        monkeypatch.setattr(archive, "execution_history_head_record", original_head)
        await state.close()
