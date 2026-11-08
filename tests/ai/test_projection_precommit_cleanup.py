#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Projection preparation releases flights without weakening submitted-write fencing."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from linktools.ai.agent import AgentBindingContract
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence.contracts import ExecutionRecord, StoredUserInput
from linktools.ai.persistence.runs import AgentRunRecord, StepEvent
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import StoredPayload

if TYPE_CHECKING:
    from typing import NoReturn

    from linktools.ai.persistence.contracts import ExecutionRunSealHead


async def _state(tmp_path: Path, backend: str) -> Persistence:
    storage = (PersistenceStorage.in_memory() if backend == "memory"
               else PersistenceStorage.sqlite(tmp_path / "state.db"))
    state = Persistence(storage)
    await state.initialize(namespace="projection-preparation", tenant_id="tenant")
    now = datetime.now(timezone.utc)
    binding = AgentBindingContract(
        agent_spec=AgentSpec("agent", tool_retries=3, output_retries=3),
        model_contract={"route_id": "default", "model_identity": "test:model"},
        selected=(), subagents=(), output_mode="text",
        output_schema={"type": "object", "properties": {"text": {"type": "string"}}},
    )
    await state.execution.executions.create_execution(ExecutionRecord(
        durable_trace=True, producer_generation=1, execution_id="execution", session_id=None,
        parent_execution_id=None, root_execution_id="execution", previous_execution_id=None,
        fork_base_execution_id=None, lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.STARTED, revision=0, event_seq=0, agent_run_seq=1,
        error_code=None, safe_error_details={}, created_at=now, updated_at=now,
        mode="run", planning=False, thinking=False, binding=binding,
        principal_id="principal", principal_kind="service",
        stored_user_input=StoredUserInput("text", StoredPayload.inline_text("input")),
    ))
    store = state.run_store
    store.bind_execution_producer("run", execution_id="execution", producer_generation=1)
    await store.register_agent_run(AgentRunRecord("run"), execution_id="execution")
    await store.append_event(StepEvent("run", "AGENT_RUN_STARTED", 0), execution_id="execution")
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("phase", ("prepare_checkpoints", "execution_history_head_record"))
async def test_projection_precommit_failure_releases_flight_without_losing_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, phase: str,
) -> None:
    state = await _state(tmp_path, backend)
    try:
        store = state.run_store
        archive = store.read_store(PersistenceDomain.TRACE)
        captured = await store.capture_execution_projection("run")
        assert captured is not None
        projection, flight = captured
        marker = AIError(ErrorCode.STORAGE_UNAVAILABLE, "precommit read failed")
        attempted = False

        async def fail(*args: object, **kwargs: object) -> "NoReturn":
            raise marker

        async def unexpected_write(*args: object, **kwargs: object) -> "NoReturn":
            nonlocal attempted
            attempted = True
            raise AssertionError("a preparation failure must not submit a write")

        with monkeypatch.context() as patch:
            patch.setattr(archive, phase, fail)
            patch.setattr(archive, "sync_prepared_projection", unexpected_write)
            with pytest.raises(AIError) as caught:
                await store.commit_captured_execution_projection(projection, flight, execution_id="execution")
        assert caught.value is marker
        assert not attempted
        assert await archive.list_events(agent_run_id="run") == []
        await asyncio.wait_for(store.wait_projection_flight("run"), 2)
        retried = await asyncio.wait_for(store.capture_execution_projection("run"), 2)
        assert retried is not None and retried[0].events == projection.events
        await store.commit_captured_execution_projection(*retried, execution_id="execution")
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert await archive.list_events(agent_run_id="run") == list(projection.events)
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("boundary", ("captured_commit", "flush"))
async def test_cancelled_precommit_read_settles_flight_before_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, boundary: str,
) -> None:
    state = await _state(tmp_path, backend)
    caller = None
    entered, release = asyncio.Event(), asyncio.Event()
    try:
        store = state.run_store
        archive = store.read_store(PersistenceDomain.TRACE)
        marker = AIError(ErrorCode.STORAGE_UNAVAILABLE, "precommit read failed")
        attempted = False

        async def held_head(*args: str, **kwargs: str) -> "NoReturn":
            entered.set()
            await release.wait()
            raise marker

        async def unexpected_write(*args: object, **kwargs: object) -> "NoReturn":
            nonlocal attempted
            attempted = True
            raise AssertionError("a failed or cancelled read must not submit a write")

        with monkeypatch.context() as patch:
            patch.setattr(archive, "execution_history_head_record", held_head)
            patch.setattr(archive, "sync_prepared_projection", unexpected_write)
            if boundary == "captured_commit":
                captured = await store.capture_execution_projection("run")
                assert captured is not None
                caller = asyncio.create_task(store.commit_captured_execution_projection(
                    *captured, execution_id="execution",
                ))
            else:
                caller = asyncio.create_task(store.flush_execution_projection(
                    "run", execution_id="execution", producer_generation=1,
                ))
            await asyncio.wait_for(entered.wait(), 2)
            caller.cancel("cancel before submission")
            if boundary == "captured_commit":
                with pytest.raises(asyncio.CancelledError) as cancelled:
                    await asyncio.wait_for(caller, 2)
                assert cancelled.value.args == ("cancel before submission",)
            else:
                await asyncio.sleep(0)
                assert not caller.done()
                release.set()
                with pytest.raises(AIError) as caught:
                    await asyncio.wait_for(caller, 2)
                assert caught.value is marker
        assert not attempted
        assert await archive.list_events(agent_run_id="run") == []
        await asyncio.wait_for(store.wait_projection_flight("run"), 2)
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        events = await archive.list_events(agent_run_id="run")
        assert len(events) == 1 and events[0].event_type == "AGENT_RUN_STARTED"
    finally:
        release.set()
        if caller is not None and not caller.done():
            caller.cancel()
            await asyncio.gather(caller, return_exceptions=True)
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("prepare_checkpoints", "execution_history_head_record"))
async def test_projection_precommit_cleanup_failure_keeps_existing_error_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    state = await _state(tmp_path, "memory")
    try:
        store = state.run_store
        archive = store.read_store(PersistenceDomain.TRACE)
        captured = await store.capture_execution_projection("run")
        assert captured is not None
        projection, flight = captured
        preparation_error = AIError(ErrorCode.STORAGE_UNAVAILABLE)
        cleanup_error = AIError(ErrorCode.STORAGE_CONFLICT)

        async def fail_prepare(*args: object, **kwargs: object) -> "NoReturn":
            raise preparation_error

        async def fail_cleanup(*args: object, **kwargs: object) -> "NoReturn":
            raise cleanup_error

        with monkeypatch.context() as patch:
            patch.setattr(archive, phase, fail_prepare)
            patch.setattr(store, "abandon_execution_projection", fail_cleanup)
            with pytest.raises(AIError) as caught:
                await store.commit_captured_execution_projection(projection, flight, execution_id="execution")
        assert caught.value is cleanup_error
        assert caught.value.__context__ is preparation_error
        assert await archive.list_events(agent_run_id="run") == []
        # The injected cleanup never ran; release this provably unsubmitted fixture flight.
        await store.abandon_execution_projection(flight)
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert await archive.list_events(agent_run_id="run") == list(projection.events)
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("readback_code", (ErrorCode.STORAGE_UNAVAILABLE, ErrorCode.STORAGE_INTEGRITY_ERROR))
async def test_submitted_projection_readback_failure_keeps_flight_fenced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, readback_code: ErrorCode,
) -> None:
    state = await _state(tmp_path, backend)
    store = state.run_store
    group = state.execution.executions.state_store.storage_group
    try:
        archive = store.read_store(PersistenceDomain.TRACE)
        original_head = archive.execution_history_head_record
        captured = await store.capture_execution_projection("run")
        assert captured is not None
        attempted = False

        async def failed_write(*args: object, **kwargs: object) -> "NoReturn":
            nonlocal attempted
            attempted = True
            raise RuntimeError("write outcome unavailable")

        async def failed_readback(*args: str, **kwargs: str) -> "ExecutionRunSealHead":
            if attempted:
                raise AIError(readback_code)
            return await original_head(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(archive, "sync_prepared_projection", failed_write)
            patch.setattr(archive, "execution_history_head_record", failed_readback)
            with pytest.raises(AIError) as caught:
                await store.commit_captured_execution_projection(*captured, execution_id="execution")
        assert attempted
        assert caught.value.code is (
            ErrorCode.STORAGE_COMMIT_UNKNOWN if readback_code is ErrorCode.STORAGE_UNAVAILABLE
            else ErrorCode.STORAGE_INTEGRITY_ERROR
        )
        for operation in (
            store.wait_projection_flight("run"),
            store.capture_execution_projection("run"),
            store.flush_execution_projection("run", execution_id="execution", producer_generation=1),
        ):
            with pytest.raises(AIError) as fenced:
                await asyncio.wait_for(operation, 2)
            assert fenced.value is caught.value
        with pytest.raises(AIError) as close_error:
            await state.close()
        assert close_error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    finally:
        # Keep the fenced owner intact; release only the fixture's physical storage.
        await group.close()
        if backend == "sqlite":
            await group.context.close()
