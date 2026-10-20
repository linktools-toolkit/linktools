#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Producer revocation is established from one authoritative metadata view."""

from linktools.ai.runtime.state import RuntimeState
import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeDomain, RuntimeStorage
from linktools.ai.runtime.state._contracts import (
    ExecutionHistoryHeadRecord, ExecutionHistorySealRecord, ExecutionHistoryState,
)
from linktools.ai.runtime.state._execution_repositories import ExecutionRepositoryImpl
from linktools.ai.runtime.state._store import active_state_scope, active_state_transaction
from linktools.ai.storage import InMemoryCacheBackend, StoredPayload

from .test_step_projection import _execution


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_hint", (False, True))
@pytest.mark.parametrize("outcome,expected", (
    ("sealed", None),
    ("closed", None),
    ("cancelled", None),
    ("missing_execution", ErrorCode.STORAGE_CONFLICT),
    ("missing_head", ErrorCode.STORAGE_CONFLICT),
    ("new_generation", None),
    ("same_generation", ErrorCode.STORAGE_CONFLICT),
    ("active", ErrorCode.STORAGE_CONFLICT),
    ("missing_seal", ErrorCode.STORAGE_INTEGRITY_ERROR),
    ("wrong_seal", ErrorCode.STORAGE_INTEGRITY_ERROR),
))
async def test_revoked_producer_check_uses_one_sql_metadata_snapshot(
    tmp_path: Path, outcome: str, expected: ErrorCode | None, terminal_hint: bool,
) -> None:
    state = RuntimeState(RuntimeStorage.sqlite(tmp_path / "producer.db"))
    await state.initialize(namespace="producer-read", tenant_id="tenant")
    repository = state.execution.executions
    archive = state.run_store.read_store(RuntimeDomain.TRACE)
    seal = ExecutionHistorySealRecord("execution", (), 0)
    open_head = outcome in {"new_generation", "same_generation"}
    closed = outcome in {"closed", "cancelled"}
    head = ExecutionHistoryHeadRecord(
        "execution", ExecutionHistoryState.OPEN if open_head else ExecutionHistoryState.CLOSED if closed else ExecutionHistoryState.SEALED,
        0, None if open_head or closed else "incorrect" if outcome == "wrong_seal" else seal.seal_digest,
        2 if outcome == "new_generation" else 1,
    )
    execution = replace(
        _execution(), durable_trace=not closed,
        status=ExecutionStatus.STARTED if outcome == "active" else ExecutionStatus.CANCELLED if outcome == "cancelled" else ExecutionStatus.SUCCEEDED,
    )
    if outcome != "missing_execution":
        await repository.create(execution)
    hint = replace(execution, status=ExecutionStatus.SUCCEEDED) if terminal_hint else None

    async def seed(transaction):
        if outcome != "missing_head":
            await repository.insert_history_head_in_transaction(transaction, head)
        if not open_head and not closed and outcome != "missing_seal":
            await repository.put_history_seal_in_transaction(transaction, seal)

    await repository.state_store.mutate(seed)
    engine = repository.state_store.storage_group.context.engine.sync_engine
    begins = []

    def begin(connection):
        begins.append(connection)

    event.listen(engine, "begin", begin)
    try:
        if expected is None:
            await archive.require_revoked_execution_producer(execution_id="execution", producer_generation=1, terminal_execution=hint)
        else:
            with pytest.raises(AIError) as raised:
                await archive.require_revoked_execution_producer(execution_id="execution", producer_generation=1, terminal_execution=hint)
            assert raised.value.code is expected
        assert len(begins) <= 1
        assert active_state_scope() is None
    finally:
        event.remove(engine, "begin", begin)
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("body_size", (32768, 1048576))
@pytest.mark.parametrize("hint_kind", ("none", "wrong_id", "active"))
async def test_newer_producer_keeps_head_only_read(
    tmp_path: Path, body_size: int, hint_kind: str,
) -> None:
    state = RuntimeState(RuntimeStorage.sqlite(tmp_path / "newer.db"))
    await state.initialize(namespace="producer-read", tenant_id="tenant")
    repository = state.execution.executions
    archive = state.run_store.read_store(RuntimeDomain.TRACE)
    execution = _execution()
    execution = replace(execution, stored_user_input=replace(
        execution.stored_user_input, payload=StoredPayload.inline_text("x" * body_size),
    ))
    await repository.create(execution)
    head = ExecutionHistoryHeadRecord("execution", ExecutionHistoryState.OPEN, 0, None, 2)
    await repository.state_store.mutate(lambda tx: repository.insert_history_head_in_transaction(tx, head))
    hint = None if hint_kind == "none" else execution if hint_kind == "active" else replace(
        execution, execution_id="different", status=ExecutionStatus.SUCCEEDED,
    )
    engine = repository.state_store.storage_group.context.engine.sync_engine
    returned_kinds = []
    selects = []

    def after(connection, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            selects.append(statement)
            names = [column[0] for column in cursor.description]
            returned_kinds.extend(dict(zip(names, row))["kind"] for row in cursor._rows)

    event.listen(engine, "after_cursor_execute", after)
    try:
        await archive.require_revoked_execution_producer(
            execution_id="execution", producer_generation=1, terminal_execution=hint,
        )
        assert len(selects) == 1
        assert returned_kinds == ["execution_history_head"]
    finally:
        event.remove(engine, "after_cursor_execute", after)
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_terminal_hint_cannot_replace_current_tenant_proof(tmp_path: Path, backend: str) -> None:
    from .test_model_control_observation import _state

    async with _state(backend, tmp_path, InMemoryCacheBackend(max_bytes=1000000)) as state:
        repository = state.execution.executions
        actual = _execution()
        await repository.create_with_history_head(actual)
        # A same-ID terminal record from elsewhere is only a fetch hint.
        foreign = RuntimeState(RuntimeStorage.in_memory())
        await foreign.initialize(namespace="model-control", tenant_id="other")
        try:
            hint = replace(actual, status=ExecutionStatus.SUCCEEDED, durable_trace=False)
            await foreign.execution.executions.create(hint)
            hint = await foreign.execution.executions.get("execution", tenant_id="other")
            await repository.observe_execution_control("execution", tenant_id="tenant")
            archive = state.run_store.read_store(RuntimeDomain.TRACE)
            with pytest.raises(AIError) as raised:
                await archive.require_revoked_execution_producer(
                    execution_id="execution", producer_generation=0, terminal_execution=hint,
                )
            assert raised.value.code is ErrorCode.STORAGE_CONFLICT
            assert active_state_scope() is None
        finally:
            await foreign.close()


@pytest.mark.asyncio
async def test_custom_repository_keeps_head_first_with_terminal_hint(tmp_path: Path) -> None:
    state = RuntimeState(RuntimeStorage.sqlite(tmp_path / "custom.db"))
    await state.initialize(namespace="producer-read", tenant_id="tenant")
    source = state.execution.executions.state_store
    calls = []

    class Custom(ExecutionRepositoryImpl):
        async def observe_execution_control(self, *args, **kwargs):
            raise AssertionError("custom owner must keep its original reads")

        async def get_history_head(self, *args, **kwargs):
            assert active_state_transaction(source) is not None
            calls.append("head")
            return await super().get_history_head(*args, **kwargs)

        async def get(self, *args, **kwargs):
            assert active_state_transaction(source) is not None
            calls.append("execution")
            return await super().get(*args, **kwargs)

    repository = Custom(source, namespace="producer-read", tenant_id="tenant")
    execution = replace(_execution(), status=ExecutionStatus.SUCCEEDED, durable_trace=False)
    await repository.create(execution)
    head = ExecutionHistoryHeadRecord("execution", ExecutionHistoryState.CLOSED, 0, None, 1)
    await source.mutate(lambda tx: repository.insert_history_head_in_transaction(tx, head))
    archive = state.run_store.read_store(RuntimeDomain.TRACE)
    archive._execution_repository = repository
    try:
        await archive.require_revoked_execution_producer(
            execution_id="execution", producer_generation=1, terminal_execution=execution,
        )
        assert calls == ["head", "execution"]
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_cancelled_control_observation_does_not_grant_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeState(RuntimeStorage.sqlite(tmp_path / "cancel.db"))
    await state.initialize(namespace="producer-read", tenant_id="tenant")
    repository = state.execution.executions
    execution = replace(_execution(), status=ExecutionStatus.SUCCEEDED, durable_trace=False)
    await repository.create(execution)
    head = ExecutionHistoryHeadRecord("execution", ExecutionHistoryState.CLOSED, 0, None, 1)
    await repository.state_store.mutate(lambda tx: repository.insert_history_head_in_transaction(tx, head))
    original = repository.observe_execution_control
    entered = asyncio.Event()

    async def observe(*args, **kwargs):
        assert active_state_transaction(repository.state_store) is not None
        await original(*args, **kwargs)
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(repository, "observe_execution_control", observe)
    archive = state.run_store.read_store(RuntimeDomain.TRACE)
    task = asyncio.create_task(archive.require_revoked_execution_producer(
        execution_id="execution", producer_generation=1, terminal_execution=execution,
    ))
    try:
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert active_state_scope() is None
        monkeypatch.setattr(repository, "observe_execution_control", original)
        await archive.require_revoked_execution_producer(
            execution_id="execution", producer_generation=1, terminal_execution=execution,
        )
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_terminal_hint_and_cached_pair_do_not_hide_missing_source(tmp_path: Path) -> None:
    path = tmp_path / "stale.db"
    state = RuntimeState(RuntimeStorage.sqlite(path, cache=InMemoryCacheBackend(max_bytes=1000000)))
    remote = RuntimeState(RuntimeStorage.sqlite(path))
    await state.initialize(namespace="producer-read", tenant_id="tenant")
    await remote.initialize(namespace="producer-read", tenant_id="tenant")
    repository = state.execution.executions
    execution = replace(_execution(), status=ExecutionStatus.SUCCEEDED, durable_trace=False)
    await repository.create(execution)
    head = ExecutionHistoryHeadRecord("execution", ExecutionHistoryState.CLOSED, 0, None, 1)
    await repository.state_store.mutate(lambda tx: repository.insert_history_head_in_transaction(tx, head))
    assert await repository.observe_execution_control("execution", tenant_id="tenant") == (execution, head)
    source = remote.execution.executions
    await source.state_store.mutate(lambda tx: tx.delete_record(source._key("execution", "execution")))
    archive = state.run_store.read_store(RuntimeDomain.TRACE)
    try:
        with pytest.raises(AIError) as raised:
            await archive.require_revoked_execution_producer(
                execution_id="execution", producer_generation=1, terminal_execution=execution,
            )
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
    finally:
        await remote.close()
        await state.close()


@pytest.mark.asyncio
async def test_terminal_hint_preserves_unresolved_durability_flight() -> None:
    from linktools.ai.runtime.state._step_contracts import AgentRunRecord

    state = RuntimeState(RuntimeStorage.in_memory())
    await state.initialize(namespace="unknown-producer", tenant_id="tenant")
    store = state.run_store
    store.bind_execution_producer("run", execution_id="execution", producer_generation=1)
    run = AgentRunRecord("run")
    await store.register_agent_run(run, execution_id="execution")
    captured = await store.capture_execution_projection("run")
    assert captured is not None
    _, flight = captured
    unknown = AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
    await store._fence_durability_flight(flight, unknown)
    with pytest.raises(AIError) as raised:
        await store.discard_revoked_producer_staging(
            "run", execution_id="execution", producer_generation=1,
            terminal_execution=replace(_execution(), status=ExecutionStatus.SUCCEEDED),
        )
    assert raised.value is unknown
    assert store._durability_flights["run"] is flight
    assert store.trace_persistence.dirty == {"run"}
    assert await store.get_agent_run(agent_run_id="run") == run
    with pytest.raises(AIError) as closed:
        await state.close()
    assert closed.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
