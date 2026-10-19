#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final owner writes retain fencing while avoiding duplicate guard updates."""

from linktools.ai.runtime.state import RuntimeState
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event, update

from linktools.ai.core import ExecutionEventType, ExecutionStatus, IdempotencyStatus, ResourceKind, StopReason, UsageMetrics
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state._contracts import (
    ExecutionHistoryState, ExecutionTerminalCommit, IdempotencyRecord, ResultRecord,
)
from linktools.ai.runtime.state._execution_repositories import ExecutionRepositoryImpl
from linktools.ai.runtime.state._runtime_commands import RuntimeStateCommands
from linktools.ai.runtime.state._sql import _SqlTransaction

from .test_checkpoint_write_batches import _snapshot
from .test_history_projection_conformance import _record
from .test_incremental_projection_boundaries import _storage

pytestmark = pytest.mark.asyncio


class _ScalarExecution(ExecutionRepositoryImpl):
    pass


def _commit(current, *, generation=1):
    return ExecutionTerminalCommit(
        current.revision, current.event_seq,
        replace(current, status=ExecutionStatus.FAILED, error_code=ErrorCode.EXECUTION_FAILED.value),
        ResultRecord(None, StopReason.ERROR, UsageMetrics(), current.updated_at),
        ExecutionEventType.EXECUTION_FAILED,
        {"error_code": ErrorCode.EXECUTION_FAILED.value, "safe_error_details": {}},
        producer_generation=generation,
    )


async def _seed(state, current):
    repository = state.execution.executions
    await repository.create_with_history_head(current)
    await repository.state_store.mutate(lambda tx: repository.admit_history_producer_in_transaction(tx, current))
    await state.execution.idempotency.reserve(IdempotencyRecord(
        "start", "a" * 64, "b" * 64, ResourceKind.EXECUTION, current.execution_id,
        IdempotencyStatus.STARTED, None, None, current.created_at, current.updated_at,
    ))


def _assert_semantic_snapshot(batch, scalar, reduced_guards):
    batch_records, batch_facts, batch_sequences = batch
    scalar_records, scalar_facts, scalar_sequences = scalar
    assert batch_facts == scalar_facts
    assert batch_sequences == scalar_sequences
    assert len(batch_records) == len(scalar_records)
    for actual, expected in zip(batch_records, scalar_records, strict=True):
        assert actual.storage_revision + reduced_guards.get(actual.kind, 0) == expected.storage_revision
        # Idempotency transition timestamps are transaction clock values.
        if actual.kind == "idempotency":
            from linktools.ai.runtime.state._codec import _decode_enveloped_domain
            actual_value = _decode_enveloped_domain(actual.data, IdempotencyRecord)
            expected_value = _decode_enveloped_domain(expected.data, IdempotencyRecord)
            assert replace(actual_value, updated_at=expected_value.updated_at) == expected_value
            assert replace(actual, storage_revision=expected.storage_revision, data=expected.data) == expected
        else:
            assert replace(actual, storage_revision=expected.storage_revision) == expected


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_final_control_cas_preserves_business_records_and_uses_fewer_writes(tmp_path: Path, backend: str) -> None:
    current = replace(_record(ExecutionStatus.STARTED, 1), revision=1, durable_trace=False)
    states = []
    snapshots = []
    counts = []
    try:
        for scalar in (False, True):
            path = tmp_path / str(scalar)
            path.mkdir()
            state = _storage(backend, path)
            state = RuntimeState(state)
            states.append(state)
            await state.initialize(namespace="owner-final", tenant_id="tenant")
            await _seed(state, current)
            source = state.execution.executions
            repository = (_ScalarExecution(source.state_store, namespace="owner-final", tenant_id="tenant")
                          if scalar else source)
            commands = RuntimeStateCommands(repository, namespace="owner-final", events=state.execution.events, background_tasks=set())
            calls = Counter()
            def capture(_connection, _cursor, statement, _parameters, _context, _many):
                calls[statement.split()[0]] += 1
            engine = repository.state_store.storage_group.context.engine.sync_engine if backend == "sqlite" else None
            if engine is not None:
                event.listen(engine, "before_cursor_execute", capture)
            try:
                result = await commands.commit_terminal_checkpoint(_commit(current), expected_execution=current)
            finally:
                if engine is not None:
                    event.remove(engine, "before_cursor_execute", capture)
            assert result.execution.status is ExecutionStatus.FAILED
            head = await repository.get_history_head(current.execution_id, tenant_id="tenant")
            assert head.state is ExecutionHistoryState.CLOSED
            assert head.producer_generation == 1
            assert await repository.get_history_seal(current.execution_id, tenant_id="tenant") is None
            snapshots.append(await _snapshot(repository.state_store))
            counts.append(calls)
        _assert_semantic_snapshot(snapshots[0], snapshots[1], {"execution_history_head": 1})
        if backend == "sqlite":
            assert counts[0]["UPDATE"] < counts[1]["UPDATE"]
            assert counts[0]["INSERT"] == counts[1]["INSERT"]
    finally:
        for state in states:
            await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_producer_final_cas_preserves_generation_and_subclass_guard(tmp_path: Path, backend: str) -> None:
    current = replace(_record(ExecutionStatus.STARTED, 1), revision=1, durable_trace=False)
    snapshots = []
    for scalar in (False, True):
        path = tmp_path / str(scalar)
        path.mkdir()
        state = _storage(backend, path)
        state = RuntimeState(state)
        await state.initialize(namespace="owner-admit", tenant_id="tenant")
        try:
            source = state.execution.executions
            repository = (_ScalarExecution(source.state_store, namespace="owner-admit", tenant_id="tenant")
                          if scalar else source)
            await repository.create_with_history_head(current)
            await repository.state_store.mutate(lambda tx: repository.admit_history_producer_in_transaction(
                tx, current, producer_claim_id="owner-claim",
            ))
            snapshots.append(await _snapshot(repository.state_store))
            head = await repository.get_history_head(current.execution_id, tenant_id="tenant")
            assert (head.revision, head.producer_generation, head.producer_claim_id) == (1, 1, "owner-claim")
        finally:
            await state.close()
    _assert_semantic_snapshot(snapshots[0], snapshots[1], {"execution_history_head": 1})


@pytest.mark.parametrize("member", ("execution_history_head", "execution", "idempotency"))
async def test_closed_terminal_any_member_conflict_rolls_back_every_final_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, member: str,
) -> None:
    state = _storage("sqlite", tmp_path)
    state = RuntimeState(state)
    await state.initialize(namespace="owner-conflict", tenant_id="tenant")
    current = replace(_record(ExecutionStatus.STARTED, 1), revision=1, durable_trace=False)
    try:
        await _seed(state, current)
        repository = state.execution.executions
        commands = RuntimeStateCommands(repository, namespace="owner-conflict", events=state.execution.events, background_tasks=set())
        original = _SqlTransaction.replace_records
        conflicts = []
        async def conflict(transaction, replacements):
            if {item.record.kind for item in replacements} == {"execution_history_head", "execution", "idempotency"}:
                target = next(item for item in replacements if item.record.kind == member)
                table = transaction._table("ai_state_records")
                await transaction._execute(update(table).where(
                    table.c.store_digest == transaction._store_hex,
                    table.c.key_digest == target.record.key_digest.hex(),
                ).values(storage_revision=target.expected_storage_revision + 1))
                conflicts.append(member)
            return await original(transaction, replacements)
        before = await _snapshot(repository.state_store)
        with monkeypatch.context() as patch:
            patch.setattr(_SqlTransaction, "replace_records", conflict)
            with pytest.raises(AIError) as raised:
                await commands.commit_terminal_checkpoint(_commit(current), expected_execution=current)
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        assert conflicts == [member]
        assert await _snapshot(repository.state_store) == before
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_closed_final_cas_is_early_fence_and_later_failure_rolls_back(tmp_path: Path, backend: str) -> None:
    state = _storage(backend, tmp_path)
    state = RuntimeState(state)
    await state.initialize(namespace="owner-rollback", tenant_id="tenant")
    current = replace(_record(ExecutionStatus.STARTED, 1), revision=1, durable_trace=False)
    try:
        await _seed(state, current)
        repository = state.execution.executions
        before = await _snapshot(repository.state_store)
        async def mutate(transaction):
            commit = _commit(current)
            identity = await repository.terminal_idempotency_in_transaction(transaction, commit)
            await repository.commit_closed_terminal_in_transaction(transaction, replace(commit, idempotency=identity))
            with pytest.raises(AIError) as fenced:
                await repository.require_open_history_head_in_transaction(
                    transaction, current.execution_id, expected_producer_generation=1,
                )
            assert fenced.value.code is ErrorCode.STORAGE_CONFLICT
            raise AIError(ErrorCode.STORAGE_CONFLICT, safe_details={"phase": "later_canonical_write"})
        with pytest.raises(AIError) as later:
            await repository.state_store.mutate(mutate)
        assert later.value.safe_details == {"phase": "later_canonical_write"}
        assert await _snapshot(repository.state_store) == before
    finally:
        await state.close()


async def test_terminal_keeps_execution_first_when_dialect_updates_members_individually(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.storage import SQLiteDialect

    state = _storage("sqlite", tmp_path)
    state = RuntimeState(state)
    await state.initialize(namespace="owner-lock-order", tenant_id="tenant")
    current = replace(_record(ExecutionStatus.STARTED, 1), execution_id="execution-0", root_execution_id="execution-0",
                      revision=1, durable_trace=False)
    try:
        await _seed(state, current)
        repository = state.execution.executions
        execution_key = repository._key("execution", current.execution_id)
        head_key = repository._key("execution_history_head", current.execution_id)
        assert head_key < execution_key
        commands = RuntimeStateCommands(repository, namespace="owner-lock-order", events=state.execution.events, background_tasks=set())
        guard = _SqlTransaction.guard_record
        execute = _SqlTransaction._execute
        writes = []

        async def record_guard(transaction, key, *, expected_storage_revision):
            result = await guard(transaction, key, expected_storage_revision=expected_storage_revision)
            writes.append(("guard", key))
            return result

        async def record_execute(transaction, statement, parameters=None):
            if getattr(statement, "is_update", False) and isinstance(parameters, list):
                writes.extend(("member", bytes.fromhex(value["_replacement_key"])) for value in parameters)
            return await execute(transaction, statement, parameters)

        with monkeypatch.context() as patch:
            patch.setattr(SQLiteDialect, "prefer_update_executemany", property(lambda _self: True))
            patch.setattr(_SqlTransaction, "guard_record", record_guard)
            patch.setattr(_SqlTransaction, "_execute", record_execute)
            await commands.commit_terminal_checkpoint(_commit(current), expected_execution=current)
        assert writes[0] == ("guard", execution_key)
        assert ("member", head_key) in writes
        assert ("member", execution_key) in writes
        assert writes.index(("member", head_key)) < writes.index(("member", execution_key))
        assert sum(kind == "guard" for kind, _key in writes) == 1
    finally:
        await state.close()


@pytest.mark.parametrize("sqlstate", ("40P01", "40001"))
async def test_aborted_terminal_batch_retries_whole_owner_write_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sqlstate: str,
) -> None:
    from sqlalchemy.exc import OperationalError

    class AbortedTransaction(Exception):
        pass

    state = _storage("sqlite", tmp_path)
    state = RuntimeState(state)
    await state.initialize(namespace="owner-retry", tenant_id="tenant")
    current = replace(_record(ExecutionStatus.STARTED, 1), revision=1, durable_trace=False)
    try:
        await _seed(state, current)
        repository = state.execution.executions
        commands = RuntimeStateCommands(repository, namespace="owner-retry", events=state.execution.events, background_tasks=set())
        original = _SqlTransaction.replace_records
        attempts = []

        async def abort_once(transaction, replacements):
            result = await original(transaction, replacements)
            if {item.record.kind for item in replacements} == {"execution_history_head", "execution", "idempotency"}:
                attempts.append(transaction)
                if len(attempts) == 1:
                    cause = AbortedTransaction("known aborted transaction")
                    cause.sqlstate = sqlstate
                    raise OperationalError("UPDATE", None, cause)
            return result

        with monkeypatch.context() as patch:
            patch.setattr(_SqlTransaction, "replace_records", abort_once)
            result = await commands.commit_terminal_checkpoint(_commit(current), expected_execution=current)
        assert len(attempts) == 2 and attempts[0] is not attempts[1]
        assert result.execution.revision == current.revision + 1
        assert result.execution.event_seq == current.event_seq + 1
        records, facts, _sequences = await _snapshot(repository.state_store)
        assert len(records) == 3 and len(facts) == 1
        assert (await repository.get_history_head(current.execution_id, tenant_id="tenant")).state is ExecutionHistoryState.CLOSED
    finally:
        await state.close()


@pytest.mark.parametrize("_server", ("postgresql", "mysql"), indirect=True)
async def test_read_committed_terminal_and_tool_guard_share_execution_first_order(
    _server: tuple[str, list[str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    import getpass
    from sqlalchemy.engine import URL
    from sqlalchemy.ext.asyncio import create_async_engine
    from linktools.ai.migrate import provision_database
    from linktools.ai.runtime import RuntimeStorage

    name, command = _server
    if name == "postgresql":
        url = URL.create("postgresql+asyncpg", username=getpass.getuser(), database=command[-1],
                         query={"host": command[command.index("-h") + 1]})
    else:
        socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
        url = URL.create("mysql+asyncmy", username="root", database=command[-1],
                         query={"unix_socket": socket, "charset": "utf8mb4"})
    engine = create_async_engine(url, isolation_level="READ COMMITTED")
    await provision_database(engine)
    state = RuntimeStorage.sql(engine)
    state = RuntimeState(state)
    await state.initialize(namespace="owner-lock-order", tenant_id="tenant")
    current = replace(_record(ExecutionStatus.STARTED, 1), execution_id="execution-0", root_execution_id="execution-0",
                      revision=1, durable_trace=False)
    first = None
    terminal = None
    entered = asyncio.Event()
    terminal_guard_started = asyncio.Event()
    release = asyncio.Event()
    try:
        await _seed(state, current)
        repository = state.execution.executions
        execution_key = repository._key("execution", current.execution_id)
        head_key = repository._key("execution_history_head", current.execution_id)
        assert head_key < execution_key
        commands = RuntimeStateCommands(repository, namespace="owner-lock-order", events=state.execution.events, background_tasks=set())
        original = _SqlTransaction.guard_record

        async def interleave(transaction, key, *, expected_storage_revision):
            if asyncio.current_task() is not first and key == execution_key:
                terminal_guard_started.set()
            result = await original(transaction, key, expected_storage_revision=expected_storage_revision)
            if asyncio.current_task() is first and key == execution_key:
                entered.set()
                await release.wait()
            return result

        async def tool_owner():
            await repository.state_store.mutate(lambda tx: repository.require_open_history_head_in_transaction(
                tx, current.execution_id, expected_producer_generation=1, allow_cancelling=False,
            ))

        with monkeypatch.context() as patch:
            patch.setattr(_SqlTransaction, "guard_record", interleave)
            first = asyncio.create_task(tool_owner())
            await asyncio.wait_for(entered.wait(), 5)
            terminal = asyncio.create_task(commands.commit_terminal_checkpoint(_commit(current), expected_execution=current))
            await asyncio.wait_for(terminal_guard_started.wait(), 5)
            release.set()
            await asyncio.wait_for(first, 5)
            try:
                await asyncio.wait_for(terminal, 5)
            except AIError as error:
                assert error.code is ErrorCode.STORAGE_CONFLICT
                await commands.commit_terminal_checkpoint(_commit(current), expected_execution=current)
        records, facts, _sequences = await _snapshot(repository.state_store)
        assert len(records) == 3 and len(facts) == 1
        assert (await repository.get_history_head(current.execution_id, tenant_id="tenant")).state is ExecutionHistoryState.CLOSED
    finally:
        release.set()
        for task in (first, terminal):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(task for task in (first, terminal) if task is not None), return_exceptions=True)
        await state.close()
        await engine.dispose()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_closed_terminal_lost_ack_uses_complete_readback_without_reapplying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    from linktools.ai.runtime.state._contracts import ExecutionTerminalCommitResult

    state = _storage(backend, tmp_path)
    state = RuntimeState(state)
    await state.initialize(namespace="owner-ack", tenant_id="tenant")
    current = replace(_record(ExecutionStatus.STARTED, 1), revision=1, durable_trace=False)
    try:
        await _seed(state, current)
        repository = state.execution.executions
        commands = RuntimeStateCommands(repository, namespace="owner-ack", events=state.execution.events, background_tasks=set())
        group_type = type(repository.state_store.storage_group)
        original = group_type.mutate
        commits = []

        async def lose_ack(group, stores, callback):
            result = await original(group, stores, callback)
            if isinstance(result, ExecutionTerminalCommitResult):
                commits.append(result)
                if len(commits) == 1:
                    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return result

        with monkeypatch.context() as patch:
            patch.setattr(group_type, "mutate", lose_ack)
            result = await commands.commit_terminal_checkpoint(_commit(current), expected_execution=current)
        assert len(commits) == 1
        assert result.execution == commits[0].execution
        records, facts, _sequences = await _snapshot(repository.state_store)
        assert len(records) == 3 and len(facts) == 1
        assert {record.kind: record.storage_revision for record in records} == {
            "execution": 2, "execution_history_head": 2, "idempotency": 1,
        }
    finally:
        await state.close()
