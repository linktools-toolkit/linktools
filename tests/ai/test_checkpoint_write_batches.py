#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpoint batches preserve scalar persistence and owner override semantics."""

from linktools.ai.runtime.state import RuntimeState
import sqlite3

from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from sqlalchemy import event, update
from sqlalchemy.exc import OperationalError

from linktools.ai.core import ExecutionEventType, ExecutionStatus, StopReason, UsageMetrics
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._contracts import (
    ContextProjection, ExecutionEventAppend, ExecutionHistoryHeadRecord,
    ExecutionHistorySealRecord, ExecutionTerminalCommit, ExecutionTerminalCommitResult,
    HistoryQuality, ResultRecord, TranscriptChunk,
)
from linktools.ai.runtime.state._execution_repositories import ExecutionRepositoryImpl
from linktools.ai.runtime.state._history import TranscriptRepository
from linktools.ai.runtime.state._archive_closure import rekey_archive_roots
from linktools.ai.runtime.state._runtime_commands import RuntimeStateCommands
from linktools.ai.runtime.state._sql import _SqlTransaction
from linktools.ai.runtime.state._step_contracts import AgentRunCheckpoint, AgentRunRecord
from linktools.ai.runtime.state._store import StateTransaction, StoredRecord

from .test_history_projection_conformance import _record
from .test_incremental_projection_boundaries import _storage

pytestmark = pytest.mark.asyncio


async def _snapshot(store, *, scope_mapping=None):
    async def read(transaction):
        return (
            tuple(sorted(await transaction.scan_records(), key=lambda item: item.key_digest)),
            tuple(sorted(await transaction.scan_facts(), key=lambda item: (item.stream_digest, item.sequence))),
            await transaction.scan_sequences(),
        )
    snapshot = await store.read(read)
    if scope_mapping is None:
        return snapshot
    records, facts, sequences = snapshot
    return (
        tuple(replace(item, data=rekey_archive_roots(dict(item.data), scope_mapping)) for item in records),
        tuple(replace(item, data=rekey_archive_roots(dict(item.data), scope_mapping)) for item in facts),
        sequences,
    )


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_checkpoint_batch_preserves_records_facts_sequences_and_revisions(tmp_path: Path, backend: str) -> None:
    calls = []

    class ScalarTranscript(TranscriptRepository):
        async def append_chunks(
            self, transaction: StateTransaction, owner_id: str,
            chunks: Sequence[TranscriptChunk], quality: HistoryQuality | None = None,
        ) -> None:
            calls.append("chunks")
            await super().append_chunks(transaction, owner_id, chunks, quality)

        async def store_projection(
            self, transaction: StateTransaction, owner_id: str, projection: ContextProjection,
        ) -> None:
            calls.append("projection")
            await super().store_projection(transaction, owner_id, projection)

    states = []
    archives = []
    try:
        for name in ("batch", "scalar"):
            path = tmp_path / name
            path.mkdir()
            state = RuntimeState(_storage(backend, path))
            await state.initialize(namespace="checkpoint-batch", tenant_id="tenant")
            states.append(state)
            archive = state.run_store.read_store(RuntimeDomain.CHECKPOINT)
            if name == "scalar":
                archive._history = ScalarTranscript(
                    archive.state_store, object_store=archive.transcript_repository._object_store, namespace="checkpoint-batch",
                    tenant_id="tenant", runtime_domain=RuntimeDomain.CHECKPOINT,
                    archive_object_store=state.archive_object_store(RuntimeDomain.CHECKPOINT),
                    archive_object_reader=lambda reference, owner=state: owner.read_archive_object(
                        RuntimeDomain.CHECKPOINT, reference,
                    ),
                )
            archives.append(archive)
        messages = [ModelRequest(parts=[UserPromptPart(str(index))]) for index in range(133)]
        checkpoints = (
            AgentRunCheckpoint("run", 1, messages[:130], transcript_message_count_before=0),
            AgentRunCheckpoint("run", 2, messages, transcript_message_count_before=130),
            AgentRunCheckpoint("run", 3, messages, transcript_message_count_before=133),
        )
        run = AgentRunRecord("run")
        counts = [Counter(), Counter()]
        for checkpoint in checkpoints:
            for index, archive in enumerate(archives):
                def capture(_connection, _cursor, statement, _parameters, _context, _many):
                    counts[index][statement.split()[0]] += 1
                engine = archive.state_store.storage_group.context.engine.sync_engine if backend == "sqlite" else None
                if engine is not None:
                    event.listen(engine, "before_cursor_execute", capture)
                try:
                    await archive.materialize_checkpoint(run, checkpoint)
                finally:
                    if engine is not None:
                        event.remove(engine, "before_cursor_execute", capture)
            batch_scope = (await archives[0].transcript_repository.content_archive()).scope_id
            scalar_scope = (await archives[1].transcript_repository.content_archive()).scope_id
            assert await _snapshot(archives[0].state_store) == await _snapshot(
                archives[1].state_store, scope_mapping={scalar_scope: batch_scope},
            )
        assert calls == ["chunks", "projection"] * 3
        if backend == "sqlite":
            assert counts[0]["INSERT"] < counts[1]["INSERT"]
            assert counts[0]["UPDATE"] < counts[1]["UPDATE"]
        for archive in archives:
            assert await archive.checkpoint_count(agent_run_id="run") == 3
            assert [message async for message in archive.iter_messages(agent_run_id="run")] == messages
            snapshot = await _snapshot(archive.state_store)
            await archive.materialize_checkpoint(run, checkpoints[-1])
            assert await _snapshot(archive.state_store) == snapshot
    finally:
        for state in states:
            await state.close()


@pytest.mark.parametrize("target", ("transcript_head", "context_projection"))
async def test_checkpoint_lost_member_cas_rolls_back_entire_write_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str,
) -> None:
    state = RuntimeState(_storage("sqlite", tmp_path))
    await state.initialize(namespace="checkpoint-cas", tenant_id="tenant")
    archive = state.run_store.read_store(RuntimeDomain.CHECKPOINT)
    run = AgentRunRecord("run")
    first = ModelRequest(parts=[UserPromptPart("first")])
    second = ModelRequest(parts=[UserPromptPart("second")])
    try:
        await archive.materialize_checkpoint(run, AgentRunCheckpoint("run", 1, [first], transcript_message_count_before=0))
        before = await _snapshot(archive.state_store)
        original = _SqlTransaction.replace_records
        lost = []

        async def interleave(transaction, replacements):
            kinds = {value.record.kind for value in replacements}
            if kinds == {"transcript_head", "context_projection"}:
                member = next(value for value in replacements if value.record.kind == target)
                table = transaction._table("ai_state_records")
                await transaction._execute(update(table).where(
                    table.c.store_digest == transaction._store_hex,
                    table.c.key_digest == member.record.key_digest.hex(),
                ).values(storage_revision=member.expected_storage_revision + 1))
                lost.append(target)
            return await original(transaction, replacements)

        with monkeypatch.context() as patch:
            patch.setattr(_SqlTransaction, "replace_records", interleave)
            with pytest.raises(AIError) as raised:
                await archive.materialize_checkpoint(run, AgentRunCheckpoint(
                    "run", 2, [first, second], transcript_message_count_before=1,
                ))
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        assert lost == [target]
        assert await _snapshot(archive.state_store) == before
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("trace", (False, True))
async def test_terminal_batch_preserves_custom_repository_path_and_durable_values(tmp_path: Path, backend: str, trace: bool) -> None:
    calls = []

    class ScalarExecution(ExecutionRepositoryImpl):
        async def put_history_seal_in_transaction(
            self, transaction: StateTransaction, seal: ExecutionHistorySealRecord,
        ) -> ExecutionHistorySealRecord:
            calls.append("seal")
            return await super().put_history_seal_in_transaction(transaction, seal)

        async def replace_history_head_in_transaction(
            self, transaction: StateTransaction, current_record: StoredRecord,
            next_head: ExecutionHistoryHeadRecord,
        ) -> ExecutionHistoryHeadRecord:
            calls.append("head")
            return await super().replace_history_head_in_transaction(transaction, current_record, next_head)

        async def commit_terminal_in_transaction(
            self, transaction: StateTransaction, commit: ExecutionTerminalCommit, *,
            pending_events: Sequence[ExecutionEventAppend] = (),
        ) -> ExecutionTerminalCommitResult:
            calls.append("terminal")
            return await super().commit_terminal_in_transaction(transaction, commit, pending_events=pending_events)

    current = replace(_record(ExecutionStatus.STARTED, 0), durable_trace=trace)
    result = ResultRecord(None, StopReason.ERROR, UsageMetrics(), current.updated_at)
    commit = ExecutionTerminalCommit(
        current.revision, current.event_seq,
        replace(current, status=ExecutionStatus.FAILED, error_code=ErrorCode.EXECUTION_FAILED.value),
        result, ExecutionEventType.EXECUTION_FAILED,
        {"error_code": ErrorCode.EXECUTION_FAILED.value, "safe_error_details": {}},
    )
    states = []
    snapshots = []
    results = []
    try:
        for name in ("batch", "scalar"):
            path = tmp_path / name
            path.mkdir()
            state = RuntimeState(_storage(backend, path))
            await state.initialize(namespace="terminal-batch", tenant_id="tenant")
            states.append(state)
            original = state.execution.executions
            repository = original if name == "batch" else ScalarExecution(
                original.state_store, namespace="terminal-batch", tenant_id="tenant",
            )
            await repository.create_with_history_head(current)
            commands = RuntimeStateCommands(
                repository, namespace="terminal-batch", events=state.execution.events,
                background_tasks=set(),
            )
            results.append(await commands.commit_terminal_checkpoint(commit))
            snapshots.append(await _snapshot(repository.state_store))
        assert calls == (["seal"] if trace else []) + ["head", "terminal"]
        assert results[0] == results[1]
        batch_records, batch_facts, batch_sequences = snapshots[0]
        scalar_records, scalar_facts, scalar_sequences = snapshots[1]
        assert batch_facts == scalar_facts
        assert batch_sequences == scalar_sequences
        for batch, scalar in zip(batch_records, scalar_records, strict=True):
            removed_guard = (int(batch.kind == "execution_history_head") - int(batch.kind == "execution")) if not trace else 0
            assert batch.storage_revision + removed_guard == scalar.storage_revision
            assert replace(batch, storage_revision=scalar.storage_revision) == scalar
    finally:
        for state in states:
            await state.close()


async def test_checkpoint_prefetch_is_bounded_and_mixed_owners_fail_closed(tmp_path: Path) -> None:
    state = RuntimeState(_storage("memory", tmp_path))
    await state.initialize(namespace="checkpoint-owner", tenant_id="tenant")
    archive = state.run_store.read_store(RuntimeDomain.CHECKPOINT)
    repository = archive.transcript_repository
    run = AgentRunRecord("run")
    try:
        checkpoint = AgentRunCheckpoint(
            "run", 1, [ModelRequest(parts=[UserPromptPart("owned")])],
            transcript_message_count_before=0,
        )
        prepared, = await archive.prepare_checkpoints(run, (checkpoint,))
        chunk, = prepared.chunks
        expected = repository.checkpoint_record_keys("run", ())
        for changed in (
            replace(chunk, owner_id="foreign"),
            replace(chunk, message_count=10**100),
            replace(chunk, first_message_index=-1),
        ):
            assert repository.checkpoint_record_keys("run", (changed,)) == expected
        before = await _snapshot(archive.state_store)
        with pytest.raises(AIError) as raised:
            await archive.sync_prepared_projection(
                run, events=(), checkpoints=(prepared, replace(prepared, owner_id="foreign", chunks=())),
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await _snapshot(archive.state_store) == before
        await archive.register_agent_run(run)
        before = await _snapshot(archive.state_store)
        with pytest.raises(AIError) as raised:
            await archive.state_store.mutate(lambda transaction: repository.append_checkpoint(
                transaction, "run", (replace(chunk, owner_id="foreign"),), prepared.projection,
            ))
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        assert await _snapshot(archive.state_store) == before
    finally:
        await state.close()


@pytest.mark.parametrize("target", ("execution_history_head", "execution"))
async def test_terminal_lost_member_cas_rolls_back_closed_head_and_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str,
) -> None:
    state = RuntimeState(_storage("sqlite", tmp_path))
    await state.initialize(namespace="terminal-cas", tenant_id="tenant")
    repository = state.execution.executions
    current = replace(_record(ExecutionStatus.STARTED, 0), durable_trace=False)
    await repository.create_with_history_head(current)
    result = ResultRecord(None, StopReason.ERROR, UsageMetrics(), current.updated_at)
    commit = ExecutionTerminalCommit(
        current.revision, current.event_seq,
        replace(current, status=ExecutionStatus.FAILED, error_code=ErrorCode.EXECUTION_FAILED.value),
        result, ExecutionEventType.EXECUTION_FAILED,
        {"error_code": ErrorCode.EXECUTION_FAILED.value, "safe_error_details": {}},
    )
    commands = RuntimeStateCommands(
        repository, namespace="terminal-cas", events=state.execution.events,
        background_tasks=set(),
    )
    original = _SqlTransaction.replace_records
    lost = []

    async def interleave(transaction, replacements):
        if {value.record.kind for value in replacements} == {"execution_history_head", "execution"}:
            member = next(value for value in replacements if value.record.kind == target)
            table = transaction._table("ai_state_records")
            await transaction._execute(update(table).where(
                table.c.store_digest == transaction._store_hex,
                table.c.key_digest == member.record.key_digest.hex(),
            ).values(storage_revision=member.expected_storage_revision + 1))
            lost.append(target)
        return await original(transaction, replacements)

    try:
        before = await _snapshot(repository.state_store)
        with monkeypatch.context() as patch:
            patch.setattr(_SqlTransaction, "replace_records", interleave)
            with pytest.raises(AIError) as raised:
                await commands.commit_terminal_checkpoint(commit)
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        assert lost == [target]
        assert await _snapshot(repository.state_store) == before
    finally:
        await state.close()


@pytest.mark.parametrize("failure_stage", ("reservation", "facts", "seek"))
@pytest.mark.parametrize("retry", (False, True))
async def test_new_checkpoint_fact_batch_rolls_back_or_retries_without_sequence_gaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str, retry: bool,
) -> None:
    state = RuntimeState(_storage("sqlite", tmp_path))
    await state.initialize(namespace="checkpoint-batch-rollback", tenant_id="tenant")
    archive = state.run_store.read_store(RuntimeDomain.CHECKPOINT)
    run = AgentRunRecord("run")
    checkpoint = AgentRunCheckpoint(
        "run", 1, [ModelRequest(parts=[UserPromptPart("owned")])],
        transcript_message_count_before=0,
    )
    method_name = {"reservation": "reserve_sequences", "facts": "insert_facts", "seek": "insert_records"}[failure_stage]
    original = getattr(_SqlTransaction, method_name)
    failures = []

    async def fail_once(transaction, values):
        result = await original(transaction, values)
        selected = (
            len(values) == 2 if failure_stage == "reservation" else
            {value.kind for value in values} == {"transcript_chunk", "step_checkpoint"} if failure_stage == "facts" else
            any(value.kind == "transcript_seek" for value in values)
        )
        if selected and not failures:
            failures.append(failure_stage)
            if retry:
                error = sqlite3.OperationalError("database is locked")
                error.sqlite_errorcode = 5
                error.sqlite_errorname = "SQLITE_BUSY"
                raise OperationalError("checkpoint batch", {}, error)
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return result

    try:
        before = await _snapshot(archive.state_store)
        with monkeypatch.context() as patch:
            patch.setattr(_SqlTransaction, method_name, fail_once)
            if retry:
                await archive.materialize_checkpoint(run, checkpoint)
            else:
                with pytest.raises(AIError) as raised:
                    await archive.materialize_checkpoint(run, checkpoint)
                assert raised.value.code is ErrorCode.STORAGE_CONFLICT
                assert await _snapshot(archive.state_store) == before
        assert failures == [failure_stage]
        if not retry:
            await archive.materialize_checkpoint(run, checkpoint)
        snapshot = await _snapshot(archive.state_store)
        _, facts, sequences = snapshot
        assert {fact.kind for fact in facts} == {"transcript_chunk", "step_checkpoint"}
        assert all(fact.sequence == 1 for fact in facts)
        assert len(sequences) == 2 and all(value == 1 for value in sequences.values())
        await archive.materialize_checkpoint(run, checkpoint)
        assert await _snapshot(archive.state_store) == snapshot
    finally:
        await state.close()
