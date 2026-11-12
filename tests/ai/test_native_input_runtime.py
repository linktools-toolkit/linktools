#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Accepted plain input is shared only after an exact prompt publication."""

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
from sqlalchemy import event

from linktools.ai.core import ExecutionEventType, ExecutionStatus, StopReason, UsageMetrics
from linktools.ai.model import StoredUserInput
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence.state import StateTransaction, StoredRecord
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage, SqlStepStore
from linktools.ai.persistence._runtime_commands import StateCommands
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint, AgentRunRecord
from linktools.ai.persistence.codec import decode_envelope, native_user_input_reference
from linktools.ai.persistence.contracts import ExecutionRecord, ExecutionTerminalCommit, ResultRecord
from linktools.ai.storage import StoredPayload

from .test_step_projection import _execution
from .test_native_text_result_runtime import _publish


async def _prepare_input(
    state: Persistence, original: StoredUserInput, source: str, output: str = "done",
) -> tuple[StateCommands, ExecutionTerminalCommit, ExecutionRecord, AgentRunCheckpoint]:
    repository = state.execution.executions
    native = SqlStepStore.for_state_store(repository.state_store, namespace="native-result", tenant_id="tenant")
    assert native is not None
    state.run_store.bind_step_store(native)
    current = replace(_execution(), revision=1, producer_generation=1, stored_user_input=original)
    await repository.create_execution(current)
    checkpoint = AgentRunCheckpoint("run", 1, [ModelRequest(parts=[UserPromptPart(source)]),
        ModelResponse(parts=[TextPart(output)])], transcript_message_count_before=0, timestamp=current.updated_at)
    commands = StateCommands(repository, namespace="native-result", events=state.execution.events,
        execution_run_store=state.run_store.read_store(PersistenceDomain.TRACE), background_tasks=set())
    result = ResultRecord(StoredPayload.inline_json(output), StopReason.END_TURN, UsageMetrics(), current.updated_at)
    commit = ExecutionTerminalCommit(current.revision, current.event_seq,
        replace(current, status=ExecutionStatus.SUCCEEDED), result, ExecutionEventType.EXECUTION_SUCCEEDED,
        {"agent_run_id": "run"}, producer_generation=1)
    return commands, commit, current, checkpoint


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ("large", "small", "different", "view"))
async def test_input_reference_preserves_complete_accepted_value(tmp_path: Path, case: str) -> None:
    body = "accepted prompt " * (7000 if case != "small" else 1)
    original = StoredUserInput("text", StoredPayload.inline_text(body), {"version": 1, "extra": "kept"} if case == "view" else None)
    config = PersistenceStorage.sqlite(tmp_path / "input.db")
    async with Persistence.open("native-result", tenant_id="tenant", storage=config) as state:
        commands, commit, current, checkpoint = await _prepare_input(state, original, body if case != "different" else body + "modified")
        await _publish(commands, commit, current, checkpoint)
        repository = state.execution.executions
        record = await repository.state_store.read_record(repository.execution_record_key("execution"))
        wrapper = decode_envelope(record.data).value["payload"]["fields"]["stored_user_input"]
        assert (native_user_input_reference(wrapper) is not None) is (case == "large")
        assert (await repository.get_fresh("execution", tenant_id="tenant")).stored_user_input == original
    async with Persistence.open("native-result", tenant_id="tenant", storage=config, read_only=True) as state:
        assert (await state.execution.executions.get_fresh("execution", tenant_id="tenant")).stored_user_input == original


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("direct", "prepared"))
async def test_nonterminal_input_reference_survives_terminal_cas(tmp_path: Path, boundary: str) -> None:
    original = StoredUserInput("text", StoredPayload.inline_text("input " * 15000))
    async with Persistence.open("native-result", tenant_id="tenant", storage=PersistenceStorage.sqlite(tmp_path / "input.db")) as state:
        commands, commit, current, checkpoint = await _prepare_input(state, original, original.payload.decode())
        trace = state.run_store.read_store(PersistenceDomain.TRACE)
        run = AgentRunRecord("run")
        if boundary == "direct":
            await trace.sync_projection(run, events=(), checkpoints=(checkpoint,), execution_id="execution")
        else:
            await trace.prepare_run_registration(run)
            prepared = await trace.prepare_checkpoints(run, (checkpoint,))
            await trace.sync_prepared_projection(run, events=(), checkpoints=prepared.checkpoints,
                                                execution_id="execution", producer_generation=1)
        repository = state.execution.executions
        record = await repository.state_store.read_record(repository.execution_record_key("execution"))
        reference = native_user_input_reference(decode_envelope(record.data).value["payload"]["fields"]["stored_user_input"])
        assert reference is not None
        # Keep the admitted run identity, including its original started_at.
        # The generic helper creates a new run and is only for first publication.
        await commands.commit_terminal_checkpoint(commit, expected_execution=current,
            execution_run=run, execution_checkpoints=(checkpoint,))
        record = await repository.state_store.read_record(repository.execution_record_key("execution"))
        assert native_user_input_reference(decode_envelope(record.data).value["payload"]["fields"]["stored_user_input"]) == reference
        assert (await repository.get_fresh("execution", tenant_id="tenant")).stored_user_input == original


@pytest.mark.asyncio
async def test_native_input_and_output_use_one_range_batch_metadata_uses_none(tmp_path: Path) -> None:
    body = "input and output " * 9000
    async with Persistence.open("native-result", tenant_id="tenant", storage=PersistenceStorage.sqlite(tmp_path / "input.db")) as state:
        commands, commit, current, checkpoint = await _prepare_input(state, StoredUserInput("text", StoredPayload.inline_text(body)), body, body)
        await _publish(commands, commit, current, checkpoint)
        counts: Counter[str] = Counter()
        def sql(connection: object, cursor: object, statement: str, parameters: object,
                context: object, executemany: bool) -> None:
            counts[statement.split()[0]] += 1
        repository = state.execution.executions
        engine = repository.state_store.storage_group.context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", sql)
        try:
            loaded = await repository.get_fresh("execution", tenant_id="tenant")
            assert loaded.stored_user_input == current.stored_user_input and loaded.result == commit.result
            assert counts["WITH"] == 1
            counts.clear()
            await repository.get_metadata_record("execution", tenant_id="tenant")
            assert counts["SELECT"] == 1 and counts["WITH"] == 0
        finally:
            event.remove(engine, "before_cursor_execute", sql)


@pytest.mark.asyncio
async def test_native_input_owner_failure_rolls_back_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_input = StoredUserInput("text", StoredPayload.inline_text("atomic input " * 10000))
    async with Persistence.open("native-result", tenant_id="tenant", storage=PersistenceStorage.sqlite(tmp_path / "input.db")) as state:
        commands, commit, current, checkpoint = await _prepare_input(state, original_input, original_input.payload.decode())
        repository = state.execution.executions
        async def transaction_type(transaction: StateTransaction) -> type[StateTransaction]:
            return type(transaction)
        target = await repository.state_store.mutate(transaction_type)
        original = target.replace_record
        failed = 0
        async def fail_after_owner(transaction: StateTransaction, record: StoredRecord, *, expected_storage_revision: int) -> bool:
            nonlocal failed
            replaced = await original(transaction, record, expected_storage_revision=expected_storage_revision)
            if record.kind == "execution":
                wrapper = decode_envelope(record.data).value["payload"]["fields"]["stored_user_input"]
                if native_user_input_reference(wrapper) is not None:
                    assert replaced and await transaction.get_record(record.key) == record
                    failed += 1
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return replaced
        with monkeypatch.context() as patch:
            patch.setattr(target, "replace_record", fail_after_owner)
            with pytest.raises(AIError):
                await _publish(commands, commit, current, checkpoint)
        assert failed == 1
        loaded = await repository.get_fresh("execution", tenant_id="tenant")
        assert loaded.status == current.status and loaded.stored_user_input == original_input
        record = await repository.state_store.read_record(repository.execution_record_key("execution"))
        assert native_user_input_reference(decode_envelope(record.data).value["payload"]["fields"]["stored_user_input"]) is None
