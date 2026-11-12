#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Atomic Native text owners preserve exact terminal results and bounded reads."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import event, text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from linktools.ai.core import ExecutionEventType, ExecutionStatus, StopReason, UsageMetrics
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage, SqlStepStore
from linktools.ai.persistence._runtime_commands import StateCommands
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint, AgentRunRecord
from linktools.ai.persistence.codec import decode_envelope, execution_payload_slots, native_text_reference, validate_native_text_payload
from linktools.ai.persistence.contracts import ExecutionEventAppend, ExecutionRecord, ExecutionTerminalCommit, ExecutionTerminalCommitResult, ResultRecord
from linktools.ai.persistence.state import StateTransaction, StoredRecord
from linktools.ai.persistence._message_root import NativeTextReference
from linktools.ai.storage import StoredPayload

from .test_step_projection import _execution


async def _prepare(state: Persistence, body: str, output: str) -> tuple[StateCommands, ExecutionTerminalCommit, ExecutionRecord, AgentRunCheckpoint]:
    repository = state.execution.executions
    native = SqlStepStore.for_state_store(repository.state_store, namespace="native-result", tenant_id="tenant")
    assert native is not None
    state.run_store.bind_step_store(native)
    current = replace(_execution(), revision=1, producer_generation=1)
    await repository.create_execution(current)
    checkpoint = AgentRunCheckpoint("run", 1, [ModelResponse(parts=[TextPart(body)])],
        transcript_message_count_before=0, timestamp=current.updated_at)
    commands = StateCommands(repository, namespace="native-result", events=state.execution.events,
        execution_run_store=state.run_store.read_store(PersistenceDomain.TRACE), background_tasks=set())
    result = ResultRecord(StoredPayload.inline_json(output), StopReason.END_TURN, UsageMetrics(), current.updated_at)
    commit = ExecutionTerminalCommit(current.revision, current.event_seq,
        replace(current, status=ExecutionStatus.SUCCEEDED), result, ExecutionEventType.EXECUTION_SUCCEEDED,
        {"agent_run_id": "run"}, producer_generation=1)
    return commands, commit, current, checkpoint


async def _publish(
    commands: StateCommands, commit: ExecutionTerminalCommit, current: ExecutionRecord, checkpoint: AgentRunCheckpoint,
) -> ExecutionTerminalCommitResult:
    return await commands.commit_terminal_checkpoint(commit, expected_execution=current,
        execution_run=AgentRunRecord("run"), execution_checkpoints=(checkpoint,))


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ("large_equal", "small_equal", "different"))
async def test_terminal_text_reuses_only_equal_published_message(tmp_path: Path, case: str) -> None:
    body = "large text " * 10000 if case != "small_equal" else "small"
    output = body if case != "different" else body + " transformed"
    config = PersistenceStorage.sqlite(tmp_path / "results.db")
    async with Persistence.open("native-result", tenant_id="tenant", storage=config) as state:
        commands, commit, current, checkpoint = await _prepare(state, body, output)
        assert (await _publish(commands, commit, current, checkpoint)).result == commit.result
        repository = state.execution.executions
        raw = await repository.state_store.read_record(repository.execution_record_key("execution"))
        references = [native_text_reference(payload) for _, payload in execution_payload_slots(decode_envelope(raw.data).value["payload"])]
        assert any(references) is (case in {"large_equal", "small_equal"})
        counts: Counter[str] = Counter()

        def sql(_connection: object, _cursor: object, statement: str, _parameters: object, _context: object, _many: bool) -> None:
            counts["SELECT" if statement.startswith("WITH") else statement.split()[0]] += 1

        engine = repository.state_store.storage_group.context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", sql)
        try:
            actual = await repository.get_result_record("execution", tenant_id="tenant", fresh=True)
            assert actual.result == commit.result
            assert counts["SELECT"] == (2 if any(references) else 1)
        finally:
            event.remove(engine, "before_cursor_execute", sql)
    # A genuinely cold owner reopens its archive itself, without fake warming.
    async with Persistence.open("native-result", tenant_id="tenant", storage=config, read_only=True) as cold:
        assert (await cold.execution.executions.get_result_record("execution", tenant_id="tenant", fresh=True)).result == commit.result
        assert (await cold.execution.executions.get("execution", tenant_id="tenant")).result == commit.result
        assert (await cold.execution.executions.get_timeline_many(("execution",), tenant_id="tenant"))["execution"].result == commit.result


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", (False, True))
async def test_terminal_writer_rejects_unverified_native_source(tmp_path: Path, missing: bool) -> None:
    async with Persistence.open("native-result", tenant_id="tenant", storage=PersistenceStorage.sqlite(tmp_path / "source.db")) as state:
        commands, commit, current, checkpoint = await _prepare(state, "source " * 12000, "source " * 12000)
        await _publish(commands, commit, current, checkpoint)
        repository = state.execution.executions
        raw = await repository.state_store.read_record(repository.execution_record_key("execution"))
        reference = next(reference for _, payload in execution_payload_slots(decode_envelope(raw.data).value["payload"])
                         if (reference := native_text_reference(payload)) is not None)
        other = replace(current, execution_id="other", root_execution_id="other")
        await repository.create_execution(other)
        output = StoredPayload.inline_json("different " * 12000)
        forged = reference
        if missing:
            forged = replace(forged, root=replace(forged.root,
                slices=(replace(forged.root.slices[0], first="f" * 64),)))
        candidate = replace(commit, execution=replace(other, status=ExecutionStatus.SUCCEEDED),
                            result=replace(commit.result, output=output))
        with pytest.raises(AIError):
            await repository.state_store.mutate(lambda transaction: repository.commit_terminal_in_transaction(
                transaction, candidate, result_source=forged))
        assert (await repository.get("other", tenant_id="tenant")).status is ExecutionStatus.STARTED


@pytest.mark.asyncio
async def test_native_message_and_result_owner_rollback_together(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with Persistence.open("native-result", tenant_id="tenant", storage=PersistenceStorage.sqlite(tmp_path / "atomic.db")) as state:
        commands, commit, current, checkpoint = await _prepare(state, "atomic " * 12000, "atomic " * 12000)
        repository = state.execution.executions
        original = repository.commit_terminal_in_transaction
        failed = 0

        async def abort(
            transaction: StateTransaction, candidate: ExecutionTerminalCommit, *,
            pending_events: Sequence[ExecutionEventAppend] = (), result_source: NativeTextReference | None = None,
        ) -> ExecutionTerminalCommitResult:
            nonlocal failed
            result = await original(transaction, candidate, pending_events=pending_events, result_source=result_source)
            assert result.execution.status is ExecutionStatus.SUCCEEDED
            async def count_messages(session: AsyncSession) -> int:
                return (await session.execute(sql_text("SELECT count(*) FROM ai_step_messages"))).scalar_one()
            assert await transaction.run_sql(repository.state_store.storage_group.context, count_messages) == 1
            failed += 1
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        monkeypatch.setattr(repository, "commit_terminal_in_transaction", abort)
        with pytest.raises(AIError):
            await _publish(commands, commit, current, checkpoint)
        assert failed == 1
        assert (await repository.get("execution", tenant_id="tenant")).status is ExecutionStatus.STARTED
        async with repository.state_store.storage_group.context.engine.connect() as connection:
            assert (await connection.execute(sql_text("SELECT count(*) FROM ai_step_messages"))).scalar_one() == 0


def test_native_payload_mapping_rejects_digest_equal_different_encoding() -> None:
    from .test_native_text_snapshot_closure import _projection
    _, _, reference, payload, _, _ = _projection()
    impostor = StoredPayload.inline_text('"' + payload.decode() + '"')
    assert (impostor.digest, impostor.size) == (payload.digest, payload.size)
    with pytest.raises(AIError):
        validate_native_text_payload(reference, impostor)


@pytest.mark.asyncio
async def test_native_slice_budget_checks_selected_bodies_before_decompression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord
    from linktools.ai.persistence._archive import ContentArchive, PreparedArchive
    from linktools.ai.persistence import _step_message_content

    async with Persistence.open("native-result", tenant_id="tenant", storage=PersistenceStorage.sqlite(tmp_path / "budget.db")) as state:
        store = state.execution.executions.state_store
        native = SqlStepStore.for_state_store(store, namespace="native-result", tenant_id="tenant")
        assert native is not None
        scope, content_store = await store.storage_group.open_archive(create=True)
        native.bind_message_archive(ContentArchive(content_store, scope, namespace="native-result", tenant_id="tenant", object_store=None))
        await native.register_run(RunRecord(run_id="run"))
        messages = [ModelResponse(parts=[TextPart(str(index) * 20000)]) for index in range(3)]
        await native.save_snapshot(ContinuableSnapshot(run_id="run", step_index=1, messages=messages))
        root = await native.prepare_archived_messages(messages, PreparedArchive())
        selected = root.slice(0, 1)
        reserved = []
        values, = await (await native.capture_message_roots((selected,), reserve=reserved.append))()
        assert len(values) == 1 and reserved == [len(values[0])]

        def no_decode(*args: object, **kwargs: object) -> bytes:
            raise AssertionError("over-budget body must not be decompressed")

        def reject(size: int) -> None:
            assert size == len(values[0])
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)

        monkeypatch.setattr(_step_message_content, "decode_payload_bytes", no_decode)
        with pytest.raises(AIError) as over_budget:
            await native.capture_message_roots((selected,), reserve=reject)
        assert over_budget.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED


@pytest.mark.asyncio
async def test_native_inspection_discards_observed_owner_before_atomic_hydration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.persistence.state import active_state_scope

    async with Persistence.open("native-result", tenant_id="tenant",
                                storage=PersistenceStorage.sqlite(tmp_path / "observation.db")) as state:
        commands, commit, current, checkpoint = await _prepare(state, "native " * 12000, "native " * 12000)
        await _publish(commands, commit, current, checkpoint)
        repository = state.execution.executions
        store = repository.state_store
        original = store.read_record
        observations = 0

        async def remove_after_observation(key: bytes) -> StoredRecord | None:
            nonlocal observations
            record = await original(key)
            if active_state_scope() is None:
                observations += 1
                assert any(native_text_reference(payload) is not None
                           for _, payload in execution_payload_slots(decode_envelope(record.data).value["payload"]))
                await store.mutate(lambda tx: tx.delete_record(key))
            return record

        monkeypatch.setattr(store, "read_record", remove_after_observation)
        # The cached/ordinary owner only selects the atomic path. Deletion before
        # that snapshot must yield absence, never hydrate the obsolete owner.
        assert await repository.get("execution", tenant_id="tenant") is None
        assert observations == 1
