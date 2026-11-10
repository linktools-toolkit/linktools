#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verified context prefixes survive new runs and cold snapshot recovery."""

from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord, StepPersistence
from sqlalchemy import event, select

from linktools.ai.capability import ExecutionCapability, TraceCapability
from linktools.ai.core import CommitPolicy
from linktools.ai.model.messages import encode_model_messages
from linktools.ai.persistence import NativeCheckpoint, Persistence, PersistenceStorage, SqlStepStore
from linktools.ai.persistence.contracts import LoadedContextMessage, LoadedModelContext
from linktools.ai.persistence._step_schema import build_step_sql_metadata
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


_STAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _NativeWrites:
    def __init__(self) -> None:
        self.identities: dict[str, list[str]] = defaultdict(list)

    def capture(self, _connection, _cursor, statement, parameters, _context, many) -> None:
        for table, width in (("ai_step_messages", 3), ("ai_step_sequences", 7)):
            if statement.startswith("INSERT INTO " + table + " "):
                for values in parameters if many else (parameters,):
                    self.identities[table].extend(values[1::width])


async def _identities(engine) -> dict[str, set[str]]:
    tables = build_step_sql_metadata().tables
    async with engine.connect() as connection:
        return {name: set((await connection.execute(select(tables[name].c.sha256))).scalars())
                for name in ("ai_step_messages", "ai_step_sequences")}


def _messages(count: int) -> list[ModelRequest]:
    return [ModelRequest(parts=[UserPromptPart(str(index), timestamp=_STAMP)], timestamp=_STAMP)
            for index in range(count)]


@pytest.mark.asyncio
async def test_session_new_runs_reuse_history_beyond_cache_and_after_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, namespace = tmp_path / "session.db", "snapshot-session-baseline"
    executions, run_ids, expected = set(), set(), {}
    transcript: list[str] = []
    prepare_snapshot = SqlStepStore._prepare_snapshot

    async def capture_snapshot(store, snapshot):
        expected[snapshot.run_id] = encode_model_messages(NativeCheckpoint.decode(snapshot).messages)
        return await prepare_snapshot(store, snapshot)

    monkeypatch.setattr(SqlStepStore, "_prepare_snapshot", capture_snapshot)

    async def model(messages, _info):
        return ModelResponse(parts=[TextPart(f"answer-{len(messages)}")], timestamp=_STAMP)

    for phase, turns in enumerate((3, 1)):
        async with Persistence.open(namespace, storage=PersistenceStorage.sqlite(path)) as state:
            native = await state.create_step_store(max_cached_messages=2)
            features = RuntimeFeatures(namespace)
            features.capability(StepPersistence(store=native, capture_frontier=True), scope="runtime")
            features.capability(ExecutionCapability(retain_history=True), scope="runtime")
            features.capability(TraceCapability(commit=CommitPolicy.sync()), scope="runtime")
            features.agent("default", model="default", allow_tools=())
            try:
                async with Runtime.attach(persistence=state,
                        features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
                    session = (await runtime.agents.get().create_session("session") if phase == 0
                               else await runtime.sessions.get("session"))
                    for turn in range(turns):
                        existing = await _identities(native.storage_context.engine)
                        writes = _NativeWrites()
                        event.listen(native.storage_context.engine.sync_engine, "before_cursor_execute", writes.capture)
                        try:
                            question = f"question-{phase}-{turn}"
                            execution = await session.start(question)
                            assert (await execution.wait(timeout_seconds=10)).result.status.value == "SUCCEEDED"
                        finally:
                            event.remove(native.storage_context.engine.sync_engine, "before_cursor_execute", writes.capture)
                        assert execution.execution_id not in executions
                        executions.add(execution.execution_id)
                        runs = await native.list_runs()
                        added = {run.run_id for run in runs} - run_ids
                        assert len(added) == 1
                        run_id, = added
                        run_ids.add(run_id)
                        snapshot = await native.latest_snapshot(run_id=run_id)
                        assert snapshot is not None
                        restored = NativeCheckpoint.decode(snapshot)
                        assert encode_model_messages(restored.messages) == expected[run_id]
                        transcript.extend((question, f"answer-{2 * len(run_ids) - 1}"))
                        assert [part.content for message in restored.messages for part in message.parts
                                if isinstance(part, (UserPromptPart, TextPart))] == transcript
                        for table in existing:
                            repeated = existing[table].intersection(writes.identities[table])
                            assert not repeated, (phase, turn, table, repeated)
                        if len(run_ids) > 1:
                            assert len(restored.messages) > 2
                    assert len((await session.timeline()).items) == len(executions)
            finally:
                await native.close()
    assert len(executions) == len(run_ids) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("new_attempt", (False, True))
async def test_cold_snapshot_readback_seeds_first_continuation_without_old_rows(
    tmp_path: Path, new_attempt: bool,
) -> None:
    path, namespace = tmp_path / "recovery.db", "snapshot-readback-baseline"
    original = ContinuableSnapshot(run_id="source", step_index=8, messages=_messages(8), idempotency_key="original")
    source = RunRecord(run_id="source", metadata={"linktools.transcript_start": "0"})
    async with Persistence.open(namespace, storage=PersistenceStorage.sqlite(path)) as state:
        native = await state.create_step_store(max_cached_messages=2)
        await native.register_run(source)
        await native.save_snapshot(original)
        await native.close()

    async with Persistence.open(namespace, storage=PersistenceStorage.sqlite(path)) as state:
        native = await state.create_step_store(max_cached_messages=2)
        state.run_store.bind_step_store(native)
        try:
            assert await native.latest_snapshot(run_id="source") == original
            assert not native._message_content._known_messages and not native._message_content._known_nodes
            run_id = "resumed" if new_attempt else "source"
            if new_attempt:
                await native.register_run(replace(source, run_id=run_id))
                state.run_store.register_context_baseline(run_id, LoadedModelContext(tuple(
                    LoadedContextMessage(message, None) for message in original.messages
                )), source_run=source)
            continued = replace(original, run_id=run_id, step_index=9, messages=_messages(9), idempotency_key="continued")
            writes = _NativeWrites()
            event.listen(native.storage_context.engine.sync_engine, "before_cursor_execute", writes.capture)
            try:
                await native.save_snapshot(continued)
            finally:
                event.remove(native.storage_context.engine.sync_engine, "before_cursor_execute", writes.capture)
            assert {name: len(values) for name, values in writes.identities.items()} == {
                "ai_step_messages": 1, "ai_step_sequences": 1,
            }
            assert await native.latest_snapshot(run_id=run_id) == continued
            counts = await _identities(native.storage_context.engine)
            assert {name: len(values) for name, values in counts.items()} == {
                "ai_step_messages": 9, "ai_step_sequences": 9,
            }
        finally:
            await native.close()


@pytest.mark.asyncio
async def test_snapshot_baseline_rejects_uncommitted_foreign_and_invalidated_roots(tmp_path: Path) -> None:
    from linktools.ai.persistence._archive import PreparedArchive
    from linktools.ai.persistence._transcript_staging import MessageArchive

    async with Persistence.open("snapshot-baseline-proof", storage=PersistenceStorage.sqlite(tmp_path / "proof.db")) as state:
        native = await state.create_step_store(max_cached_messages=2)
        try:
            bundle = PreparedArchive()
            messages = _messages(4)
            root = await native.prepare_archived_messages(messages, bundle)
            digest, _size = MessageArchive().snapshot(messages).encoded_digest()
            assert not native.seed_snapshot_baseline("new", root, raw_digest=digest)
            assert not native.inherit_snapshot_baseline("new", source_run_id="missing")
            await native.save_snapshot(ContinuableSnapshot(run_id="source", step_index=4, messages=messages))
            assert native.seed_snapshot_baseline("new", root, raw_digest=digest)
            assert not native.seed_snapshot_baseline("foreign", replace(root, group_id="foreign"), raw_digest=digest)
            assert not native.seed_snapshot_baseline("changed", root, raw_digest="0" * 64)
            native.invalidate_message_content()
            assert not native.seed_snapshot_baseline("stale", root, raw_digest=digest)
            assert not native.inherit_snapshot_baseline("stale", source_run_id="source")
        finally:
            await native.close()
