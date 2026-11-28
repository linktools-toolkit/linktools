#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verified context prefixes survive new runs and cold snapshot recovery."""

from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord
from sqlalchemy import event, select
from sqlalchemy.dialects import mysql

from linktools.ai.capability import SessionCapability
from linktools.ai.core import TenantAuthorizationPolicy
from linktools.ai.model.messages import encode_model_messages
from linktools.ai.persistence import NativeCheckpoint, Persistence, PersistenceStorage, SqlStepStore
from linktools.ai.persistence.contracts import LoadedContextMessage, LoadedModelContext
from linktools.ai.persistence._step_schema import build_step_sql_metadata
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.steps import native_store_locator

from ._runtime_test_helpers import configure_execution
from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from ._sqlite_insert_measurement import bound_insert_rows


_STAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _HostAuthorization(TenantAuthorizationPolicy):
    """Explicit SQL session policy for the test actor."""


class _NativeWrites:
    def __init__(self) -> None:
        self.identities: dict[str, list[str]] = defaultdict(list)
        self.selects = 0
        self.mysql_prefix_sql = None

    def capture(self, _connection, _cursor, statement, parameters, _context, many) -> None:
        if statement.lstrip().upper().startswith(("SELECT ", "WITH ")):
            self.selects += 1
            if "prior_complete" in statement:
                dialect = mysql.dialect()
                dialect.server_version_info = (5, 7, 44)
                self.mysql_prefix_sql = str(_context.compiled.statement.compile(dialect=dialect))
        for table in ("ai_messages", "ai_message_sequences"):
            if statement.startswith("INSERT INTO " + table + " "):
                self.identities[table].extend(
                    row["sha256"] for row in bound_insert_rows(statement, parameters, many)
                )


async def _identities(engine) -> dict[str, set[str]]:
    tables = build_step_sql_metadata().tables
    async with engine.connect() as connection:
        return {name: set((await connection.execute(select(tables[name].c.sha256))).scalars())
                for name in ("ai_messages", "ai_message_sequences")}


def _messages(count: int) -> list[ModelRequest]:
    return [ModelRequest(parts=[UserPromptPart(str(index), timestamp=_STAMP)], timestamp=_STAMP)
            for index in range(count)]


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_source", ("complete", "foreign_session", "foreign_tenant", "pruned", "interrupted", "invalid_descriptor", "cache_disabled"))
async def test_session_new_runs_reuse_history_beyond_cache_and_after_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prior_source: str,
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
            native = await state.create_step_store(max_cached_messages=0 if prior_source == "cache_disabled" else 2)

            async def open_bounded_steps():
                return native

            monkeypatch.setattr(state, "create_step_store", open_bounded_steps)
            features = RuntimeFeatures(namespace)
            features.capability(SessionCapability(checkpoints=True), scope="runtime")
            configure_execution(features, retain_history=True)
            features.agent("default", model="default", allow_tools=())
            try:
                async with Runtime.open(persistence=state,
                        features=model_features(_Models(_UsageFunctionModel(model)), features),
                        authorization=_HostAuthorization()) as runtime:
                    session = (await runtime.agents.get().create_session("session") if phase == 0
                               else await runtime.sessions.get("session"))
                    if phase and prior_source not in {"complete", "cache_disabled"}:
                        foreign = SqlStepStore.scope_key(namespace=namespace, tenant_id="other-tenant")
                        snapshots = build_step_sql_metadata().tables["ai_step_snapshots"]
                        change = {
                            "foreign_session": {"conversation_id": "other-session"},
                            "foreign_tenant": {"store_key": foreign, "store_locator": native_store_locator(foreign)},
                            "invalid_descriptor": {"messages": "{}"},
                            "pruned": {"pruned": True, "messages": "{}", "sequence_root": None},
                            "interrupted": {"state": "interrupted"},
                        }[prior_source]
                        async with native.storage_context.engine.begin() as connection:
                            await connection.execute(snapshots.update().where(snapshots.c.run_id == run_id).values(**change))
                        candidates = await state.session_turns.read_context_range(
                            session.session_id, after=0, through=3, checkpoint=native,
                        )
                        assert all(item.seed_checkpoint_prefix is None for item in candidates)
                    captured_runs = []
                    for turn in range(turns):
                        existing = await _identities(native.storage_context.engine)
                        writes = _NativeWrites()
                        event.listen(native.storage_context.engine.sync_engine, "before_cursor_execute", writes.capture)
                        try:
                            question = f"question-{phase}-{turn}"
                            turn_handle = await runtime.sessions.start(
                                session.session_id, question, agent="default", principal=runtime.default_principal,
                                idempotency_key=question,
                            )
                            assert (await turn_handle.wait_final(timeout_seconds=10)).status.value == "SUCCEEDED"
                        finally:
                            event.remove(native.storage_context.engine.sync_engine, "before_cursor_execute", writes.capture)
                        assert turn_handle.key.turn_id not in executions
                        executions.add(turn_handle.key.turn_id)
                        runs = await native.list_runs()
                        added = {run.run_id for run in runs} - run_ids
                        assert len(added) == 1
                        run_id, = added
                        run_ids.add(run_id)
                        captured_runs.append((run_id, question))
                        if prior_source != "cache_disabled" and (not phase or prior_source == "complete"):
                            for table in existing:
                                repeated = existing[table].intersection(writes.identities[table])
                                assert not repeated, (phase, turn, table, repeated)
                        if phase:
                            assert writes.selects == 10
                            if prior_source == "cache_disabled":
                                assert writes.mysql_prefix_sql is None
                                continue
                            assert "prior_complete.run_id = CAST(ai_session_turns.turn_id AS CHAR)" in writes.mysql_prefix_sql
                            assert "CAST(prior_complete.run_id AS BINARY) = ai_session_turns.turn_id" in writes.mysql_prefix_sql
                            assert "CAST(prior_complete.conversation_id AS BINARY) = ai_session_turns.session_key" in writes.mysql_prefix_sql
                        else:
                            assert writes.selects == 8 and writes.mysql_prefix_sql is None
                    for run_id, question in captured_runs:
                        snapshot = await native.latest_snapshot(run_id=run_id)
                        assert snapshot is not None
                        restored = NativeCheckpoint.decode(snapshot)
                        assert encode_model_messages(restored.messages) == expected[run_id]
                        transcript.extend((question, f"answer-{len(transcript) + 1}"))
                        assert [part.content for message in restored.messages for part in message.parts
                                if isinstance(part, (UserPromptPart, TextPart))] == transcript
                        assert len(restored.messages) == len(transcript)
                    assert len((await session.timeline())["items"]) == len(executions)
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
                "ai_messages": 1, "ai_message_sequences": 1,
            }
            assert await native.latest_snapshot(run_id=run_id) == continued
            counts = await _identities(native.storage_context.engine)
            assert {name: len(values) for name, values in counts.items()} == {
                "ai_messages": 9, "ai_message_sequences": 9,
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
