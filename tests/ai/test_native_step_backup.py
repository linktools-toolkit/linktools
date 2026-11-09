#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict owner backups preserve append occurrences, tombstones, and retry state."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, InMemoryStepStore, RunRecord, StepEvent
from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.persistence import SqlStepStore
from linktools.ai.persistence._step_backup import NativeStepBackup
from linktools.ai.persistence._step_schema import build_step_sql_metadata
from linktools.ai.persistence.runs import NativeCheckpoint


class _Reader:
    def __init__(self, session):
        self.session = session

    async def read_sql_rows(self, context, statement):
        return tuple((await self.session.execute(statement)).mappings().all())


@pytest.mark.asyncio
async def test_raw_backup_preserves_occurrences_order_and_pruned_identities(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'native.db'}")
    store = SqlStepStore(engine, namespace="backup", tenant_id="tenant", max_snapshots_per_run=1)
    await store.initialize(local=True)
    try:
        await store.register_run(RunRecord(run_id="run"))
        event = StepEvent(run_id="run", kind="run_started", step_index=3)
        await store.append_event(event)
        await store.append_event(event)
        first = ContinuableSnapshot(run_id="run", step_index=7,
                                    messages=[ModelRequest(parts=[UserPromptPart("recover me")])], state="interrupted",
                                    idempotency_key="retired")
        await store.save_snapshot(NativeCheckpoint.encode(first, output_retries_used=1))
        latest = replace(first, step_index=2, state="complete", idempotency_key=None)
        await store.save_snapshot(NativeCheckpoint.encode(latest, output_retries_used=4))
        async with store.storage_context.sessions() as session, session.begin():
            backup = await store.capture_backup(_Reader(session))
        assert backup.snapshots[0]["pruned"] is True
        assert backup.snapshots[0]["messages"] == "{}"
        assert backup.events[0]["key_digest"] != backup.events[1]["key_digest"]
        assert [row["step_index"] for row in backup.snapshots] == [7, 2]
        wire = backup.to_json()
        assert NativeStepBackup.from_json(wire).to_json() == wire
        metadata = build_step_sql_metadata()
        async with store.storage_context.sessions() as session, session.begin():
            for name in ("runs", "events", "snapshots"):
                await session.execute(delete(metadata.tables[f"ai_step_{name}"]))
            await store.restore_backup(session, NativeStepBackup.from_json(wire))
        assert await store.list_events(run_id="run") == [event, event]
        restored = NativeCheckpoint.decode(await store.latest_snapshot(run_id="run"))
        assert restored.step_index == 2 and restored.output_retries_used == 4
        assert restored.messages == first.messages
        # A retired keyed save remains a replay tombstone after restore.
        await store.save_snapshot(NativeCheckpoint.encode(first, output_retries_used=1))
        async with store.storage_context.sessions() as session, session.begin():
            after = await store.capture_backup(_Reader(session))
        assert after.to_json() == wire
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ("descriptor", "retry_digest", "occurrence", "tombstone", "run_reference"))
async def test_raw_backup_rejects_corruption_instead_of_skipping_rows(tmp_path: Path, damage: str) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'native.db'}")
    store = SqlStepStore(engine, namespace="backup", tenant_id="tenant")
    await store.initialize(local=True)
    try:
        await store.register_run(RunRecord(run_id="run"))
        await store.append_event(StepEvent(run_id="run", kind="run_started", step_index=0))
        await store.save_snapshot(NativeCheckpoint.encode(
            ContinuableSnapshot(run_id="run", step_index=0, messages=[]), output_retries_used=3,
        ))
        async with engine.begin() as connection:
            if damage == "descriptor":
                await connection.execute(text("UPDATE ai_step_snapshots SET messages = '{}'"))
            elif damage == "retry_digest":
                value = json.loads(await connection.scalar(text("SELECT messages FROM ai_step_snapshots")))
                value["checkpoint"]["message_sha256"] = "0" * 64
                await connection.execute(text("UPDATE ai_step_snapshots SET messages = :value"),
                                         {"value": json.dumps(value)})
            elif damage == "occurrence":
                await connection.execute(text("UPDATE ai_step_events SET key_digest = 'invalid'"))
            elif damage == "tombstone":
                await connection.execute(text("UPDATE ai_step_snapshots SET pruned = 1"))
            else:
                await connection.execute(text("UPDATE ai_step_events SET run_fields = 1, conversation_id = NULL"))
        async with store.storage_context.sessions() as session, session.begin():
            with pytest.raises(ValueError):
                await store.capture_backup(_Reader(session))
    finally:
        await store.close()
        await engine.dispose()


def test_checkpoint_descriptor_rejects_unknown_state_and_invalid_retry_types() -> None:
    for descriptor in (
        {"output_retries_used": True, "message_sha256": "0" * 64},
        {"output_retries_used": -1, "message_sha256": "0" * 64},
        {"output_retries_used": 0, "message_sha256": "G" * 64},
        {"output_retries_used": 0, "message_sha256": "0" * 64, "pending": {}},
    ):
        with pytest.raises(ValueError):
            NativeCheckpoint.validate_descriptor(descriptor)


@pytest.mark.asyncio
async def test_external_effect_owner_is_rejected_before_any_query(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'unused.db'}")
    store = SqlStepStore(engine, namespace="backup", tenant_id="tenant", tool_effect_store=InMemoryStepStore())
    try:
        assert not store.backup_supported and store.backup_source is None
        with pytest.raises(ValueError, match="external effect"):
            await store.capture_backup(None)
    finally:
        await store.close()
        await engine.dispose()


def test_message_closure_enumerates_ordered_ranges_and_repeated_occurrences() -> None:
    from linktools.ai.persistence._archive import ImmutableSequenceNode
    from linktools.ai.persistence._archive_closure import NativeMessageClosure
    from linktools.ai.persistence._message_root import NativeMessageSlice

    scope = "0" * 64
    first = ImmutableSequenceNode("1" * 64)
    second = ImmutableSequenceNode("2" * 64)
    third = ImmutableSequenceNode("3" * 64, 3, second.identity, first.identity)
    fourth = ImmutableSequenceNode("4" * 64, next_forest=third.identity)
    rows = [{"store_digest": scope, "sha256": node.identity, "message_sha256": node.content_id,
             "tree_count": node.tree_count, "left_sha256": node.left, "right_sha256": node.right,
             "next_sha256": node.next_forest} for node in (first, second, third, fourth)]
    closure = NativeMessageClosure(rows)
    assert closure.message_identities(scope, (NativeMessageSlice(fourth.identity, 4, 0, 4),)) == (
        "1" * 64, "2" * 64, "3" * 64, "4" * 64,
    )
    assert closure.message_identities(scope, (
        NativeMessageSlice(fourth.identity, 4, 1, 3), NativeMessageSlice(third.identity, 3, 0, 3),
    )) == ("2" * 64, "3" * 64, "1" * 64, "2" * 64, "3" * 64)
    assert closure.message_identities(scope, ()) == ()


def test_backup_snapshot_decodes_retry_binding_without_store_or_database() -> None:
    import hashlib
    from linktools.ai.model.messages import encode_model_messages

    snapshot = ContinuableSnapshot(run_id="run", step_index=2, messages=[], idempotency_key="checkpoint")
    encoded = NativeCheckpoint.encode(snapshot, output_retries_used=3)
    row = {"run_id": snapshot.run_id, "step_index": snapshot.step_index,
           "timestamp": snapshot.timestamp.isoformat(), "state": snapshot.state,
           "idempotency_key": snapshot.idempotency_key, "sequence_root": None,
           "messages": json.dumps({"stored_messages": 0, "checkpoint": {
               "output_retries_used": 3,
               "message_sha256": hashlib.sha256(encode_model_messages(encoded.messages[-1:])).hexdigest(),
           }})}
    original = dict(row)
    assert SqlStepStore.decode_backup_snapshot(row, ()) == encoded
    assert row == original
    with pytest.raises(ValueError, match="corrupt or misbound"):
        SqlStepStore.decode_backup_snapshot(dict(row, step_index=3), ())
