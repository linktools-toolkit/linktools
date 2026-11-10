#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native restore reads shared prefix bodies linearly and still fails closed."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.errors import AIError
from linktools.ai.persistence import SqlStepStore
from linktools.ai.persistence._archive_closure import NativeMessageClosure
from linktools.ai.persistence._step_backup import NativeStepBackup
from linktools.ai.persistence._step_schema import build_step_sql_metadata
from linktools.ai.persistence.runs import NativeCheckpoint


class _Reader:
    def __init__(self, session):
        self.session = session

    async def read_sql_rows(self, context, statement):
        return tuple((await self.session.execute(statement)).mappings().all())


class _BodyRows:
    """Observe result volume, including repeated bodies in joined range queries."""

    def __init__(self, session):
        self.session = session
        self.count = 0

    async def execute(self, *args, **kwargs):
        result = await self.session.execute(*args, **kwargs)
        return _Result(result, self)


class _Result:
    def __init__(self, result, observer):
        self.result = result
        self.observer = observer

    def mappings(self):
        return _Mappings(self.result.mappings(), self.observer)

    def first(self):
        return self.result.first()


class _Mappings:
    def __init__(self, result, observer):
        self.result = result
        self.observer = observer

    def all(self):
        rows = self.result.all()
        self.observer.count += sum("message" in row and row["message"] is not None for row in rows)
        return rows


async def _prefix_backup(store, count):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    messages = [ModelRequest(parts=[UserPromptPart(f"message-{index}")], timestamp=now)
                for index in range(count + 1)]
    await store.register_run(RunRecord(run_id="run", started_at=now))
    expected = []
    # The first body is never a checkpoint tail, so validating only tails must
    # not silently omit its integrity check.
    for length in range(2, count + 2):
        snapshot = ContinuableSnapshot(run_id="run", step_index=count + 2 - length,
            messages=messages[:length], timestamp=now, idempotency_key=f"snapshot-{length}")
        encoded = NativeCheckpoint.encode(snapshot, output_retries_used=length % 4)
        await store.save_snapshot(encoded)
        expected.append(encoded)
    async with store.storage_context.sessions() as session, session.begin():
        backup = await store.capture_backup(_Reader(session), validate_messages=False)
    return backup, expected


async def _empty_owner(session, metadata):
    for name in ("runs", "events", "snapshots"):
        await session.execute(delete(metadata.tables[f"ai_step_{name}"]))


@pytest.mark.asyncio
@pytest.mark.parametrize("count", (16, 64))
async def test_restore_shared_prefixes_reads_linear_body_volume_and_preserves_snapshots(tmp_path: Path, count: int) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'native.db'}")
    store = SqlStepStore(engine, namespace="restore-cost", tenant_id="tenant")
    await store.initialize(local=True)
    try:
        backup, expected = await _prefix_backup(store, count)
        metadata = build_step_sql_metadata()
        async with store.storage_context.sessions() as session, session.begin():
            await _empty_owner(session, metadata)
            observed = _BodyRows(session)
            await store.restore_backup(observed, backup)
        # This is a result-volume bound, not a fixed SELECT count: a bounded
        # number of passes over unique bodies is allowed, replaying prefixes is not.
        assert count + 1 <= observed.count <= 2 * (count + 1)
        actual = await store.list_snapshots(run_id="run", include_interrupted=True)
        assert actual == expected
        assert [NativeCheckpoint.decode(item).output_retries_used for item in actual] == [
            length % 4 for length in range(2, count + 2)
        ]
        async with store.storage_context.sessions() as session, session.begin():
            restored = await store.capture_backup(_Reader(session), validate_messages=False)
        assert restored.to_json() == backup.to_json()
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ("early_body", "missing_node", "header_digest"))
async def test_restore_rejects_corrupt_prefix_content_nodes_and_checkpoint_headers(tmp_path: Path, damage: str) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'native.db'}")
    store = SqlStepStore(engine, namespace="restore-cost", tenant_id="tenant")
    await store.initialize(local=True)
    try:
        backup, _ = await _prefix_backup(store, 8)
        metadata = build_step_sql_metadata()
        nodes, bodies = metadata.tables["ai_step_sequences"], metadata.tables["ai_step_messages"]
        async with store.storage_context.sessions() as session, session.begin():
            await _empty_owner(session, metadata)
            if damage == "early_body":
                rows = (await session.execute(select(nodes))).mappings().all()
                closure = NativeMessageClosure(rows)
                first = closure.message_identities(store.store_digest, backup.message_sequences[0].slices)[0]
                await session.execute(update(bodies).where(bodies.c.sha256 == first).values(message="{}"))
            elif damage == "missing_node":
                await session.execute(delete(nodes).where(nodes.c.sha256 == backup.snapshots[0]["sequence_root"]))
            else:
                wire = backup.to_json()
                descriptor = json.loads(wire["snapshots"][0]["messages"])
                descriptor["checkpoint"]["message_sha256"] = "0" * 64
                wire["snapshots"][0]["messages"] = json.dumps(descriptor)
                backup = NativeStepBackup.from_json(wire)
        async with store.storage_context.sessions() as session, session.begin():
            with pytest.raises((AIError, ValueError)):
                await store.restore_backup(session, backup)
        async with store.storage_context.sessions() as session:
            for name in ("runs", "events", "snapshots"):
                table = metadata.tables[f"ai_step_{name}"]
                assert (await session.execute(select(table.c.id))).first() is None
    finally:
        await store.close()
        await engine.dispose()
