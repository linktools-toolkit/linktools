#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL run counters live only in their admission and preserve portable sequences."""

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import text

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence._run_counters import run_counter_streams
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.persistence.state import StateTransaction, stream_key
from linktools.ai.storage import InMemoryObjectStore

from .test_memory_run_counter_owner import _admission
from .test_state_batch_delete_backends import _open_store
from .test_native_text_result_runtime import _prepare, _publish


@pytest.mark.asyncio
async def test_sql_admission_counters_cas_rollback_delete_and_global_pages(tmp_path: Path) -> None:
    async with _open_store("sql", tmp_path) as store:
        record = _admission()
        event, checkpoint, interaction = run_counter_streams(record.key)
        await store.mutate(lambda tx: tx.insert_record(record))
        async def write(tx: StateTransaction) -> None:
            assert await tx.reserve_sequences({event: 2, checkpoint: 3, b"opaque": 4}) == {event: 2, checkpoint: 3, b"opaque": 4}
            assert await tx.replace_record(replace(record, storage_revision=1, data={"run": "changed"}), expected_storage_revision=0)
            assert await tx.get_sequence(event) == 2
            assert await tx.advance_sequence(event, 2) == 3
        await store.mutate(write)
        missing = stream_key("run-counter", "tenant", "trace", "event", ("missing",))
        with pytest.raises(AIError) as error:
            await store.mutate(lambda tx: tx.reserve_sequences({event: 4, missing: 1, b"opaque": 2}))
        assert error.value.code is ErrorCode.STORAGE_CONFLICT
        values = {event: 3, checkpoint: 3, b"opaque": 4}
        assert await store.read(lambda tx: tx.scan_sequences()) == dict(sorted(values.items()))
        pages = {}
        after = None
        while page := await store.read(lambda tx: tx.scan_sequences_page(after=after, limit=1)):
            pages.update(page)
            after = next(iter(page))
        assert pages == dict(sorted(values.items()))
        async with store.context.engine.connect() as connection:
            assert await connection.scalar(text("SELECT count(*) FROM ai_state_counters")) == 1
            row = (await connection.execute(text("SELECT event_high_water,checkpoint_high_water,interaction_high_water FROM ai_step_run_admissions"))).one()
            assert tuple(row) == (3, 3, None)
        async def recreate(tx: StateTransaction) -> None:
            assert await tx.get_sequence(event) == 3
            await tx.delete_record(record.key)
            assert await tx.get_sequence(event) == 0
            await tx.delete_sequences((event, checkpoint, interaction))
            await tx.insert_record(record)
            assert await tx.reserve_sequence(interaction, 1) == 1
        await store.mutate(recreate)
        await store.mutate(lambda tx: tx.delete_sequence(interaction))
        assert interaction not in await store.read(lambda tx: tx.scan_sequences())
        with pytest.raises(AIError) as error:
            await store.mutate(lambda tx: tx.advance_sequence(interaction, 0))
        assert error.value.code is ErrorCode.STORAGE_CONFLICT


@pytest.mark.asyncio
async def test_runtime_admission_counter_snapshot_restores_unique_owner(tmp_path: Path) -> None:
    objects = InMemoryObjectStore("run-control")
    limits = SnapshotLimits(10000, 20_000_000)
    config = PersistenceStorage.sqlite(tmp_path / "source.db")
    async with Persistence.open("native-result", tenant_id="tenant", storage=config) as state:
        commands, commit, current, checkpoint = await _prepare(state, "output", "output")
        await _publish(commands, commit, current, checkpoint)
        store = state.run_store.read_store(PersistenceDomain.TRACE).state_store
        before = await store.read(lambda tx: tx.scan_sequences())
        assert before
    async with Persistence.open("native-result", tenant_id="tenant", storage=config, read_only=True) as state:
        snapshot = await state.export_snapshot(object_store=objects, limits=limits)
    root = tmp_path / "restored"
    await Persistence.restore_snapshot(snapshot, object_store=objects, root=root, limits=limits)
    async with Persistence.open("native-result", tenant_id="tenant", storage=PersistenceStorage.from_root(root)) as state:
        store = state.run_store.read_store(PersistenceDomain.TRACE).state_store
        assert await store.read(lambda tx: tx.scan_sequences()) == before
        async with store.storage_group.context.engine.connect() as connection:
            rows = (await connection.execute(text("SELECT event_high_water,checkpoint_high_water FROM ai_step_run_admissions WHERE store_key=:store"), {"store": store.store_key})).all()
            assert rows and any(row[0] is not None or row[1] is not None for row in rows)
            assert await connection.scalar(text("SELECT count(*) FROM ai_state_counters WHERE store_key=:store"), {"store": store.store_key}) == 0
        assert (await state.execution.executions.get_fresh("execution", tenant_id="tenant")).result == commit.result
