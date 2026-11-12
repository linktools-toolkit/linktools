#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL run counters live only in their admission and preserve portable sequences."""

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import text

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence._run_counters import run_counter_address, run_counter_streams
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.persistence.state import StateTransaction, stream_key, state_store_key
from linktools.ai.storage import InMemoryObjectStore

from .test_memory_run_counter_owner import _admission
from .test_state_batch_delete_backends import _open_store
from .test_native_text_result_runtime import _prepare, _publish


@pytest.mark.asyncio
async def test_sql_admission_counters_cas_rollback_delete_and_global_pages(tmp_path: Path) -> None:
    async with _open_store("sql", tmp_path) as bare:
        store = await bare.storage_group.open_partition(state_store_key("run-counter", "tenant", "trace"), create=True)
        assert store is not None
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
            ordinary = (await connection.execute(text("SELECT stream_key FROM ai_state_counters WHERE store_key=:store"),
                {"store": store.store_key})).scalars().all()
            assert all(run_counter_address(key) is None for key in ordinary)
        assert (await state.execution.executions.get_fresh("execution", tenant_id="tenant")).result == commit.result


def test_run_counter_cursor_mapping_preserves_arbitrary_binary_order() -> None:
    from linktools.ai.persistence._run_counters import RUN_COUNTER_COLUMNS, run_counter_page_start
    from linktools.ai.persistence.state import record_key
    owners = tuple(record_key(namespace, "tenant", domain, "agent_run", (run,))
        for namespace in ("a", "long-name", "é") for domain in ("trace", "checkpoint", "conversation")
        for run in ("a", "z", "long-run"))
    streams = tuple(key for owner in owners for key in run_counter_streams(owner))
    cursors = {b"\x00", b"opaque", b"\xff", *(key[:end] for key in streams for end in range(1, len(key) + 1)),
               *(key + b"\x00" for key in streams), *(key + b"\xff" for key in streams)}
    for after in cursors:
        for index, column in enumerate(RUN_COUNTER_COLUMNS):
            bound = run_counter_page_start(after, column)
            for owner in owners:
                if bound is None:
                    actual = False
                else:
                    start, inclusive = bound
                    actual = start is None or (owner >= start if inclusive else owner > start)
                assert actual is (run_counter_streams(owner)[index] > after), (after, column, owner)


@pytest.mark.asyncio
async def test_sql_run_counter_pages_bound_each_physical_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from linktools.ai.persistence._sql import _SqlTransaction
    from linktools.ai.persistence.state import record_key
    async with _open_store("sql", tmp_path) as bare:
        store = await bare.storage_group.open_partition(state_store_key("a", "tenant", "trace"), create=True)
        assert store is not None
        records = tuple(replace(_admission(run), key=record_key(namespace, "tenant", domain, "agent_run", (run,)))
            for namespace, domain, run in (("a", "trace", "one"), ("long", "checkpoint", "two"),
                                            ("é", "conversation", "three")))
        async def publish(tx: StateTransaction) -> None:
            await tx.insert_records(records)
            await tx.reserve_sequences({key: index + 1 for index, key in enumerate(
                (b"\x00", b"opaque", *(key for record in records for key in run_counter_streams(record.key))))})
        await store.mutate(publish)
        all_values = await store.read(lambda tx: tx.scan_sequences())
        original = _SqlTransaction._execute
        reads = []
        async def bounded(self: _SqlTransaction, statement: object) -> object:
            if str(statement).lstrip().upper().startswith("SELECT"):
                assert statement._limit_clause is not None
                assert statement._limit_clause.value == 2
                reads.append(statement)
            return await original(self, statement)
        monkeypatch.setattr(_SqlTransaction, "_execute", bounded)
        cursors = (None, b"\x00", b"opaque", b"\xff", *all_values,
                   *(key[:-1] for key in all_values if len(key) > 1))
        for after in cursors:
            reads.clear()
            actual = await store.read(lambda tx: tx.scan_sequences_page(after=after, limit=2))
            expected = dict(tuple((key, value) for key, value in all_values.items() if after is None or key > after)[:2])
            assert actual == expected
            assert 1 <= len(reads) <= 4


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sql"))
async def test_run_counter_routing_requires_matching_step_partition(backend: str, tmp_path: Path) -> None:
    async with _open_store(backend, tmp_path) as bare:
        store = await bare.storage_group.open_partition(
            state_store_key("run-counter", "tenant", "trace"), create=True,
        )
        assert store is not None
        owned = stream_key("run-counter", "tenant", "trace", "event", ("run",))
        foreign = tuple(stream_key(namespace, tenant, domain, "event", ("run",))
                        for namespace, tenant, domain in (("other", "tenant", "trace"),
                            ("run-counter", "other", "trace"), ("run-counter", "tenant", "checkpoint")))
        # A generic State owner accepts all legal stream tuples without Step admissions.
        assert await bare.mutate(lambda tx: tx.reserve_sequence(owned, 1)) == 1
        with pytest.raises(AIError) as caught:
            await store.mutate(lambda tx: tx.reserve_sequence(owned, 1))
        assert caught.value.code is ErrorCode.STORAGE_CONFLICT
        assert await store.mutate(lambda tx: tx.reserve_sequences(dict.fromkeys(foreign, 2))) == dict.fromkeys(foreign, 2)
        await store.mutate(lambda tx: tx.insert_record(_admission()))
        assert await store.mutate(lambda tx: tx.reserve_sequence(owned, 3)) == 3
        await store.mutate(lambda tx: tx.delete_record(_admission().key))
        assert await store.read(lambda tx: tx.scan_sequences()) == dict(sorted(dict.fromkeys(foreign, 2).items()))
        assert await bare.read(lambda tx: tx.get_sequence(owned)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("dialect_name", ("sqlite", "postgresql", "mysql"))
async def test_run_counter_reservation_uses_authoritative_dialect_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dialect_name: str,
) -> None:
    from sqlalchemy.dialects import mysql, postgresql, sqlite
    from linktools.ai.persistence._sql import _SqlTransaction
    async with _open_store("sql", tmp_path) as bare:
        store = await bare.storage_group.open_partition(
            state_store_key("run-counter", "tenant", "trace"), create=True,
        )
        assert store is not None
        record = _admission()
        await store.mutate(lambda tx: tx.insert_record(record))
        event, checkpoint, _ = run_counter_streams(record.key)
        statements = []
        original = _SqlTransaction._execute

        async def execute(self: _SqlTransaction, statement: object) -> object:
            statements.append(statement)
            return await original(self, statement)

        async def reserve(tx: StateTransaction) -> None:
            # Exercise each dialect choice on SQLite; compile the exact statements
            # for its server dialect too. Server execution remains covered by CI.
            with monkeypatch.context() as patch:
                patch.setattr(type(store.context.dialect), "name", property(lambda self: dialect_name))
                patch.setattr(_SqlTransaction, "_execute", execute)
                assert await tx.reserve_sequences({event: 2, checkpoint: 3}) == {event: 2, checkpoint: 3}
                assert await tx.reserve_sequence(event, 4) == 6
        await store.mutate(reserve)
        assert sum(bool(getattr(statement, "is_select", False)) for statement in statements) == (2 if dialect_name == "mysql" else 0)
        dialect = {"sqlite": sqlite, "postgresql": postgresql, "mysql": mysql}[dialect_name].dialect()
        for statement in statements:
            sql = str(statement.compile(dialect=dialect))
            if getattr(statement, "is_update", False):
                assert ("RETURNING" in sql) == (dialect_name != "mysql")
        assert await store.read(lambda tx: tx.get_sequences((event, checkpoint))) == {event: 6, checkpoint: 3}
