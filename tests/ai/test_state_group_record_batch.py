#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional group reads preserve exact partitions and transaction-local writes."""

import asyncio
import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event, update
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.persistence.state import GroupRecordReader
from linktools.ai.persistence._memory import InMemoryStateStore
from linktools.ai.persistence._sql import SqlStateStore, _SqlGroupTransaction
from linktools.ai.persistence.state import StateGroupTransaction, StateStore, StateTransaction, StoredRecord
from linktools.ai.storage import CacheBackend, InMemoryCacheBackend

from .test_state_group_archive import _filesystem


def _key(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def _record(value: str, *, key: bytes | None = None) -> StoredRecord:
    return StoredRecord(key or _key(value), None, None, "probe", value, None, 0,
                        None, 0, None, {"value": value})


@asynccontextmanager
async def _database(path: Path, *, cache: CacheBackend | None = None) -> AsyncIterator[tuple[AsyncEngine, SqlStateStore, SqlStateStore]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await provision_database(engine)
    first = SqlStateStore(engine, runtime_domain="first", cache=cache)
    await first.initialize()
    second = SqlStateStore(engine, context=first.context, metadata=first.storage_group.metadata,
                           runtime_domain="second", group=first.storage_group)
    await second.initialize()
    try:
        yield engine, first, second
    finally:
        await second.close()
        await first.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_group_reads_bound_parameters_and_keep_partition_keys_exact(tmp_path: Path) -> None:
    async with _database(tmp_path / "bounded.db") as (engine, first, second):
        key = _key("common")
        one, two = _record("one", key=key), _record("two", key=key)
        await first.mutate(lambda tx: tx.insert_record(one))
        await second.mutate(lambda tx: tx.insert_record(two))
        requests = {first: (key, key, *(_key(f"first-{n}") for n in range(1100))),
                    second: (key, *(_key(f"second-{n}") for n in range(600)))}
        parameter_counts: list[int] = []

        def capture(connection, cursor, statement, parameters, context, many) -> None:
            if statement.startswith("SELECT"):
                parameter_counts.append(len(parameters))
                assert len(parameters) <= 999

        async def read(group: StateGroupTransaction) -> None:
            assert isinstance(group, GroupRecordReader)
            assert await group.get_records(requests) == {first: {key: one}, second: {key: two}}
            assert await group.transaction(first).get_record(_key("first-0")) is None
            assert await group.transaction(second).get_records((key,)) == {key: two}
            assert await group.get_records({}) == {}
            assert await group.get_records({first: (), second: ()}) == {first: {}, second: {}}

        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            await first.storage_group.read_many((first, second), read)
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
        assert len(parameter_counts) > 1


@pytest.mark.asyncio
async def test_group_reads_keep_insert_replace_guard_delete_visible(tmp_path: Path) -> None:
    async with _database(tmp_path / "own-writes.db") as (_, first, second):
        existing, new = _record("existing"), _record("new")
        await first.mutate(lambda tx: tx.insert_record(existing))

        async def write(group: StateGroupTransaction) -> None:
            assert isinstance(group, GroupRecordReader)
            assert await group.get_records({first: (existing.key_digest,), second: (new.key_digest,)}) == {
                first: {existing.key_digest: existing}, second: {},
            }
            tx, other = group.transaction(first), group.transaction(second)
            guarded = await tx.guard_record(existing.key_digest, expected_storage_revision=0)
            assert guarded is not None and guarded.storage_revision == 1
            assert await tx.get_record(existing.key_digest) == guarded
            await other.insert_record(new)
            assert await other.get_record(new.key_digest) == new
            changed = replace(guarded, data={"value": "updated"}, storage_revision=2)
            assert await tx.replace_record(changed, expected_storage_revision=1)
            assert await group.get_records({first: (existing.key_digest,), second: (new.key_digest,)}) == {
                first: {existing.key_digest: changed}, second: {new.key_digest: new},
            }
            await other.delete_record(new.key_digest)
            assert await group.get_records({second: (new.key_digest,)}) == {second: {}}
            async def nested(nested_group: StateGroupTransaction) -> None:
                assert isinstance(nested_group, GroupRecordReader)
                assert await nested_group.get_records({first: (existing.key_digest,)}) == {
                    first: {existing.key_digest: changed},
                }
            await first.storage_group.read_many((first,), nested)

        await first.storage_group.mutate((first, second), write)


@pytest.mark.asyncio
async def test_group_prefetch_does_not_replace_physical_cas_or_partial_rollback(tmp_path: Path) -> None:
    async with _database(tmp_path / "cas.db") as (_, first, second):
        existing, staged = _record("existing"), _record("staged")
        await first.mutate(lambda tx: tx.insert_record(existing))

        async def fail(group: StateGroupTransaction) -> None:
            assert isinstance(group, GroupRecordReader)
            await group.get_records({first: (existing.key_digest,), second: (staged.key_digest,)})
            await group.transaction(second).insert_record(staged)
            transaction = group.transaction(first)
            table = first.storage_group.metadata.tables["ai_state_records"]
            # Change physical revision without refreshing the local observation.
            await transaction._execute(update(table).where(
                table.c.store_digest == first.store_digest.hex(),
                table.c.key_digest == existing.key_digest.hex(),
            ).values(storage_revision=1))
            assert await transaction.guard_record(existing.key_digest, expected_storage_revision=0) is None
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        with pytest.raises(AIError) as error:
            await first.storage_group.mutate((first, second), fail)
        assert error.value.code is ErrorCode.STORAGE_CONFLICT
        assert await first.read(lambda tx: tx.get_record(existing.key_digest)) == existing
        assert await second.read(lambda tx: tx.get_record(staged.key_digest)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("writable", (False, True))
async def test_group_reader_rejects_foreign_unenlisted_child_and_expired_scope(tmp_path: Path, writable: bool) -> None:
    async with _database(tmp_path / "one.db") as (_, first, second):
        async with _database(tmp_path / "two.db") as (_, foreign, unused):
            captured = []
            assert isinstance(first.storage_group, GroupRecordReader)
            with pytest.raises(RuntimeError):
                await first.storage_group.get_records({first: (_key("x"),)})

            async def inspect(group: StateGroupTransaction) -> None:
                assert isinstance(group, GroupRecordReader)
                captured.append(group)
                assert await first.storage_group.get_records({first: (_key("x"),)}) == {first: {}}
                with pytest.raises(RuntimeError):
                    await foreign.storage_group.get_records({foreign: (_key("x"),)})
                with pytest.raises(RuntimeError):
                    await group.get_records({first: (_key("x"),), foreign: (_key("x"),)})
                with pytest.raises(RuntimeError):
                    await group.get_records({second: (_key("x"),)})
                with pytest.raises(RuntimeError):
                    await first.storage_group.get_records({second: (_key("x"),)})
                with pytest.raises(RuntimeError):
                    await asyncio.create_task(first.storage_group.get_records({first: (_key("x"),)}))
                with pytest.raises(RuntimeError):
                    await asyncio.create_task(group.get_records({first: (_key("x"),)}))
                with pytest.raises(AIError) as error:
                    await group.get_records({first: (b"bad",)})
                assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
                if not writable:
                    with pytest.raises(RuntimeError):
                        await group.transaction(first).insert_record(_record("forbidden"))

            callback = first.storage_group.mutate if writable else first.storage_group.read_many
            await callback((first,), inspect)
            with pytest.raises(RuntimeError):
                await captured[0].get_records({first: (_key("x"),)})

            async def next_scope(group: StateGroupTransaction) -> None:
                with pytest.raises(RuntimeError):
                    await captured[0].get_records({first: (_key("x"),)})
            await callback((first,), next_scope)
            assert await first.read(lambda tx: tx.get_record(_key("forbidden"))) is None


@pytest.mark.asyncio
async def test_group_reads_demultiplex_aliases_of_one_physical_partition(tmp_path: Path) -> None:
    async with _database(tmp_path / "aliases.db", cache=InMemoryCacheBackend(max_bytes=65536)) as (engine, first, second):
        alias = SqlStateStore(engine, context=first.context, metadata=first.storage_group.metadata,
                              runtime_domain="alias", group=first.storage_group, store_digest=first.store_digest)
        await alias.initialize()
        one, two = _record("one"), _record("two")
        await first.mutate(lambda tx: tx.insert_records((one, two)))
        assert await first.read_records((one.key_digest,)) == {one.key_digest: one}
        try:
            async def read(group: StateGroupTransaction) -> None:
                assert isinstance(group, GroupRecordReader)
                assert await group.get_records({first: (one.key_digest,), alias: (two.key_digest,)}) == {
                    first: {one.key_digest: one}, alias: {two.key_digest: two},
                }
            await first.storage_group.read_many((first, alias), read)
            async def write(group: StateGroupTransaction) -> None:
                assert isinstance(group, GroupRecordReader)
                await group.get_records({first: (one.key_digest,), alias: (one.key_digest,)})
                changed = replace(one, data={"value": "changed"}, storage_revision=1)
                assert await group.transaction(alias).replace_record(changed, expected_storage_revision=0)
                assert await group.transaction(first).get_record(one.key_digest) == changed
                assert await group.get_records({first: (one.key_digest,), alias: (one.key_digest,)}) == {
                    first: {one.key_digest: changed}, alias: {one.key_digest: changed},
                }
            await first.storage_group.mutate((first, alias), write)
            assert (await first.read_records((one.key_digest,)))[one.key_digest].data == {"value": "changed"}
        finally:
            await alias.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem"))
async def test_unsupported_and_custom_groups_keep_original_transaction_reads(tmp_path: Path, backend: str) -> None:
    if backend == "memory":
        store = InMemoryStateStore()
    else:
        store = _filesystem(tmp_path / "unsupported")
    await store.initialize()
    assert not isinstance(store.storage_group, GroupRecordReader)
    record = _record("ordinary")

    class CustomGroup:
        def __init__(self, group: StateGroupTransaction) -> None:
            self.group = group

        def transaction(self, requested: StateStore) -> StateTransaction:
            return self.group.transaction(requested)

    async def read(group: StateGroupTransaction) -> None:
        assert not isinstance(group, GroupRecordReader)
        custom = CustomGroup(group)
        assert not isinstance(custom, GroupRecordReader)
        assert await custom.transaction(store).get_records((record.key_digest,)) == {record.key_digest: record}

    try:
        await store.mutate(lambda tx: tx.insert_record(record))
        await store.storage_group.read_many((store,), read)
        await store.storage_group.mutate((store,), read)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_group_batch_queries_compile_for_supported_sql_dialects(tmp_path: Path) -> None:
    async with _database(tmp_path / "compile.db") as (_, first, second):
        requests = {first: tuple(_key(f"first-{n}") for n in range(1800)),
                    second: tuple(_key(f"second-{n}") for n in range(1000))}
        async def compile_queries(group: StateGroupTransaction) -> None:
            assert isinstance(group, _SqlGroupTransaction)
            dialects = (sqlite.dialect(), mysql.dialect(), postgresql.dialect())
            dialects[1].server_version_info = (5, 7, 44)
            for dialect in dialects:
                batches = 0
                for batch in group._record_read_batches(requests):
                    batches += 1
                    statement = group._record_read_statement(batch)
                    compiled = statement.compile(dialect=dialect, compile_kwargs={"render_postcompile": True})
                    count = len(compiled.params) if compiled.positiontup is None else len(compiled.positiontup)
                    assert count <= 999
                    assert "FOR UPDATE" not in str(compiled)
                    assert "WITH RECURSIVE" not in str(compiled)
                assert batches > 1
        await first.storage_group.mutate((first, second), compile_queries)
