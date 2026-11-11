#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional group reads preserve exact partitions and transaction-local writes."""

import asyncio
import hashlib
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event, update
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.persistence.state import GroupRecordReader
from linktools.ai.persistence._memory import InMemoryStateStore
from linktools.ai.persistence._sql import SqlStateStore, _SqlGroupTransaction
from linktools.ai.persistence.state import (
    RecordReplacement, StateGroupTransaction, StateStore, StateTransaction,
    StoredAlias, StoredFact, StoredRecord,
)
from linktools.ai.storage import CacheBackend, InMemoryCacheBackend, SqlStorageContext



def _key(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def _record(value: str, *, key: bytes | None = None) -> StoredRecord:
    return StoredRecord(key or _key(value), None, None, "probe", (value).encode("ascii"), None, 0,
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
            assert await group.get_records({first: (existing.key,), second: (new.key,)}) == {
                first: {existing.key: existing}, second: {},
            }
            tx, other = group.transaction(first), group.transaction(second)
            guarded = await tx.guard_record(existing.key, expected_storage_revision=0)
            assert guarded is not None and guarded.storage_revision == 1
            assert await tx.get_record(existing.key) == guarded
            await other.insert_record(new)
            assert await other.get_record(new.key) == new
            changed = replace(guarded, data={"value": "updated"}, storage_revision=2)
            assert await tx.replace_record(changed, expected_storage_revision=1)
            assert await group.get_records({first: (existing.key,), second: (new.key,)}) == {
                first: {existing.key: changed}, second: {new.key: new},
            }
            await other.delete_record(new.key)
            assert await group.get_records({second: (new.key,)}) == {second: {}}
            async def nested(nested_group: StateGroupTransaction) -> None:
                assert isinstance(nested_group, GroupRecordReader)
                assert await nested_group.get_records({first: (existing.key,)}) == {
                    first: {existing.key: changed},
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
            await group.get_records({first: (existing.key,), second: (staged.key,)})
            await group.transaction(second).insert_record(staged)
            transaction = group.transaction(first)
            table = first.storage_group.metadata.tables["ai_state_records"]
            # Change physical revision without refreshing the local observation.
            await transaction._execute(update(table).where(
                table.c.store_key == first.store_key,
                table.c.key == existing.key,
            ).values(storage_revision=1))
            assert await transaction.guard_record(existing.key, expected_storage_revision=0) is None
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        with pytest.raises(AIError) as error:
            await first.storage_group.mutate((first, second), fail)
        assert error.value.code is ErrorCode.STORAGE_CONFLICT
        assert await first.read(lambda tx: tx.get_record(existing.key)) == existing
        assert await second.read(lambda tx: tx.get_record(staged.key)) is None


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
                with pytest.raises(ValueError):
                    await group.get_records({first: (b"",)})
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
                              runtime_domain="alias", group=first.storage_group, store_key=first.store_key)
        await alias.initialize()
        one, two = _record("one"), _record("two")
        await first.mutate(lambda tx: tx.insert_records((one, two)))
        assert await first.read_records((one.key,)) == {one.key: one}
        try:
            async def read(group: StateGroupTransaction) -> None:
                assert isinstance(group, GroupRecordReader)
                assert await group.get_records({first: (one.key,), alias: (two.key,)}) == {
                    first: {one.key: one}, alias: {two.key: two},
                }
            await first.storage_group.read_many((first, alias), read)
            async def write(group: StateGroupTransaction) -> None:
                assert isinstance(group, GroupRecordReader)
                await group.get_records({first: (one.key,), alias: (one.key,)})
                changed = replace(one, data={"value": "changed"}, storage_revision=1)
                assert await group.transaction(alias).replace_record(changed, expected_storage_revision=0)
                assert await group.transaction(first).get_record(one.key) == changed
                assert await group.get_records({first: (one.key,), alias: (one.key,)}) == {
                    first: {one.key: changed}, alias: {one.key: changed},
                }
            await first.storage_group.mutate((first, alias), write)
            assert (await first.read_records((one.key,)))[one.key].data == {"value": "changed"}
        finally:
            await alias.close()


@asynccontextmanager
async def _partition_handles(
    path: Path, *, cache: CacheBackend | None = None, alias_cache: CacheBackend | None = None,
) -> AsyncIterator[tuple[SqlStateStore, SqlStateStore]]:
    async with _database(path, cache=cache) as (engine, first, _):
        alias = SqlStateStore(engine, context=first.context, metadata=first.storage_group.metadata,
                              runtime_domain="alias", group=first.storage_group, store_key=first.store_key,
                              cache=alias_cache)
        await alias.initialize()
        try:
            yield first, alias
        finally:
            await alias.close()


@pytest.mark.asyncio
async def test_physical_partition_handles_observe_each_others_writes(tmp_path: Path) -> None:
    async with _partition_handles(tmp_path / "views.db") as (first, alias):
        record = _record("owner")
        alias_key, sequence_key = _key("alias"), _key("sequence")
        await first.mutate(lambda tx: tx.insert_record(record))

        async def write(group: StateGroupTransaction) -> None:
            tx = group.transaction(first)
            assert await tx.resolve_alias(alias_key) is None
            assert await tx.get_sequence(sequence_key) == 0
            assert await tx.guard_record_revision(record.key, expected_storage_revision=0) == 1

            async def nested(other: StateTransaction) -> None:
                current = await other.get_record(record.key)
                assert current is not None and current.storage_revision == 1
                assert await other.replace_record(
                    replace(current, storage_revision=2, data={"value": "changed"}),
                    expected_storage_revision=1,
                )
                await other.insert_alias(StoredAlias(alias_key, record.key))
                assert await other.next_sequence(sequence_key) == 1

            await alias.mutate(nested)
            assert await tx.resolve_alias(alias_key) == record.key
            assert await first.read(lambda current: current.get_sequence(sequence_key)) == 1
            assert await tx.guard_record_revision(record.key, expected_storage_revision=0) == 2
            current = await tx.get_record(record.key)
            assert current is not None and current.data == {"value": "changed"}
            assert await tx.advance_sequence(sequence_key, 1) == 2
            assert await alias.read(lambda other: other.get_sequence(sequence_key)) == 2

        await first.storage_group.mutate((first, alias), write)

        async def read(group: StateGroupTransaction) -> None:
            for store in (first, alias):
                tx = group.transaction(store)
                assert await tx.get_sequence(sequence_key) == 2
                assert await tx.resolve_alias(alias_key) == record.key
                current = await tx.get_record(record.key)
                assert current is not None and current.storage_revision == 2

        await first.storage_group.read_many((first, alias), read)


@pytest.mark.asyncio
@pytest.mark.parametrize("batch", (False, True))
async def test_physical_partition_owner_deletion_revokes_guards(tmp_path: Path, batch: bool) -> None:
    async with _partition_handles(tmp_path / "owner.db") as (first, alias):
        record = _record("owner")
        alias_key = _key("alias")

        async def seed(tx: StateTransaction) -> None:
            await tx.insert_record(record)
            await tx.insert_alias(StoredAlias(alias_key, record.key))

        await first.mutate(seed)

        async def delete(group: StateGroupTransaction) -> None:
            tx, other = group.transaction(first), group.transaction(alias)
            assert await tx.resolve_alias(alias_key) == record.key
            assert await tx.guard_record_revision(record.key, expected_storage_revision=0) == 1
            if batch:
                await other.delete_records((record.key,))
            else:
                assert await other.delete_record(record.key)
            with pytest.raises(RuntimeError, match="owner must be guarded"):
                await tx.insert_fact(StoredFact(_key("stream"), 1, record.key, "probe", None, None, {}))
            with pytest.raises(RuntimeError, match="owner must be guarded"):
                await tx.insert_alias(StoredAlias(_key("new-alias"), record.key))
            assert await tx.guard_record_revision(record.key, expected_storage_revision=1) is None
            assert await tx.resolve_alias(alias_key) is None

        await first.storage_group.mutate((first, alias), delete)
        assert await first.read(lambda tx: tx.get_record(record.key)) is None
        assert await first.read(lambda tx: tx.scan_facts()) == ()
        assert await first.read(lambda tx: tx.scan_aliases()) == ()
        await first.validate_integrity()


@pytest.mark.asyncio
async def test_physical_partition_handles_preserve_caught_batch_abort(tmp_path: Path) -> None:
    async with _partition_handles(tmp_path / "abort.db") as (first, alias):
        records = (_record("first"), _record("second"))
        unrelated = _record("unrelated")
        await first.mutate(lambda tx: tx.insert_records(records))

        async def write(group: StateGroupTransaction) -> None:
            async def conflict(tx: StateTransaction) -> None:
                with pytest.raises(AIError) as caught:
                    await tx.replace_records((
                        RecordReplacement(replace(records[0], storage_revision=1), 0),
                        RecordReplacement(replace(records[1], storage_revision=2), 1),
                    ))
                assert caught.value.code is ErrorCode.STORAGE_CONFLICT

            await alias.mutate(conflict)
            await group.transaction(first).insert_record(unrelated)

        with pytest.raises(AIError) as caught:
            await first.storage_group.mutate((first, alias), write)
        assert caught.value.code is ErrorCode.STORAGE_CONFLICT
        assert await first.read(lambda tx: tx.get_records(tuple(record.key for record in records))) == {
            record.key: record for record in records
        }
        assert await alias.read(lambda tx: tx.get_record(unrelated.key)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", (False, True))
async def test_physical_partition_unknown_mutation_invalidates_each_handle_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, committed: bool,
) -> None:
    async with _partition_handles(
        tmp_path / "unknown.db", cache=InMemoryCacheBackend(max_bytes=65536),
        alias_cache=InMemoryCacheBackend(max_bytes=65536),
    ) as (first, alias):
        old = replace(_record("owner"), kind="execution")
        new = replace(old, storage_revision=1, data={"value": "changed"})
        await first.mutate(lambda tx: tx.insert_record(old))
        for store in (first, alias):
            assert await store.read_record(old.key) == old

        context = first.context
        original = type(context).run_mutation
        uncertain = AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        async def unknown(
            owner: SqlStorageContext, callback: Callable[[AsyncSession], Awaitable[object]],
            *, retry_limit: int | None = None, domain: str = "sql",
        ) -> object:
            if owner is not context:
                return await original(owner, callback, retry_limit=retry_limit, domain=domain)
            if committed:
                await original(owner, callback, retry_limit=retry_limit, domain=domain)
            else:
                async def rollback(session: AsyncSession) -> object:
                    await callback(session)
                    raise uncertain
                await original(owner, rollback, retry_limit=retry_limit, domain=domain)
            raise uncertain

        async def change(group: StateGroupTransaction) -> None:
            assert await group.transaction(first).get_record(old.key) == old
            assert await alias.mutate(lambda tx: tx.replace_record(new, expected_storage_revision=0))
            assert await group.transaction(first).get_record(old.key) == new

        with monkeypatch.context() as patch:
            patch.setattr(type(context), "run_mutation", unknown)
            with pytest.raises(AIError) as caught:
                await first.storage_group.mutate((first, alias), change)
            assert caught.value is uncertain
        expected = new if committed else old
        for store in (first, alias):
            assert await store.read_record(old.key) == expected
            assert await store.read(lambda tx: tx.get_record(old.key)) == expected


@pytest.mark.asyncio
async def test_unsupported_and_custom_groups_keep_original_transaction_reads() -> None:
    store = InMemoryStateStore()
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
        assert await custom.transaction(store).get_records((record.key,)) == {record.key: record}

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
