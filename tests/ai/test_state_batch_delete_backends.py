#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bulk owner deletion preserves transaction semantics with linear cleanup work."""

import hashlib
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event, update
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.runtime.state._filesystem import FilesystemStateStore
from linktools.ai.runtime.state._filesystem_layout import _CowMap, _FactStreamInfo
from linktools.ai.runtime.state._memory import InMemoryStateStore
from linktools.ai.runtime.state._sql import SqlStateStore, _SqlTransaction
from linktools.ai.runtime.state._store import (
    FactQuery, RecordQuery, StateStore, StateTransaction, StateTransactionNestingError,
    StoredAlias, StoredFact, StoredOperation, StoredRecord,
)

pytestmark = pytest.mark.asyncio
_BACKENDS = ("memory", "filesystem", "sql")


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("ascii")).digest()


def _record(index: int) -> StoredRecord:
    return StoredRecord(
        _digest(f"record-{index}"), _digest("scope"), None, "probe", f"{index:04d}",
        None, 0, None, 0, None, {"index": index},
    )


def _alias(record: StoredRecord) -> StoredAlias:
    return StoredAlias(_digest(f"alias-{record.sort_key}"), record.key_digest)


def _fact(record: StoredRecord, sequence: int = 1) -> StoredFact:
    return StoredFact(
        _digest(f"stream-{record.sort_key}"), sequence, record.key_digest, "probe",
        _digest(f"subject-{sequence}"), None, {"sequence": sequence},
    )


@asynccontextmanager
async def _open_store(backend: str, root: Path) -> AsyncIterator[StateStore]:
    engine = None
    if backend == "memory":
        store = InMemoryStateStore()
    elif backend == "filesystem":
        store = FilesystemStateStore(
            root, namespace="batch-delete", tenant_id="tenant",
            runtime_domain="conversation", _range_index=True,
        )
    else:
        root.mkdir(parents=True, exist_ok=True)
        engine = create_async_engine(f"sqlite+aiosqlite:///{root / 'state.db'}")
        await provision_database(engine)
        store = SqlStateStore(engine)
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()
        if engine is not None:
            await engine.dispose()


async def _seed(store: StateStore, records: tuple[StoredRecord, ...]) -> None:
    async def seed(transaction: StateTransaction) -> None:
        await transaction.insert_records(records)
        await transaction.insert_aliases(tuple(_alias(record) for record in records))
        await transaction.insert_facts(tuple(_fact(record) for record in records))

    await store.mutate(seed)


@pytest.mark.parametrize("backend", _BACKENDS)
async def test_batch_delete_cascades_owners_and_reads_staged_deletes(
    backend: str, tmp_path: Path,
) -> None:
    deleted = _record(1)
    retained = replace(_record(2), parent_digest=deleted.key_digest)
    staged = _record(3)
    staged_alias = StoredAlias(_digest("staged-alias"), deleted.key_digest)
    counter = _digest("counter")
    operation = StoredOperation(_digest("operation"), counter, 1, "DONE", False, {})
    async with _open_store(backend, tmp_path / backend) as store:
        await _seed(store, (deleted, retained))

        async def remove(transaction: StateTransaction) -> None:
            assert await transaction.get_record(deleted.key_digest) == deleted
            assert await transaction.resolve_alias(_alias(deleted).alias_digest) == deleted.key_digest
            assert await transaction.list_facts(FactQuery(_fact(deleted).stream_digest)) == (_fact(deleted),)
            assert await transaction.guard_record(deleted.key_digest, expected_storage_version=0)
            await transaction.insert_alias(staged_alias)
            await transaction.insert_fact(_fact(deleted, 2))
            await transaction.insert_record(staged)
            await transaction.insert_alias(_alias(staged))
            await transaction.insert_fact(_fact(staged))
            await transaction.reserve_sequence(counter, 1)
            await transaction.insert_operation(operation)
            await transaction.delete_records((
                deleted.key_digest, staged.key_digest, deleted.key_digest, _digest("missing"),
            ))
            assert await transaction.get_records((deleted.key_digest, staged.key_digest)) == {}
            assert await transaction.resolve_aliases((
                _alias(deleted).alias_digest, staged_alias.alias_digest, _alias(staged).alias_digest,
            )) == {}
            assert await transaction.list_facts(FactQuery(_fact(deleted).stream_digest)) == ()
            assert await transaction.list_facts(FactQuery(_fact(staged).stream_digest)) == ()
            assert await transaction.list_records(RecordQuery(kind="probe")) == (retained,)
            assert await transaction.get_sequence(counter) == 1
            assert await transaction.get_operation(operation.key_digest) == operation
            await transaction.delete_records(())
            await transaction.delete_records((deleted.key_digest, staged.key_digest, _digest("missing")))

        await store.mutate(remove)
        assert await store.read(lambda transaction: transaction.scan_records()) == (retained,)
        assert await store.read(lambda transaction: transaction.scan_aliases()) == (_alias(retained),)
        assert await store.read(lambda transaction: transaction.scan_facts()) == (_fact(retained),)
        await store.validate_integrity()


@pytest.mark.parametrize("backend", _BACKENDS)
async def test_batch_delete_rolls_back_owners_aliases_and_facts(
    backend: str, tmp_path: Path,
) -> None:
    records = (_record(1), _record(2))
    async with _open_store(backend, tmp_path / backend) as store:
        await _seed(store, records)

        async def abort(transaction: StateTransaction) -> None:
            await transaction.delete_records(tuple(record.key_digest for record in records))
            assert await transaction.scan_records() == ()
            assert await transaction.scan_aliases() == ()
            assert await transaction.scan_facts() == ()
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        with pytest.raises(AIError) as raised:
            await store.mutate(abort)
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        assert await store.read(lambda transaction: transaction.get_records(
            tuple(record.key_digest for record in records)
        )) == {record.key_digest: record for record in records}
        assert set(await store.read(lambda transaction: transaction.scan_aliases())) == {
            _alias(record) for record in records
        }
        facts = await store.read(lambda transaction: transaction.scan_facts())
        assert {fact.stream_digest: fact for fact in facts} == {
            _fact(record).stream_digest: _fact(record) for record in records
        }
        await store.validate_integrity()


@pytest.mark.parametrize("backend", _BACKENDS)
async def test_batch_delete_allows_reusing_deleted_identity(
    backend: str, tmp_path: Path,
) -> None:
    record = _record(1)
    replacement = replace(record, data={"replacement": True})
    replacement_fact = replace(_fact(record), data={"replacement": True})
    async with _open_store(backend, tmp_path / backend) as store:
        await _seed(store, (record,))

        async def recreate(transaction: StateTransaction) -> None:
            await transaction.delete_records((record.key_digest,))
            await transaction.insert_record(replacement)
            await transaction.insert_alias(_alias(replacement))
            await transaction.insert_fact(replacement_fact)
            assert await transaction.get_record(record.key_digest) == replacement
            assert await transaction.resolve_alias(_alias(record).alias_digest) == record.key_digest
            assert await transaction.list_facts(FactQuery(replacement_fact.stream_digest)) == (replacement_fact,)

        await store.mutate(recreate)
        assert await store.read(lambda transaction: transaction.get_record(record.key_digest)) == replacement
        assert await store.read(lambda transaction: transaction.scan_facts()) == (replacement_fact,)
        await store.validate_integrity()


@pytest.mark.parametrize("backend", _BACKENDS)
async def test_batch_delete_is_rejected_in_read_only_callbacks(
    backend: str, tmp_path: Path,
) -> None:
    record = _record(1)
    async with _open_store(backend, tmp_path / backend) as store:
        await _seed(store, (record,))
        with pytest.raises(StateTransactionNestingError):
            await store.read(lambda transaction: transaction.delete_records((record.key_digest,)))
        assert await store.read(lambda transaction: transaction.get_record(record.key_digest)) == record


async def test_sql_batch_delete_later_chunk_conflict_rolls_back_entire_batch(tmp_path: Path) -> None:
    records = tuple(_record(index) for index in range(260))
    async with _open_store("sql", tmp_path / "sql") as store:
        await _seed(store, records)

        async def conflict(transaction: StateTransaction) -> None:
            assert isinstance(transaction, _SqlTransaction)
            keys = tuple(record.key_digest for record in records)
            await transaction.get_records(keys)
            table = transaction._table("ai_state_records")
            # Leave the read version stale to exercise the same CAS failure as a competing writer.
            await transaction._execute(update(table).where(
                table.c.store_digest == store.store_digest.hex(),
                table.c.key_digest == records[-1].key_digest.hex(),
            ).values(storage_version=1))
            await transaction.delete_records(keys)

        with pytest.raises(AIError) as raised:
            await store.mutate(conflict)
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        actual = await store.read(lambda transaction: transaction.get_records(
            tuple(record.key_digest for record in records)
        ))
        assert actual == {record.key_digest: record for record in records}
        assert len(await store.read(lambda transaction: transaction.scan_aliases())) == len(records)
        assert len(await store.read(lambda transaction: transaction.scan_facts())) == len(records)
        await store.validate_integrity()


async def test_sql_batch_delete_statement_count_does_not_grow_per_owner(tmp_path: Path) -> None:
    counts = []
    for size in (8, 64, 513):
        async with _open_store("sql", tmp_path / str(size)) as store:
            assert isinstance(store, SqlStateStore)
            records = tuple(_record(index) for index in range(size))
            await _seed(store, records)
            statements: list[str] = []
            parameter_counts: list[int] = []

            def capture(
                _connection: object, _cursor: object, statement: str,
                _parameters: Sequence[object] | Mapping[str, object],
                _context: object, _executemany: bool,
            ) -> None:
                if "ai_state_" in statement:
                    statements.append(statement)
                    parameter_counts.append(len(_parameters))

            event.listen(store.context.engine.sync_engine, "before_cursor_execute", capture)
            try:
                await store.mutate(lambda transaction: transaction.delete_records(
                    tuple(record.key_digest for record in records)
                ))
            finally:
                event.remove(store.context.engine.sync_engine, "before_cursor_execute", capture)
            counts.append(len(statements))
            assert await store.read(lambda transaction: transaction.scan_records()) == ()
            assert await store.read(lambda transaction: transaction.scan_aliases()) == ()
            assert await store.read(lambda transaction: transaction.scan_facts()) == ()
            assert max(parameter_counts) < 1000
    assert counts[1] <= counts[0] <= 4
    assert counts[2] <= 4 * 3


async def test_sql_batch_delete_preserves_other_logical_stores(tmp_path: Path) -> None:
    record = _record(1)
    async with _open_store("sql", tmp_path / "sql") as store:
        assert isinstance(store, SqlStateStore)
        other = SqlStateStore(
            store.context.engine, context=store.context, store_digest=_digest("other-store"),
        )
        await other.initialize()
        try:
            await _seed(store, (record,))
            await _seed(other, (record,))
            await store.mutate(lambda transaction: transaction.delete_records((record.key_digest,)))
            assert await store.read(lambda transaction: transaction.get_record(record.key_digest)) is None
            assert await other.read(lambda transaction: transaction.get_record(record.key_digest)) == record
            assert await other.read(lambda transaction: transaction.scan_aliases()) == (_alias(record),)
            assert await other.read(lambda transaction: transaction.scan_facts()) == (_fact(record),)
            await other.validate_integrity()
        finally:
            await other.close()


async def test_filesystem_batch_delete_relationship_work_grows_linearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    examined = []
    for size in (8, 64):
        async with _open_store("filesystem", tmp_path / str(size)) as store:
            assert isinstance(store, FilesystemStateStore)
            records = tuple(_record(index) for index in range(2 * size))
            await _seed(store, records)
            cache = store._require_index().cache
            aliases = cache.list_aliases
            streams = cache.list_fact_streams
            deleted = _CowMap.deleted
            work = {"aliases": 0, "streams": 0, "tombstones": 0}

            def count_aliases() -> tuple[tuple[bytes, bytes], ...]:
                values = aliases()
                work["aliases"] += len(values)
                return values

            def count_streams() -> tuple[_FactStreamInfo, ...]:
                values = streams()
                work["streams"] += len(values)
                return values

            def count_deleted(mapping: _CowMap) -> frozenset[object]:
                values = deleted(mapping)
                work["tombstones"] += len(values)
                return values

            with monkeypatch.context() as patch:
                patch.setattr(cache, "list_aliases", count_aliases)
                patch.setattr(cache, "list_fact_streams", count_streams)
                patch.setattr(_CowMap, "deleted", count_deleted)
                await store.mutate(lambda transaction: transaction.delete_records(
                    tuple(record.key_digest for record in records[:size])
                ))
            examined.append(work)
            assert len(await store.read(lambda transaction: transaction.scan_records())) == size
            await store.validate_integrity()
    for name in examined[0]:
        assert examined[1][name] <= 8 * examined[0][name] + 64, (name, examined)
    assert examined[1]["aliases"] <= 2 * 64
    assert examined[1]["streams"] <= 2 * 64
