#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed archive edges preserve scope, readonly access, and SQL bounds."""

import asyncio
import hashlib
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

import pytest
from sqlalchemy import event
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.persistence.runs import ArchiveFrontierReader, ArchiveNodeRead, ArchiveNodeSlot
from linktools.ai.persistence._archive import decode_archive_scope, ArchiveRoot, ContentArchive, PreparedArchive
from linktools.ai.persistence._plan import PersistenceDomain
from linktools.ai.persistence._schema import build_runtime_sql_metadata
from linktools.ai.persistence._sql import SqlStateStore, _SqlTransaction, _ARCHIVE_READ_FIXED_BINDS
from linktools.ai.persistence.state import StateTransaction, StoredRecord, _ReadOnlyStateTransaction
from linktools.ai.storage import create_sql_storage_context, dialect_for_name


ALL = ArchiveNodeSlot.SELF | ArchiveNodeSlot.LEFT | ArchiveNodeSlot.RIGHT | ArchiveNodeSlot.NEXT
# Each metadata and content slot binds both a locator and the raw key.
_NODE_BINDS = 4 * int(ALL).bit_count()
# Fill a node batch, then a mixed batch, and leave a direct-content tail.
_NODE_REQUESTS = (999 - _ARCHIVE_READ_FIXED_BINDS) // _NODE_BINDS + 1
_CONTENT_REQUESTS = (999 - _ARCHIVE_READ_FIXED_BINDS - _NODE_BINDS) // 2 + 1


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


@asynccontextmanager
async def _archive_database(path: Path) -> AsyncIterator[tuple[AsyncEngine, ContentArchive, PreparedArchive, ArchiveRoot]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await provision_database(engine)
    store = SqlStateStore(engine)
    await store.initialize()
    archive = ContentArchive(store, "frontier", namespace="namespace", tenant_id="tenant", object_store=None)
    prepared = PreparedArchive()
    identities = await archive.prepare_content(tuple(("scheme:a/b:界", f"body-{index}".encode()) for index in range(10)), prepared)
    root = await archive.append(archive.empty_root(), identities, prepared)
    await store.mutate(lambda transaction: archive.publish(transaction, prepared))
    try:
        yield engine, archive, prepared, root
    finally:
        await store.close()
        await engine.dispose()


def test_content_slots_include_their_metadata_dependencies() -> None:
    request = ArchiveNodeRead(_digest("node"), ArchiveNodeSlot(0), ArchiveNodeSlot.LEFT | ArchiveNodeSlot.NEXT)
    assert request.metadata_slots == ArchiveNodeSlot.SELF | ArchiveNodeSlot.LEFT | ArchiveNodeSlot.NEXT
    assert request.content_slots == ArchiveNodeSlot.LEFT | ArchiveNodeSlot.NEXT
    for bad_key in (b"", "not-bytes", b"x" * 1153):
        with pytest.raises(ValueError):
            ArchiveNodeRead(bad_key)
    for invalid in (-1, 16, True):
        with pytest.raises(ValueError):
            ArchiveNodeRead(_digest("node"), content_slots=invalid)


class _UnsupportedReader:
    async def get_records(self, keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord]:
        return {}


class _OptionalReader(_UnsupportedReader):
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.calls: list[tuple[bytes, Sequence[ArchiveNodeRead], Sequence[bytes]]] = []

    async def read_archive_frontiers(
        self, *, scope_key: bytes, requests: Sequence[ArchiveNodeRead], content_keys: Sequence[bytes] = (),
    ) -> Mapping[bytes, StoredRecord] | None:
        self.calls.append((scope_key, requests, content_keys))
        if self.error is not None:
            raise self.error
        return {}


@pytest.mark.asyncio
async def test_readonly_optional_reader_distinguishes_unsupported_empty_and_errors() -> None:
    unsupported = _ReadOnlyStateTransaction(cast(StateTransaction, _UnsupportedReader()))
    assert isinstance(unsupported, ArchiveFrontierReader)
    assert await unsupported.read_archive_frontiers(scope_key=_digest("scope"), requests=()) is None
    source = _OptionalReader()
    supported = _ReadOnlyStateTransaction(cast(StateTransaction, source))
    request = (ArchiveNodeRead(_digest("node")),)
    content = (_digest("body"),)
    assert await supported.read_archive_frontiers(scope_key=_digest("scope"), requests=request, content_keys=content) == {}
    assert source.calls == [(_digest("scope"), request, content)]
    for error in (AIError(ErrorCode.STORAGE_INTEGRITY_ERROR), asyncio.CancelledError()):
        failing = _ReadOnlyStateTransaction(cast(StateTransaction, _OptionalReader(error)))
        with pytest.raises(type(error)) as caught:
            await failing.read_archive_frontiers(scope_key=_digest("scope"), requests=request)
        assert caught.value is error


@pytest.mark.asyncio
async def test_fixed_frontier_returns_only_requested_bodies_and_reuses_transaction_records(tmp_path: Path) -> None:
    async with _archive_database(tmp_path / "frontier.db") as (engine, archive, prepared, root):
        first_id = root.slices[0].first
        first = prepared.nodes[first_id]
        first_key = archive._key("archive_sequence", first_id)
        identities = (first_id, first.left, first.right, first.next_forest)
        expected_nodes = {archive._key("archive_sequence", identity) for identity in identities}
        expected_content = archive._key("archive_content", first.content_id)
        statements = []

        def record_query(connection, cursor, statement, parameters, context, executemany) -> None:
            if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                statements.append((statement, parameters))

        event.listen(engine.sync_engine, "before_cursor_execute", record_query)

        async def read(transaction: StateTransaction) -> Mapping[bytes, StoredRecord]:
            assert isinstance(transaction, ArchiveFrontierReader)
            records = await transaction.read_archive_frontiers(
                scope_key=decode_archive_scope(archive.content_scope),
                requests=(ArchiveNodeRead(first_key, ALL, ArchiveNodeSlot.SELF),),
            )
            assert records is not None
            count = len(statements)
            assert await transaction.get_records(tuple(records)) == records
            assert len(statements) == count
            return records

        records = await archive.store.read(read)
        event.remove(engine.sync_engine, "before_cursor_execute", record_query)
        assert set(records) == expected_nodes | {expected_content}
        assert len(statements) == 1
        assert all(record.scope_key == decode_archive_scope(archive.content_scope) for record in records.values())
        sql, parameters = statements[0]
        async with engine.connect() as connection:
            plan = (await connection.exec_driver_sql("EXPLAIN QUERY PLAN " + sql, parameters)).all()
        assert any("SEARCH ai_state_records USING INDEX" in row[3] and "sort_prefix=?" in row[3] for row in plan), plan
        assert not any("SCAN ai_state_records" in row[3] for row in plan), plan
        assert all("sort_prefix=?" in row[3] or "key_locator=?" in row[3]
                   for row in plan if "SEARCH" in row[3] and "ai_state_records" in row[3]), plan


@pytest.mark.asyncio
async def test_frontier_unions_overlaps_direct_keys_and_preserves_scope(tmp_path: Path) -> None:
    async with _archive_database(tmp_path / "scope.db") as (engine, archive, prepared, root):
        first_id = root.slices[0].first
        first = prepared.nodes[first_id]
        first_key = archive._key("archive_sequence", first_id)
        direct = archive._key("archive_content", first.content_id)
        extra = next(record.key for record in prepared.records.values()
                     if record.kind == "archive_content" and record.key != direct)
        requests = (ArchiveNodeRead(first_key, ArchiveNodeSlot.NEXT, ArchiveNodeSlot.SELF),
                    ArchiveNodeRead(first_key, ArchiveNodeSlot.SELF, ArchiveNodeSlot.NEXT))

        async def read(transaction: StateTransaction, scope: bytes) -> Mapping[bytes, StoredRecord] | None:
            assert isinstance(transaction, ArchiveFrontierReader)
            return await transaction.read_archive_frontiers(scope_key=scope, requests=requests,
                                                            content_keys=(direct, extra, extra, _digest("missing")))

        records = await archive.store.read(lambda tx: read(tx, decode_archive_scope(archive.content_scope)))
        expected = {first_key, archive._key("archive_sequence", first.next_forest), direct, extra,
                    archive._key("archive_content", prepared.nodes[first.next_forest].content_id)}
        assert set(records) == expected
        assert await archive.store.read(lambda tx: read(tx, _digest("other-scope"))) == {}
        other = SqlStateStore(engine, group=archive.store.storage_group, store_key=_digest("other-store"))
        await other.initialize()
        try:
            assert await other.read(lambda tx: read(tx, decode_archive_scope(archive.content_scope))) == {}
        finally:
            await other.close()


@pytest.mark.asyncio
async def test_sql_frontier_batches_stay_within_999_parameters(tmp_path: Path) -> None:
    async with _archive_database(tmp_path / "bounded.db") as (engine, archive, prepared, root):
        actual = ArchiveNodeRead(archive._key("archive_sequence", root.slices[0].first), ALL, ALL)
        requests = (actual,) + tuple(ArchiveNodeRead(_digest(f"missing-node-{index}"), ALL, ALL) for index in range(_NODE_REQUESTS - 1))
        contents = tuple(record.key for record in prepared.records.values() if record.kind == "archive_content")
        missing_contents = tuple(_digest(f"missing-content-{index}") for index in range(_CONTENT_REQUESTS - len(contents)))
        contents = contents[1:] + missing_contents + contents[:1]
        parameter_counts = []

        def check_query(connection, cursor, statement, parameters, context, executemany) -> None:
            if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                parameter_counts.append(len(parameters))
                assert len(parameters) <= 999

        event.listen(engine.sync_engine, "before_cursor_execute", check_query)

        async def read(transaction: StateTransaction) -> Mapping[bytes, StoredRecord] | None:
            assert isinstance(transaction, ArchiveFrontierReader)
            return await transaction.read_archive_frontiers(scope_key=decode_archive_scope(archive.content_scope),
                                                            requests=requests, content_keys=contents)

        records = await archive.store.read(read)
        event.remove(engine.sync_engine, "before_cursor_execute", check_query)
        assert len(parameter_counts) > 1
        assert {record.sort_key for record in records.values() if record.kind == "archive_content"} == {
            record.sort_key for record in prepared.records.values() if record.kind == "archive_content"
        }


def test_fixed_frontier_compiles_for_supported_server_dialects_with_bounded_parameters() -> None:
    engine = create_async_engine("sqlite+aiosqlite://")
    metadata = build_runtime_sql_metadata(frozenset((PersistenceDomain.EXECUTION,)))
    transaction = _SqlTransaction(AsyncSession(engine), metadata, create_sql_storage_context(engine), _digest("store"))
    requests = tuple(ArchiveNodeRead(_digest(f"node-{index}"), ALL, ALL) for index in range(_NODE_REQUESTS))
    content_keys = tuple(_digest(f"body-{index}") for index in range(_CONTENT_REQUESTS))
    for dialect in (sqlite.dialect(), mysql.dialect(server_version_info=(5, 7, 44)), postgresql.dialect()):
        transaction._context.dialect = dialect_for_name(dialect.name)
        for nodes, contents in transaction._archive_read_batches(requests, content_keys):
            statement = transaction._archive_read_statement(_digest("scope"), nodes, contents)
            compiled = statement.compile(dialect=dialect, compile_kwargs={"render_postcompile": True})
            parameters = len(compiled.params) if compiled.positiontup is None else len(compiled.positiontup)
            assert parameters <= 999
            assert "WITH RECURSIVE" not in str(compiled).upper()
            sql = str(compiled).upper()
            assert "NOT IN" not in sql
            frontier = statement.selects[0] if hasattr(statement, "selects") else statement
            if nodes:
                identities = frontier.get_final_froms()[0].left
                assert identities.c.identity.type.dialect_impl(dialect).python_type is bytes
            if nodes:
                if dialect.name == "postgresql":
                    assert "CONVERT_TO(" in sql
                else:
                    assert " AS BINARY)" in sql if dialect.name == "mysql" else " AS BLOB)" in sql


@pytest.mark.asyncio
async def test_direct_step_content_uses_exact_index_without_sequence_union(tmp_path: Path) -> None:
    async with _archive_database(tmp_path / "direct-step.db") as (engine, archive, prepared, root):
        scope, store = await archive.store.storage_group.open_archive(create=True)
        archive = ContentArchive(store, scope, namespace="namespace", tenant_id="tenant", object_store=None)
        prepared = PreparedArchive()
        root = await archive.prepare_append(archive.empty_root(), (("model-message", b"message"),), prepared)
        await store.mutate(lambda tx: archive.publish(tx, prepared))
        identities = await archive.prepare_content((("execution-payload", b"payload"),), prepared)
        await archive.store.mutate(lambda tx: archive.publish(tx, prepared))
        key = archive._key("archive_content", identities[0])
        statements = []

        def record_query(connection, cursor, statement, parameters, context, executemany) -> None:
            if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                statements.append((statement, parameters))

        event.listen(engine.sync_engine, "before_cursor_execute", record_query)
        records = await archive.store.read(lambda tx: tx.read_archive_frontiers(
            scope_key=decode_archive_scope(archive.content_scope), requests=(), content_keys=(key,),
        ))
        assert set(records) == {key}
        assert len(statements) == 1
        sql, parameters = statements.pop()
        assert "ai_state_records" not in sql
        async with engine.connect() as connection:
            plan = (await connection.exec_driver_sql("EXPLAIN QUERY PLAN " + sql, parameters)).all()
        assert any("SEARCH ai_step_messages" in row[3] and "store_locator=? AND sha256=?" in row[3] for row in plan), plan
        await archive.store.read(lambda tx: tx.read_archive_frontiers(
            scope_key=decode_archive_scope(archive.content_scope),
            requests=(ArchiveNodeRead(archive._key("archive_sequence", root.slices[0].first)),),
        ))
        event.remove(engine.sync_engine, "before_cursor_execute", record_query)
        assert len(statements) == 1
        assert "ai_step_messages" not in statements[0][0]
