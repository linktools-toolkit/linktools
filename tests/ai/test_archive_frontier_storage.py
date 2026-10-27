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
from sqlalchemy import event, text
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.persistence.runs import ArchiveFrontierReader, ArchiveNodeRead, ArchiveNodeSlot
from linktools.ai.persistence._archive import ArchiveRoot, ContentArchive, PreparedArchive
from linktools.ai.persistence._plan import PersistenceDomain
from linktools.ai.persistence._schema import build_runtime_sql_metadata
from linktools.ai.persistence._sql import SqlStateStore, _SqlTransaction
from linktools.ai.persistence.state import StateTransaction, StoredRecord, _ReadOnlyStateTransaction
from linktools.ai.storage import create_sql_storage_context


ALL = ArchiveNodeSlot.SELF | ArchiveNodeSlot.LEFT | ArchiveNodeSlot.RIGHT | ArchiveNodeSlot.NEXT


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
    identities = await archive.prepare_content(tuple(("bytes", f"body-{index}".encode()) for index in range(10)), prepared)
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
    for bad_key in (b"", b"short"):
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
        self, *, scope_digest: bytes, requests: Sequence[ArchiveNodeRead], content_keys: Sequence[bytes] = (),
    ) -> Mapping[bytes, StoredRecord] | None:
        self.calls.append((scope_digest, requests, content_keys))
        if self.error is not None:
            raise self.error
        return {}


@pytest.mark.asyncio
async def test_readonly_optional_reader_distinguishes_unsupported_empty_and_errors() -> None:
    unsupported = _ReadOnlyStateTransaction(cast(StateTransaction, _UnsupportedReader()))
    assert isinstance(unsupported, ArchiveFrontierReader)
    assert await unsupported.read_archive_frontiers(scope_digest=_digest("scope"), requests=()) is None
    source = _OptionalReader()
    supported = _ReadOnlyStateTransaction(cast(StateTransaction, source))
    request = (ArchiveNodeRead(_digest("node")),)
    content = (_digest("body"),)
    assert await supported.read_archive_frontiers(scope_digest=_digest("scope"), requests=request, content_keys=content) == {}
    assert source.calls == [(_digest("scope"), request, content)]
    for error in (AIError(ErrorCode.STORAGE_INTEGRITY_ERROR), asyncio.CancelledError()):
        failing = _ReadOnlyStateTransaction(cast(StateTransaction, _OptionalReader(error)))
        with pytest.raises(type(error)) as caught:
            await failing.read_archive_frontiers(scope_digest=_digest("scope"), requests=request)
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
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(context.compiled.statement)

        event.listen(engine.sync_engine, "before_cursor_execute", record_query)

        async def read(transaction: StateTransaction) -> Mapping[bytes, StoredRecord]:
            assert isinstance(transaction, ArchiveFrontierReader)
            records = await transaction.read_archive_frontiers(
                scope_digest=bytes.fromhex(archive.content_scope),
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
        assert all(record.scope_digest == bytes.fromhex(archive.content_scope) for record in records.values())
        sql = str(statements[0].compile(dialect=sqlite.dialect(), compile_kwargs={"literal_binds": True}))
        async with engine.connect() as connection:
            plan = (await connection.execute(text("EXPLAIN QUERY PLAN " + sql))).all()
        assert any("SEARCH ai_state_records USING INDEX" in row[3] and "sort_key=?" in row[3] for row in plan), plan
        assert not any("SCAN ai_state_records" in row[3] for row in plan), plan
        assert all("sort_key=?" in row[3] or "key_digest=?" in row[3]
                   for row in plan if "SEARCH" in row[3] and "ai_state_records" in row[3]), plan


@pytest.mark.asyncio
async def test_frontier_unions_overlaps_direct_keys_and_preserves_scope(tmp_path: Path) -> None:
    async with _archive_database(tmp_path / "scope.db") as (engine, archive, prepared, root):
        first_id = root.slices[0].first
        first = prepared.nodes[first_id]
        first_key = archive._key("archive_sequence", first_id)
        direct = archive._key("archive_content", first.content_id)
        extra = next(record.key_digest for record in prepared.records.values()
                     if record.kind == "archive_content" and record.key_digest != direct)
        requests = (ArchiveNodeRead(first_key, ArchiveNodeSlot.NEXT, ArchiveNodeSlot.SELF),
                    ArchiveNodeRead(first_key, ArchiveNodeSlot.SELF, ArchiveNodeSlot.NEXT))

        async def read(transaction: StateTransaction, scope: bytes) -> Mapping[bytes, StoredRecord] | None:
            assert isinstance(transaction, ArchiveFrontierReader)
            return await transaction.read_archive_frontiers(scope_digest=scope, requests=requests,
                                                            content_keys=(direct, extra, extra, _digest("missing")))

        records = await archive.store.read(lambda tx: read(tx, bytes.fromhex(archive.content_scope)))
        expected = {first_key, archive._key("archive_sequence", first.next_forest), direct, extra,
                    archive._key("archive_content", prepared.nodes[first.next_forest].content_id)}
        assert set(records) == expected
        assert await archive.store.read(lambda tx: read(tx, _digest("other-scope"))) == {}
        other = SqlStateStore(engine, group=archive.store.storage_group, store_digest=_digest("other-store"))
        await other.initialize()
        try:
            assert await other.read(lambda tx: read(tx, bytes.fromhex(archive.content_scope))) == {}
        finally:
            await other.close()


@pytest.mark.asyncio
async def test_sql_frontier_batches_stay_within_999_parameters(tmp_path: Path) -> None:
    async with _archive_database(tmp_path / "bounded.db") as (engine, archive, prepared, root):
        actual = ArchiveNodeRead(archive._key("archive_sequence", root.slices[0].first), ALL, ALL)
        requests = (actual,) + tuple(ArchiveNodeRead(_digest(f"missing-node-{index}"), ALL, ALL) for index in range(500))
        contents = tuple(record.key_digest for record in prepared.records.values() if record.kind == "archive_content")
        contents += tuple(_digest(f"missing-content-{index}") for index in range(1200))
        parameter_counts = []

        def check_query(connection, cursor, statement, parameters, context, executemany) -> None:
            if statement.lstrip().upper().startswith("SELECT"):
                parameter_counts.append(len(parameters))
                assert len(parameters) <= 999

        event.listen(engine.sync_engine, "before_cursor_execute", check_query)

        async def read(transaction: StateTransaction) -> Mapping[bytes, StoredRecord] | None:
            assert isinstance(transaction, ArchiveFrontierReader)
            return await transaction.read_archive_frontiers(scope_digest=bytes.fromhex(archive.content_scope),
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
    requests = tuple(ArchiveNodeRead(_digest(f"node-{index}"), ALL, ALL) for index in range(500))
    content_keys = tuple(_digest(f"body-{index}") for index in range(500))
    for dialect in (sqlite.dialect(), mysql.dialect(server_version_info=(5, 7, 44)), postgresql.dialect()):
        for nodes, contents in transaction._archive_read_batches(requests, content_keys):
            statement = transaction._archive_read_statement(_digest("scope").hex(), nodes, contents)
            compiled = statement.compile(dialect=dialect, compile_kwargs={"render_postcompile": True})
            parameters = len(compiled.params) if compiled.positiontup is None else len(compiled.positiontup)
            assert parameters <= 999
            assert "WITH RECURSIVE" not in str(compiled).upper()
            assert "NOT IN" not in str(compiled).upper()
