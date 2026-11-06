#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MySQL bulk writes retain driver packet limits and outer transaction ownership."""

from dataclasses import replace

import pytest
from asyncmy.connection import Connection
from asyncmy.cursors import Cursor
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord, StepEvent
from sqlalchemy import Column, Integer, MetaData, String, Table, Text, UniqueConstraint, event, select, text
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.dialects.mysql.asyncmy import dialect as mysql_dialect
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.sql import ClauseElement

from linktools.ai.migrate import provision_database
from linktools.ai.persistence import SqlStepStore
from linktools.ai.storage import MySQLDialect, SqlValue


_PACKET_LIMIT = 4 * 1024 * 1024
_OPERATIONS = ("insert_ignore_conflict_many", "upsert_many", "upsert_increment_many")


def _table() -> Table:
    return Table(
        "ai_packet_entries", MetaData(),
        Column("id", Integer, primary_key=True),
        Column("key", String(32), nullable=False),
        Column("body", Text().with_variant(LONGTEXT(), "mysql"), nullable=False),
        Column("revision", Integer, nullable=False),
        UniqueConstraint("key"),
    )


async def _write(
    dialect: MySQLDialect, session: "AsyncSession | _PacketSession", table: Table,
    rows: list[dict[str, SqlValue]], operation: str,
) -> int | dict[str, int] | None:
    arguments = dict(table=table, rows=rows, index_elements=("key",))
    if operation == "upsert_many":
        return await dialect.upsert_many(session, **arguments, set_columns=("body",))
    if operation == "upsert_increment_many":
        return await dialect.upsert_increment_many(
            session, **arguments, column="revision", returning_key="key",
        )
    return await dialect.insert_ignore_conflict_many(session, **arguments)


class _PacketCursor(Cursor):
    def __init__(self, *, duplicate: bool = False) -> None:
        connection = Connection(charset="utf8mb4")
        connection.server_status = 0
        super().__init__(connection)
        self.packets: list[bytes] = []
        self.duplicate = duplicate

    async def execute(
        self, query: str | bytes, args: tuple[SqlValue, ...] | None = None,
    ) -> int:
        wire = self.mogrify(query, args) if args is not None else query
        if isinstance(wire, str):
            wire = wire.encode("utf8")
        assert len(wire) + 1 <= _PACKET_LIMIT
        self.packets.append(bytes(wire))
        if self.duplicate and len(self.packets) == 2:
            raise IntegrityError("insert", {}, Exception(1062, "Duplicate entry"))
        self.rowcount = 1
        return 1


class _PacketSession:
    def __init__(self, cursor: _PacketCursor) -> None:
        self.cursor = cursor
        self.selects = 0

    async def execute(
        self, statement: ClauseElement, parameters: list[dict[str, SqlValue]] | None = None,
    ) -> "_PacketSession":
        if statement.is_select:
            self.selects += 1
            return self
        compiler = statement.compile(
            dialect=mysql_dialect(),
            column_keys=list(parameters[0]) if parameters else None,
        )
        rows = [compiler.construct_params(row) for row in parameters] if parameters else [compiler.params]
        bound = [tuple(row[key] for key in compiler.positiontup) for row in rows]
        if len(bound) > 1:
            await self.cursor.executemany(str(compiler), bound)
        else:
            await self.cursor.execute(str(compiler), bound[0])
        return self

    def all(self) -> list[tuple[str, int]]:
        return []


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("body,count", (
    ("body", 64),
    ("x" * (80 * 1024), 64),
    ("\\'\"\0測🙂" * (10 * 1024), 64),
    ("x" * (2 * 1024 * 1024), 1),
), ids=("small", "large", "escaped-unicode", "large-individual-row"))
async def test_mysql_bulk_packets_fit_without_splitting_small_batches_or_single_rows(
    operation: str, body: str, count: int,
) -> None:
    cursor = _PacketCursor()
    rows = [{"key": str(index), "body": body, "revision": 1} for index in range(count)]
    session = _PacketSession(cursor)
    await _write(MySQLDialect(), session, _table(), rows, operation)
    assert session.selects == (1 if operation == "upsert_increment_many" else 0)
    if body == "body" or count == 1:
        assert len(cursor.packets) == 1
    else:
        assert len(cursor.packets) > 1
        assert all(len(packet) <= cursor.max_stmt_length for packet in cursor.packets)
    # Check escaped, encoded bytes, including every row, rather than Python string lengths.
    expected = cursor.mogrify("%s", (body,)).encode("utf8")
    assert sum(packet.count(expected) for packet in cursor.packets) == count


@pytest.mark.asyncio
async def test_mysql_duplicate_after_partial_batch_replays_all_rows_with_bounded_packets() -> None:
    cursor = _PacketCursor(duplicate=True)
    body = "x" * (80 * 1024)
    rows = [{"key": str(index), "body": body, "revision": 1} for index in range(64)]
    result = await _write(MySQLDialect(), _PacketSession(cursor), _table(), rows, _OPERATIONS[0])
    assert result is None
    retried = [packet for packet in cursor.packets if b"ON DUPLICATE KEY UPDATE" in packet]
    assert retried
    assert all(len(packet) <= cursor.max_stmt_length for packet in retried)
    expected = cursor.mogrify("%s", (body,)).encode("utf8")
    assert sum(packet.count(expected) for packet in retried) == len(rows)


@pytest.mark.asyncio
async def test_mysql_event_retry_does_not_replay_neighboring_unkeyed_events() -> None:
    class EventCursor(_PacketCursor):
        async def execute(
            self, query: str | bytes, args: tuple[SqlValue, ...] | None = None,
        ) -> int:
            result = await super().execute(query, args)
            if len(self.packets) == 3:
                raise IntegrityError("insert", {}, Exception(1062, "Duplicate entry"))
            return result

    engine = create_async_engine("mysql+asyncmy://")
    store = SqlStepStore(engine, namespace="events", tenant_id="one")
    cursor = EventCursor()
    unkeyed = StepEvent(run_id="run", kind="model_request_started", step_index=0,
                       error="unkeyed-marker")
    keyed = [replace(unkeyed, idempotency_key=str(index), error="x" * (80 * 1024))
             for index in range(64)]
    try:
        await store._write_events(_PacketSession(cursor), (unkeyed, *keyed, unkeyed))
        assert any(b"ON DUPLICATE KEY UPDATE" in packet for packet in cursor.packets)
        assert sum(packet.count(b"unkeyed-marker") for packet in cursor.packets) == 2
        assert all(len(packet) <= cursor.max_stmt_length for packet in cursor.packets)
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("_server", ("mysql",), indirect=True)
async def test_mysql_mixed_event_batch_preserves_occurrences_after_packet_conflict(
    _server: tuple[str, list[str]],
) -> None:
    _, command = _server
    socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
    url = URL.create("mysql+asyncmy", username="root", database=command[-1],
                     query={"unix_socket": socket, "charset": "utf8mb4"})
    engine = create_async_engine(url, hide_parameters=True)
    store = SqlStepStore(engine, namespace="mixed-events", tenant_id="one")
    unkeyed = StepEvent(run_id="run", kind="model_request_started", step_index=0)
    keyed = [replace(unkeyed, idempotency_key=str(index), error="x" * (80 * 1024))
             for index in range(64)]
    try:
        await provision_database(engine)
        await store.append_event(keyed[-1])
        await store.commit_step(events=(unkeyed, *keyed[:-1], replace(keyed[-1], error="ignored"), unkeyed))
        assert await store.list_events(run_id="run") == [keyed[-1], unkeyed, *keyed[:-1], unkeyed]
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("_server", ("mysql",), indirect=True)
async def test_mysql_large_native_snapshot_rolls_back_and_retries_at_packet_limit(
    _server: tuple[str, list[str]],
) -> None:
    _, command = _server
    socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
    url = URL.create("mysql+asyncmy", username="root", database=command[-1],
                     query={"unix_socket": socket, "charset": "utf8mb4"})
    admin = create_async_engine(url, hide_parameters=True)
    engine = None
    store = None
    try:
        async with admin.connect() as connection:
            original = await connection.scalar(text("SELECT @@GLOBAL.max_allowed_packet"))
            try:
                await connection.execute(text(f"SET GLOBAL max_allowed_packet = {_PACKET_LIMIT}"))
                engine = create_async_engine(url, hide_parameters=True, pool_size=1, max_overflow=0)
                await provision_database(engine)
                async with engine.connect() as probe:
                    assert await probe.scalar(text("SELECT @@SESSION.max_allowed_packet")) == _PACKET_LIMIT
                store = SqlStepStore(engine, namespace="packets", tenant_id="one")
                run = RunRecord(run_id="run", registration_id="registration")
                snapshot = ContinuableSnapshot(run_id="run", step_index=0, idempotency_key="snapshot",
                    messages=[ModelRequest(parts=[UserPromptPart(f"{index}:" + "x" * (80 * 1024))])
                              for index in range(64)])

                def fail_refs(connection, cursor, statement, parameters, context, many):
                    if statement.startswith("INSERT INTO ai_step_message_refs "):
                        raise RuntimeError("reference write failed")

                event.listen(engine.sync_engine, "before_cursor_execute", fail_refs)
                try:
                    with pytest.raises(RuntimeError, match="reference write failed"):
                        await store.commit_step(run=run, snapshot=snapshot)
                finally:
                    event.remove(engine.sync_engine, "before_cursor_execute", fail_refs)
                async with engine.connect() as probe:
                    for name in ("runs", "snapshots", "messages", "message_refs"):
                        assert await probe.scalar(text(f"SELECT count(*) FROM ai_step_{name}")) == 0
                await store.commit_step(run=run, snapshot=snapshot)
                assert await store.latest_snapshot(run_id="run") == snapshot
                await store.close()
                store = SqlStepStore(engine, namespace="packets", tenant_id="one")
                next_snapshot = replace(snapshot, step_index=1, idempotency_key="next")
                await store.save_snapshot(next_snapshot)
                assert await store.latest_snapshot(run_id="run") == next_snapshot
            finally:
                await connection.execute(text(f"SET GLOBAL max_allowed_packet = {int(original)}"))
    finally:
        if store is not None:
            await store.close()
        if engine is not None:
            await engine.dispose()
        await admin.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("_server", ("mysql",), indirect=True)
@pytest.mark.parametrize("operation", _OPERATIONS)
async def test_mysql_partial_bulk_conflicts_and_failures_preserve_outer_transaction(
    _server: tuple[str, list[str]], operation: str,
) -> None:
    _, command = _server
    socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
    url = URL.create("mysql+asyncmy", username="root", database=command[-1],
                     query={"unix_socket": socket, "charset": "utf8mb4"})
    engine = create_async_engine(url, hide_parameters=True)
    table = _table()
    rows = [{"key": str(index), "body": "x" * (80 * 1024), "revision": 1} for index in range(64)]
    # Place the existing row after multiple successful driver batches.
    original = {**rows[-1], "body": "retained", "revision": 10}
    try:
        async with engine.begin() as connection:
            await connection.run_sync(table.metadata.create_all)
            await connection.execute(table.insert(), original)
        async with AsyncSession(engine) as session:
            with pytest.raises(RuntimeError, match="outer failure"):
                async with session.begin():
                    await _write(MySQLDialect(), session, table, rows, operation)
                    raise RuntimeError("outer failure")
            stored = (await session.execute(select(table.c.key, table.c.body, table.c.revision))).all()
            assert stored == [(original["key"], "retained", 10)]
        async with AsyncSession(engine) as session, session.begin():
            result = await _write(MySQLDialect(), session, table, rows, operation)
            if operation == "insert_ignore_conflict_many":
                assert result is None
            elif operation == "upsert_increment_many":
                assert result == {row["key"]: (11 if row["key"] == original["key"] else 1) for row in rows}
        async with engine.connect() as connection:
            stored = (await connection.execute(select(table.c.key, table.c.body, table.c.revision))).all()
        assert len(stored) == len(rows)
        for key, body, revision in stored:
            if key != original["key"]:
                assert (body, revision) == (rows[0]["body"], 1)
            else:
                assert body == (rows[0]["body"] if operation == "upsert_many" else "retained")
                assert revision == (11 if operation == "upsert_increment_many" else 10)
        # A non-duplicate error in a later batch also rolls back its earlier rows.
        async with engine.begin() as connection:
            await connection.execute(table.delete())
        invalid = [dict(row) for row in rows]
        invalid[-1]["body"] = None
        async with AsyncSession(engine) as session:
            with pytest.raises(IntegrityError):
                async with session.begin():
                    await _write(MySQLDialect(), session, table, invalid, operation)
        async with engine.connect() as connection:
            assert (await connection.execute(select(table))).all() == []
    finally:
        async with engine.begin() as connection:
            await connection.run_sync(table.metadata.drop_all)
        await engine.dispose()
