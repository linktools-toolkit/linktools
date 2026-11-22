#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exercise binary SQL boundaries with aiomysql on an isolated local MySQL server."""

import hashlib
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import event, text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._sql import SqlStateStore
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import SqlObjectStore

from ._metrics_server_test_helpers import _server_database
from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


@pytest_asyncio.fixture(params=(False, True), ids=("escaped", "no-backslash-escapes"))
async def mysql_binary_engine(
    request: pytest.FixtureRequest, _mysql_server: tuple[str, list[str]],
) -> AsyncIterator[AsyncEngine]:
    pytest.importorskip("aiomysql")
    with _server_database(_mysql_server) as (_, command):
        socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
        url = URL.create("mysql+aiomysql", username="root", database=command[-1],
                         query={"unix_socket": socket, "charset": "utf8mb4"})
        engine = create_async_engine(url, pool_size=1, max_overflow=0, hide_parameters=True)
        try:
            await provision_database(engine)
            async with engine.connect() as connection:
                original_mode = await connection.scalar(text("SELECT @@SESSION.sql_mode"))
                modes = set(original_mode.split(",")) - {"", "NO_BACKSLASH_ESCAPES"}
                if request.param:
                    modes.add("NO_BACKSLASH_ESCAPES")
                await connection.execute(text("SET SESSION sql_mode = :mode"), {"mode": ",".join(sorted(modes))})
                assert ("NO_BACKSLASH_ESCAPES" in (await connection.scalar(
                    text("SELECT @@SESSION.sql_mode")
                )).split(",")) is request.param
            try:
                yield engine
            finally:
                async with engine.connect() as connection:
                    await connection.execute(text("SET SESSION sql_mode = :mode"), {"mode": original_mode})
                    assert await connection.scalar(text("SELECT @@SESSION.sql_mode")) == original_mode
        finally:
            await engine.dispose()


@pytest.mark.asyncio
async def test_aiomysql_session_history_finds_committed_turn_and_returns_none_when_missing(
    mysql_binary_engine: AsyncEngine,
) -> None:
    async def model(messages, info):
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("binary-history")
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.sync()), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    features = model_features(_Models(_UsageFunctionModel(model)), features)
    storage = PersistenceStorage.sql(mysql_binary_engine)
    async with Runtime.open("binary-history", storage=storage, features=features) as owner:
        principal = owner.default_principal
        assert await owner.history.lookup_session("missing", principal=principal) is None
        await owner.agents.get().create_session("present")
        execution = await owner.sessions.start("present", "question", principal=principal)
        assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED

    async with Runtime.open("binary-history", storage=storage, features=features) as reopened:
        session = await reopened.history.lookup_session("present", principal=reopened.default_principal)
        assert session is not None and session.session_id == "present"
        detail = await reopened.history.session_detail("present", principal=reopened.default_principal, max_age=0)
        assert detail.session.session_id == "present"
        turn, = detail.timeline.items
        assert turn.execution_id == execution.execution_id
        assert turn.conversation_committed
        assert any(item.item_kind == "assistant" and item.content == "answer" for item in turn.items)


@pytest.mark.asyncio
async def test_aiomysql_counter_matches_existing_literal_bytes_and_reserves_batch(
    mysql_binary_engine: AsyncEngine,
) -> None:
    async with Persistence.open("binary-counter", storage=PersistenceStorage.sql(mysql_binary_engine)) as state:
        probe = await state.capability_store("probe")
        for key in (b"test\x00\xff", b"test00ff"):
            assert await probe.mutate(lambda tx: tx.next_sequence(key)) == 1
            assert await probe.read(lambda tx: tx.get_sequence(key)) == 1

    # These fixtures reach MySQL directly, bypassing SDK write-bind conversion.
    stream = bytes.fromhex("00FF5C271A80")
    statements: list[tuple[str, bool]] = []

    def capture(_conn, _cursor, statement: str, _parameters, _context, many: bool) -> None:
        if "ai_state_counters" in statement:
            statements.append((statement, many))

    async with mysql_binary_engine.begin() as connection:
        await connection.execute(text(
            "INSERT INTO ai_state_counters (store_key, stream_key, value, stream_locator) "
            "VALUES (X'007F80FF', X'00FF5C271A80', 7, SHA2(X'00FF5C271A80', 256))"
        ))
    store = SqlStateStore(mysql_binary_engine, store_key=bytes.fromhex("007F80FF"))
    await store.initialize()
    event.listen(mysql_binary_engine.sync_engine, "before_cursor_execute", capture)
    try:
        assert await store.read(lambda tx: tx.get_sequence(stream)) == 7
        assert await store.read(lambda tx: tx.get_sequence(stream + b"x")) == 0
        requested = {b"new-" + bytes((index, 255, 92, 39, 0)): index + 1 for index in range(8)}
        statements.clear()
        assert await store.mutate(lambda tx: tx.reserve_sequences(requested)) == requested
        writes = [(statement, many) for statement, many in statements if statement.lstrip().upper().startswith("INSERT")]
        assert len(writes) == 1, "reserving eight independent counters must not issue eight statements"
        assert await store.read(lambda tx: tx.scan_sequences()) == {stream: 7, **requested}
    finally:
        event.remove(mysql_binary_engine.sync_engine, "before_cursor_execute", capture)
        await store.close()


@pytest.mark.asyncio
async def test_aiomysql_object_multichunk_round_trip_keeps_exact_binary(
    mysql_binary_engine: AsyncEngine,
) -> None:
    store = SqlObjectStore(mysql_binary_engine)
    key = "opaque/数🌿\\'"
    payload = bytes(range(256)) * (3 * 1024 * 1024 // 256 + 7)
    digest = hashlib.sha256(payload).hexdigest()

    async def chunks() -> AsyncIterator[bytes]:
        for offset in range(0, len(payload), 77_777):
            yield payload[offset:offset + 77_777]

    await store.put(key, chunks(), expected_size=len(payload), expected_digest=digest)
    info = await store.stat(key)
    assert info is not None and (info.size, info.digest) == (len(payload), digest)
    async with mysql_binary_engine.connect() as connection:
        physical_chunks = await connection.scalar(text("SELECT COUNT(*) FROM ai_object_chunks"))
    assert physical_chunks > 1
    assert b"".join([chunk async for chunk in store.open(key)]) == payload


@pytest.mark.asyncio
async def test_aiomysql_binary_state_batch_insert_replace_and_delete(mysql_binary_engine: AsyncEngine) -> None:
    from dataclasses import replace

    from linktools.ai.persistence.state import RecordReplacement, StoredRecord

    async with Persistence.open("binary-records", storage=PersistenceStorage.sql(mysql_binary_engine)) as state:
        store = await state.capability_store("probe")
        rows = tuple(
            StoredRecord(
                key=key, scope_key=None, parent_key=None, kind="probe", sort_key=key,
                state=None, storage_revision=0, lease_owner=None, lease_fence=0,
                lease_expires_at=None, data={"key": key.hex()}, payload_bytes=payload,
            )
            for key, payload in ((b"\x00\xff", b""), (b"00ff", None), (b"all", bytes(range(256))))
        )
        keys = tuple(row.key for row in rows)
        await store.mutate(lambda tx: tx.insert_records(rows))
        assert await store.read(lambda tx: tx.get_records(keys)) == {row.key: row for row in rows}

        changed_rows = tuple(replace(row, storage_revision=1, payload_bytes=body)
                             for row, body in zip(rows, (None, b"", b"\x00\xff\\'")))
        await store.mutate(lambda tx: tx.replace_records(tuple(RecordReplacement(row, 0) for row in changed_rows)))
        assert await store.read(lambda tx: tx.get_records(keys)) == {row.key: row for row in changed_rows}

        await store.mutate(lambda tx: tx.delete_records(keys[:2]))
        assert await store.read(lambda tx: tx.get_records(keys)) == {changed_rows[2].key: changed_rows[2]}
