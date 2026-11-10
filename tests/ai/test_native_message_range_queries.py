#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CI range correctness and SQL-work budgets, including MySQL 5.7 SQL shape."""

import getpass
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from sqlalchemy import event
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.migrate import provision_database
from linktools.ai.persistence import SqlStepStore
from linktools.ai.persistence._payload import encode_payload_bytes
from linktools.ai.persistence._step_message_content import (
    NativeMessageContent, NativeMessageRoot, NativeMessageSlice, NativeRangeBudget, PreparedMessagePayload,
)
from linktools.ai.persistence._transcript_staging import MessageArchive
from linktools.ai.storage import StoredPayload


_STAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


async def _payload(raw: bytes) -> PreparedMessagePayload:
    compression, _, _, encoded = encode_payload_bytes(raw)
    return PreparedMessagePayload(raw, encoded=encoded, compression=compression, payload=StoredPayload.inline_bytes(encoded))


class _Transaction:
    def __init__(self, native, session) -> None:
        self.native, self.session = native, session

    async def run_sql(self, context, callback):
        assert context is self.native.storage_context
        return await callback(self.session)


async def _publish(native, content, prepared):
    receipt = await native.storage_context.run_mutation(lambda session: content.publish(
        _Transaction(native, session), prepared, owner_inserted=True,
    ))
    return content.confirm_committed(receipt)


def _owner(dialect) -> NativeMessageContent:
    context = SimpleNamespace(engine=SimpleNamespace(dialect=dialect))
    return NativeMessageContent(context, group_id="execution", store_digest="a" * 64)


@pytest.mark.parametrize("dialect", [sqlite.dialect(), postgresql.dialect(), mysql.dialect()])
def test_range_query_limits_paths_binds_sql_and_unselected_bodies(dialect) -> None:
    content = _owner(dialect)
    root = NativeMessageRoot("execution", "a" * 64, (NativeMessageSlice("b" * 64, 1023, 501, 504),))
    plan = content.range_query((root,))
    compiled = str(plan.statement.compile(dialect=dialect)).upper()
    assert ("WITH RECURSIVE" in compiled) == (dialect.name != "mysql")
    assert "LEFT OUTER JOIN AI_STEP_MESSAGES" in compiled
    assert "POSITION >=" in compiled and "POSITION <" in compiled
    assert plan.bind_parameters <= 900 and plan.sql_bytes <= 256 * 1024
    assert plan.path_units < 1023 if dialect.name != "mysql" else plan.path_units < 1023 * (plan.depth + 1)
    if dialect.name == "mysql":
        assert "SELECT DISTINCT" in compiled
        assert plan.path_units > _owner(sqlite.dialect()).range_query((root,)).path_units
    for budget in (NativeRangeBudget(max_bind_parameters=1), NativeRangeBudget(max_sql_bytes=64),
                   NativeRangeBudget(max_path_units=1)):
        with pytest.raises(ValueError, match="budget"):
            content.range_query((root,), budget=budget)
    assert content.range_query((content.empty_root(),)) is None
    with pytest.raises(ValueError, match="seed budget"):
        content.range_query((root, replace(root, slices=(NativeMessageSlice("c" * 64, 1, 0, 1),))),
                            budget=NativeRangeBudget(max_seeds=1))
    with pytest.raises(ValueError, match="scope"):
        content.range_query((replace(root, group_id="other"),))
    with pytest.raises(ValueError, match="scope"):
        content.range_query((replace(root, store_digest="c" * 64),))


def test_mysql_tiny_ranges_do_not_expand_the_entire_forest_height() -> None:
    content = _owner(mysql.dialect())
    for count in (10_000, 1_000_000):
        tail = NativeMessageRoot("execution", "a" * 64,
                                 (NativeMessageSlice("b" * 64, count, count - 1, count),))
        plan = content.range_query((tail,))
        assert plan.depth == 0 and plan.path_units == 1
        assert plan.bind_parameters <= 4 and plan.sql_bytes < 2048
    first = replace(tail, slices=(NativeMessageSlice("b" * 64, 1_000_000, 0, 1),))
    plan = content.range_query((first,))
    assert plan.path_units >= plan.depth + 1
    assert plan.sql_bytes < 128 * 1024
    separated = replace(tail, slices=(NativeMessageSlice("b" * 64, 1024, 1, 2),))
    batch = content.range_query(tuple(replace(separated, slices=(NativeMessageSlice(f"{index:064x}", 1024, 1, 2),))
                                      for index in range(16)))
    assert batch.bind_parameters < 100 and batch.sql_bytes < 64 * 1024


@pytest.mark.parametrize("dialect", [sqlite.dialect(), postgresql.dialect(), mysql.dialect()])
def test_repeated_span_work_is_shared_and_singleton_batches_fit_one_read(dialect) -> None:
    content = _owner(dialect)
    single = NativeMessageRoot("execution", "a" * 64, (NativeMessageSlice("b" * 64, 1, 0, 1),))
    once, repeated = content.range_query((single,)), content.range_query((single,) * 1000)
    assert (once.bind_parameters, once.sql_bytes, once.path_units) == (
        repeated.bind_parameters, repeated.sql_bytes, repeated.path_units,
    )
    for count in (128, 446):
        distinct = tuple(replace(single, slices=(NativeMessageSlice(f"{index:064x}", 1, 0, 1),)) for index in range(count))
        batch = content.range_query(distinct)
        assert batch.path_units == count and batch.bind_parameters < count + 4
        assert batch.sql_bytes < 32 * 1024


@pytest.mark.parametrize("dialect", [sqlite.dialect(), postgresql.dialect(), mysql.dialect()])
def test_immutable_content_insert_never_updates_on_duplicate(dialect) -> None:
    content = _owner(dialect)
    for table in (content._messages, content._nodes):
        sql = str(content._content_insert(table).compile(dialect=dialect)).upper()
        assert "UPDATE" not in sql and "SELECT" not in sql and "FOR UPDATE" not in sql
        if dialect.name == "mysql":
            assert sql.startswith("INSERT IGNORE")
        else:
            assert "DO NOTHING" in sql


@pytest.mark.asyncio
async def test_range_hydration_returns_only_selected_bodies_and_one_read(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'ranges.db'}")
    native = SqlStepStore(engine, namespace="messages", tenant_id="one")
    await native.initialize(local=True)
    content = NativeMessageContent(native.storage_context, group_id="execution", store_digest=native.store_digest)
    archive = MessageArchive()
    frozen = archive.snapshot([ModelRequest(parts=[UserPromptPart(str(index), timestamp=_STAMP)], timestamp=_STAMP)
                               for index in range(255)])
    try:
        prepared = await content.prepare_append(frozen, prepare_payload=_payload)
        all_body_bytes = sum(len(body.encode()) for _, body in prepared.messages)
        proof = await _publish(native, content, prepared)
        selected = content.select(proof, 101, 105).root
        reads = []

        def capture(connection, cursor, statement, parameters, context, many) -> None:
            if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                reads.append(statement)

        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        plan = content.range_query((selected,))
        async with engine.connect() as connection:
            rows = (await connection.execute(plan.statement)).mappings().all()
        decoded, = await content.hydrate_ranges((selected,), rows)
        assert len(reads) == 1
        assert decoded == tuple(frozen[101:105].encoded_payloads())
        assert sum(row["message"] is not None for row in rows) == 4
        assert len(rows) < 4 + 4 * (255).bit_length()
        returned_body_bytes = sum(len(row["message"].encode()) for row in rows if row["message"] is not None)
        assert returned_body_bytes > 0
        assert returned_body_bytes < all_body_bytes
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        for bad in (rows[:-1], [*rows, rows[0]]):
            with pytest.raises(ValueError, match="incomplete|duplicate"):
                await content.hydrate_ranges((selected,), bad)
        bad = [dict(row) for row in rows]
        body_row = next(row for row in bad if row["message"] is not None)
        body_row["message_sha256"] = "f" * 64
        with pytest.raises(ValueError, match="corrupt"):
            await content.hydrate_ranges((selected,), bad)
    finally:
        await native.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_trim_branch_and_fragmented_selections_keep_exact_occurrence_order(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'branch.db'}")
    native = SqlStepStore(engine, namespace="messages", tenant_id="one")
    await native.initialize(local=True)
    content = NativeMessageContent(native.storage_context, group_id="execution", store_digest=native.store_digest)
    archive = MessageArchive()
    frozen = archive.snapshot([ModelRequest(parts=[UserPromptPart(value, timestamp=_STAMP)], timestamp=_STAMP)
                               for value in ("a", "b", "a", "c", "a", "d", "a", "e", "a", "f")])
    try:
        proof = await _publish(native, content, await content.prepare_append(frozen, prepare_payload=_payload))
        trimmed = content.select(proof, 1, 5)
        branch = await content.prepare_append(frozen[8:10], prepare_payload=_payload, base=trimmed)
        assert len(branch.nodes) == 2
        branch_proof = await _publish(native, content, branch)
        fragments = NativeMessageRoot("execution", native.store_digest,
            proof.root.slice(0, 2).slices + proof.root.slice(6, 9).slices)
        roots = (trimmed.root, branch_proof.root, fragments, proof.root.slice(2, 2))
        plan = content.range_query(roots)
        async with engine.connect() as connection:
            rows = (await connection.execute(plan.statement)).mappings().all()
        groups = await content.hydrate_ranges(roots, rows)
        payloads = tuple(frozen.encoded_payloads())
        assert groups == (payloads[1:5], payloads[1:5] + payloads[8:10], payloads[:2] + payloads[6:9], ())
        unique_slices = set(item for root in roots for item in root.slices)
        assert sum(row["message"] is not None for row in rows) == sum(item.end - item.start for item in unique_slices)
        foreign = NativeMessageContent(native.storage_context, group_id="other", store_digest=native.store_digest)
        with pytest.raises(ValueError, match="scope"):
            await foreign.hydrate_ranges(tuple(root if isinstance(root, NativeMessageRoot) else content.scoped_root(root)
                                                for root in roots), rows)
    finally:
        await native.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_server_native_cas_and_range_execution_preserve_occurrences(_server: tuple[str, list[str]]) -> None:
    name, command = _server
    if name == "postgresql":
        url = URL.create("postgresql+asyncpg", username=getpass.getuser(), database=command[-1],
                         query={"host": command[command.index("-h") + 1]})
    else:
        socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
        url = URL.create("mysql+asyncmy", username="root", database=command[-1],
                         query={"unix_socket": socket, "charset": "utf8mb4"})
    engine = create_async_engine(url)
    await provision_database(engine)
    native = SqlStepStore(engine, namespace="native-range-contract", tenant_id="one")
    await native.initialize()
    archive = MessageArchive()
    frozen = archive.snapshot([ModelRequest(parts=[UserPromptPart(str(index % 3), timestamp=_STAMP)], timestamp=_STAMP)
                               for index in range(35)])
    statements = []

    def capture(connection, cursor, statement, parameters, context, many) -> None:
        statements.append(statement.lstrip().upper())

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        for _ in range(2):
            content = NativeMessageContent(native.storage_context, group_id="execution",
                                           store_digest=native.store_digest, max_cached_messages=0, max_cached_nodes=0)
            prepared = await content.prepare_append(frozen, prepare_payload=_payload)
            proof = await _publish(native, content, prepared)
        assert not any(sql.startswith(("SELECT", "WITH", "UPDATE")) for sql in statements)
        roots = (proof.root.slice(0, 3), proof.root.slice(14, 18), proof.root.slice(34, 35), proof.root.slice(14, 18))
        statements.clear()
        plan = content.range_query(roots)
        async with engine.connect() as connection:
            rows = (await connection.execute(plan.statement)).mappings().all()
        payloads = tuple(frozen.encoded_payloads())
        assert await content.hydrate_ranges(roots, rows) == (
            payloads[:3], payloads[14:18], payloads[34:], payloads[14:18],
        )
        assert sum(sql.startswith(("SELECT", "WITH")) for sql in statements) == 1
        assert sum(row["message"] is not None for row in rows) == 8
        singleton = await _publish(native, content, await content.prepare_append(frozen[:1], prepare_payload=_payload))
        singleton_plan = content.range_query((singleton.root, singleton.root))
        async with engine.connect() as connection:
            singleton_rows = (await connection.execute(singleton_plan.statement)).mappings().all()
        assert await content.hydrate_ranges((singleton.root, singleton.root), singleton_rows) == (payloads[:1], payloads[:1])
        assert len(singleton_rows) == 1
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await native.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_actual_readback_resumes_frozen_append_without_another_select(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'readback.db'}")
    native = SqlStepStore(engine, namespace="native-readback", tenant_id="one")
    await native.initialize(local=True)
    archive = MessageArchive()
    frozen = archive.snapshot([ModelRequest(parts=[UserPromptPart(str(index), timestamp=_STAMP)], timestamp=_STAMP)
                               for index in range(35)])
    warm = NativeMessageContent(native.storage_context, store_digest=native.store_digest)
    statements = []

    def capture(connection, cursor, statement, parameters, context, many) -> None:
        statements.append(statement.lstrip().upper())

    try:
        original = await _publish(native, warm, await warm.prepare_append(frozen, prepare_payload=_payload))
        cold = NativeMessageContent(native.storage_context, store_digest=native.store_digest)
        roots = (original.root, original.root.slice(4, 9))
        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        async with engine.connect() as connection:
            plan = cold.range_query(roots)
            rows = (await connection.execute(plan.statement)).mappings().all()
        readbacks = await cold.read_hydrated_ranges(roots, rows)
        assert sum(sql.startswith(("SELECT", "WITH")) for sql in statements) == 1
        assert readbacks[0].payloads == tuple(frozen.encoded_payloads())
        assert readbacks[1].payloads == tuple(frozen[4:9].encoded_payloads())
        assert len(readbacks[0].frontier) <= len(frozen).bit_length()
        assert not readbacks[1].frontier
        with pytest.raises(ValueError, match="committed proof"):
            await cold.prepare_append(frozen[:1], prepare_payload=_payload, base=readbacks[0])
        proof = cold.adopt_read(readbacks[0])
        with pytest.raises(ValueError, match="validated readback"):
            cold.adopt_read(replace(readbacks[0]))
        with pytest.raises(ValueError, match="validated readback"):
            warm.adopt_read(readbacks[0])
        assert not cold._known_messages and not cold._known_nodes
        prepared = await cold.prepare_append(frozen[:1], prepare_payload=_payload, base=proof)
        assert len(prepared.nodes) == len(prepared.messages) == 1
        assert len(prepared.root.slices) == 1 and prepared.root.count == 36
        await _publish(native, cold, prepared)
        assert sum(sql.startswith(("SELECT", "WITH")) for sql in statements) == 1
        selected = cold.adopt_read(readbacks[1])
        branch = await cold.prepare_append(frozen[:1], prepare_payload=_payload, base=selected)
        assert len(branch.root.slices) == 2 and len(branch.nodes) == 1
        with pytest.raises(ValueError, match="incomplete"):
            await cold.read_hydrated_ranges(roots, rows[:-1])
        cold.invalidate()
        with pytest.raises(ValueError, match="validated readback"):
            cold.adopt_read(readbacks[0])
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    finally:
        await native.close()
        await engine.dispose()


@pytest.mark.parametrize("dialect", [sqlite.dialect(), postgresql.dialect(), mysql.dialect()])
def test_range_seed_cast_keeps_binary_scope_comparison_in_legal_sql(dialect) -> None:
    content = _owner(dialect)
    root = NativeMessageRoot("execution", "a" * 64, (NativeMessageSlice("b" * 64, 4, 0, 4),))
    query = content.range_query((root,))
    compiled = query.statement.compile(dialect=dialect)
    sql = str(compiled).upper()
    assert compiled.params["native_root_0"] == "b" * 64
    assert compiled.params["native_store_digest"] == "a" * 64
    if dialect.name == "mysql":
        assert "AS CHAR(64) CHARACTER SET UTF8MB4) COLLATE UTF8MB4_BIN" in sql
        assert "COLLATE UTF8MB4_BIN)" not in sql
    else:
        assert "AS CHAR(64))" in sql
