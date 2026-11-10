#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CI range correctness and SQL-work budgets, including MySQL 5.7 SQL shape."""

import getpass
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from sqlalchemy import event, select, update
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from linktools.ai.migrate import provision_database
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import SqlStepStore
from linktools.ai.persistence._payload import encode_payload_bytes
from linktools.ai.persistence._step_message_content import (
    NativeMessageContent, NativeMessageRoot, NativeMessageSlice, NativeRangeBudget, PreparedMessagePayload,
)
from linktools.ai.persistence._transcript_staging import MessageArchive
from linktools.ai.storage import dialect_for_name


_STAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


async def _payload(raw: bytes) -> PreparedMessagePayload:
    compression, _, _, encoded = encode_payload_bytes(raw)
    return PreparedMessagePayload(raw, encoded=encoded, compression=compression, payload=None)


class _Transaction:
    def __init__(self, native, session) -> None:
        self.native, self.session = native, session

    async def run_sql(self, context, callback):
        assert context is self.native.storage_context
        return await callback(self.session)


async def _publish(native, content, prepared):
    # This low-level publication fixture models an owner write in the same
    # transaction; it does not assert retention across collection or restore.
    receipt = await native.storage_context.run_mutation(lambda session: content.publish(
        _Transaction(native, session), prepared, owner_inserted=True,
    ))
    return content.confirm_committed(receipt)


def _owner(dialect) -> NativeMessageContent:
    context = SimpleNamespace(engine=SimpleNamespace(dialect=dialect), dialect=dialect_for_name(dialect.name))
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
@pytest.mark.asyncio
async def test_immutable_content_insert_compares_winners_without_body_reads(dialect) -> None:
    content = _owner(dialect)
    prepared = await content.prepare_append(MessageArchive().snapshot([
        ModelRequest(parts=[UserPromptPart("exact", timestamp=_STAMP)], timestamp=_STAMP),
    ]), prepare_payload=_payload)
    identity, body = prepared.messages[0]
    node_id, node = prepared.nodes[0]
    rows = ((content._messages, {"store_digest": "a" * 64, "sha256": identity, **body.to_native_values()}),
            (content._nodes, {"store_digest": "a" * 64, "sha256": node_id,
                "message_sha256": node.content_id, "tree_count": node.tree_count,
                "left_sha256": node.left, "right_sha256": node.right, "next_sha256": node.next_forest}))

    class Session:
        rowcount = 1
        sql = ""

        async def execute(self, statement, parameters=None):
            self.sql = str(statement.compile(dialect=dialect)).upper()
            return SimpleNamespace(rowcount=self.rowcount)

    session = Session()
    for table, row in rows:
        await content._insert_rows(session, table, (row,))
        sql = session.sql
        assert re.search(r"\b(?:SELECT|IGNORE)\b", sql) is None
        assert all(name.upper() in sql for name in row)
        if dialect.name == "mysql":
            assert "ON DUPLICATE KEY UPDATE ID = CASE WHEN" in sql
            assert "THEN AI_STEP_" in sql and ".ID ELSE CASE" in sql
        else:
            assert "DO UPDATE SET ID = AI_STEP_" in sql and ".ID WHERE" in sql
        session.rowcount = 0
        with pytest.raises(AIError) as caught:
            await content._insert_rows(session, table, (row,))
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        session.rowcount = 1


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
        all_body_bytes = sum(len(body.body) for _, body in prepared.messages)
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
        returned_body_bytes = sum(len(row["message"]) for row in rows if row["message"] is not None)
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
        # The previously published "a" singleton already owns the same immutable node.
        assert len(branch.nodes) == 1
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
    await _check_native_cas_and_range_execution(create_async_engine(url))


@pytest.mark.asyncio
async def test_sqlite_native_cas_and_range_execution_preserve_occurrences(tmp_path: Path) -> None:
    await _check_native_cas_and_range_execution(
        create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'native-cas.db'}"),
    )


async def _check_native_cas_and_range_execution(engine: AsyncEngine) -> None:
    await provision_database(engine)
    native = SqlStepStore(engine, namespace="native-range-contract", tenant_id="one")
    await native.initialize()
    archive = MessageArchive()
    frozen = archive.snapshot([ModelRequest(parts=[UserPromptPart(str(index % 3), timestamp=_STAMP)], timestamp=_STAMP)
                               for index in range(35)])
    statements = []
    publication_io = []

    def capture(connection, cursor, statement, parameters, context, many) -> None:
        sql = statement.lstrip().upper()
        statements.append(sql)
        if sql.startswith(("SELECT", "WITH", "UPDATE")):
            publication_io.append(context.compiled.statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        for _ in range(2):
            publication_io.clear()
            content = NativeMessageContent(native.storage_context, group_id="execution",
                                           store_digest=native.store_digest, max_cached_messages=0, max_cached_nodes=0)
            prepared = await content.prepare_append(frozen, prepare_payload=_payload)
            proof = await _publish(native, content, prepared)
            # Each fresh publisher resolves bounded surrogate IDs, then fences
            # those exact dependency sets against concurrent garbage collection.
            # No message body or sequence frontier is reread during publication.
            assert len(publication_io) == 4
            sequence_ids = sorted({value for identity, node in prepared.nodes
                                   for value in (identity, node.left, node.right, node.next_forest)
                                   if value is not None})
            message_ids = sorted({identity for identity, _body in prepared.messages})
            for actual, table, identities in zip(
                publication_io[:2], (content._nodes, content._messages),
                (sequence_ids, message_ids), strict=True,
            ):
                expected = select(table.c.sha256, table.c.id).where(
                    table.c.store_digest == native.store_digest, table.c.sha256.in_(identities),
                )
                assert str(actual.compile(dialect=engine.dialect, compile_kwargs={"literal_binds": True})) == str(
                    expected.compile(dialect=engine.dialect, compile_kwargs={"literal_binds": True}),
                )
            dependency_counts = (
                len({value for _identity, node in prepared.nodes
                     for value in (node.left, node.right, node.next_forest) if value is not None}),
                len(message_ids),
            )
            for actual, table, count in zip(
                publication_io[2:], (content._nodes, content._messages), dependency_counts, strict=True,
            ):
                targets = actual.whereclause.right.value
                assert len(targets) == len(set(targets)) == count
                assert 0 < count <= len(frozen)
                assert all(type(value) is int and value > 0 for value in targets)
                expected = update(table).where(table.c.id.in_(targets)).values(
                    gc_revision=table.c.gc_revision + 1, updated_at=table.c.updated_at,
                )
                assert str(actual.compile(dialect=engine.dialect, compile_kwargs={"literal_binds": True})) == str(
                    expected.compile(dialect=engine.dialect, compile_kwargs={"literal_binds": True}),
                )
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
@pytest.mark.parametrize("history_count", (35, 255))
async def test_actual_readback_resumes_frozen_append_without_another_content_read(
    tmp_path: Path, history_count: int,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'readback.db'}")
    native = SqlStepStore(engine, namespace="native-readback", tenant_id="one")
    await native.initialize(local=True)
    archive = MessageArchive()
    frozen = archive.snapshot([ModelRequest(parts=[UserPromptPart(str(index), timestamp=_STAMP)], timestamp=_STAMP)
                               for index in range(history_count)])
    warm = NativeMessageContent(native.storage_context, store_digest=native.store_digest)
    statements = []
    read_calls = []

    def capture(connection, cursor, statement, parameters, context, many) -> None:
        sql = statement.lstrip().upper()
        statements.append(sql)
        if sql.startswith(("SELECT", "WITH")):
            read_calls.append((sql, parameters))

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
        assert len(prepared.root.slices) == 1 and prepared.root.count == history_count + 1
        await _publish(native, cold, prepared)
        reads = [sql for sql in statements if sql.startswith(("SELECT", "WITH"))]
        # One hydration read remains sufficient at either history size. The
        # publication adds only bounded surrogate-ID resolution for GC edges.
        assert len(reads) == 3
        allowed = {identity for identity, _body in prepared.messages}
        for identity, node in prepared.nodes:
            allowed.update(value for value in (identity, node.left, node.right, node.next_forest)
                           if value is not None)
        for sql, parameters in read_calls[1:]:
            normalized = " ".join(sql.split())
            assert re.fullmatch(
                r"SELECT (AI_STEP_MESSAGES|AI_STEP_SEQUENCES)\.SHA256, \1\.ID "
                r"FROM \1 WHERE \1\.STORE_DIGEST = \? AND \1\.SHA256 IN \([?, ]+\)",
                normalized,
            ), normalized
            assert parameters[0] == native.store_digest
            assert 1 <= len(parameters[1:]) <= 4
            assert set(parameters[1:]) <= allowed
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
