#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistent snapshot sequences share prefixes without extra read round trips."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot
from sqlalchemy import event, text
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.steps import native_store_locator, encode_native_replay_key, native_store_key
from linktools.ai.persistence import SqlStepStore
from linktools.ai.persistence._step_schema import build_step_sql_metadata
from linktools.ai.persistence._step_message_content import NativeMessageContent


class _ReadTraffic:
    def __init__(self) -> None:
        self.count = 0
        self.reads: list[tuple[str, tuple[object, ...]]] = []

    def capture(self, connection, cursor, statement, parameters, context, many) -> None:
        # A recursive expansion is one database read, even when SQL starts WITH.
        if statement.lstrip().upper().startswith(("SELECT", "WITH")):
            self.count += 1
            self.reads.append((statement, tuple(parameters)))

    def assert_publication_metadata(
        self, start: int, *, store_key: bytes, node_bound: int, message_bound: int,
    ) -> None:
        reads = self.reads[start:]
        assert len(reads) == 3
        seen = set()
        for statement, parameters in reads:
            projection = " ".join(statement.split("FROM", 1)[0].split())
            for table, bound in (("ai_step_sequences", node_bound), ("ai_step_messages", message_bound)):
                if projection == f"SELECT {table}.sha256, {table}.id":
                    assert table not in seen
                    seen.add(table)
                    assert f"{table}.store_key =" in statement and f"{table}.sha256 IN" in statement
                    assert parameters[:2] == (native_store_locator(store_key), store_key)
                    assert 0 < len(parameters) - 2 <= bound
                    break
            else:
                assert "collector" not in seen
                seen.add("collector")
                columns = projection.removeprefix("SELECT ").split(", ")
                assert all(column.startswith("ai_native_content_edges.") for column in columns)
                assert {column.split(".")[-1] for column in columns} <= {
                    "id", "edge_kind", "source_id", "target_id", "revision", "updated_at", "created_at",
                }
                assert "LIMIT" in statement and 0 < parameters[-2] <= 32
        assert seen == {"ai_step_sequences", "ai_step_messages", "collector"}


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [10, 140])
async def test_cumulative_snapshot_sequences_grow_linearly_and_reopen_in_two_reads(
    tmp_path: Path, size: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'growing.db'}"
    engine = create_async_engine(url)
    store = SqlStepStore(engine, namespace="steps", tenant_id="one")
    messages = [ModelRequest(parts=[UserPromptPart(str(index))]) for index in range(size)]
    await store.initialize(local=True)
    traffic = _ReadTraffic()
    event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
    try:
        for length in range(1, size + 1):
            latest = ContinuableSnapshot(run_id="run", step_index=length, idempotency_key=str(length),
                                         messages=messages[:length])
            start = len(traffic.reads)
            await store.save_snapshot(latest)
            # One appended node references at most three existing frontier IDs.
            # Crossing the SQL batch size must not turn this into a history scan.
            traffic.assert_publication_metadata(
                start, store_key=store.store_key, node_bound=4, message_bound=1,
            )
        assert traffic.count == 3 * size
        async with engine.connect() as connection:
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_snapshots")) == size
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_messages")) == size
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_sequences")) == size
            assert await connection.scalar(text(
                "SELECT count(DISTINCT sequence_root) FROM ai_step_snapshots"
            )) == size
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        await store.close()
        await engine.dispose()

    # A new engine and store cannot depend on the writer's confirmed body cache.
    engine = create_async_engine(url)
    cold = SqlStepStore(engine, namespace="steps", tenant_id="one", max_cached_messages=0)
    traffic = _ReadTraffic()
    event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
    try:
        from linktools.ai.persistence import _step_store as step_module

        decoded_occurrences = 0
        decode_payloads = step_module.decode_model_message_payloads

        def decode_once(payloads):
            nonlocal decoded_occurrences
            decoded_occurrences += len(payloads)
            return decode_payloads(payloads)

        monkeypatch.setattr(step_module, "decode_model_message_payloads", decode_once)
        assert await cold.latest_snapshot(run_id="run") == latest
        assert traffic.count == 2
        assert decoded_occurrences == size
        decoded_occurrences = 0
        history = await cold.list_snapshots(run_id="run")
        assert len(history) == size
        assert decoded_occurrences == sum(range(1, size + 1))
        for length, snapshot in enumerate(history, 1):
            assert snapshot.messages == messages[:length]
            assert snapshot.step_index == length
            assert snapshot.idempotency_key == str(length)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        await cold.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_sequence_sharing_preserves_repetition_branches_truncation_and_cross_run_prefixes(
    tmp_path: Path,
) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'branches.db'}"
    engine = create_async_engine(url)
    store = SqlStepStore(engine, namespace="steps", tenant_id="one", max_cached_messages=0)
    await store.initialize(local=True)
    messages = [ModelRequest(parts=[UserPromptPart(value)]) for value in ("a", "b", "c", "branch")]
    base = ContinuableSnapshot(run_id="base", step_index=0, idempotency_key="first", messages=messages[:3])
    repeated = replace(base, run_id="repeated", messages=[*base.messages, messages[0], messages[0]])
    branch = replace(base, run_id="branch", messages=[*messages[:2], messages[3]])
    shortened = replace(base, run_id="shortened", messages=messages[:2])
    other_run = replace(base, run_id="other")
    cases = [base, repeated, branch, shortened, other_run]
    traffic = _ReadTraffic()
    event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
    try:
        for snapshot in cases:
            start = len(traffic.reads)
            await store.save_snapshot(snapshot)
            # This uncached writer receives each full branch as new input;
            # ID reads stay bounded by that input, not unrelated stored history.
            traffic.assert_publication_metadata(
                start, store_key=store.store_key,
                node_bound=4 * len(snapshot.messages), message_bound=len(snapshot.messages),
            )
            after = traffic.count
            await store.save_snapshot(snapshot)
            # Duplicate owners verify raw scope once, without publishing dependencies.
            assert traffic.count == after + 1
            statement, parameters = traffic.reads[-1]
            assert " ".join(statement.split("FROM", 1)[0].split()) == "SELECT ai_step_snapshots.store_key"
            assert "ai_step_snapshots.store_locator =" in statement
            assert parameters == (native_store_locator(store.store_key), snapshot.run_id,
                                  encode_native_replay_key(snapshot.idempotency_key))
        assert traffic.count == 4 * len(cases)
        async with engine.connect() as connection:
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_snapshots")) == len(cases)
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_messages")) == 4
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_sequences")) == 6
            assert await connection.scalar(text(
                "SELECT count(DISTINCT sequence_root) FROM ai_step_snapshots"
            )) == 4
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        await store.close()
        await engine.dispose()
    engine = create_async_engine(url)
    cold = SqlStepStore(engine, namespace="steps", tenant_id="one", max_cached_messages=0)
    traffic = _ReadTraffic()
    event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
    try:
        for snapshot in cases:
            traffic.count = 0
            assert await cold.latest_snapshot(run_id=snapshot.run_id) == snapshot
            assert traffic.count == 2
            assert await cold.list_snapshots(run_id=snapshot.run_id) == [snapshot]
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        await cold.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["missing-child", "wrong-count", "wrong-message"])
async def test_corrupt_sequence_fails_closed_after_reopening(tmp_path: Path, damage: str) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'corrupt-sequence.db'}"
    engine = create_async_engine(url)
    store = SqlStepStore(engine, namespace="steps", tenant_id="one")
    await store.initialize(local=True)
    snapshot = ContinuableSnapshot(run_id="run", step_index=0, idempotency_key="first",
        messages=[ModelRequest(parts=[UserPromptPart(str(index))]) for index in range(7)])
    try:
        await store.save_snapshot(snapshot)
    finally:
        await store.close()
        await engine.dispose()
    engine = create_async_engine(url)
    cold = SqlStepStore(engine, namespace="steps", tenant_id="one")
    try:
        async with engine.begin() as connection:
            root = await connection.scalar(text("SELECT sequence_root FROM ai_step_snapshots"))
            if damage == "missing-child":
                await connection.execute(text(
                    "DELETE FROM ai_step_sequences WHERE sha256 = "
                    "(SELECT left_sha256 FROM ai_step_sequences WHERE sha256 = :root)"
                ), {"root": root})
            elif damage == "wrong-count":
                await connection.execute(text(
                    "UPDATE ai_step_sequences SET tree_count = tree_count + 2 WHERE sha256 = :root"
                ), {"root": root})
            else:
                other = await connection.scalar(text(
                    "SELECT message_sha256 FROM ai_step_sequences WHERE sha256 <> :root LIMIT 1"
                ), {"root": root})
                await connection.execute(text(
                    "UPDATE ai_step_sequences SET message_sha256 = :other WHERE sha256 = :root"
                ), {"other": other, "root": root})
        with pytest.raises(ValueError):
            await cold.latest_snapshot(run_id="run")
        assert await cold.list_snapshots(run_id="run") == []
    finally:
        await cold.close()
        await engine.dispose()


@pytest.mark.parametrize("dialect_name", ["mysql", "sqlite", "postgresql"])
@pytest.mark.parametrize("count", [10, 1000])
def test_sequence_read_compiles_for_supported_dialects_without_unbounded_mysql_expansion(
    dialect_name: str, count: int,
) -> None:
    dialect = {"mysql": mysql, "sqlite": sqlite, "postgresql": postgresql}[dialect_name].dialect()
    if dialect_name == "mysql":
        dialect.server_version_info = (5, 7, 44)
    # Only query construction is exercised: no engine, driver or connection.
    store = object.__new__(SqlStepStore)
    store._context = SimpleNamespace(engine=SimpleNamespace(dialect=dialect))
    store._metadata = build_step_sql_metadata()
    store._store_key = native_store_key("a", "tenant")
    store._message_content = NativeMessageContent(store._context, store_key=store._store_key)
    query = store._sequence_query([{
        "id": 1, "sequence_root": "b" * 64,
        "messages": json.dumps({"stored_messages": count, "checkpoint": None}),
    }])
    assert query is not None
    sql = str(query.compile(dialect=dialect, compile_kwargs={"literal_binds": True})).upper()
    assert "AI_STEP_SEQUENCES" in sql and "AI_STEP_MESSAGES" in sql
    if dialect_name == "mysql":
        assert "WITH" not in sql and "RECURSIVE" not in sql
        assert "UNION ALL" in sql
        # Derived frontiers must grow with tree height, not message count.
        # Generous bounds avoid fixing alias names or exact compiler formatting.
        depth = maximum = 0
        for character in sql:
            if character == "(":
                depth += 1
                maximum = max(maximum, depth)
            elif character == ")":
                depth -= 1
        assert depth == 0
        assert maximum <= 4 * count.bit_length() + 12
        assert sql.count("SELECT ") <= 8 * count.bit_length() ** 2
    else:
        assert "WITH RECURSIVE" in sql


@pytest.mark.asyncio
@pytest.mark.parametrize("descriptor, clear_root", [
    ("{", False),
    (json.dumps({"stored_messages": -1, "checkpoint": None}), False),
    (json.dumps({"stored_messages": "1", "checkpoint": None}), False),
    (json.dumps({"stored_messages": True, "checkpoint": None}), False),
    (json.dumps({"stored_messages": 1.0, "checkpoint": None}), False),
    (json.dumps({"stored_messages": 0, "checkpoint": None}), False),
    (json.dumps({"stored_messages": 1, "checkpoint": None}), True),
    (json.dumps({"stored_messages": 2 ** 63, "checkpoint": None}), False),
], ids=["invalid-json", "negative", "string", "boolean", "float", "unexpected-root", "missing-root", "overflow"])
async def test_invalid_snapshot_descriptor_does_not_hide_other_valid_history(
    tmp_path: Path, descriptor: str, clear_root: bool,
) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'descriptors.db'}"
    engine = create_async_engine(url)
    store = SqlStepStore(engine, namespace="steps", tenant_id="one")
    await store.initialize(local=True)
    first = ContinuableSnapshot(run_id="run", step_index=0, idempotency_key="first",
        messages=[ModelRequest(parts=[UserPromptPart("preserved")])])
    middle = replace(first, step_index=1, idempotency_key="bad-middle")
    last = replace(first, step_index=2, idempotency_key="last")
    terminal = replace(first, step_index=3, idempotency_key="bad-terminal")
    try:
        for snapshot in (first, middle, last, terminal):
            await store.save_snapshot(snapshot)
    finally:
        await store.close()
        await engine.dispose()
    engine = create_async_engine(url)
    cold = SqlStepStore(engine, namespace="steps", tenant_id="one")
    try:
        table = build_step_sql_metadata().tables["ai_step_snapshots"]
        damage = {"messages": descriptor}
        if clear_root:
            damage["sequence_root"] = None
        async with engine.begin() as connection:
            changed = await connection.execute(table.update().where(
                table.c.store_key == cold.store_key,
                table.c.idempotency_key.in_([
                    encode_native_replay_key("bad-middle"), encode_native_replay_key("bad-terminal"),
                ]),
            ).values(**damage))
            assert changed.rowcount == 2
        with pytest.raises(ValueError):
            await cold.latest_snapshot(run_id="run")
        assert await cold.list_snapshots(run_id="run") == [first, last]
    finally:
        await cold.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("rollback", (False, True))
async def test_publication_ids_are_reused_only_within_their_live_owner_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rollback: bool,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'publication-ids.db'}")
    store = SqlStepStore(engine, namespace="steps", tenant_id="one")
    await store.initialize(local=True)
    snapshot = ContinuableSnapshot(run_id="run", step_index=1, idempotency_key="snapshot",
                                    messages=[ModelRequest(parts=[UserPromptPart("body")])])
    publish = NativeMessageContent.publish_in_session
    receipts, sessions, projections = [], [], []
    fail_once = rollback

    async def observe(content, session, prepared, **kwargs):
        nonlocal fail_once
        roots = tuple(span.first for span in prepared.root.slices)
        for previous in receipts:
            with pytest.raises(ValueError, match="active transaction"):
                previous.publication_targets(session, store_key=store.store_key, roots=roots)
        receipt = await publish(content, session, prepared, **kwargs)
        targets = receipt.publication_targets(session, store_key=store.store_key, roots=roots)
        assert len(targets) == 1 and targets[0] > 0
        for scope, selected in ((native_store_key("0", "tenant"), roots), (store.store_key, ("0" * 64,))):
            with pytest.raises(ValueError, match="store, and roots"):
                receipt.publication_targets(session, store_key=scope, roots=selected)
        receipts.append(receipt)
        sessions.append(session)
        if fail_once:
            fail_once = False
            raise RuntimeError("rollback publication")
        return receipt

    def capture(_connection, _cursor, statement, _parameters, _context, _many):
        if statement.startswith("SELECT"):
            projections.append(statement.split("FROM", 1)[0])

    monkeypatch.setattr(NativeMessageContent, "publish_in_session", observe)
    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        if rollback:
            with pytest.raises(RuntimeError, match="rollback publication"):
                await store.save_snapshot(snapshot)
            projections.clear()
        await store.save_snapshot(snapshot)
        # Two bounded ID lookups index the node/message, then the collector
        # checks durable candidates. Neither publication nor snapshot rereads the root.
        assert len(projections) == 3
        assert not any("ai_step_messages.message" in sql or "ai_step_messages.payload_json" in sql
                       for sql in projections)
        for receipt, session in zip(receipts, sessions):
            with pytest.raises(ValueError, match="active transaction"):
                receipt.publication_targets(
                    session, store_key=store.store_key,
                    roots=tuple(span.first for span in receipt.root.slices),
                )
        assert await store.latest_snapshot(run_id="run") == snapshot
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("rollback", (False, True))
async def test_publication_ids_reject_a_completed_savepoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rollback: bool,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'savepoint-ids.db'}")
    store = SqlStepStore(engine, namespace="steps", tenant_id="one")
    await store.initialize(local=True)
    snapshot = ContinuableSnapshot(run_id="run", step_index=1, idempotency_key="snapshot",
                                    messages=[ModelRequest(parts=[UserPromptPart("body")])])
    publish = NativeMessageContent.publish_in_session

    async def nested_publication(content, session, prepared, **kwargs):
        roots = tuple(span.first for span in prepared.root.slices)
        try:
            async with session.begin_nested():
                receipt = await publish(content, session, prepared, **kwargs)
                assert receipt.publication_targets(session, store_key=store.store_key, roots=roots)
                if rollback:
                    raise RuntimeError("rollback savepoint")
        except RuntimeError as error:
            assert rollback and str(error) == "rollback savepoint"
        assert session.in_transaction()
        with pytest.raises(ValueError, match="active transaction"):
            receipt.publication_targets(session, store_key=store.store_key, roots=roots)
        return receipt

    monkeypatch.setattr(NativeMessageContent, "publish_in_session", nested_publication)
    try:
        with pytest.raises(ValueError, match="active transaction"):
            await store.save_snapshot(snapshot)
        assert await store.latest_snapshot(run_id="run") is None
    finally:
        await store.close()
        await engine.dispose()
