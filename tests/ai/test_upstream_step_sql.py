#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public upstream StepStore behavior over the project SQL adapter."""

import asyncio
import re
import sqlite3
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest
import pytest_asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryContent, ModelRequest, ModelResponse, ModelMessagesTypeAdapter,
    TextPart, ToolCallPart, ToolReturnPart, UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot, InMemoryStepStore, RunRecord, SqliteStepStore,
    StepEvent, StepPersistence, StepStore, ToolEffectRecord, continue_run,
)
from sqlalchemy import Text, UniqueConstraint, event, text
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.schema import CreateIndex, CreateTable

from linktools.ai.core import RESOURCE_ID_MAX_BYTES, validate_resource_id
from linktools.ai.errors import AIError
from linktools.ai.persistence._step_schema import build_step_sql_metadata
from linktools.ai.persistence._step_store import SqlStepStore


@pytest_asyncio.fixture(params=["sql", "upstream"])
async def store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[StepStore]:
    if request.param == "upstream":
        yield SqliteStepStore(database=tmp_path / "official.db", media_store=None, max_snapshots_per_run=1)
        return
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'adapter.db'}")
    adapter = SqlStepStore(engine, namespace="steps", tenant_id="one",
                           tool_effect_store=InMemoryStepStore(), max_snapshots_per_run=1)
    await adapter.initialize(local=True)
    try:
        yield adapter
    finally:
        await adapter.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_run_lineage_filters_sort_by_instant_and_keep_original_records(store: StepStore) -> None:
    records = [
        RunRecord(run_id="later", conversation_id="conversation", parent_run_id="parent",
                  started_at=datetime.fromisoformat("2026-01-01T01:30:00+00:00"),
                  agent_name="agent", metadata={"label": "original"}, registration_id="registration"),
        RunRecord(run_id="earlier", conversation_id="conversation", parent_run_id="parent",
                  started_at=datetime.fromisoformat("2026-01-01T00:00:00+00:00")),
        RunRecord(run_id="other", conversation_id="conversation", parent_run_id="other"),
    ]
    for record in records:
        await store.register_run(record)
    assert await store.get_run(run_id="later") == records[0]
    assert await store.get_run(run_id="missing") is None
    assert await store.list_runs(parent_run_id="parent", conversation_id="conversation") == [records[1], records[0]]
    with pytest.raises((ValueError, sqlite3.IntegrityError)):
        await store.register_run(records[0])


@pytest.mark.asyncio
async def test_sql_lineage_filters_preserve_bounded_exact_identifiers_under_padded_comparisons(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'lineage.db'}")
    metadata = build_step_sql_metadata()
    runs = metadata.tables["ai_step_runs"]
    # SQLite RTRIM models MySQL PAD SPACE comparisons for the accepted IDs.
    for name in ("run_id", "conversation_id", "parent_run_id"):
        runs.c[name].type = Text(collation="RTRIM")
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    store = SqlStepStore(engine, namespace="steps", tenant_id="one")
    other = SqlStepStore(engine, namespace="steps", tenant_id="two")
    prefix = "共同前缀" + "x" * (RESOURCE_ID_MAX_BYTES - len("共同前缀".encode()) - 1)
    records = [
        RunRecord(run_id="exact", conversation_id=prefix + "a", parent_run_id=prefix + "a"),
        RunRecord(run_id="other-parent", conversation_id=prefix + "a", parent_run_id=prefix + "b"),
        RunRecord(run_id="other-conversation", conversation_id=prefix + "b", parent_run_id=prefix + "a"),
        RunRecord(run_id="missing"),
    ]
    try:
        for record in records:
            await store.register_run(record)
        await other.register_run(records[0])
        assert await store.list_runs(conversation_id=prefix + "a", parent_run_id=prefix + "a") == records[:1]
        assert {record.run_id for record in await store.list_runs(conversation_id=prefix + "a")} == {"exact", "other-parent"}
        assert {record.run_id for record in await store.list_runs(parent_run_id=prefix + "a")} == {"exact", "other-conversation"}
        assert await other.list_runs() == records[:1]
        assert len(await store.list_runs()) == len(records)
        for field in ("run_id", "conversation_id", "parent_run_id"):
            with pytest.raises(AIError):
                await store.register_run(replace(records[0], **{field: "padded "}))
        with pytest.raises(AIError):
            await store.list_runs(conversation_id="padded ")
        with pytest.raises(AIError):
            await store.list_runs(parent_run_id="padded ")
    finally:
        await store.close()
        await other.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ("same-key", ""))
async def test_boundary_events_are_append_only_and_keyed_retries_are_suppressed(store: StepStore, key: str) -> None:
    first = StepEvent(run_id="run", kind="run_started", step_index=4,
                      conversation_id="conversation", agent_name="agent",
                      metadata={"test": "value"}, idempotency_key=key)
    second = StepEvent(run_id="run", kind="tool_call_failed", step_index=0,
                       tool_call_id="call", tool_name="tool", error="failed")
    await store.append_event(first)
    await store.append_event(replace(first, metadata={"test": "must-not-replace"}))
    await store.append_event(second)
    await store.append_event(second)
    assert await store.list_events(run_id="run") == [first, second, second]
    assert await store.list_events(run_id="missing") == []


@pytest.mark.asyncio
async def test_snapshot_missing_replay_key_appends_and_empty_key_deduplicates(store: StepStore) -> None:
    snapshot = ContinuableSnapshot(run_id="run", step_index=0, messages=[])
    await store.save_snapshot(snapshot)
    appended = replace(snapshot, step_index=1)
    await store.save_snapshot(appended)
    assert await store.latest_snapshot(run_id="run") == appended
    keyed = replace(snapshot, step_index=2, idempotency_key="")
    await store.save_snapshot(keyed)
    await store.save_snapshot(replace(keyed, step_index=3))
    assert await store.latest_snapshot(run_id="run") == keyed
    other_run = replace(keyed, run_id="other")
    await store.save_snapshot(other_run)
    assert await store.latest_snapshot(run_id="other") == other_run


@pytest.mark.asyncio
async def test_snapshot_retention_keeps_latest_complete_and_latest_interrupted(store: StepStore) -> None:
    complete = ContinuableSnapshot(run_id="run", step_index=7, idempotency_key="7:complete",
                                   messages=[ModelRequest(parts=[UserPromptPart("settled")])])
    interrupted = replace(complete, step_index=0, idempotency_key="8:interrupted", state="interrupted")
    newest = replace(interrupted, idempotency_key="9:interrupted")
    await store.save_snapshot(complete)
    await store.save_snapshot(interrupted)
    await store.save_snapshot(newest)
    assert await store.latest_snapshot(run_id="run") == complete
    assert await store.latest_snapshot(run_id="run", include_interrupted=True) == newest
    await store.save_snapshot(interrupted)
    assert await store.latest_snapshot(run_id="run", include_interrupted=True) == newest
    snapshots = await store.list_snapshots(run_id="run", include_interrupted=True)
    assert snapshots == [complete, newest]


@pytest.mark.asyncio
async def test_snapshot_preserves_native_messages_and_official_continue_run(store: StepStore) -> None:
    messages = [
        ModelRequest(parts=[UserPromptPart(["inspect", BinaryContent(data=b"binary", media_type="image/png")])]),
        ModelResponse(parts=[ToolCallPart("work", {"x": 3}, tool_call_id="call")]),
        ModelRequest(parts=[ToolReturnPart("work", {"answer": [3, 4]}, tool_call_id="call")]),
        ModelResponse(parts=[TextPart("complete")]),
    ]
    snapshot = ContinuableSnapshot(run_id="native", step_index=3, messages=messages)
    await store.save_snapshot(snapshot)
    result = await continue_run(store, run_id="native")
    assert ModelMessagesTypeAdapter.dump_json(result) == ModelMessagesTypeAdapter.dump_json(messages)
    empty = ContinuableSnapshot(run_id="empty", step_index=0, messages=[])
    await store.save_snapshot(empty)
    assert await store.latest_snapshot(run_id="empty") == empty


@pytest.mark.asyncio
async def test_sql_partitions_isolate_run_and_message_keys(tmp_path: Path) -> None:
    run_id = "run-" + "x" * (RESOURCE_ID_MAX_BYTES - 4)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'scopes.db'}")
    stores = [SqlStepStore(engine, namespace=namespace, tenant_id=tenant, tool_effect_store=InMemoryStepStore())
              for namespace, tenant in [("one", "a"), ("one", "b"), ("two", "a")]]
    try:
        await stores[0].initialize(local=True)
        for index, item in enumerate(stores):
            await item.register_run(RunRecord(run_id=run_id, metadata={"scope": str(index)}))
            await item.save_snapshot(ContinuableSnapshot(run_id="same", step_index=0,
                messages=[ModelRequest(parts=[UserPromptPart(str(index))])], idempotency_key="same"))
        await stores[0].register_run(RunRecord(run_id="offset", parent_run_id="parent",
            started_at=datetime.fromisoformat("2026-01-01T02:00:00+02:00")))
        await stores[0].register_run(RunRecord(run_id="utc", parent_run_id="parent",
            started_at=datetime.fromisoformat("2026-01-01T01:00:00+00:00")))
        assert [run.run_id for run in await stores[0].list_runs(parent_run_id="parent")] == ["offset", "utc"]
        for index, item in enumerate(stores):
            assert (await item.get_run(run_id=run_id)).metadata == {"scope": str(index)}
            snapshot = await item.latest_snapshot(run_id="same")
            assert snapshot.messages[0].parts[0].content == str(index)
    finally:
        for item in stores:
            await item.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_sql_failed_reference_publication_rolls_back_snapshot_body_and_replay_key(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'atomic.db'}")
    store = SqlStepStore(engine, namespace="steps", tenant_id="one", tool_effect_store=InMemoryStepStore())
    await store.initialize(local=True)
    snapshot = ContinuableSnapshot(run_id="run", step_index=0, idempotency_key="atomic",
                                    messages=[ModelRequest(parts=[UserPromptPart("message")])])

    def fail_reference(connection, cursor, statement, parameters, context, many):
        if statement.startswith("INSERT INTO ai_step_sequences"):
            raise RuntimeError("reference publication interrupted")

    try:
        event.listen(engine.sync_engine, "before_cursor_execute", fail_reference)
        with pytest.raises(RuntimeError, match="reference publication interrupted"):
            await store.save_snapshot(snapshot)
        event.remove(engine.sync_engine, "before_cursor_execute", fail_reference)
        assert await store.latest_snapshot(run_id="run") is None
        async with engine.connect() as connection:
            for table in ("snapshots", "messages", "sequences"):
                assert await connection.scalar(text(f"SELECT count(*) FROM ai_step_{table}")) == 0
        await store.save_snapshot(snapshot)
        assert await store.latest_snapshot(run_id="run") == snapshot
    finally:
        if event.contains(engine.sync_engine, "before_cursor_execute", fail_reference):
            event.remove(engine.sync_engine, "before_cursor_execute", fail_reference)
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_snapshot_pruning_and_body_lookup_stay_in_their_partition(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'scoped-refs.db'}")
    stores = [SqlStepStore(engine, namespace="steps", tenant_id=tenant, max_snapshots_per_run=1)
              for tenant in ("one", "two")]
    first = ContinuableSnapshot(run_id="run", step_index=0, idempotency_key="first",
                                messages=[ModelRequest([UserPromptPart("shared")])])
    second = replace(first, step_index=1, idempotency_key="second",
                     messages=[ModelRequest([UserPromptPart("new shared")])])
    try:
        await stores[0].initialize(local=True)
        for store in stores:
            await store.save_snapshot(first)
        await stores[0].save_snapshot(second)
        assert await stores[1].latest_snapshot(run_id="run") == first
        await stores[1].save_snapshot(second)
        async with engine.begin() as connection:
            await connection.execute(text("DELETE FROM ai_step_messages WHERE store_digest = :scope"),
                                     {"scope": stores[0]._store_digest})
        with pytest.raises(ValueError, match="missing or corrupt"):
            await stores[0].latest_snapshot(run_id="run")
        assert await stores[1].latest_snapshot(run_id="run") == second
    finally:
        for store in stores:
            await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_sql_concurrent_boundary_retries_publish_once(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'concurrent.db'}")
    stores = [SqlStepStore(engine, namespace="steps", tenant_id="one", tool_effect_store=InMemoryStepStore()) for _ in range(2)]
    try:
        await stores[0].initialize(local=True)
        record = RunRecord(run_id="run")
        results = await asyncio.gather(*(store.register_run(record) for store in stores), return_exceptions=True)
        assert sum(result is None for result in results) == 1
        assert sum(isinstance(result, ValueError) for result in results) == 1
        boundary = StepEvent(run_id="run", kind="run_started", step_index=0, idempotency_key="key")
        await asyncio.gather(*(store.append_event(boundary) for store in stores))
        assert await stores[0].list_events(run_id="run") == [boundary]
        snapshot = ContinuableSnapshot(run_id="run", step_index=0, messages=[], idempotency_key="key")
        await asyncio.gather(*(store.save_snapshot(snapshot) for store in stores))
        assert await stores[0].list_snapshots(run_id="run") == [snapshot]
    finally:
        for store in stores:
            await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_sql_media_round_trip_and_deduplicated_message_batches(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'media.db'}")
    store = SqlStepStore(engine, namespace="steps", tenant_id="one", tool_effect_store=InMemoryStepStore())
    try:
        await store.initialize(local=True)
        messages = [ModelRequest(parts=[UserPromptPart("x" * 81920)])]
        messages.extend(ModelResponse(parts=[TextPart(str(i))]) for i in range(1100))
        first = ContinuableSnapshot(run_id="run", step_index=0, messages=messages, idempotency_key="first")
        second = replace(first, step_index=1, idempotency_key="second")
        await store.save_snapshot(first)
        await store.save_snapshot(second)
        assert await store.latest_snapshot(run_id="run") == second
        async with engine.connect() as connection:
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_messages")) == len(messages)
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_sequences")) == len(messages)
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_sql_corrupt_or_missing_message_reference_fails_closed(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'corrupt.db'}")
    store = SqlStepStore(engine, namespace="steps", tenant_id="one", tool_effect_store=InMemoryStepStore())
    try:
        await store.initialize(local=True)
        snapshot = ContinuableSnapshot(run_id="run", step_index=0, messages=[ModelResponse(parts=[TextPart("original")])])
        await store.save_snapshot(snapshot)
        async with engine.begin() as connection:
            await connection.execute(text("UPDATE ai_step_messages SET message = :body"), {"body": "{}"})
        with pytest.raises(ValueError, match="missing or corrupt"):
            await store.latest_snapshot(run_id="run")
        assert await store.list_snapshots(run_id="run") == []
        async with engine.begin() as connection:
            await connection.execute(text("DELETE FROM ai_step_sequences"))
        with pytest.raises(ValueError, match="incomplete"):
            await store.latest_snapshot(run_id="run")
        assert await store.list_snapshots(run_id="run") == []
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_sql_effect_methods_delegate_without_another_durable_ledger(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'effects.db'}")
    owner = InMemoryStepStore()
    store = SqlStepStore(engine, namespace="steps", tenant_id="one", tool_effect_store=owner)
    try:
        record = ToolEffectRecord(run_id="run", tool_call_id="call", tool_name="work", status="started",
                                  idempotency_key="provider-key", effect_summary="write")
        await store.record_tool_effect(record)
        assert await owner.get_tool_effect(run_id="run", tool_call_id="call") == record
        assert await store.list_unresolved_tool_effects(run_id="run") == [record]
        done = replace(record, status="completed")
        await owner.record_tool_effect(done)
        assert await store.get_tool_effect(run_id="run", tool_call_id="call") == done
        assert await store.list_unresolved_tool_effects(run_id="run") == []
        async with engine.connect() as connection:
            assert (await connection.execute(text("SELECT name FROM sqlite_master WHERE type = 'table'"))).all() == []
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_sql_initialize_does_not_create_or_inspect_external_schema(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'unprovisioned.db'}")
    store = SqlStepStore(engine, namespace="steps", tenant_id="one", tool_effect_store=InMemoryStepStore())
    statements = []

    def capture(connection, cursor, statement, parameters, context, many):
        statements.append(statement.lower())

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        await store.initialize()
        assert not any("create " in sql or "table_info" in sql or "sqlite_master" in sql for sql in statements)
        async with engine.connect() as connection:
            assert (await connection.execute(text("SELECT name FROM sqlite_master WHERE type = 'table'"))).all() == []
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_sql_cold_store_continues_official_capability_after_settled_tool(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'cold.db'}")
    effects = InMemoryStepStore()
    store = SqlStepStore(engine, namespace="steps", tenant_id="one", tool_effect_store=effects)
    await store.initialize(local=True)
    calls = []

    def failing_model(messages, info):
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            raise RuntimeError("model failed after tool")
        return ModelResponse(parts=[ToolCallPart("work", {}, tool_call_id="call")])

    agent = Agent(FunctionModel(failing_model), capabilities=[StepPersistence(store=store, capture_frontier=True)])

    @agent.tool_plain
    def work() -> str:
        calls.append("effect")
        return "saved-result"

    try:
        with pytest.raises(RuntimeError, match="model failed after tool"):
            await agent.run("question")
        original = (await store.list_runs())[0]
        for generated in (original.run_id, original.conversation_id):
            assert validate_resource_id(generated) == generated
        await store.close()
        cold = SqlStepStore(engine, namespace="steps", tenant_id="one", tool_effect_store=effects)
        history = await continue_run(cold, run_id=original.run_id)
        assert [part.content for message in history for part in message.parts if isinstance(part, ToolReturnPart)] == ["saved-result"]
        resumed = Agent(FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("done")])),
                         capabilities=[StepPersistence(store=cold)])
        assert (await resumed.run(message_history=history, conversation_id="conversation")).output == "done"
        assert calls == ["effect"]
        assert len(await cold.list_runs()) == 2
        await cold.close()
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_sql_effect_binding_borrows_partition_without_owning_its_lifetime(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'binding.db'}")
    store = SqlStepStore(engine, namespace="steps", tenant_id="one")
    first_owner, second_owner = InMemoryStepStore(), InMemoryStepStore()
    record = ToolEffectRecord(run_id="run", tool_call_id="call", tool_name="tool", status="started")
    try:
        await store.initialize(local=True)
        with pytest.raises(RuntimeError, match="requires a bound tool-effect store"):
            await store.record_tool_effect(record)
        with pytest.raises(RuntimeError, match="requires a bound tool-effect store"):
            await store.get_tool_effect(run_id="run", tool_call_id="call")
        with pytest.raises(RuntimeError, match="requires a bound tool-effect store"):
            await store.list_unresolved_tool_effects(run_id="run")
        first = store.with_tool_effect_store(first_owner)
        second = store.with_tool_effect_store(second_owner)
        await first.record_tool_effect(record)
        assert await second.get_tool_effect(run_id="run", tool_call_id="call") is None
        await first.register_run(RunRecord(run_id="run"))
        assert await second.get_run(run_id="run") == await store.get_run(run_id="run")
        await first.close()
        await second.register_run(RunRecord(run_id="next"))
        assert await store.get_run(run_id="next") is not None
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.parametrize("dialect", (mysql.dialect(), postgresql.dialect(), sqlite.dialect()))
def test_native_indexes_use_bounded_original_identifiers(dialect) -> None:
    metadata = build_step_sql_metadata()
    kind = "mysql" if dialect.name == "mysql" else "portable"
    expected = {
        "runs": {("store_digest", "run_id"): True,
                 ("store_digest", "conversation_id"): False, ("store_digest", "parent_run_id"): False},
        "events": {("store_digest", "run_id"): False},
        "snapshots": {("store_digest", "run_id", "pruned"): False},
    }
    for name, lookups in expected.items():
        table = metadata.tables["ai_step_" + name]
        for column in ("run_id", "conversation_id", "parent_run_id"):
            assert table.c[column].type.length == RESOURCE_ID_MAX_BYTES
            assert "VARCHAR(128)" in str(table.c[column].type.compile(dialect=dialect))
        indexes = {tuple(column.name for column in index.columns): index for index in table.indexes
                   if index.info.get("ddl_dialect") == kind}
        for columns, unique in lookups.items():
            if unique and dialect.name != "mysql":
                assert columns in {tuple(column.name for column in constraint.columns)
                                   for constraint in table.constraints if isinstance(constraint, UniqueConstraint)}
                ddl = str(CreateTable(table).compile(dialect=dialect))
            else:
                index = indexes[columns]
                assert index.unique is unique
                ddl = str(CreateIndex(index).compile(dialect=dialect))
            assert "(" + ", ".join(columns) + ")" in ddl
            if dialect.name == "mysql":
                # MySQL budgets declared characters, not the application's byte limit.
                key_bytes = sum(column.type.length * 4 if getattr(column.type, "length", None) else 1
                                for column in (table.c[name] for name in columns))
                assert key_bytes == (769 if name == "snapshots" else 768)
                assert key_bytes <= 3072


def test_step_sql_metadata_compiles_for_mysql57_and_obeys_project_table_policy() -> None:
    metadata = build_step_sql_metadata()
    dialect = mysql.dialect()
    dialect.server_version_info = (5, 7, 44)
    assert metadata.tables["ai_step_snapshots"].c.messages.type.compile(dialect=dialect) == (
        "VARCHAR(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin"
    )
    for table in metadata.tables.values():
        ddl = str(CreateTable(table).compile(dialect=dialect))
        assert table.name.startswith("ai_") and table.comment
        assert list(table.c)[-2].name == "updated_at" and list(table.c)[-1].name == "created_at"
        assert "id BIGINT NOT NULL" in ddl and "AUTO_INCREMENT" in ddl
        assert not table.foreign_keys and "FLOAT" not in ddl
        assert "ON UPDATE CURRENT_TIMESTAMP" in ddl
        indexes = [index for index in table.indexes if index.info.get("ddl_dialect") == "mysql"]
        assert {"ix_updated_at", "ix_created_at"}.issubset({index.name for index in indexes})
        for index in indexes:
            assert len(index.columns) <= 3
            if index.unique:
                assert all(not column.nullable for column in index.columns)
            else:
                assert not any(column.primary_key for column in index.columns)
            assert index.name == ("uk_" if index.unique else "ix_") + "_".join(column.name for column in index.columns)
            assert "(" in str(CreateIndex(index).compile(dialect=dialect))
        for column in table.c:
            assert column.comment
            if column.name.endswith("digest") or column.name == "sha256":
                assert "CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin" == column.type.compile(dialect=dialect)


def test_step_development_ddl_and_complete_metadata_match_canonical_schema() -> None:
    from linktools.ai.migrate import build_sql_schema_metadata

    metadata = build_step_sql_metadata()
    complete = build_sql_schema_metadata()
    migration = (Path(__file__).resolve().parents[2] / "linktools-ai" / "migrations" / "init_schema.sql").read_text()
    compact = " ".join(migration.split())

    def normalize_nullable_columns(sql: str) -> str:
        # MySQL's implicit nullable default and explicit NULL are equivalent.
        # Preserve NOT NULL, defaults, and the complete quoted comment text.
        chunks = re.split(r"('(?:[^']|'')*')", sql)
        return " ".join("".join(
            re.sub(r"(?<!NOT )(?<!DEFAULT )\bNULL\s+(?=COMMENT\s*$)", "", chunk) if index % 2 == 0 else chunk
            for index, chunk in enumerate(chunks)
        ).split())
    column = "value TEXT COMMENT 'literal NULL COMMENT ''nested''; data'"
    assert normalize_nullable_columns(column.replace("TEXT COMMENT", "TEXT NULL COMMENT")) == column
    for declaration in ("NOT NULL", "DEFAULT NULL"):
        assert normalize_nullable_columns(column.replace("TEXT COMMENT", f"TEXT {declaration} COMMENT")) != column
    dialect = mysql.dialect()
    for name, table in metadata.tables.items():
        assert list(complete.tables[name].c.keys()) == list(table.c.keys())
        ddl = str(CreateTable(table).compile(dialect=dialect))
        # MySQL's independent table options may be emitted in any order.
        columns = ddl.rsplit(")", 1)[0]
        statement = re.search(rf"CREATE TABLE {re.escape(name)} \((?:[^;']|'(?:[^']|'')*')*;", compact)
        assert statement is not None
        actual_columns = re.sub(
            r", (?:UNIQUE )?(?:KEY|INDEX) \w+ \((?:[^()]|\([^()]*\))+\)",
            "",
            statement.group(),
        ).rsplit(")", 1)[0]
        assert normalize_nullable_columns(columns) == normalize_nullable_columns(actual_columns)
        assert f"COMMENT='{table.comment}'" in compact
    assert all(line == line.rstrip() for line in migration.splitlines())
