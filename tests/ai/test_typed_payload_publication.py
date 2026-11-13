#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Content addresses reuse the winning representation without a body pre-read."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._archive import ContentArchive, ContentRecord, PreparedArchive
from linktools.ai.persistence._step_payload_owner import PAYLOAD_CODEC


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_payload_duplicate_keeps_winner_bytes_and_gc_revision(tmp_path: Path, backend: str) -> None:
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "payload.db")
    async with Persistence.open("payload-publish", tenant_id="tenant", storage=storage) as state:
        scope, store = await state.execution.executions.state_store.storage_group.open_archive(create=True)
        archive = ContentArchive(store, scope, namespace="payload-publish", tenant_id="tenant", object_store=None)
        raw = b"compressible result " * 2000
        prepared = PreparedArchive()
        identity, = await archive.prepare_content(((PAYLOAD_CODEC, raw),), prepared)
        candidate, = prepared.records.values()
        encoded = ContentRecord.from_record(candidate)
        assert encoded.compression == "zlib"
        winner_content = replace(encoded, compression="raw", body=raw)
        winner = replace(candidate, storage_revision=7, data=winner_content.to_json(), payload_bytes=raw)
        await store.mutate(lambda tx: tx.insert_record(winner))

        async def reuse(transaction):
            # An ignored candidate is not an observation of the winning GC revision.
            await archive.publish(transaction, prepared)
            actual = await transaction.get_record(winner.key)
            assert actual == winner
            assert await transaction.guard_record_revision(winner.key, expected_storage_revision=0) is None

        await store.mutate(reuse)
        assert await store.read_record(winner.key) == winner
        assert await archive.read_contents((identity,), codec=PAYLOAD_CODEC) == (raw,)

        # Repeated/concurrent publication preserves both bytes and revision.
        await asyncio.gather(*(archive.ensure_content(((PAYLOAD_CODEC, raw),)) for _ in range(2)))
        assert await store.read_record(winner.key) == winner

        async def guard_then_reuse(transaction):
            await transaction.get_record(winner.key)
            guarded = await transaction.guard_record_revision(winner.key, expected_storage_revision=7)
            assert guarded == 8
            reads = []
            def observe(connection, cursor, statement, parameters, context, executemany):
                if statement.lstrip().upper().startswith(("SELECT", "WITH", "INSERT", "UPDATE")):
                    reads.append(statement)
            engine = store.storage_group.context.engine.sync_engine if backend == "sqlite" else None
            if engine is not None:
                event.listen(engine, "before_cursor_execute", observe)
            try:
                await archive.publish(transaction, prepared)
                assert await transaction.guard_record_revision(winner.key, expected_storage_revision=7) == guarded
                assert (await transaction.get_record(winner.key)).storage_revision == guarded
            finally:
                if engine is not None:
                    event.remove(engine, "before_cursor_execute", observe)
            assert reads == []

        await store.mutate(guard_then_reuse)
        assert (await store.read_record(winner.key)).storage_revision == 8


@pytest.mark.asyncio
async def test_payload_new_and_duplicate_publish_do_not_pre_read(tmp_path: Path) -> None:
    async with Persistence.open("payload-publish", tenant_id="tenant",
                                storage=PersistenceStorage.sqlite(tmp_path / "payload.db")) as state:
        scope, store = await state.execution.executions.state_store.storage_group.open_archive(create=True)
        archive = ContentArchive(store, scope, namespace="payload-publish", tenant_id="tenant", object_store=None)
        statements = []

        def observe(connection, cursor, statement, parameters, context, executemany):
            if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                statements.append(statement)

        engine = store.storage_group.context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", observe)
        try:
            first = await archive.ensure_content(((PAYLOAD_CODEC, b"body"),))
            second = await archive.ensure_content(((PAYLOAD_CODEC, b"body"),))
        finally:
            event.remove(engine, "before_cursor_execute", observe)
        assert first == second
        assert statements == []
        assert await archive.read_contents(first, codec=PAYLOAD_CODEC) == (b"body",)


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown_count", (False, True))
async def test_mixed_payload_publish_invalidates_cached_absence(tmp_path: Path, monkeypatch, unknown_count: bool) -> None:
    from linktools.ai.storage import InMemoryCacheBackend

    cache = InMemoryCacheBackend(max_bytes=1_000_000)
    async with Persistence.open("payload-publish", tenant_id="tenant",
                                storage=PersistenceStorage.sqlite(tmp_path / "mixed.db", cache=cache)) as state:
        scope, store = await state.execution.executions.state_store.storage_group.open_archive(create=True)
        archive = ContentArchive(store, scope, namespace="payload-publish", tenant_id="tenant", object_store=None)
        await archive.ensure_content(((PAYLOAD_CODEC, b"old"),))
        key, = archive.content_record_keys(((PAYLOAD_CODEC, b"new"),))
        assert await store.read_record(key) is None
        if unknown_count:
            dialect = store.storage_group.context.dialect
            original = dialect.insert_ignore_conflict_many

            async def insert(*args, **kwargs):
                await original(*args, **kwargs)
                return None  # MySQL may not report which members of a batch were new.

            async def unsupported(*args, **kwargs):
                return None

            monkeypatch.setattr(dialect, "insert_ignore_conflict_many_returning", unsupported)
            monkeypatch.setattr(dialect, "insert_ignore_conflict_many", insert)
        await archive.ensure_content(((PAYLOAD_CODEC, b"old"), (PAYLOAD_CODEC, b"new")))
        assert await store.read_record(key) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("rollback", (False, True))
async def test_mixed_payload_batch_reuses_only_returned_rows(tmp_path: Path, rollback: bool) -> None:
    async with Persistence.open("payload-publish", tenant_id="tenant",
                                storage=PersistenceStorage.sqlite(tmp_path / "returned.db")) as state:
        scope, store = await state.execution.executions.state_store.storage_group.open_archive(create=True)
        archive = ContentArchive(store, scope, namespace="payload-publish", tenant_id="tenant", object_store=None)
        await archive.ensure_content(((PAYLOAD_CODEC, b"old"),))
        mixed, fresh = PreparedArchive(), PreparedArchive()
        await archive.prepare_content(((PAYLOAD_CODEC, b"old"), (PAYLOAD_CODEC, b"new")), mixed)
        await archive.prepare_content(((PAYLOAD_CODEC, b"new"),), fresh)
        candidate, = fresh.records.values()
        statements = []

        def observe(connection, cursor, statement, parameters, context, executemany):
            if statement.lstrip().upper().startswith(("SELECT", "WITH", "INSERT")):
                statements.append(statement)

        async def publish(transaction):
            await archive.publish(transaction, mixed)
            assert await transaction.get_record(candidate.key) == candidate
            await archive.publish(transaction, fresh)
            if rollback:
                raise ValueError("roll back the new content")

        engine = store.storage_group.context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", observe)
        try:
            if rollback:
                with pytest.raises(ValueError, match="roll back"):
                    await store.mutate(publish)
            else:
                await store.mutate(publish)
        finally:
            event.remove(engine, "before_cursor_execute", observe)
        assert len(statements) == 1
        assert statements[0].startswith("INSERT INTO ai_step_messages")
        assert await store.read_record(candidate.key) == (None if rollback else candidate)
        if rollback:
            await archive.ensure_content(((PAYLOAD_CODEC, b"new"),))
            assert await store.read_record(candidate.key) == candidate


@pytest.mark.asyncio
async def test_mysql_payload_returning_fallback_does_not_execute() -> None:
    from linktools.ai.storage import MySQLDialect
    from linktools.ai.persistence._step_schema import build_step_sql_metadata

    class NoExecute:
        async def execute(self, statement):
            pytest.fail("unsupported RETURNING must leave execution to the count fallback")

    table = build_step_sql_metadata().tables["ai_step_messages"]
    assert await MySQLDialect().insert_ignore_conflict_many_returning(
        NoExecute(), table=table, rows=({"sha256": "a" * 64},),
        index_elements=("store_locator", "sha256"), returning=("sha256",),
    ) is None
