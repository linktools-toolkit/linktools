#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable publication rejects byte changes without a metadata pre-read."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence._archive import ContentArchive, PreparedArchive
from linktools.ai.persistence._memory import InMemoryStateStore
from linktools.ai.persistence.state import StateStore, StateTransaction, StateTransactionNestingError

from .test_state_group_record_batch import _database, _record


@asynccontextmanager
async def _store(backend: str, path: Path) -> AsyncIterator[tuple[StateStore, AsyncEngine | None]]:
    if backend == "memory":
        store = InMemoryStateStore()
        await store.initialize()
        try:
            yield store, None
        finally:
            await store.close()
    else:
        async with _database(path) as (engine, store, _):
            yield store, engine


async def _assert_exact_publication(store: StateStore) -> None:
    value = replace(_record("exact"), data={"text": "\\\\unicode 測🙂 ", "number": -0.0})
    await store.mutate(lambda tx: tx.insert_immutable_records((value,)))
    await asyncio.gather(*(store.mutate(lambda tx: tx.insert_immutable_records((value,))) for _ in range(2)))
    found = await store.read_record(value.key_digest)
    assert found is not None
    assert canonical_json_bytes(dict(found.data)) == canonical_json_bytes(dict(value.data))
    assert found.storage_revision == 0


async def _assert_rejected_batch(store: StateStore, *, difference: str) -> None:
    winner = replace(_record("existing", key=b"\xff" * 32), data={"number": -0.0})
    if difference == "float":
        candidate = replace(winner, data={"number": 0.0})
    elif difference == "integer":
        winner = replace(winner, data={"number": 1})
        candidate = replace(winner, data={"number": 1.0})
    elif difference == "boolean":
        winner = replace(winner, data={"number": True})
        candidate = replace(winner, data={"number": 1})
    elif difference == "trailing-space":
        candidate = replace(winner, sort_key=winner.sort_key + b" ")
    else:
        candidate = replace(winner, sort_key=winner.sort_key.upper())
    await store.mutate(lambda tx: tx.insert_immutable_records((winner,)))
    fresh = tuple(_record(f"new-{index}", key=index.to_bytes(32, "big")) for index in range(70))

    async def swallow(transaction: StateTransaction) -> None:
        try:
            await transaction.insert_immutable_records((*fresh, candidate))
        except AIError as error:
            assert error.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        else:
            pytest.fail("changed immutable content was accepted")
        await transaction.insert_record(_record("after-error"))

    with pytest.raises(AIError) as caught:
        await store.mutate(swallow)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    stored = await store.read(lambda tx: tx.scan_records())
    assert len(stored) == 1
    assert stored[0].sort_key == winner.sort_key
    assert stored[0].storage_revision == winner.storage_revision
    assert canonical_json_bytes(dict(stored[0].data)) == canonical_json_bytes(dict(winner.data))


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_exact_immutable_publication_accepts_repeated_and_concurrent_writers(tmp_path: Path, backend: str) -> None:
    async with _store(backend, tmp_path / "exact.db") as (store, _):
        await _assert_exact_publication(store)
        with pytest.raises(StateTransactionNestingError):
            await store.read(lambda tx: tx.insert_immutable_records((_record("read-only"),)))


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("difference", ("float", "integer", "boolean", "trailing-space", "case"))
async def test_immutable_batch_conflict_rolls_back_even_after_caller_catches(
    tmp_path: Path, backend: str, difference: str,
) -> None:
    async with _store(backend, tmp_path / "conflict.db") as (store, _):
        await _assert_rejected_batch(store, difference=difference)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_complete_archive_publication_adds_no_select_and_retains_missing_dependency_checks(
    tmp_path: Path, backend: str,
) -> None:
    async with _store(backend, tmp_path / "archive.db") as (owner, engine):
        scope, store = await owner.storage_group.open_archive(create=True)
        archive = ContentArchive(store, scope, namespace="archive", tenant_id="tenant", object_store=None)
        prepared = PreparedArchive()
        root = await archive.prepare_append(archive.empty_root(), (("bytes", b"body"),), prepared)
        selects = []

        def observe(connection, cursor, statement, parameters, context, many) -> None:
            if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                selects.append(statement)

        if engine is not None:
            event.listen(engine.sync_engine, "before_cursor_execute", observe)
        try:
            await store.mutate(lambda tx: archive.publish(tx, prepared))
            await store.mutate(lambda tx: archive.publish(tx, prepared))
        finally:
            if engine is not None:
                event.remove(engine.sync_engine, "before_cursor_execute", observe)
        assert selects == []
        assert await archive.read_ranges((root,), codec="bytes") == ((b"body",),)
        content = next(record for record in prepared.records.values() if record.kind == "archive_content")
        partial = PreparedArchive(owner=prepared.owner, nodes=dict(prepared.nodes),
                                  records={key: value for key, value in prepared.records.items() if key != content.key_digest})
        await store.mutate(lambda tx: tx.delete_record(content.key_digest))
        with pytest.raises(AIError) as caught:
            await store.mutate(lambda tx: archive.publish(tx, partial))
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_cancelled_immutable_write_cannot_be_committed_by_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _database(tmp_path / "cancelled.db") as (_, store, _):
        dialect = store.context.dialect
        original = type(dialect).insert_immutable_many

        async def cancelled(self, *args, **kwargs):
            await original(self, *args, **kwargs)
            raise asyncio.CancelledError()

        monkeypatch.setattr(type(dialect), "insert_immutable_many", cancelled)

        async def swallowed(transaction: StateTransaction) -> None:
            try:
                await transaction.insert_immutable_records((_record("cancelled"),))
            except asyncio.CancelledError:
                pass

        with pytest.raises(asyncio.CancelledError):
            await store.mutate(swallowed)
        assert await store.read(lambda tx: tx.scan_records()) == ()
