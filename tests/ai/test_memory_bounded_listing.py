#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime memory listing must keep prefix queries bounded at the repository."""

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.memory import PersistentMemoryStore
from linktools.ai.memory._store import _MutationReceipt
from linktools.ai.persistence.contracts import MemoryRecord
from linktools.ai.persistence._memory import InMemoryStateStorageGroup, InMemoryStateStore
from linktools.ai.persistence.repositories import MemoryRepositoryImpl
from linktools.ai.storage import StoredPayload

pytestmark = pytest.mark.asyncio


class _RecordingMemoryRepository:
    def __init__(self, paths: tuple[str, ...]) -> None:
        self._paths = paths
        self.calls: list[dict[str, Any]] = []

    async def list(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        limit = kwargs["limit"]
        start = 0 if kwargs["cursor"] is None else int(kwargs["cursor"])
        selected = self._paths[start : start + limit]
        end = start + len(selected)
        return SimpleNamespace(
            items=tuple(SimpleNamespace(metadata={"path": path}) for path in selected),
            next_cursor=str(end) if end < len(self._paths) else None,
        )


def _runtime_store(records: _RecordingMemoryRepository) -> PersistentMemoryStore:
    store = object.__new__(PersistentMemoryStore)
    store._state = SimpleNamespace(records=records)
    store._tenant_id = "tenant"
    store._memory_scope_digest = "scope-digest"
    return store


async def test_list_paths_pushes_prefix_and_limit_to_repository_once() -> None:
    records = _RecordingMemoryRepository(("memory/a.md",))
    store = _runtime_store(records)

    assert await store.list_paths("memory", limit=10) == ["memory/a.md"]
    assert records.calls == [
        {
            "tenant_id": "tenant",
            "memory_scope_digest": "scope-digest",
            "prefix": "memory",
            "cursor": None,
            "limit": 10,
        }
    ]


async def test_list_paths_pages_for_large_harness_limit() -> None:
    records = _RecordingMemoryRepository(
        tuple(f"memory/{index:04d}.md" for index in range(1600))
    )
    store = _runtime_store(records)

    paths = await store.list_paths("memory/", limit=1500)

    assert len(paths) == 1500
    assert paths[0] == "memory/0000.md"
    assert paths[-1] == "memory/1499.md"
    assert [call["limit"] for call in records.calls] == [1000, 500]
    assert records.calls[1]["cursor"] == "1000"


async def test_memory_receipt_rejects_unwritten_appended_status() -> None:
    payload = json.dumps(
        {
            "version": 1,
            "result": {
                "file": "memory/a.md",
                "version": "m1:" + "a" * 64,
                "status": "appended",
            },
        }
    )

    with pytest.raises(AIError) as raised:
        _MutationReceipt.decode(payload)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


async def test_list_paths_fails_closed_if_repository_breaks_prefix_contract() -> None:
    records = _RecordingMemoryRepository(("archive/a.md",))
    store = _runtime_store(records)

    with pytest.raises(AIError) as raised:
        await store.list_paths("memory", limit=10)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert len(records.calls) == 1


async def test_memory_repository_projects_logical_path_as_sort_key() -> None:
    repository = MemoryRepositoryImpl(
        InMemoryStateStore(InMemoryStateStorageGroup()),
        namespace="memory-ordering",
        tenant_id="tenant",
    )
    now = datetime.now(timezone.utc)
    value = MemoryRecord(
        "memory-id",
        "scope-digest",
        StoredPayload.inline_text("content"),
        {"path": "memory/z.md"},
        1,
        now,
        now,
    )

    stored = repository._stored("memory", value.memory_id, value)

    assert stored.sort_key == "memory/z.md"


async def test_memory_repository_accepts_full_harness_path_length() -> None:
    repository = MemoryRepositoryImpl(
        InMemoryStateStore(InMemoryStateStorageGroup()),
        namespace="memory-long-path",
        tenant_id="tenant",
    )
    now = datetime.now(timezone.utc)
    logical_path = "memory/" + "x" * 197 + ".md"
    value = MemoryRecord(
        "memory-id",
        "scope-digest",
        StoredPayload.inline_text("content"),
        {"path": logical_path},
        1,
        now,
        now,
    )

    stored = repository._stored("memory", value.memory_id, value)

    assert len(logical_path) > 128
    assert stored.sort_key == logical_path


async def test_memory_repository_applies_prefix_before_limit() -> None:
    state = InMemoryStateStore(InMemoryStateStorageGroup())
    await state.initialize()
    try:
        repository = MemoryRepositoryImpl(
            state,
            namespace="memory-prefix",
            tenant_id="tenant",
        )
        now = datetime.now(timezone.utc)
        values = tuple(
            MemoryRecord(
                memory_id,
                "scope-digest",
                StoredPayload.inline_text("content"),
                {"path": path},
                1,
                now,
                now,
            )
            for memory_id, path in (
                ("archive-a", "archive/a.md"),
                ("memory-a", "memory/a.md"),
                ("memory-b", "memory/b.md"),
            )
        )
        stored = tuple(
            repository._stored("memory", value.memory_id, value) for value in values
        )
        await state.mutate(lambda transaction: transaction.insert_records(stored))

        first = await repository.list(
            tenant_id="tenant",
            memory_scope_digest="scope-digest",
            prefix="memory/",
            cursor=None,
            limit=1,
        )
        assert [value.metadata["path"] for value in first.items] == ["memory/a.md"]
        assert first.next_cursor is not None

        second = await repository.list(
            tenant_id="tenant",
            memory_scope_digest="scope-digest",
            prefix="memory/",
            cursor=first.next_cursor,
            limit=10,
        )
        assert [value.metadata["path"] for value in second.items] == ["memory/b.md"]
    finally:
        await state.close()






async def test_list_paths_preserves_lexical_prefix() -> None:
    records = _RecordingMemoryRepository(("memory/001.md", "memory/010.md"))
    store = _runtime_store(records)

    assert await store.list_paths("memory/0", limit=2) == [
        "memory/001.md",
        "memory/010.md",
    ]
    assert records.calls == [
        {
            "tenant_id": "tenant",
            "memory_scope_digest": "scope-digest",
            "prefix": "memory/0",
            "cursor": None,
            "limit": 2,
        }
    ]


async def test_list_paths_preserves_empty_prefix() -> None:
    records = _RecordingMemoryRepository(("memory/a.md",))
    store = _runtime_store(records)

    assert await store.list_paths("", limit=1) == ["memory/a.md"]
    assert records.calls[0]["prefix"] == ""


@pytest.mark.parametrize("prefix", ("", "memory/"))
async def test_sql_memory_listing_reopens_and_reads_only_requested_page(
    tmp_path: Path, prefix: str,
) -> None:
    from sqlalchemy import event
    from linktools.ai.persistence import Persistence, PersistenceStorage

    storage = PersistenceStorage.sqlite(tmp_path / "memory.db")
    now = datetime.now(timezone.utc)
    values = tuple(
        MemoryRecord(
            f"memory-{index}", "scope-digest", StoredPayload.inline_text("content"),
            {"path": f"memory/{index:03d}.md"}, 1, now, now,
        ) for index in range(3)
    )
    async with Persistence.open("memory-bounded", storage=storage, tenant_id="tenant") as state:
        repository = state.memory.records
        await repository.state_store.mutate(lambda transaction: transaction.insert_records(
            tuple(repository._stored("memory", value.memory_id, value) for value in values)
        ))
    async with Persistence.open("memory-bounded", storage=storage, tenant_id="tenant", read_only=True) as state:
        repository = state.memory.records
        engine = repository.state_store.context.engine
        queries = []

        def capture(connection, cursor, statement, parameters, context, many):
            if statement.lstrip().upper().startswith("SELECT") and "ai_state_records" in statement:
                queries.append((statement, parameters))

        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            page = await repository.list(
                tenant_id="tenant", memory_scope_digest="scope-digest", prefix=prefix,
                cursor=None, limit=1,
            )
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
        assert [value.metadata["path"] for value in page.items] == ["memory/000.md"]
        assert page.next_cursor is not None
        assert len(queries) == 1
        assert "LIMIT" in queries[0][0]
        assert queries[0][1][-2:] == (2, 0)
        second = await repository.list(
            tenant_id="tenant", memory_scope_digest="scope-digest", prefix="memory/0",
            cursor=page.next_cursor, limit=1,
        )
        assert [value.metadata["path"] for value in second.items] == ["memory/001.md"]
