#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime memory listing must keep prefix queries bounded at the repository."""

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._memory import RuntimeMemoryStore
from linktools.ai.runtime.state._contracts import MemoryRecord
from linktools.ai.runtime.state._memory import MemoryStateStorageGroup, MemoryStateStore
from linktools.ai.runtime.state._repositories import MemoryRepositoryImpl
from linktools.ai.storage import StoredPayload

pytestmark = pytest.mark.asyncio


class _RecordingMemoryRepository:
    def __init__(self, paths: tuple[str, ...]) -> None:
        self._paths = paths
        self.calls: list[dict[str, Any]] = []

    async def list(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        limit = kwargs["limit"]
        return SimpleNamespace(
            items=tuple(
                SimpleNamespace(metadata={"path": path})
                for path in self._paths[:limit]
            ),
            next_cursor=None,
        )


def _runtime_store(records: _RecordingMemoryRepository) -> RuntimeMemoryStore:
    store = object.__new__(RuntimeMemoryStore)
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
            "prefix": "memory/",
            "cursor": None,
            "limit": 10,
        }
    ]


async def test_list_paths_fails_closed_if_repository_breaks_prefix_contract() -> None:
    records = _RecordingMemoryRepository(("archive/a.md",))
    store = _runtime_store(records)

    with pytest.raises(AIError) as raised:
        await store.list_paths("memory", limit=10)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert len(records.calls) == 1


async def test_memory_repository_projects_logical_path_as_sort_key() -> None:
    repository = MemoryRepositoryImpl(
        MemoryStateStore(MemoryStateStorageGroup()),
        namespace="memory-ordering",
        tenant_id="tenant",
    )
    now = datetime.now(timezone.utc)
    value = MemoryRecord(
        "memory-id",
        "tenant",
        "scope-digest",
        StoredPayload.inline_text("content"),
        {"path": "memory/z.md"},
        1,
        now,
        now,
    )

    stored = repository._stored("memory", value.memory_id, value)

    assert stored.sort_key == "memory/z.md"


async def test_memory_repository_applies_prefix_before_limit() -> None:
    state = MemoryStateStore(MemoryStateStorageGroup())
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
                "tenant",
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
            repository._stored("memory", value.memory_id, value)
            for value in values
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
