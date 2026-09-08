#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime MemoryStore listing must never perform an unbounded scan."""

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


async def test_list_paths_pushes_caller_limit_to_path_ordered_repository() -> None:
    store = object.__new__(RuntimeMemoryStore)
    observed: list[int | None] = []

    async def list_records(*, limit: int | None) -> tuple[list[Any], bool]:
        observed.append(limit)
        return [SimpleNamespace(metadata={"path": "memory/a.md"})], True

    store._list_records = list_records  # type: ignore[method-assign]

    assert await store.list_paths("memory", limit=10) == ["memory/a.md"]
    assert observed == [10]


async def test_list_paths_fails_closed_when_fallback_budget_is_exceeded() -> None:
    store = object.__new__(RuntimeMemoryStore)
    observed: list[int | None] = []

    async def list_records(*, limit: int | None) -> tuple[list[Any], bool]:
        observed.append(limit)
        if len(observed) == 1:
            return [SimpleNamespace(metadata={"path": "archive/a.md"})], True
        return [], True

    store._list_records = list_records  # type: ignore[method-assign]

    with pytest.raises(AIError) as raised:
        await store.list_paths("memory", limit=10)

    assert observed[0] == 10
    assert isinstance(observed[1], int)
    assert observed[1] > observed[0]
    assert raised.value.code is ErrorCode.STORAGE_UNAVAILABLE
    assert raised.value.safe_details == {"reason": "memory_listing_capacity"}
    assert raised.value.retryable is False


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
