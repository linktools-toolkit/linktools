#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime MemoryStore listing must never perform an unbounded scan."""

from typing import Any

import pytest

from linktools.ai.runtime._memory import RuntimeMemoryStore

pytestmark = pytest.mark.asyncio


async def test_list_paths_uses_finite_scan_budget() -> None:
    store = object.__new__(RuntimeMemoryStore)
    observed: list[int | None] = []

    async def list_records(*, limit: int | None) -> tuple[list[Any], bool]:
        observed.append(limit)
        return [], False

    store._list_records = list_records  # type: ignore[method-assign]

    assert await store.list_paths("memory", limit=10) == []
    assert len(observed) == 1
    assert isinstance(observed[0], int)
    assert observed[0] > 10


async def test_list_paths_fails_closed_when_scan_budget_is_exceeded() -> None:
    store = object.__new__(RuntimeMemoryStore)

    async def list_records(*, limit: int | None) -> tuple[list[Any], bool]:
        assert isinstance(limit, int)
        return [], True

    store._list_records = list_records  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="bounded scan capacity"):
        await store.list_paths("memory", limit=10)
