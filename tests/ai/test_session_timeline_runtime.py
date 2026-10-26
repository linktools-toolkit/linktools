#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""End-to-end Session timeline coverage through the public Runtime API."""
from ._runtime_test_helpers import model_features

from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path

import pytest

from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus
from linktools.ai.capability import ExecutionCapability
from linktools.ai.history import History
from linktools.ai.core import Page
from linktools.ai.execution import SessionTurn
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage

from ._runtime_test_helpers import RuntimeUsageModels


@pytest.mark.asyncio
async def test_in_memory_session_run_restores_timeline() -> None:
    application = RuntimeFeatures("application")
    application.capability(ExecutionCapability(retain_history=True), scope="runtime")
    application.agent("default", model="default", allow_tools=())

    async with Persistence.open('default', storage=PersistenceStorage.in_memory()) as persistence_27, Runtime.open(persistence=persistence_27, features=model_features(RuntimeUsageModels(), application)) as runtime:
        session = await runtime.agents.get("default").create_session("session")
        first = (await session.run("hello", timeout_seconds=10)).result
        second = (await session.run("again", timeout_seconds=10)).result
        assert first.status is ExecutionStatus.SUCCEEDED
        assert second.status is ExecutionStatus.SUCCEEDED

        page = await session.timeline()
        assert [turn.execution_id for turn in page.items] == [
            first.execution_id,
            second.execution_id,
        ]
        assert [turn.status for turn in page.items] == [
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.SUCCEEDED,
        ]
        assert [turn.user_input["prompt"] for turn in page.items] == [
            {"kind": "text", "text": "hello"},
            {"kind": "text", "text": "again"},
        ]
        assert all(turn.conversation_committed for turn in page.items)
        assert [
            [item.item_kind for item in turn.items] for turn in page.items
        ] == [["assistant"], ["assistant"]]
        assert page.next_cursor is None


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem"))
async def test_session_timeline_fork_reads_match_live_and_reopened_history(
    tmp_path: Path, backend: str,
) -> None:
    application = RuntimeFeatures("application")
    application.capability(ExecutionCapability(retain_history=True), scope="runtime")
    application.agent("default", model="default", allow_tools=())
    storage = (
        PersistenceStorage.in_memory()
        if backend == "memory"
        else PersistenceStorage.filesystem(tmp_path)
    )

    async def check_pages(
        read: Callable[..., Awaitable[Page[SessionTurn]]],
        expected: tuple[SessionTurn, ...],
    ) -> None:
        full = await read()
        assert full.items == expected
        assert full.next_cursor is None
        newest = await read(limit=1)
        assert newest.items == expected[-1:]
        assert newest.next_cursor is not None
        oldest = await read(limit=1, cursor=newest.next_cursor)
        assert oldest.items == expected[:1]
        assert oldest.next_cursor is None

    async with Persistence.open('default', storage=storage) as persistence_86, Runtime.open(persistence=persistence_86, features=model_features(RuntimeUsageModels(), application)) as runtime:
        principal = runtime.default_principal
        parent = await runtime.agents.get("default").create_session("parent")
        assert (await parent.timeline()).items == ()
        assert (await runtime.history.session_timeline(
            "parent", principal=principal,
        )).items == ()
        parent_turn = (await parent.run("before fork", timeout_seconds=10)).result
        assert parent_turn.status is ExecutionStatus.SUCCEEDED

        child = await parent.fork("child")
        await parent.close()
        child_turn = (await child.run("after fork", timeout_seconds=10)).result
        assert child_turn.status is ExecutionStatus.SUCCEEDED

        page = await child.timeline()
        assert [turn.execution_id for turn in page.items] == [
            parent_turn.execution_id,
            child_turn.execution_id,
        ]
        assert [turn.user_input["prompt"] for turn in page.items] == [
            {"kind": "text", "text": "before fork"},
            {"kind": "text", "text": "after fork"},
        ]
        assert all(turn.conversation_committed for turn in page.items)
        assert all(turn.items for turn in page.items)
        await check_pages(child.timeline, page.items)
        await check_pages(
            partial(runtime.history.session_timeline, "child", principal=principal),
            page.items,
        )

    if backend == "filesystem":
        async with Persistence.open('default', storage=PersistenceStorage.filesystem(tmp_path), tenant_id="default", read_only=True) as history_persistence_118, History.open(persistence=history_persistence_118) as history:
            await check_pages(
                partial(history.session_timeline, "child", principal=principal),
                page.items,
            )
