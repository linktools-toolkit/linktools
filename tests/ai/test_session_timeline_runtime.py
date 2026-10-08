#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end Session timeline coverage through the public Runtime API."""

from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path

import pytest

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import ExecutionStatus
from linktools.ai.runtime import Page, Runtime, RuntimeHistory, SessionTurn
from linktools.ai.runtime.state import RuntimeStorage

from ._runtime_test_helpers import RuntimeUsageModels


@pytest.mark.asyncio
async def test_in_memory_session_run_restores_timeline() -> None:
    application = CapabilityGroup("application")
    application.agent("default", model="default", allow_tools=())

    async with Runtime.open(
        "default",
        models=RuntimeUsageModels(),  # type: ignore[arg-type]
        storage=RuntimeStorage.in_memory(),
        capabilities=(application,),
    ) as runtime:
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
    application = CapabilityGroup("application")
    application.agent("default", model="default", allow_tools=())
    storage = (
        RuntimeStorage.in_memory()
        if backend == "memory"
        else RuntimeStorage.filesystem(tmp_path)
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

    async with Runtime.open(
        "default",
        models=RuntimeUsageModels(),  # type: ignore[arg-type]
        storage=storage,
        capabilities=(application,),
    ) as runtime:
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
        async with RuntimeHistory.open(
            "default", storage=RuntimeStorage.filesystem(tmp_path),
        ) as history:
            await check_pages(
                partial(history.session_timeline, "child", principal=principal),
                page.items,
            )
