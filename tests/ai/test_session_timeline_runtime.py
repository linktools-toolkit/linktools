#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end Session timeline coverage through the public Runtime API."""

import pytest

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import ExecutionStatus
from linktools.ai.runtime import Runtime
from linktools.ai.runtime.state import RuntimeStorage

from ._runtime_test_helpers import RuntimeUsageModels


@pytest.mark.asyncio
async def test_in_memory_session_run_restores_timeline() -> None:
    application = CapabilityGroup("application")
    application.agent("default", model_route="default", allow_tools=())

    async with Runtime.open(
        "default",
        models=RuntimeUsageModels(),  # type: ignore[arg-type]
        storage=RuntimeStorage.in_memory(),
        capabilities=(application,),
    ) as runtime:
        session = await runtime.agent("default").create_session("session")
        first = await session.run("hello", timeout_seconds=10)
        second = await session.run("again", timeout_seconds=10)
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
async def test_in_memory_fork_survives_parent_close() -> None:
    application = CapabilityGroup("application")
    application.agent("default", model_route="default", allow_tools=())

    async with Runtime.open(
        "default",
        models=RuntimeUsageModels(),  # type: ignore[arg-type]
        storage=RuntimeStorage.in_memory(),
        capabilities=(application,),
    ) as runtime:
        parent = await runtime.agent("default").create_session("parent")
        parent_turn = await parent.run("before fork", timeout_seconds=10)
        assert parent_turn.status is ExecutionStatus.SUCCEEDED

        child = await parent.fork("child")
        await parent.close()
        child_turn = await child.run("after fork", timeout_seconds=10)
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
