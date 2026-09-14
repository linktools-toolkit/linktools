#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end Session timeline coverage through the public Runtime API."""

from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus
from linktools.ai.runtime import Runtime
from linktools.ai.runtime.state import (
    RuntimeDomain,
    RuntimeState,
    RuntimeStatePlan,
    RuntimeStateRoute,
)

from .test_runtime_composition_regressions import (
    _RuntimeUsageModels,
    _runtime_usage_workspace,
)


@pytest.mark.asyncio
async def test_in_memory_session_run_restores_timeline(tmp_path: Path) -> None:
    workspace = _runtime_usage_workspace(tmp_path / "workspace")

    async with Runtime.open(
        workspace,
        models=_RuntimeUsageModels(),  # type: ignore[arg-type]
        state=RuntimeState.in_memory(),
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
async def test_transient_fork_survives_parent_close_and_releases_on_child_close(
    tmp_path: Path,
) -> None:
    workspace = _runtime_usage_workspace(tmp_path / "workspace-fork")
    state = RuntimeState.from_plan(
        RuntimeStatePlan(conversation=RuntimeStateRoute.transient())
    )

    async with Runtime.open(
        workspace,
        models=_RuntimeUsageModels(),  # type: ignore[arg-type]
        state=state,
    ) as runtime:
        parent = await runtime.agent("default").create_session("parent")
        parent_turn = await parent.run("before fork", timeout_seconds=10)
        assert parent_turn.status is ExecutionStatus.SUCCEEDED

        tenant_id = runtime.default_principal.tenant_id
        parent_record = await state.conversation.sessions.get(
            "parent", tenant_id=tenant_id
        )
        assert parent_record is not None
        assert parent_record.continuation is not None
        parent_run_id = parent_record.continuation.step_run_id

        child = await parent.fork("child")
        await parent.close()
        archive = state.steps.read_store(RuntimeDomain.CONVERSATION)
        assert await archive.get_run(run_id=parent_run_id) is not None

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

        child_record = await state.conversation.sessions.get(
            "child", tenant_id=tenant_id
        )
        assert child_record is not None
        assert child_record.continuation is not None
        child_run_id = child_record.continuation.step_run_id

        await child.close()
        assert await archive.get_run(run_id=parent_run_id) is None
        assert await archive.get_run(run_id=child_run_id) is None
