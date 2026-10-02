#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session tool result recovery across process termination boundaries."""

from pathlib import Path

import pytest
from linktools.ai.core import ExecutionEventType, ExecutionStatus
from linktools.ai.runtime import Runtime, RuntimeStorage
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine

from ._session_tool_test_helpers import (
    _ToolModels,
    _application,
    _exit_at_boundary,
    _relevant_kinds,
    _split_sqlite_storage,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("sqlite", "sql", "split_sqlite"))
@pytest.mark.parametrize(
    "phase",
    (
        "tool_completed",
        "tool_checkpoint",
        "projected_tool_checkpoint",
        "before_terminal",
        "after_terminal",
    ),
)
async def test_session_tool_turn_recovers_after_process_exit_without_replaying_effect(
    tmp_path: Path,
    backend: str,
    phase: str,
) -> None:
    database = tmp_path / "runtime.db"
    effect_log = tmp_path / "effects.txt"
    await _exit_at_boundary(database, effect_log, backend, phase)
    committed_effects = effect_log.read_text().splitlines()
    assert len(committed_effects) == 1
    execution_id = committed_effects[0]
    engine = None
    if backend == "sqlite":
        state = RuntimeStorage.sqlite(database)
    elif backend == "split_sqlite":
        state = _split_sqlite_storage(database)
    else:
        engine = create_async_engine(
            URL.create("sqlite+aiosqlite", database=str(database))
        )
        state = RuntimeStorage.sql(engine)
    calls: list[str] = []
    application = _application(calls, effect_policy="non_replay_safe", effect_log=effect_log)
    try:
        async with Runtime.open(
            "session-tool-crash",
            models=_ToolModels(),
            storage=state,
            capabilities=(application,),
        ) as runtime:
            session = runtime.agents.get("default").session("session")
            same = await session.start("inspect", idempotency_key="turn-1")
            result = await same.wait(timeout_seconds=15)
            assert same.execution_id == execution_id
            assert result.status is ExecutionStatus.SUCCEEDED, result
            assert calls == []
            assert effect_log.read_text().splitlines() == committed_effects
            watched = [
                item
                async for item in same.watch(include_content=True)
                if item.depth == 0
            ]
            assert (
                watched[-1].event.event_type == ExecutionEventType.EXECUTION_SUCCEEDED
            )
            history = await session.history()
            assert _relevant_kinds(history.items) == [
                "user",
                "tool_call",
                "tool_result",
                "assistant",
            ]
            execution_history = await runtime.history.history(
                execution_id, principal=runtime.default_principal, include_content=True
            )
            execution_history_kinds = _relevant_kinds(execution_history.items)
            session_history_kinds = _relevant_kinds(history.items)
            if backend == "split_sqlite":
                assert set(session_history_kinds) <= set(execution_history_kinds)
            else:
                assert execution_history_kinds == session_history_kinds
            interactions = await same.model_interactions(include_content=True)
            assert len(interactions.items) >= 2, interactions.items
            assert interactions.items[-1].response is not None
            assert "done" in str(interactions.items[-1].response)
            repeated = await session.start("inspect", idempotency_key="turn-1")
            assert repeated.execution_id == execution_id
            assert (
                await repeated.wait(timeout_seconds=15)
            ).status is ExecutionStatus.SUCCEEDED
            following = await session.run(
                "continue", idempotency_key="turn-2", timeout_seconds=15
            )
            assert following.status is ExecutionStatus.SUCCEEDED
            assert effect_log.read_text().splitlines() == committed_effects
            if backend == "split_sqlite":
                repeated_history = await session.history()
                assert _relevant_kinds(repeated_history.items) == [
                    "user",
                    "tool_call",
                    "tool_result",
                    "assistant",
                    "user",
                    "assistant",
                ]
    finally:
        await state.close()
        if engine is not None:
            await engine.dispose()
