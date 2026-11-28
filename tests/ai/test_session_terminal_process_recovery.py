#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical SQL tool result recovery around terminal commit process exits."""

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.core import ExecutionStatus, ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.persistence import Persistence, PersistencePlan, PersistenceStorage
from linktools.ai.persistence.turns import SessionTurnKey
from linktools.ai.runtime import Runtime
from linktools.ai.runtime._regenerable_turn import _RegenerableTurnRunner
from linktools.ai.storage import StorageRoute

from .test_session_tool_process_recovery import (
    _HostAuthorization,
    _SESSION,
    _expire,
    _features,
    _run_until_crash,
    _send,
)

_NAMESPACE = "terminal-tool-crash"


@asynccontextmanager
async def _storage(root: Path, backend: str, *, provision: bool):
    if backend == "sqlite":
        yield PersistenceStorage.sqlite(root / "runtime.db")
    elif backend == "split_sqlite":
        conversation = StorageRoute.sqlite(root / "conversation.db")
        step = StorageRoute.sqlite(root / "step.db")
        yield PersistenceStorage(PersistencePlan(conversation=conversation, execution=step, checkpoint=step))
    else:
        engine = create_async_engine(URL.create("sqlite+aiosqlite", database=str(root / "runtime.db")))
        try:
            if provision:
                await provision_runtime_database(engine)
            yield PersistenceStorage.sql(engine)
        finally:
            await engine.dispose()


def _exit_at_native_terminal(root_path: str, backend: str, phase: str) -> None:
    original = _RegenerableTurnRunner._finish_retained_turn

    async def finish(runner, turn, **kwargs):
        assert kwargs["state"] == "succeeded"
        if phase == "before_terminal":
            os._exit(91)
        committed = await original(runner, turn, **kwargs)
        assert committed
        os._exit(91)

    _RegenerableTurnRunner._finish_retained_turn = finish

    async def run():
        root = Path(root_path)
        async with (
            _storage(root, backend, provision=True) as storage,
            Persistence.open(_NAMESPACE, storage=storage) as state,
            Runtime.open(persistence=state, features=_features(root, traced=True),
                         authorization=_HostAuthorization()) as runtime,
        ):
            await runtime.agents.get().create_session(_SESSION)
            await (await _send(runtime)).wait_final(timeout_seconds=15)
        raise AssertionError("canonical terminal commit boundary not reached")

    asyncio.run(run())


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", (
    "sqlite",
    pytest.param("sql", marks=pytest.mark.merge),
))
@pytest.mark.parametrize("phase", ("before_terminal", "after_terminal"))
async def test_session_tool_turn_recovers_at_terminal_commit_without_replaying_effect(
    tmp_path: Path,
    backend: str,
    phase: str,
) -> None:
    await _run_until_crash(_exit_at_native_terminal, str(tmp_path), backend, phase)
    effects = tmp_path / "effects.log"
    provider = tmp_path / "provider.log"
    [turn_id] = effects.read_text().splitlines()
    original_provider_calls = provider.read_text().splitlines()
    assert original_provider_calls == ["initial", "continuation"]
    key = SessionTurnKey(_SESSION, turn_id)
    async with (
        _storage(tmp_path, backend, provision=False) as storage,
        Persistence.open(_NAMESPACE, storage=storage) as state,
        Runtime.open(persistence=state, features=_features(tmp_path, traced=True),
                     authorization=_HostAuthorization()) as runtime,
    ):
        turns, tools = state.session_turns, state.checkpoint.tools
        tenant, principal = runtime.tenant_id, runtime.default_principal
        before = await turns.get_turn(key)
        [receipt] = await tools.list_by_execution(turn_id, tenant_id=tenant)
        assert receipt.status is ToolOperationStatus.COMPLETED
        repeated = await _send(runtime)
        assert repeated.key == key and await turns.get_turn(key) == before
        if phase == "before_terminal":
            assert before["state"] == "accepted" and before["generation"] == 1
            assert not await runtime.sessions.recover_turn(_SESSION, turn_id, mode="resume", principal=principal)
            await _expire(turns, key, generation=1)
            assert await runtime.sessions.recover_turn(_SESSION, turn_id, mode="resume", principal=principal)
        else:
            assert before["state"] == "succeeded" and before["generation"] == 1
            assert not await runtime.sessions.recover_turn(_SESSION, turn_id, mode="resume", principal=principal)
        result = await repeated.wait_final(timeout_seconds=15)
        assert result.key == key and result.status is ExecutionStatus.SUCCEEDED
        assert result.view["output"] == {"text": "done"}
        assert (await turns.get_turn(key))["generation"] == (2 if phase == "before_terminal" else 1)
        assert (await turns.get_session(_SESSION))["active_turn_id"] is None
        [published] = (await runtime.history.turn_session(_SESSION, principal=principal))["timeline"]["items"]
        assert published["turn_id"] == turn_id and published["conversation_committed"]
        [operation] = published["process"]
        assert operation["key"] == "tool:stable-call" and operation["status"] == "SUCCEEDED"
        assert await tools.list_by_execution(turn_id, tenant_id=tenant) == (receipt,)
        assert await state.execution.executions.get(turn_id, tenant_id=tenant) is None
        # The crash occurs before the optional asynchronous Trace offer;
        # recovery must not invent an attempt or repair diagnostics.
        page = await runtime.history.turn_trace(_SESSION, turn_id, principal=principal)
        assert page.items == () and page.next_cursor is None
    assert effects.read_text().splitlines() == [turn_id]
    assert provider.read_text().splitlines() == original_provider_calls


@pytest.mark.asyncio
@pytest.mark.merge
async def test_split_sqlite_rejects_an_uncoordinated_effect_before_dispatch(tmp_path: Path) -> None:
    async with (
        _storage(tmp_path, "split_sqlite", provision=True) as storage,
        Persistence.open(_NAMESPACE, storage=storage) as state,
        Runtime.open(persistence=state, features=_features(tmp_path),
                     authorization=_HostAuthorization()) as runtime,
    ):
        await runtime.agents.get().create_session(_SESSION)
        with pytest.raises(AIError) as denied:
            await _send(runtime)
        assert denied.value.code is ErrorCode.STORAGE_CAPABILITY_MISSING
        row = await state.session_turns.get_session(_SESSION)
        assert row["active_turn_id"] is None and row["last_turn_no"] == 0
    assert not (tmp_path / "effects.log").exists()
    assert not (tmp_path / "provider.log").exists()
