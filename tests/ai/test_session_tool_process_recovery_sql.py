#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native session tool recovery with a caller-provided SQL engine."""

import asyncio
import os
from pathlib import Path

import pytest
from pydantic_ai.messages import ToolReturnPart

from linktools.ai.core import ExecutionStatus, ToolOperationStatus
from linktools.ai.persistence import Persistence
from linktools.ai.persistence.turns import SessionTurnKey
from linktools.ai.runtime import Runtime
from linktools.ai.runtime._step_store import RuntimeStepStore
from linktools.ai.runtime._tool import RuntimeToolOperationBridge

from .test_session_terminal_process_recovery import _storage
from .test_session_tool_process_recovery import (
    _HostAuthorization,
    _SESSION,
    _expire,
    _features,
    _run_until_crash,
    _send,
)

_NAMESPACE = "explicit-sql-tool-crash"


def _exit_after_native_tool(root_path: str, phase: str) -> None:
    original_complete = RuntimeToolOperationBridge.complete
    original_step = RuntimeStepStore.commit_step

    async def complete(bridge, decision, result):
        completed = await original_complete(bridge, decision, result)
        if phase == "tool_completed":
            os._exit(91)
        return completed

    async def commit_step(store, **kwargs):
        await original_step(store, **kwargs)
        snapshot = kwargs.get("snapshot")
        if (phase != "tool_completed" and snapshot is not None and snapshot.messages
                and any(isinstance(part, ToolReturnPart) for part in snapshot.messages[-1].parts)):
            os._exit(91)

    RuntimeToolOperationBridge.complete = complete
    RuntimeStepStore.commit_step = commit_step

    async def run():
        root = Path(root_path)
        traced = phase == "wrapped_tool_checkpoint_trace"
        async with (
            _storage(root, "sql", provision=True) as storage,
            Persistence.open(_NAMESPACE, storage=storage) as state,
            Runtime.open(persistence=state, features=_features(root, traced=traced, wrapped_return=traced),
                         authorization=_HostAuthorization()) as runtime,
        ):
            await runtime.agents.get().create_session(_SESSION)
            await (await _send(runtime)).wait_final(timeout_seconds=15)
        raise AssertionError("native tool commit boundary not reached")

    asyncio.run(run())


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", (
    "tool_completed",
    pytest.param("tool_checkpoint", marks=pytest.mark.merge),
    pytest.param("wrapped_tool_checkpoint_trace", marks=pytest.mark.merge),
))
async def test_session_tool_turn_recovers_after_process_exit_without_replaying_effect(
    tmp_path: Path, phase: str,
) -> None:
    await _run_until_crash(_exit_after_native_tool, str(tmp_path), phase)
    effect_log, provider_log = tmp_path / "effects.log", tmp_path / "provider.log"
    [turn_id] = effect_log.read_text().splitlines()
    assert provider_log.read_text().splitlines() == ["initial"]
    key = SessionTurnKey(_SESSION, turn_id)
    traced = phase == "wrapped_tool_checkpoint_trace"

    async with (
        _storage(tmp_path, "sql", provision=False) as storage,
        Persistence.open(_NAMESPACE, storage=storage) as state,
        Runtime.open(persistence=state, features=_features(tmp_path, traced=traced, wrapped_return=traced),
                     authorization=_HostAuthorization()) as runtime,
    ):
        principal, tenant = runtime.default_principal, runtime.tenant_id
        turns, tools = state.session_turns, state.checkpoint.tools
        accepted = await turns.get_turn(key)
        assert accepted["state"] == "accepted" and accepted["generation"] == 1
        [receipt] = await tools.list_by_execution(turn_id, tenant_id=tenant)
        assert receipt.status is ToolOperationStatus.COMPLETED and receipt.tool_call_id == "stable-call"
        repeated = await _send(runtime)
        assert repeated.key == key and await turns.get_turn(key) == accepted
        assert not await runtime.sessions.recover_turn(_SESSION, turn_id, mode="resume", principal=principal)
        await _expire(turns, key, generation=1)
        assert await runtime.sessions.recover_turn(_SESSION, turn_id, mode="resume", principal=principal)
        result = await repeated.wait_final(timeout_seconds=15)
        assert result.key == key and result.status is ExecutionStatus.SUCCEEDED
        assert result.view["output"] == {"text": "done"}
        assert (await turns.get_turn(key))["generation"] == 2
        assert (await turns.get_session(_SESSION))["active_turn_id"] is None
        [published] = (await runtime.history.turn_session(_SESSION, principal=principal))["timeline"]["items"]
        [operation] = published["process"]
        assert published["turn_id"] == turn_id and published["conversation_committed"]
        assert operation["key"] == "tool:stable-call" and operation["status"] == "SUCCEEDED"
        assert await tools.list_by_execution(turn_id, tenant_id=tenant) == (receipt,)
        assert await state.execution.executions.get(turn_id, tenant_id=tenant) is None
        if traced:
            # Crashing before the asynchronous Trace offer does not roll back Step.
            assert not (await runtime.history.turn_trace(_SESSION, turn_id, principal=principal)).items
    assert effect_log.read_text().splitlines() == [turn_id]
    assert provider_log.read_text().splitlines() == ["initial", "continuation"]
