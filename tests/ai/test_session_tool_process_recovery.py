#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native SQLite turn recovery across actual process loss after a tool effect."""

import asyncio
import multiprocessing
import os
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from linktools.ai.capability import AgentContext, ExecutionCapability, StepPersistenceCapability
from linktools.ai.core import ExecutionStatus, TenantAuthorizationPolicy, ToolOperationStatus
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.turns import SessionTurnKey
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._step_store import RuntimeStepStore
from linktools.ai.runtime._tool import RuntimeToolOperationBridge

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from ._session_tool_test_helpers import _assert_tool_turn_recovers_without_replaying_effect


_NAME = "native-session-tool-crash"
_SESSION = "session"
_REQUEST = "turn-1"


class _HostAuthorization(TenantAuthorizationPolicy):
    """Supply explicit authorization for the live and recovered SQL hosts."""


def _append(path: Path, value: str) -> None:
    with path.open("a", encoding="utf-8") as log:
        log.write(value + "\n")
        log.flush()
        os.fsync(log.fileno())


def _features(root: Path) -> RuntimeFeatures:
    async def effect(context: AgentContext[None]) -> str:
        _append(root / "effects.log", context.execution_id)
        return "confirmed effect"

    async def model(messages, _info):
        results = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        if results:
            assert [(part.tool_call_id, part.content) for part in results] == [("stable-call", "confirmed effect")]
            _append(root / "provider.log", "continuation")
            return ModelResponse(parts=[TextPart("done")])
        _append(root / "provider.log", "initial")
        return ModelResponse(parts=[ToolCallPart("effect", {}, tool_call_id="stable-call")])

    features = RuntimeFeatures(_NAME)
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.capability(StepPersistenceCapability(), scope="runtime")
    features.tool(effect, effect_policy="non_replay_safe")
    features.agent("default", model="default", allow_tools=("effect",), allow_skills=(), allow_subagents=())
    return model_features(_Models(_UsageFunctionModel(model)), features)


async def _send(runtime: Runtime):
    return await runtime.sessions.start(
        _SESSION, "inspect", agent=runtime.agents.get(), principal=runtime.default_principal, idempotency_key=_REQUEST,
    )


def _crash_native_process(root_path: str, phase: str) -> None:
    root = Path(root_path)
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
        if (phase == "tool_checkpoint" and snapshot is not None and snapshot.messages
                and any(isinstance(part, ToolReturnPart) for part in snapshot.messages[-1].parts)):
            os._exit(91)

    RuntimeToolOperationBridge.complete = complete
    RuntimeStepStore.commit_step = commit_step

    async def run() -> None:
        async with (
            Persistence.open(_NAME, storage=PersistenceStorage.from_root(root / "runtime")) as state,
            Runtime.open(persistence=state, features=_features(root), authorization=_HostAuthorization()) as runtime,
        ):
            await runtime.agents.get().create_session(_SESSION)
            turn = await _send(runtime)
            await turn.wait_final(timeout_seconds=15)
        raise AssertionError("native SQLite crash boundary was not reached")

    asyncio.run(run())


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("tool_completed", "tool_checkpoint"))
async def test_native_sqlite_tool_turn_recovers_after_process_exit_without_replaying_effect(tmp_path: Path, phase: str):
    process = multiprocessing.get_context("spawn").Process(target=_crash_native_process, args=(str(tmp_path), phase))
    process.start()
    try:
        await asyncio.to_thread(process.join, 25)
        assert process.exitcode == 91, (phase, process.exitcode)
    finally:
        if process.is_alive():
            process.kill()
            await asyncio.to_thread(process.join, 5)
        process.close()

    effect_log = tmp_path / "effects.log"
    provider_log = tmp_path / "provider.log"
    [turn_id] = effect_log.read_text().splitlines()
    assert provider_log.read_text().splitlines() == ["initial"]
    key = SessionTurnKey(_SESSION, turn_id)
    async with (
        Persistence.open(_NAME, storage=PersistenceStorage.from_root(tmp_path / "runtime")) as state,
        Runtime.open(persistence=state, features=_features(tmp_path), authorization=_HostAuthorization()) as runtime,
    ):
        turns = state.session_turns
        tools = state.checkpoint.tools
        tenant, principal = runtime.tenant_id, runtime.default_principal
        before = await turns.get_turn(key)
        assert before["state"] == "accepted" and before["generation"] == 1
        [receipt] = await tools.list_by_execution(turn_id, tenant_id=tenant)
        assert receipt.tool_call_id == "stable-call" and receipt.status is ToolOperationStatus.COMPLETED
        repeated = await _send(runtime)
        assert repeated.key == key
        assert await turns.get_turn(key) == before
        assert provider_log.read_text().splitlines() == ["initial"]
        assert not await runtime.sessions.recover_turn(_SESSION, turn_id, principal=principal)
        # Simulate the DB lease reaching its expiry without sleeping the worker
        # lease duration. Recovery must acquire a new generation itself.
        async with turns._context.engine.begin() as connection:
            expired = await connection.execute(turns._turns.update().filter_by(
                **turns._turn_values(key), generation=1, state="accepted",
            ).values(lease_until=0))
            assert expired.rowcount == 1
        assert await runtime.sessions.recover_turn(_SESSION, turn_id, principal=principal)
        result = await repeated.wait_final(timeout_seconds=10)
        assert result.key == key and result.status is ExecutionStatus.SUCCEEDED
        assert result.view["output"] == {"text": "done"}
        assert (await turns.get_turn(key))["generation"] == 2
        assert (await turns.get_session(_SESSION))["active_turn_id"] is None
        assert await tools.list_by_execution(turn_id, tenant_id=tenant) == (receipt,)
        timeline = (await runtime.history.turn_session(_SESSION, principal=principal))["timeline"]
        [published] = timeline["items"]
        assert published["turn_id"] == turn_id and published["conversation_committed"]
        assert published["user_input"]["prompt"] == "inspect"
        [operation] = published["process"]
        assert operation["key"] == "tool:stable-call" and operation["status"] == "SUCCEEDED"
        assert await state.execution.executions.get(turn_id, tenant_id=tenant) is None
        assert (await _send(runtime)).key == key
    assert effect_log.read_text().splitlines() == [turn_id]
    assert provider_log.read_text().splitlines() == ["initial", "continuation"]


@pytest.mark.asyncio
async def test_session_tool_turn_recovers_after_projected_tool_checkpoint(tmp_path: Path) -> None:
    await _assert_tool_turn_recovers_without_replaying_effect(tmp_path, "sqlite", "projected_tool_checkpoint")
