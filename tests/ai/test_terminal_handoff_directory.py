#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sealed native Session publication needs durable facts, not a live model route."""

import asyncio
import json
import os
import shutil
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import select

from linktools.ai.capability import SessionCapability
from linktools.ai.core import Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistencePlan, PersistenceStorage
from linktools.ai.persistence.turns import SessionTurnKey
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._canonical_split_step import _CanonicalSplitStep
from linktools.ai.storage import StorageRoute

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_session_tool_process_recovery import _append, _run_until_crash

_NAMESPACE = "native-terminal-directory"
_OWNER = Principal("operator", "default")
_OTHER = Principal("other", "default")
_SESSION = "session"


class _HostAuthorization(TenantAuthorizationPolicy):
    """Exercise actual owner authorization on native SQL Session recovery."""


def _storage(root: Path) -> PersistenceStorage:
    source = StorageRoute.sqlite(root / "step.db")
    return PersistenceStorage(PersistencePlan(
        conversation=StorageRoute.sqlite(root / "conversation.db"), execution=source, checkpoint=source,
    ))


def _features(model=None) -> RuntimeFeatures:
    features = RuntimeFeatures(_NAMESPACE)
    features.capability(SessionCapability(checkpoints=True), scope="runtime")
    if model is None:
        return features
    features.agent("default", model="default", allow_tools=(), allow_subagents=())
    return model_features(_Models(_UsageFunctionModel(model)), features)


def _crash_sealed_owner(root_dir: str) -> None:
    root = Path(root_dir)

    async def model(_messages, info):
        _append(root / "model.log", "called")
        return ModelResponse(parts=[TextPart("owner answer")])

    async def stop_before_publish(_adapter, key, *, turn_no, state, source, **_kwargs):
        assert state == "succeeded" and source.phase == "sealed"
        assert source.turn_id == key.turn_id and turn_no == 1
        _append(root / "turn.log", key.turn_id)
        _append(root / "turn.log", source.pin.receipt_id)
        _append(root / "body-root.json", json.dumps(source.pin.transcript.to_json()))
        os._exit(91)

    _CanonicalSplitStep.publish_terminal = stop_before_publish

    async def run() -> None:
        async with Runtime.open(
            _NAMESPACE, storage=_storage(root), features=_features(model), authorization=_HostAuthorization(),
        ) as runtime:
            await runtime.sessions.create_turn_session(_SESSION, principal=_OWNER)
            turn = await runtime.sessions.start_turn(
                _SESSION, "question", agent="default", principal=_OWNER, idempotency_key="original",
            )
            await turn.wait_final(timeout_seconds=12)
        raise AssertionError("native crash boundary was not reached")

    asyncio.run(run())


def _crash_active_owner(root_dir: str) -> None:
    root = Path(root_dir)

    async def model(_messages, _info):
        row = await runtime._state.session_turns.get_session(_SESSION)
        turn_id = row["active_turn_id"].decode("utf-8")
        _append(root / "model.log", "called")
        _append(root / "turn.log", turn_id)
        os._exit(91)

    runtime = Runtime.open(
        _NAMESPACE, storage=_storage(root), features=_features(model), authorization=_HostAuthorization(),
    )

    async def run() -> None:
        async with runtime:
            await runtime.sessions.create_turn_session(_SESSION, principal=_OWNER)
            turn = await runtime.sessions.start_turn(
                _SESSION, "question", agent="default", principal=_OWNER, idempotency_key="original",
            )
            await turn.wait_final(timeout_seconds=12)
        raise AssertionError("active model crash boundary was not reached")

    asyncio.run(run())


@pytest.fixture(scope="module")
def terminal_handoff_copy(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("native-terminal-directory-owner")

    async def prepare(destination: Path):
        if not (root / "turn.log").exists():
            await _run_until_crash(_crash_sealed_owner, str(root))
        assert (root / "model.log").read_text().splitlines() == ["called"]
        assert all((root / filename).is_file() for filename in ("conversation.db", "step.db"))
        # Copy from the stopped process's unopened directory, including WALs.
        copied = Path(shutil.copytree(root, destination / "crash"))
        turn_id, receipt_id = (copied / "turn.log").read_text().splitlines()
        return copied, SessionTurnKey(_SESSION, turn_id), receipt_id

    return prepare


async def _expire_and_check_target(state, key):
    turns = state.session_turns
    target = await turns.get_turn(key)
    assert target["generation"] == 1 and target["state"] == "accepted" and target["split_source_receipt"] is None
    assert (await turns.get_session(key.session_id))["active_turn_id"] == key.turn_id.encode()
    async with turns._context.engine.begin() as connection:
        expired = await connection.execute(turns._turns.update().filter_by(
            **turns._turn_values(key), generation=1,
        ).values(lease_until=0))
        assert expired.rowcount == 1
    return await turns.get_turn(key)


@pytest.mark.asyncio
async def test_terminal_handoff_survives_removed_model_route(tmp_path, terminal_handoff_copy):
    copied, key, receipt_id = await terminal_handoff_copy(tmp_path)
    async with (
        Persistence.open(_NAMESPACE, storage=_storage(copied)) as state,
        Runtime.open(persistence=state, features=_features(), authorization=_HostAuthorization()) as runtime,
    ):
        before = await _expire_and_check_target(state, key)
        with pytest.raises(AIError) as denied:
            await runtime.sessions.recover_turn(key.session_id, key.turn_id, mode="resume", principal=_OTHER)
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert await state.session_turns.get_turn(key) == before
        assert await runtime.sessions.recover_turn(key.session_id, key.turn_id, mode="resume", principal=_OWNER)
        row = await state.session_turns.get_turn(key)
        assert row["state"] == "succeeded" and row["generation"] == 1
        assert row["split_source_receipt"] == receipt_id.encode()
        assert row["result_json"]["transcript"] == json.loads((copied / "body-root.json").read_text())
        assert not await runtime.sessions.recover_turn(key.session_id, key.turn_id, mode="resume", principal=_OWNER)
        history = await runtime.history.turn_session(key.session_id, principal=_OWNER)
        assert history["session"]["active_turn_id"] is None
        [item] = history["timeline"]["items"]
        assert item["turn_id"] == key.turn_id and item["output"] == {"text": "owner answer"}
        assert await runtime.sessions.fork_turn_session(
            key.session_id, "fork", principal=_OWNER,
            expected_revision=history["session"]["revision"], idempotency_key="fork",
        )
        fork = await runtime.history.turn_session("fork", principal=_OWNER)
        assert fork["timeline"]["items"] == history["timeline"]["items"]
        assert await state.execution.executions.get(key.turn_id, tenant_id="default") is None
    assert (copied / "model.log").read_text().splitlines() == ["called"]


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ("binding", "outcome"))
async def test_terminal_handoff_rejects_corruption_without_model_route(tmp_path, terminal_handoff_copy, damage):
    copied, key, receipt_id = await terminal_handoff_copy(tmp_path)
    async with (
        Persistence.open(_NAMESPACE, storage=_storage(copied)) as state,
        Runtime.open(persistence=state, features=_features(), authorization=_HostAuthorization()) as runtime,
    ):
        await _expire_and_check_target(state, key)
        turns = state.session_turns
        if damage == "binding":
            async with turns._context.engine.begin() as connection:
                changed = await connection.execute(turns._turns.update().filter_by(
                    **turns._turn_values(key), generation=1,
                ).values(binding_key=b"\xa5" * 32))
                assert changed.rowcount == 1
        else:
            store = state.execution.executions.state_store
            table = store.storage_group.metadata.tables["ai_step_runs"]
            columns = table.c
            criteria = (columns.run_id == key.turn_id, columns.conversation_id == key.session_id,
                        columns.turn_source_generation == 1, columns.turn_source_receipt == receipt_id)
            async with store.context.engine.begin() as connection:
                pin = (await connection.execute(select(columns.turn_source_pin).where(*criteria))).scalar_one()
                changed = await connection.execute(table.update().where(*criteria).values(
                    turn_source_pin={**pin, "output_ref": [999, 0]},
                ))
                assert changed.rowcount == 1
        before = await turns.get_turn(key)
        with pytest.raises(AIError) as corrupt:
            await runtime.sessions.recover_turn(key.session_id, key.turn_id, mode="resume", principal=_OWNER)
        assert corrupt.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await turns.get_turn(key) == before
        assert (await turns.get_session(key.session_id))["active_turn_id"] == key.turn_id.encode()
    assert (copied / "model.log").read_text().splitlines() == ["called"]


@pytest.mark.asyncio
async def test_active_recovery_still_requires_model_route(tmp_path):
    await _run_until_crash(_crash_active_owner, str(tmp_path))
    [turn_id] = (tmp_path / "turn.log").read_text().splitlines()
    key = SessionTurnKey(_SESSION, turn_id)
    async with (
        Persistence.open(_NAMESPACE, storage=_storage(tmp_path)) as state,
        Runtime.open(persistence=state, features=_features(), authorization=_HostAuthorization()) as runtime,
    ):
        before = await _expire_and_check_target(state, key)
        with pytest.raises(AIError) as unavailable:
            await runtime.sessions.recover_turn(key.session_id, key.turn_id, mode="resume", principal=_OWNER)
        assert unavailable.value.code is ErrorCode.AGENT_BINDING_UNAVAILABLE
        assert await state.session_turns.get_turn(key) == before
    assert (tmp_path / "model.log").read_text().splitlines() == ["called"]
