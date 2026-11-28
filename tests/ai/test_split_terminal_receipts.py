#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split handoff completion uses confirmed execution receipts."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord, StepEvent

from linktools.ai.capability import StepPersistenceCapability
from linktools.ai.core import Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistencePlan, PersistenceStorage
from linktools.ai.persistence._session_turn_sql import SessionTurnKey, _SplitTargetObservation
from linktools.ai.persistence._split_step_source import SqlSplitStepSource
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._canonical_split_step import _CanonicalSplitStep
from linktools.ai.runtime._regenerable_turn import _RegenerableTurnRunner
from linktools.ai.storage import StorageRoute

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from ._session_tool_test_helpers import _split_sqlite_storage
from .test_session_turn_sql import accept


class _HostAuthorization(TenantAuthorizationPolicy):
    """Use the normal native SQL session ownership checks."""


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("unconfirmed", "wrong-receipt", "lost-ack"))
@pytest.mark.parametrize("drop_notification", (False, True), ids=("notify", "no-notify"))
async def test_split_handoff_unknown_requires_source_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str, drop_notification: bool,
) -> None:
    principal = Principal("operator", "default")
    calls: list[str] = []
    terminal_attempts: list[str] = []
    source_terminal = []
    finished = asyncio.Event()
    source = StorageRoute.sqlite(tmp_path / "step.db")
    storage = PersistenceStorage(PersistencePlan(
        conversation=StorageRoute.sqlite(tmp_path / "conversation.db"), execution=source, checkpoint=source,
    ))

    async def model(_messages, _info):
        calls.append("model")
        return ModelResponse(parts=[TextPart("durable answer")])

    features = RuntimeFeatures("application")
    features.capability(StepPersistenceCapability(), scope="runtime")
    features.agent("default", model="default", allow_tools=(), allow_subagents=())
    original_publish = _CanonicalSplitStep.publish_terminal
    original_terminal = _RegenerableTurnRunner._terminal

    async def publish(self, key, **kwargs):
        try:
            return await original_publish(self, key, **kwargs)
        finally:
            source_terminal.append((self, kwargs["source"]))
            finished.set()

    def terminal(self, turn, state, **kwargs):
        terminal_attempts.append(state)
        if not drop_notification:
            return original_terminal(self, turn, state, **kwargs)

    async with (
        Persistence.open("split-receipt", storage=storage) as persistence,
        Runtime.open(persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features),
                     authorization=_HostAuthorization()) as runtime,
    ):
        turns = persistence.session_turns
        original_commit = turns.publish_split_terminal
        original_observe = turns.observe_split_target

        async def commit(key, **kwargs):
            assert kwargs["source"].phase == "sealed" and kwargs["state"] == "succeeded"
            if outcome == "lost-ack":
                assert await original_commit(key, **kwargs)
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        async def observe(key):
            observed = await original_observe(key)
            if outcome == "wrong-receipt":
                assert observed.state == "accepted" and observed.receipt_id is None
                # Even an apparent terminal outcome isn't proof for this source.
                return _SplitTargetObservation(observed.generation, "succeeded", "unrelated")
            return observed

        with monkeypatch.context() as fault:
            fault.setattr(_CanonicalSplitStep, "publish_terminal", publish)
            fault.setattr(_RegenerableTurnRunner, "_terminal", terminal)
            fault.setattr(turns, "publish_split_terminal", commit)
            fault.setattr(turns, "observe_split_target", observe)
            await runtime.sessions.create_turn_session("session", principal=principal)
            turn = await runtime.sessions.start(
                "session", "question", agent="default", principal=principal, idempotency_key="question",
            )
            await asyncio.wait_for(finished.wait(), timeout=10)
            key = turn.key
            [(adapter, sealed)] = source_terminal
            assert terminal_attempts == (["succeeded"] if outcome == "lost-ack" else [])
            pin = await adapter._read_pin(adapter._identity(key, 1))
            assert pin.pin == sealed.pin
            assert pin.phase == ("published" if outcome == "lost-ack" else "sealed")
            row = await turns.get_turn(key)
            assert row["state"] == ("succeeded" if outcome == "lost-ack" else "accepted")
            assert row["split_source_receipt"] == (
                sealed.pin.receipt_id.encode() if outcome == "lost-ack" else None
            )
            if outcome != "lost-ack":
                assert (await turns.get_session(key.session_id))["active_turn_id"] == key.turn_id.encode()

    step_only = RuntimeFeatures("application")
    step_only.capability(StepPersistenceCapability(), scope="runtime")
    async with (
        Persistence.open("split-receipt", storage=storage) as persistence,
        Runtime.open(persistence=persistence, features=step_only, authorization=_HostAuthorization()) as runtime,
    ):
        turns = persistence.session_turns
        if outcome != "lost-ack":
            async with turns._context.engine.begin() as connection:
                expired = await connection.execute(turns._turns.update().filter_by(
                    **turns._turn_values(key), generation=1,
                ).values(lease_until=0))
                assert expired.rowcount == 1
            assert await runtime.sessions.recover_turn(key.session_id, key.turn_id, principal=principal)
        assert not await runtime.sessions.recover_turn(key.session_id, key.turn_id, principal=principal)
        row = await turns.get_turn(key)
        assert (row["generation"], row["state"]) == (1, "succeeded")
        assert row["split_source_receipt"] == sealed.pin.receipt_id.encode()
        assert row["result_json"]["transcript"] == sealed.pin.transcript.to_json()
        history = await runtime.history.turn_session(key.session_id, principal=principal)
        assert history["session"]["active_turn_id"] is None
        [item] = history["timeline"]["items"]
        assert item["turn_id"] == key.turn_id and item["output"] == {"text": "durable answer"}
        assert calls == ["model"]


@pytest.mark.asyncio
@pytest.mark.parametrize("ordering", ("stop-first", "registration-first", "lost-ack-exact", "lost-ack-unrelated"))
async def test_split_cold_stop_requires_exact_source_owner(tmp_path: Path, monkeypatch, ordering: str) -> None:
    key = SessionTurnKey("chat", "turn")
    cancel_result = {"error_code": ErrorCode.EXECUTION_CANCELLED.value}
    async with Persistence.open("split-stop", storage=_split_sqlite_storage(tmp_path / "state.db")) as state:
        turns = state.session_turns
        assert turns is not None
        source_store = state._stores[PersistenceDomain.EXECUTION]
        assert source_store.storage_group is not state._stores[PersistenceDomain.CONVERSATION].storage_group
        native = await state.create_step_store()
        try:
            owner = _CanonicalSplitStep(turns, source_store, native)
            await turns.create(key.session_id, config={})
            assert await accept(turns, key.turn_id)
            run = RunRecord(run_id=key.turn_id, conversation_id=key.session_id, registration_id="old")
            start = StepEvent(
                run_id=key.turn_id, conversation_id=key.session_id, kind="run_started",
                step_index=0, idempotency_key="initial",
            )
            request = StepEvent(
                run_id=key.turn_id, conversation_id=key.session_id, kind="model_request_started",
                step_index=1, idempotency_key="request",
            )
            checkpoint = ContinuableSnapshot(
                run_id=key.turn_id, conversation_id=key.session_id, step_index=1, state="complete",
                idempotency_key="checkpoint", messages=[ModelRequest(parts=[UserPromptPart("body")])],
            )
            prepared = await native.prepare_step(run=run, events=(start, request), snapshot=checkpoint)
            assert await turns.request_cancel(key)
            table = turns._turns
            async with turns._context.engine.begin() as connection:
                await connection.execute(table.update().where(
                    table.c.turn_id == key.turn_id.encode(),
                ).values(lease_until=0))

            if ordering == "registration-first":
                original_write = owner._write
                registered_first = False

                async def register_between_guard_and_insert(callback):
                    nonlocal registered_first
                    try:
                        return await original_write(callback)
                    except AIError as error:
                        if not registered_first and error.code is ErrorCode.STORAGE_CONFLICT:
                            await owner.commit_step(prepared, key=key, generation=1)
                            registered_first = True
                        raise

                monkeypatch.setattr(owner, "_write", register_between_guard_and_insert)

            if ordering.startswith("lost-ack"):
                original_write = owner._write
                original_claim = SqlSplitStepSource.cancel_unregistered
                claim_committed = False

                async def claim(source, db, **kwargs):
                    if ordering == "lost-ack-unrelated":
                        kwargs["pin"] = replace(kwargs["pin"], receipt_id="unrelated")
                    return await original_claim(source, db, **kwargs)

                async def write(callback):
                    nonlocal claim_committed
                    value = await original_write(callback)
                    if not claim_committed:
                        claim_committed = True
                        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
                    return value

                monkeypatch.setattr(SqlSplitStepSource, "cancel_unregistered", claim)
                monkeypatch.setattr(owner, "_write", write)

            async def recover():
                return await owner.recover_open_turn(
                    key, generation=1, turn_no=1, lease_seconds=60, cancelling=True, cancel_result=cancel_result,
                )

            if ordering in ("registration-first", "lost-ack-unrelated"):
                with pytest.raises(AIError) as failure:
                    await recover()
                assert failure.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
                assert failure.value.safe_details == {
                    "recovery_stage": "split_step", "reason": "split_unregistered_stop_unconfirmed",
                }
                assert (await turns.get_turn(key))["state"] == "cancelling"
                if ordering == "lost-ack-unrelated":
                    return
                assert registered_first

            assert await recover() == ("settled", None, None)
            target = await turns.get_turn(key)
            source = await owner._read_pin(owner._identity(key, 1))
            assert (target["state"], target["generation"]) == ("cancelled", 1)
            assert source is not None and source.phase == "published" and source.pin.result == cancel_result
            assert target["split_source_receipt"] == source.pin.receipt_id.encode()
            assert (await turns.get_session(key.session_id))["active_turn_id"] is None
            registered = await native.get_run(run_id=key.turn_id)
            assert registered is not None
            assert registered.registration_id == ("old" if ordering == "registration-first" else None)
            events = await native.list_events(run_id=key.turn_id)
            checkpoints = await native.list_snapshots(run_id=key.turn_id, include_interrupted=True)
            if ordering == "registration-first":
                assert [event.kind for event in events] == ["run_started", "model_request_started"]
                assert [snapshot.idempotency_key for snapshot in checkpoints] == ["checkpoint"]
            else:
                assert events == [] and checkpoints == []
            if ordering == "stop-first":
                with pytest.raises(AIError) as conflict:
                    await owner.commit_step(prepared, key=key, generation=1)
                assert conflict.value.code is ErrorCode.STORAGE_CONFLICT
                assert await owner._read_pin(owner._identity(key, 1)) == source
                assert await native.list_events(run_id=key.turn_id) == []
                assert await native.list_snapshots(run_id=key.turn_id, include_interrupted=True) == []
        finally:
            await native.close()
