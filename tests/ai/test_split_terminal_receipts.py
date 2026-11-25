#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split handoff completion uses confirmed execution receipts."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai_harness.step_persistence import RunRecord, StepEvent

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain
from linktools.ai.persistence._session_turn_sql import SessionTurnKey
from linktools.ai.persistence._split_step_source import SqlSplitStepSource
from linktools.ai.persistence.contracts import ExecutionRecoveryState, RecoveryHandoffPhase
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._canonical_split_step import _CanonicalSplitStep
from linktools.ai.runtime._local import LocalExecutionBackend

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from ._session_tool_test_helpers import _split_sqlite_storage
from .test_session_turn_sql import accept


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", (False, True), ids=("unconfirmed", "lost-ack"))
@pytest.mark.parametrize("drop_notification", (False, True), ids=("notify", "no-notify"))
async def test_split_handoff_unknown_requires_source_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, committed: bool, drop_notification: bool,
) -> None:
    calls: list[str] = []
    publications: list[str] = []
    completion_writes: list[str] = []
    database = tmp_path / "terminal.db"

    async def model(messages, info):
        calls.append("model")
        return ModelResponse(parts=[TextPart("durable answer")])

    features = RuntimeFeatures("application")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    models = _Models(_UsageFunctionModel(model))
    original_publish = LocalExecutionBackend._publish_terminal_event

    def publish(self, execution_id, **kwargs):
        publications.append(execution_id)
        if not drop_notification:
            return original_publish(self, execution_id, **kwargs)

    async with Persistence.open("split-receipt", storage=_split_sqlite_storage(database)) as persistence:
        async with Runtime.open(persistence=persistence, features=model_features(models, features)) as runtime:
            repository = runtime._state.execution.executions
            original_swap = repository.compare_and_swap

            async def swap(execution_id, **kwargs):
                following = kwargs["next_record"]
                if (following.recovery is not None
                        and following.recovery.handoff_phase is RecoveryHandoffPhase.COMPLETED):
                    completion_writes.append(execution_id)
                    if committed:
                        await original_swap(execution_id, **kwargs)
                    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
                return await original_swap(execution_id, **kwargs)

            with monkeypatch.context() as fault:
                fault.setattr(repository, "compare_and_swap", swap)
                fault.setattr(LocalExecutionBackend, "_publish_terminal_event", publish)
                session = await runtime.agents.get().create_session("session")
                execution = await session.start("question", idempotency_key="question")
                assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
                await runtime.close()
                stored = await repository.get(execution.execution_id, tenant_id=runtime.tenant_id)
                assert stored is not None
                assert stored.recovery_state is (
                    ExecutionRecoveryState.COMPLETED if committed else ExecutionRecoveryState.HANDOFF
                )
                assert publications == [execution.execution_id]
                if not committed:
                    assert not stored.retention_closed
                assert completion_writes == [execution.execution_id]
                assert calls == ["model"]

    async with Persistence.open("split-receipt", storage=_split_sqlite_storage(database)) as persistence:
        async with Runtime.open(persistence=persistence, features=model_features(models, features), auto_recover=not committed) as runtime:
            stored = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
            assert stored is not None and stored.recovery_state is ExecutionRecoveryState.COMPLETED
            session = runtime.agents.get().session("session")
            replay = await session.start("question", idempotency_key="question")
            assert replay.execution_id == execution.execution_id
            assert (await replay.wait(timeout_seconds=10)).result.output == {"text": "durable answer"}
            retained = await runtime._state.conversation.sessions.get("session", tenant_id=runtime.tenant_id)
            assert retained is not None and retained.active_execution_id is None
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
            prepared = await native.prepare_step(run=run, events=(start,))
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

                async def claim(source, db, context, **kwargs):
                    if ordering == "lost-ack-unrelated":
                        kwargs["pin"] = replace(kwargs["pin"], receipt_id="unrelated")
                    return await original_claim(source, db, context, **kwargs)

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
            if ordering == "stop-first":
                with pytest.raises(AIError) as conflict:
                    await owner.commit_step(prepared, key=key, generation=1)
                assert conflict.value.code is ErrorCode.STORAGE_CONFLICT
                assert await owner._read_pin(owner._identity(key, 1)) == source
        finally:
            await native.close()
