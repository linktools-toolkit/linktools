#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retained conversation publication recovery boundaries."""

from dataclasses import replace

import pytest
from tests.ai.test_conversation_output_retention import _NAMESPACE, _features, _models
from linktools.ai.runtime import Runtime, RuntimeHistory
from linktools.ai.runtime.state import RuntimeStorage, RuntimeStoragePlan
from linktools.ai.storage import StorageRoute
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state._contracts import (
    RecoveryCheckpointState,
    RecoveryHandoffPhase,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_turn", [False, True])
async def test_unknown_release_must_not_report_success_without_envelope(
    tmp_path, monkeypatch, prior_turn
):
    storage = RuntimeStorage.from_plan(
        RuntimeStoragePlan(
            execution=StorageRoute.sqlite(tmp_path / "e.db"),
            conversation=StorageRoute.sqlite(tmp_path / "c.db"),
        )
    )
    try:
        async with Runtime.open(
            _NAMESPACE, storage=storage, features=_features(), models=_models()
        ) as runtime:
            session = await runtime.agents.get().create_session("s")
            if prior_turn:
                await session.run("prior committed turn", timeout_seconds=5)
            seq = 2 if prior_turn else 1
            sessions = storage.conversation.sessions
            original = sessions.state_store.mutate
            calls = 0

            async def lose_release(callback, *args, **kwargs):
                nonlocal calls
                if callback.__name__ == "release_conversation":
                    calls += 1
                    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
                return await original(callback, *args, **kwargs)

            monkeypatch.setattr(sessions.state_store, "mutate", lose_release)
            execution = await session.start("prompt")
            with pytest.raises(AIError) as raised:
                await execution.wait(timeout_seconds=5)
            assert raised.value.code in {
                ErrorCode.STORAGE_COMMIT_UNKNOWN,
                ErrorCode.STORAGE_RECOVERY_REQUIRED,
            }
            tenant = runtime.default_principal.tenant_id
            principal = runtime.default_principal
            execution_id = execution.execution_id
            commits = await sessions.list_timeline_commits(
                "s", tenant_id=tenant, start_turn_seq=seq, end_turn_seq=seq + 1
            )
            assert commits == ()
            raw = await storage.execution.executions.get(execution_id, tenant_id=tenant)
            header = await sessions.get("s", tenant_id=tenant)
            recovery = await storage.recovery.checkpoints.get(
                execution_id, tenant_id=tenant
            )
            assert header.active_execution_id == execution_id
            assert not raw.retention_closed
            assert recovery.state is RecoveryCheckpointState.HANDOFF
            assert recovery.handoff_phase is RecoveryHandoffPhase.PREPARED
            with pytest.raises(AIError) as result_error:
                await runtime.executions.result(execution_id, principal=principal)
            assert result_error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
            with pytest.raises(AIError) as size_error:
                await runtime._execution_service.result_payload_size(
                    execution_id, principal=principal
                )
            assert size_error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
            async with RuntimeHistory.open(
                _NAMESPACE,
                storage=RuntimeStorage.from_plan(storage.plan),
            ) as history:
                with pytest.raises(AIError) as history_error:
                    await history.result(execution_id, principal=principal)
                assert history_error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
            monkeypatch.setattr(sessions.state_store, "mutate", original)
    except AIError as close_error:
        assert close_error.code in {
            ErrorCode.STORAGE_COMMIT_UNKNOWN,
            ErrorCode.STORAGE_RECOVERY_REQUIRED,
        }
    storage = RuntimeStorage.from_plan(
        RuntimeStoragePlan(
            execution=StorageRoute.sqlite(tmp_path / "e.db"),
            conversation=StorageRoute.sqlite(tmp_path / "c.db"),
        )
    )
    async with Runtime.open(
        _NAMESPACE, storage=storage, features=_features(), models=_models()
    ) as runtime:
        recovered_commits = await storage.conversation.sessions.list_timeline_commits(
            "s", tenant_id=principal.tenant_id, start_turn_seq=seq, end_turn_seq=seq + 1
        )
        result = (
            await runtime.executions.wait(
                execution_id, principal=principal, timeout_seconds=5
            )
        ).result
        page = await runtime.sessions.timeline("s", principal=principal)
        assert len(page.items) == seq and page.items[-1].items
        assert page.items[-1].output == result.output
        assert recovered_commits[0].start_message_index == (2 if prior_turn else 0)
        assert recovered_commits[0].end_message_index == (4 if prior_turn else 2)
        assert len(page.items[-1].items) == 1


@pytest.mark.asyncio
async def test_rejected_admitted_start_retains_input(tmp_path, monkeypatch):
    storage = RuntimeStorage.from_plan(
        RuntimeStoragePlan(
            execution=StorageRoute.sqlite(tmp_path / "e.db"),
            conversation=StorageRoute.sqlite(tmp_path / "c.db"),
        )
    )
    async with Runtime.open(
        _NAMESPACE, storage=storage, features=_features(), models=_models()
    ) as runtime:
        session = await runtime.agents.get().create_session("s")

        async def reject(*args, **kwargs):
            raise AIError(ErrorCode.SESSION_CONFLICT)

        original_claim = storage.execution.executions.claim_start
        monkeypatch.setattr(storage.execution.executions, "claim_start", reject)
        with pytest.raises(AIError):
            await session.start("original rejected input")
        sessions = storage.conversation.sessions
        head = await sessions.timeline_head(
            "s", tenant_id=runtime.default_principal.tenant_id
        )
        commits = await sessions.list_timeline_commits(
            "s",
            tenant_id=runtime.default_principal.tenant_id,
            start_turn_seq=1,
            end_turn_seq=head + 1,
        )
        assert head == 1 and len(commits) == 1
        assert (
            commits[0].stored_user_input.payload.decode() == "original rejected input"
        )
        assert commits[0].output is None
        assert commits[0].start_message_index == commits[0].end_message_index == 0
        monkeypatch.setattr(storage.execution.executions, "claim_start", original_claim)
        await session.run("next real input", timeout_seconds=5)
        page = await session.timeline()
        assert len(page.items) == 2 and page.items[0].items == ()
        history = await session.history()
        assert "original rejected input" not in repr(history.items)
        assert "next real input" in repr(history.items)


@pytest.mark.asyncio
async def test_corrupt_retained_receipt_remains_integrity_error(tmp_path):
    storage = RuntimeStorage.sqlite(tmp_path / "state.db")
    async with Runtime.open(
        _NAMESPACE,
        storage=storage,
        features=_features(),
        models=_models(),
    ) as runtime:
        session = await runtime.agents.get().create_session("s")
        execution = await session.start("original input")
        await execution.wait(timeout_seconds=5)
        principal = runtime.default_principal
        sessions = storage.conversation.sessions

        async def corrupt(transaction):
            record = await transaction.get_record(sessions._timeline_commit_key("s", 1))
            assert record is not None
            malformed = replace(
                record,
                data=dict(record.data, status="invalid"),
                storage_revision=record.storage_revision + 1,
            )
            assert await transaction.replace_record(
                malformed,
                expected_storage_revision=record.storage_revision,
            )

        await sessions.state_store.mutate(corrupt)
        with pytest.raises(AIError) as live_error:
            await runtime.executions.result(execution.execution_id, principal=principal)
        assert live_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        async with RuntimeHistory.open(
            _NAMESPACE,
            storage=RuntimeStorage.sqlite(tmp_path / "state.db"),
        ) as history:
            with pytest.raises(AIError) as history_error:
                await history.result(execution.execution_id, principal=principal)
            assert history_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_completed_receipt_old_handoff_after_later_turn(tmp_path, monkeypatch):
    from linktools.ai.runtime._local import LocalExecutionBackend

    plan = RuntimeStoragePlan(
        execution=StorageRoute.sqlite(tmp_path / "e.db"),
        conversation=StorageRoute.sqlite(tmp_path / "c.db"),
    )
    storage = RuntimeStorage.from_plan(plan)
    old_complete = LocalExecutionBackend._complete_handoff
    first_id = None

    async def fail_first(self, checkpoint):
        nonlocal first_id
        if first_id is None:
            first_id = checkpoint.execution_id
        if checkpoint.execution_id == first_id:
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        return await old_complete(self, checkpoint)

    async with Runtime.open(
        _NAMESPACE, storage=storage, features=_features(), models=_models()
    ) as runtime:
        session = await runtime.agents.get().create_session("s")
        monkeypatch.setattr(LocalExecutionBackend, "_complete_handoff", fail_first)
        await session.run("first", timeout_seconds=5)
        await session.run("second", timeout_seconds=5)
        principal = runtime.default_principal
        expected_page = await session.timeline()
        expected_session = await storage.conversation.sessions.get("s", tenant_id=principal.tenant_id)
    monkeypatch.setattr(LocalExecutionBackend, "_complete_handoff", old_complete)
    reopened = RuntimeStorage.from_plan(plan)
    async with Runtime.open(
        _NAMESPACE,
        storage=reopened,
        features=_features(),
        models=_models(),
    ) as runtime:
        page = await runtime.sessions.timeline("s", principal=principal)
        assert len(page.items) == 2
        assert page.items == expected_page.items
        current = await reopened.conversation.sessions.get("s", tenant_id=principal.tenant_id)
        assert current.continuation == expected_session.continuation
        assert current.active_execution_id is None
        checkpoint = await reopened.recovery.checkpoints.get(first_id, tenant_id=principal.tenant_id)
        assert checkpoint.state is RecoveryCheckpointState.COMPLETED
        commits = await reopened.conversation.sessions.list_timeline_commits(
            "s", tenant_id=principal.tenant_id, start_turn_seq=1, end_turn_seq=3,
        )
        assert [(item.start_message_index, item.end_message_index) for item in commits] == [(0, 2), (2, 4)]
