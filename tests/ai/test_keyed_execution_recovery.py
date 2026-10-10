#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native keyed recovery preserves one producer through uncertain outcomes."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.models.function import FunctionModel

from linktools.ai.core import (
    ExecutionStatus, OperationKind, OperationLedgerInput, OperationStatus,
    ResourceKind, SessionStatus, idempotency_key_digest,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeStorage

from .test_live_history_readback_integration import _Models
from .test_runtime_recovery_ownership import _NAMESPACE, _RuntimeProcess, _capabilities


@asynccontextmanager
async def _recoverable(
    tmp_path: Path, *, phase: str = "active", model_error: bool = False,
    model_started: asyncio.Event | None = None,
) -> AsyncIterator[tuple[Runtime, Runtime, RuntimeStorage, str, list[int], asyncio.Event]]:
    database = tmp_path / "runtime.db"
    owner = _RuntimeProcess(database, phase=phase)
    execution_id = owner.initial["execution_id"]
    owner.crash()
    owner.close()
    release = asyncio.Event()
    calls = []

    async def model(messages, info):
        del messages, info
        calls.append(1)
        if model_started is not None:
            model_started.set()
        await release.wait()
        if model_error:
            raise RuntimeError("offline business failure")
        yield "recovered"

    state = RuntimeStorage.sqlite(database)
    other_state = RuntimeStorage.sqlite(database)
    try:
        async with Runtime.open(
            _NAMESPACE, models=_Models(FunctionModel(stream_function=model)),
            storage=state, capabilities=(_capabilities(),),
        ) as runtime, Runtime.open(
            _NAMESPACE, models=_Models(FunctionModel(stream_function=model)),
            storage=other_state, capabilities=(_capabilities(),),
        ) as other:
            try:
                yield runtime, other, state, execution_id, calls, release
            finally:
                release.set()
    finally:
        release.set()


async def _head(state: RuntimeStorage, execution_id: str):
    return await state.execution.executions.get_history_head(execution_id, tenant_id="default")


async def _receipt(state: RuntimeStorage, key: str):
    return await state.execution.operations.get(idempotency_key_digest(key), tenant_id="default")


async def _recover(runtime: Runtime, execution_id: str, key: str):
    return await runtime.executions.recover(
        execution_id, principal=runtime.default_principal, idempotency_key=key,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("active", "admitted"))
async def test_keyed_recovery_replays_across_runtimes_and_after_completion(tmp_path: Path, phase: str) -> None:
    async with _recoverable(tmp_path, phase=phase) as (runtime, other, state, execution_id, calls, release):
        original = await _head(state, execution_id)
        await _recover(runtime, execution_id, "recovery")
        claimed = await _head(state, execution_id)
        assert claimed.producer_generation > original.producer_generation
        receipt = await _receipt(state, "recovery")
        assert receipt.status is OperationStatus.SUCCEEDED
        assert receipt.result_ref == claimed.producer_claim_id
        await asyncio.gather(_recover(runtime, execution_id, "recovery"), _recover(other, execution_id, "recovery"))
        assert await _head(state, execution_id) == claimed
        release.set()
        result = await runtime.executions.wait(execution_id, principal=runtime.default_principal, timeout_seconds=10)
        assert result.result.status is ExecutionStatus.SUCCEEDED
        assert calls == [1]
        terminal = await _head(state, execution_id)
        await _recover(other, execution_id, "recovery")
        assert await _head(state, execution_id) == terminal
        assert calls == [1]


@pytest.mark.asyncio
async def test_running_receipt_rejects_replay_and_caller_cancel_does_not_abandon_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _recoverable(tmp_path) as (runtime, other, state, execution_id, calls, release):
        backend = runtime._execution_service.runtime_backend()
        launch = backend.launch
        entered = asyncio.Event()
        continue_launch = asyncio.Event()

        async def pause_launch(*args, **kwargs):
            entered.set()
            await continue_launch.wait()
            return await launch(*args, **kwargs)

        monkeypatch.setattr(backend, "launch", pause_launch)
        recovering = asyncio.create_task(_recover(runtime, execution_id, "recovery"))
        try:
            await asyncio.wait_for(entered.wait(), 10)
            claimed = await _head(state, execution_id)
            assert (await _receipt(state, "recovery")).status is OperationStatus.RUNNING
            with pytest.raises(AIError) as unknown:
                await _recover(other, execution_id, "recovery")
            assert unknown.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
            assert await _head(state, execution_id) == claimed
            assert calls == []
            recovering.cancel()
            await asyncio.sleep(0)
            assert not recovering.done()
            continue_launch.set()
            with pytest.raises(asyncio.CancelledError):
                await recovering
            assert (await _receipt(state, "recovery")).status is OperationStatus.SUCCEEDED
            await _recover(other, execution_id, "recovery")
            assert await _head(state, execution_id) == claimed
            release.set()
            result = await runtime.executions.wait(execution_id, principal=runtime.default_principal, timeout_seconds=10)
            assert result.result.status is ExecutionStatus.SUCCEEDED
            assert calls == [1]
        finally:
            continue_launch.set()
            await asyncio.gather(recovering, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("unknown", "launched_ack_lost", "rejected", "settled_ack_lost", "claims_changed"))
async def test_launch_and_settlement_outcomes_do_not_repeat_takeover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    async with _recoverable(tmp_path) as (runtime, other, state, execution_id, _calls, _release):
        backend = runtime._execution_service.runtime_backend()
        launch = backend.launch
        if outcome == "settled_ack_lost":
            cas = state.execution.operations.compare_and_swap

            async def lose_ack(*args, **kwargs):
                result = await cas(*args, **kwargs)
                if result.operation_kind is OperationKind.EXECUTION_RECOVER:
                    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
                return result

            monkeypatch.setattr(state.execution.operations, "compare_and_swap", lose_ack)
            await _recover(runtime, execution_id, "recovery")
            expected = OperationStatus.SUCCEEDED
        else:
            async def fail_launch(*args, **kwargs):
                if outcome == "launched_ack_lost":
                    await launch(*args, **kwargs)
                if outcome in {"unknown", "launched_ack_lost"}:
                    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
                raise AIError(ErrorCode.STORAGE_CONFLICT, safe_details={"phase": "recovery_launch_rejected"})

            if outcome == "claims_changed":
                checks = 0

                async def changing_claims(*args, **kwargs):
                    nonlocal checks
                    checks += 1
                    return (), int(checks > 1)

                def reject_background_retry(execution_id):
                    pytest.fail("Keyed recovery must not schedule an unkeyed retry")

                monkeypatch.setattr(backend, "_reconcile_tool_effects", changing_claims)
                monkeypatch.setattr(backend, "_defer_recovery_reconcile", reject_background_retry)
            else:
                monkeypatch.setattr(backend, "launch", fail_launch)
            with pytest.raises(AIError):
                await _recover(runtime, execution_id, "recovery")
            expected = OperationStatus.FAILED if outcome == "rejected" else OperationStatus.RUNNING
        claimed = await _head(state, execution_id)
        assert (await _receipt(state, "recovery")).status is expected
        if expected is OperationStatus.SUCCEEDED:
            await _recover(other, execution_id, "recovery")
        else:
            with pytest.raises(AIError) as replay:
                await _recover(other, execution_id, "recovery")
            assert replay.value.code is (
                ErrorCode.STORAGE_RECOVERY_REQUIRED if expected is OperationStatus.RUNNING else ErrorCode.STORAGE_CONFLICT
            )
        assert await _head(state, execution_id) == claimed


@pytest.mark.asyncio
async def test_keyed_recovery_observes_cancel_intent_after_many_unfinished_receipts(tmp_path: Path) -> None:
    async with _recoverable(tmp_path) as (runtime, other, state, execution_id, calls, _release):
        now = datetime.now(timezone.utc)
        candidate = OperationLedgerInput(
            "", "default", ResourceKind.EXECUTION, execution_id, execution_id,
            OperationKind.EXECUTION_RECOVER, OperationStatus.RUNNING,
            "a" * 64, "previous-claim", None, None, False, now, now,
        )
        for index in range(257):
            await state.execution.operations.append(replace(candidate, operation_id=f"old-recover-{index}"))
        await state.execution.operations.append(replace(
            candidate, operation_id="pending-cancel", operation_kind=OperationKind.EXECUTION_CANCEL,
            status=OperationStatus.PENDING, result_ref=None,
        ))
        original = await _head(state, execution_id)
        await _recover(runtime, execution_id, "recovery")
        receipt = await _receipt(state, "recovery")
        assert receipt.status is OperationStatus.CANCELLED
        assert receipt.result_ref is None
        current = await state.execution.executions.get(execution_id, tenant_id="default")
        assert current.status is ExecutionStatus.CANCELLED
        assert (await _head(state, execution_id)).producer_generation == original.producer_generation
        await _recover(other, execution_id, "recovery")
        assert calls == []


@pytest.mark.asyncio
async def test_runtime_close_drains_recovery_control_without_starting_a_new_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _recoverable(tmp_path) as (runtime, _other, state, execution_id, calls, _release):
        backend = runtime._execution_service.runtime_backend()
        launch = backend.launch
        entered = asyncio.Event()
        continue_launch = asyncio.Event()

        async def pause_launch(*args, **kwargs):
            entered.set()
            await continue_launch.wait()
            return await launch(*args, **kwargs)

        monkeypatch.setattr(backend, "launch", pause_launch)
        recovering = asyncio.create_task(_recover(runtime, execution_id, "recovery"))
        closing = None
        try:
            await asyncio.wait_for(entered.wait(), 10)
            closing = asyncio.create_task(backend.close())
            await asyncio.sleep(0)
            assert not closing.done()
            continue_launch.set()
            with pytest.raises(AIError) as declined:
                await recovering
            assert declined.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
            await closing
            assert (await _receipt(state, "recovery")).status is OperationStatus.FAILED
            assert calls == []
        finally:
            continue_launch.set()
            await asyncio.gather(recovering, *(() if closing is None else (closing,)), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("admitted", "active"))
@pytest.mark.parametrize("cancel_source", ("between_scans", "closing_session"))
async def test_keyed_recovery_retains_receipt_when_cancellation_prevents_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, cancel_source: str,
) -> None:
    async with _recoverable(tmp_path, phase=phase) as (runtime, other, state, execution_id, calls, _release):
        backend = runtime._execution_service.runtime_backend()
        if cancel_source == "between_scans":
            pending = backend._pending_cancel_operations
            scans = 0

            async def cancel_on_second_scan(*args, **kwargs):
                nonlocal scans
                scans += 1
                if scans == 2:
                    now = datetime.now(timezone.utc)
                    await state.execution.operations.append(OperationLedgerInput(
                        "racing-cancel", "default", ResourceKind.EXECUTION, execution_id, execution_id,
                        OperationKind.EXECUTION_CANCEL, OperationStatus.PENDING,
                        "b" * 64, None, None, None, False, now, now,
                    ))
                return await pending(*args, **kwargs)

            monkeypatch.setattr(backend, "_pending_cancel_operations", cancel_on_second_scan)
        else:
            session = await state.conversation.sessions.get("session", tenant_id="default")
            assert session is not None
            await state.conversation.sessions.compare_and_swap(
                session.session_id, tenant_id="default", expected_revision=session.revision,
                next_record=replace(session, revision=session.revision + 1, status=SessionStatus.CLOSING),
            )
        original = await _head(state, execution_id)
        await _recover(runtime, execution_id, "recovery")
        receipt = await _receipt(state, "recovery")
        assert receipt.status is OperationStatus.CANCELLED
        current = await state.execution.executions.get(execution_id, tenant_id="default")
        assert current.status is ExecutionStatus.CANCELLED
        final_head = await _head(state, execution_id)
        if phase == "admitted":
            assert receipt.result_ref is None
            assert final_head.producer_generation == original.producer_generation
        else:
            assert receipt.result_ref == final_head.producer_claim_id
        await _recover(other, execution_id, "recovery")
        assert await _head(state, execution_id) == final_head
        assert calls == []


@pytest.mark.asyncio
async def test_competing_cancel_recovery_replays_the_no_producer_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _recoverable(tmp_path, phase="admitted") as (runtime, other, state, execution_id, calls, _release):
        now = datetime.now(timezone.utc)
        await state.execution.operations.append(OperationLedgerInput(
            "pending-cancel", "default", ResourceKind.EXECUTION, execution_id, execution_id,
            OperationKind.EXECUTION_CANCEL, OperationStatus.PENDING,
            "b" * 64, None, None, None, False, now, now,
        ))
        coordinator = other._execution_service.runtime_backend()._recovery_coordinator
        recover = coordinator.recover_execution
        stale_entered = asyncio.Event()
        release_stale = asyncio.Event()

        async def paused_recover(*args, **kwargs):
            stale_entered.set()
            await release_stale.wait()
            return await recover(*args, **kwargs)

        monkeypatch.setattr(coordinator, "recover_execution", paused_recover)
        stale = asyncio.create_task(_recover(other, execution_id, "recovery"))
        try:
            await asyncio.wait_for(stale_entered.wait(), 10)
            winner = await _recover(runtime, execution_id, "recovery")
            receipt = await _receipt(state, "recovery")
            assert receipt.status is OperationStatus.CANCELLED and receipt.result_ref is None
            head = await _head(state, execution_id)
            release_stale.set()
            assert (await stale).execution_id == winner.execution_id
            assert await _head(state, execution_id) == head
            assert await _receipt(state, "recovery") == receipt
            assert calls == []
        finally:
            release_stale.set()
            await asyncio.gather(stale, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("admitted", "active"))
async def test_keyed_recovery_replays_session_conflict_terminal_outcome(tmp_path: Path, phase: str) -> None:
    async with _recoverable(tmp_path, phase=phase) as (runtime, other, state, execution_id, calls, _release):
        execution = await state.execution.executions.get(execution_id, tenant_id="default")
        session = await state.conversation.sessions.get("session", tenant_id="default")
        assert execution is not None and session is not None
        sibling = replace(execution, execution_id="other-session-owner", root_execution_id="other-session-owner")
        await state.execution.executions.create_with_history_head(sibling)
        await state.conversation.sessions.release_execution(
            session.session_id, tenant_id="default", execution_id=execution_id,
        )
        reassigned = await state.conversation.sessions.admit_execution(
            session.session_id, tenant_id="default", execution_id=sibling.execution_id,
            expected=session.continuation,
        )
        assert reassigned.active_execution_id == sibling.execution_id
        with pytest.raises(AIError) as failed_control:
            await _recover(runtime, execution_id, "recovery")
        assert failed_control.value.code is ErrorCode.SESSION_BUSY
        receipt = await _receipt(state, "recovery")
        assert receipt.status is OperationStatus.FAILED
        if phase == "admitted":
            assert receipt.result_ref is None
        else:
            assert receipt.result_ref == (await _head(state, execution_id)).producer_claim_id
        assert receipt.error_code == ErrorCode.SESSION_BUSY.value
        failed = await state.execution.executions.get(execution_id, tenant_id="default")
        assert failed.status is ExecutionStatus.FAILED and failed.error_code == ErrorCode.SESSION_BUSY.value
        with pytest.raises(AIError) as replay:
            await _recover(other, execution_id, "recovery")
        assert replay.value.code is ErrorCode.SESSION_BUSY
        assert await _receipt(state, "recovery") == receipt
        assert await state.execution.executions.get(sibling.execution_id, tenant_id="default") == sibling
        assert calls == []


@pytest.mark.asyncio
async def test_late_cancel_recovery_cannot_control_another_producer_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    async with _recoverable(
        tmp_path, phase="admitted", model_started=started,
    ) as (runtime, other, state, execution_id, calls, _release):
        winner_backend = runtime._execution_service.runtime_backend()
        contender_backend = other._execution_service.runtime_backend()
        launch = winner_backend.launch
        pending = contender_backend._pending_cancel_operations
        stale_loaded = asyncio.Event()
        scan_cancel = asyncio.Event()
        launch_confirmed = asyncio.Event()
        finish_launch = asyncio.Event()

        async def pause_cancel_scan(*args, **kwargs):
            stale_loaded.set()
            await scan_cancel.wait()
            return await pending(*args, **kwargs)

        async def pause_launch_return(*args, **kwargs):
            await launch(*args, **kwargs)
            await asyncio.wait_for(started.wait(), 10)
            launch_confirmed.set()
            await finish_launch.wait()

        async def reject_foreign_cancel(*args, **kwargs):
            pytest.fail("A losing recovery controller cannot write cancellation for the winning producer")

        monkeypatch.setattr(contender_backend, "_pending_cancel_operations", pause_cancel_scan)
        monkeypatch.setattr(contender_backend, "commit_cancel_checkpoint", reject_foreign_cancel)
        monkeypatch.setattr(winner_backend, "launch", pause_launch_return)
        stale = asyncio.create_task(_recover(other, execution_id, "recovery"))
        winner = None
        try:
            await asyncio.wait_for(stale_loaded.wait(), 10)
            winner = asyncio.create_task(_recover(runtime, execution_id, "recovery"))
            await asyncio.wait_for(launch_confirmed.wait(), 10)
            receipt = await _receipt(state, "recovery")
            head = await _head(state, execution_id)
            assert receipt.status is OperationStatus.RUNNING
            assert receipt.result_ref == head.producer_claim_id
            now = datetime.now(timezone.utc)
            await state.execution.operations.append(OperationLedgerInput(
                "pending-cancel", "default", ResourceKind.EXECUTION, execution_id, execution_id,
                OperationKind.EXECUTION_CANCEL, OperationStatus.PENDING,
                "b" * 64, None, None, None, False, now, now,
            ))
            scan_cancel.set()
            with pytest.raises(AIError) as losing_control:
                await stale
            assert losing_control.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
            assert await _receipt(state, "recovery") == receipt
            assert await _head(state, execution_id) == head
            finish_launch.set()
            await winner
            settled = await _receipt(state, "recovery")
            assert settled.status is OperationStatus.SUCCEEDED
            assert settled.result_ref == receipt.result_ref
            await _recover(other, execution_id, "recovery")
            assert await _receipt(state, "recovery") == settled
            assert await _head(state, execution_id) == head
            assert calls == [1]
        finally:
            scan_cancel.set()
            finish_launch.set()
            await asyncio.gather(stale, *(() if winner is None else (winner,)), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("failed", "cancelled"))
async def test_confirmed_launch_receipt_is_independent_of_fast_business_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    started = asyncio.Event()
    async with _recoverable(
        tmp_path, phase="admitted", model_error=outcome == "failed", model_started=started,
    ) as (runtime, other, state, execution_id, calls, release):
        backend = runtime._execution_service.runtime_backend()
        launch = backend.launch
        expected = ExecutionStatus.FAILED if outcome == "failed" else ExecutionStatus.CANCELLED

        async def finish_worker_before_readback(*args, **kwargs):
            await launch(*args, **kwargs)
            worker = backend._tasks[execution_id]
            await asyncio.wait_for(started.wait(), 10)
            if outcome == "failed":
                release.set()
            else:
                execution = await runtime.executions.get(execution_id)
                await execution.cancel(idempotency_key="business-cancel")
            await asyncio.wait_for(
                asyncio.gather(asyncio.shield(worker), return_exceptions=True), 10,
            )
            current = await state.execution.executions.get(execution_id, tenant_id="default")
            assert current.status is expected

        monkeypatch.setattr(backend, "launch", finish_worker_before_readback)
        await _recover(runtime, execution_id, "recovery")
        receipt = await _receipt(state, "recovery")
        assert receipt.status is OperationStatus.SUCCEEDED
        head = await _head(state, execution_id)
        assert receipt.result_ref == head.producer_claim_id
        current = await state.execution.executions.get(execution_id, tenant_id="default")
        assert current.status is expected
        previous_calls = list(calls)
        await _recover(other, execution_id, "recovery")
        assert await _receipt(state, "recovery") == receipt
        assert await _head(state, execution_id) == head
        assert calls == previous_calls
