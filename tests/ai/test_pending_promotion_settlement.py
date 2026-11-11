#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A live pending owner retains known outcomes across uncertain promotion."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus, ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import Execution, ExecutionRecord
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.commands import StateCommands
from linktools.ai.persistence.contracts import ToolOperationAdmission, ToolOperationRecord
from linktools.ai.persistence.state import active_state_scope
from linktools.ai.runtime._tool import RuntimeToolOperationBridge, ToolOperationDecision
from linktools.ai.storage import StoredPayload

from .test_pending_tool_staging import _ToolRuntime, _assert_known_result, _tool_runtime


@dataclass
class _PromotionLoss:
    admitted: ToolOperationRecord
    source: Callable[[], Awaitable[tuple[ExecutionRecord | None, tuple[ToolOperationRecord, ...]]]]
    submitted: asyncio.Event = field(default_factory=asyncio.Event)
    release_commit: asyncio.Event = field(default_factory=asyncio.Event)
    payload_ready: asyncio.Event = field(default_factory=asyncio.Event)
    readable: bool = True
    lose_settlement_ack: bool = False
    settlement_ack_lost: bool = False
    server_tasks: list[asyncio.Task[object]] = field(default_factory=list)

    async def commit(self) -> None:
        self.release_commit.set()
        await asyncio.wait_for(asyncio.gather(*self.server_tasks), 5)


@asynccontextmanager
async def _lost_promotion(
    case: _ToolRuntime, execution: Execution[None], monkeypatch: pytest.MonkeyPatch,
    *, readback_unavailable: bool,
) -> AsyncIterator[_PromotionLoss]:
    backend = case.runtime._execution_service.runtime_backend()
    turn = backend._pending_turns.get(execution.execution_id, tenant_id=case.runtime.tenant_id)
    assert turn is not None and len(turn.tool_operations) == 1
    admitted = next(iter(turn.tool_operations.values()))
    assert admitted.status is ToolOperationStatus.CLAIMED
    executions, tools = case.persistence.execution.executions, case.persistence.checkpoint.tools
    group = executions.state_store.storage_group
    original_mutate, original_read = type(group).mutate, type(group).read_many
    original_payload = RuntimeToolOperationBridge._result_payload

    async def source() -> tuple[ExecutionRecord | None, tuple[ToolOperationRecord, ...]]:
        # Inspect the server independently of the injected client read outage.
        async def read(_transaction: object) -> tuple[ExecutionRecord | None, tuple[ToolOperationRecord, ...]]:
            return (
                await executions.get(execution.execution_id, tenant_id=case.runtime.tenant_id),
                await tools.list_by_execution(execution.execution_id, tenant_id=case.runtime.tenant_id),
            )

        return await original_read(group, (executions.state_store, tools.state_store), read)

    loss = _PromotionLoss(admitted, source, readable=not readback_unavailable)

    async def delayed(owner: object, stores: object, callback: object) -> object:
        if owner is group and active_state_scope() is None and not loss.submitted.is_set():
            async def server_commit() -> object:
                await loss.release_commit.wait()
                return await original_mutate(owner, stores, callback)

            loss.server_tasks.append(asyncio.create_task(server_commit()))
            loss.submitted.set()
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        if owner is group and active_state_scope() is None and loss.lose_settlement_ack:
            loss.lose_settlement_ack = False
            await original_mutate(owner, stores, callback)
            loss.settlement_ack_lost = True
            loss.readable = False
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        return await original_mutate(owner, stores, callback)

    async def unavailable(owner: object, stores: object, callback: object) -> object:
        if (owner is group and active_state_scope() is None
                and loss.submitted.is_set() and not loss.readable):
            raise OSError("source promotion readback temporarily unavailable")
        return await original_read(owner, stores, callback)

    async def payload(
        bridge: RuntimeToolOperationBridge, decision: ToolOperationDecision, result: object,
    ) -> StoredPayload:
        encoded = await original_payload(bridge, decision, result)
        if bridge._execution_id == execution.execution_id:
            loss.payload_ready.set()
        return encoded

    try:
        with monkeypatch.context() as patch:
            patch.setattr(type(group), "mutate", delayed)
            patch.setattr(type(group), "read_many", unavailable)
            patch.setattr(RuntimeToolOperationBridge, "_result_payload", payload)
            yield loss
    finally:
        loss.readable = True
        loss.release_commit.set()
        case.gate.release.set()
        case.gate.finish.set()
        await asyncio.gather(*loss.server_tasks, return_exceptions=True)


async def _return_during_unknown_promotion(
    case: _ToolRuntime, execution: Execution[None], loss: _PromotionLoss,
) -> None:
    with pytest.raises(AIError) as promotion:
        await asyncio.wait_for(execution.cancel(idempotency_key="cancel-with-lost-promotion"), 5)
    assert promotion.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
    assert loss.submitted.is_set()
    case.gate.release.set()
    await asyncio.wait_for(case.gate.stopped.wait(), 5)
    await asyncio.wait_for(loss.payload_ready.wait(), 5)
    assert case.gate.returned.is_set()
    assert case.gate.effects == [execution.execution_id]
    assert await loss.source() == (None, ())
    # A returned and encoded value is not a source acknowledgement. Neither a
    # waiter nor a repeated request may treat this unknown boundary as success.
    with pytest.raises(AIError) as waiting:
        await execution.wait(timeout_seconds=5)
    assert waiting.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
    if loss.readable:
        with pytest.raises(AIError) as repeated:
            await case.session.start("work", idempotency_key="request")
        assert repeated.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
    assert len(loss.server_tasks) == 1
    assert case.gate.effects == [execution.execution_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("readback_unavailable,lose_settlement_ack", (
    (False, False), (True, False), (True, True),
))
async def test_uncertain_cancel_promotion_preserves_late_known_tool_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    readback_unavailable: bool, lose_settlement_ack: bool,
) -> None:
    async with _tool_runtime(tmp_path) as case:
        execution = await case.start()
        async with _lost_promotion(case, execution, monkeypatch,
                readback_unavailable=readback_unavailable) as loss:
            await _return_during_unknown_promotion(case, execution, loss)
            await loss.commit()
            if readback_unavailable:
                source, operations = await loss.source()
                assert source is not None and source.status is ExecutionStatus.STARTED
                assert operations == (loss.admitted,)
                assert operations[0].result_payload is None
                with pytest.raises(AIError) as waiting:
                    await execution.wait(timeout_seconds=5)
                assert waiting.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
                assert await loss.source() == (source, operations)
            loss.readable = True
            if lose_settlement_ack:
                loss.lose_settlement_ack = True
                with pytest.raises(AIError) as settlement:
                    await execution.wait(timeout_seconds=5)
                assert settlement.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
                assert loss.settlement_ack_lost
                _, operations = await loss.source()
                assert len(operations) == 1 and operations[0].status is ToolOperationStatus.COMPLETED
                _assert_known_result(operations[0])
                # The original CLAIMED receipt is already confirmed. A retry
                # must settle the late result, not revalidate that old tuple.
                loss.readable = True
            case.gate.finish.set()
            result = await execution.wait(timeout_seconds=10)
            assert result.result.status in {
                ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
            }
            operations = await case.operations(execution.execution_id)
            assert len(operations) == 1
            completed = operations[0]
            assert completed.status is ToolOperationStatus.COMPLETED
            assert (completed.tool_operation_id, completed.owner, completed.fence) == (
                loss.admitted.tool_operation_id, loss.admitted.owner, loss.admitted.fence,
            )
            _assert_known_result(completed)
            assert case.gate.effects == [execution.execution_id]
            replay = await case.session.start("work", idempotency_key="request")
            assert replay.execution_id == execution.execution_id
            assert (await replay.wait(timeout_seconds=5)).result == result.result
        await case.runtime.close()
        assert await case.operations(execution.execution_id) == operations
        assert case.gate.effects == [execution.execution_id]


@pytest.mark.asyncio
async def test_uncertain_promotion_late_result_cannot_overwrite_successor_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    successor_operation = None
    execution_id = None
    failures: list[ErrorCode] = []
    try:
        async with _tool_runtime(tmp_path) as case:
            execution = await case.start()
            execution_id = execution.execution_id
            backend = case.runtime._execution_service.runtime_backend()
            worker = backend._tasks[execution_id]
            async with _lost_promotion(case, execution, monkeypatch, readback_unavailable=True) as loss:
                await _return_during_unknown_promotion(case, execution, loss)
                await loss.commit()
                current, operations = await loss.source()
                assert current is not None and operations == (loss.admitted,)
                successor = await case.persistence.execution.executions.claim_next_agent_run(
                    execution_id, tenant_id=case.runtime.tenant_id, expected_revision=current.revision,
                    expected_agent_run_seq=current.agent_run_seq, producer_claim_id="successor",
                )
                commands = StateCommands(case.persistence.execution.executions,
                    namespace="pending-tool-staging", events=case.persistence.execution.events,
                    tools=case.persistence.checkpoint.tools, background_tasks=set())
                pending = await commands.reconcile_tool_turn_takeover(
                    loss.admitted.tool_operation_id, execution_id=execution_id,
                    producer_generation=successor.producer_generation,
                    producer_claim_id=successor.producer_claim_id,
                )
                assert pending.status is ToolOperationStatus.PENDING
                request = ToolOperationAdmission(
                    execution_id=execution_id, tool_operation_id=pending.tool_operation_id,
                    agent_run_id=pending.agent_run_id, recovery_agent_run_id=None,
                    tool_call_id=pending.tool_call_id,
                    tool_name=pending.tool_name, arguments_digest=pending.arguments_digest,
                    binding_digest=pending.binding_digest, replay_safe=True,
                    owner=successor.producer_claim_id, producer_generation=successor.producer_generation,
                    arguments_payload=pending.arguments_payload, capability_ref=pending.capability_ref,
                )
                successor_operation = (await commands.commit_tool_admission(request)).value
                assert successor_operation.owner == "successor"
                assert successor_operation.fence == loss.admitted.fence + 1
                loss.readable = True
                case.gate.finish.set()
                with pytest.raises(AIError) as stale:
                    await execution.wait(timeout_seconds=5)
                failures.append(stale.value.code)
                await asyncio.wait_for(asyncio.gather(worker, return_exceptions=True), 5)
                assert await case.operations(execution_id) == (successor_operation,)
                assert case.gate.effects == [execution_id]
    except AIError as closing:
        failures.append(closing.code)
    assert successor_operation is not None and execution_id is not None
    assert failures and set(failures) <= {
        ErrorCode.STORAGE_COMMIT_UNKNOWN, ErrorCode.STORAGE_CONFLICT,
        ErrorCode.TOOL_OPERATION_CONFLICT, ErrorCode.TOOL_RESULT_CONFLICT,
    }
    # Losing the producer cannot make cleanup overwrite the successor either.
    async with Persistence.open("pending-tool-staging",
            storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as state:
        assert await state.checkpoint.tools.list_by_execution(
            execution_id, tenant_id=state.checkpoint.tools.tenant_id,
        ) == (successor_operation,)
