#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution turns fence dispatch while preserving definite external outcomes."""
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest
import pytest_asyncio

from linktools.ai.core import ExecutionStatus, ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.commands import StateCommands
from linktools.ai.persistence.contracts import ExecutionRecord, ToolOperationAdmission, ToolOperationRecord
from linktools.ai.persistence.codec import decode_domain, encode_domain
from linktools.ai.persistence.state import Observed
from linktools.ai.storage import StoredPayload
from .test_step_projection import _execution
from .test_tool_command_convergence import _admission
from .test_execution_recovery_commands import _commands as _recovery, _resolution_operation


ToolTurn = tuple[Persistence, StateCommands, ExecutionRecord]


@pytest_asyncio.fixture(params=("memory", "sqlite"))
async def turn(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[ToolTurn]:
    storage = (PersistenceStorage.in_memory() if request.param == "memory"
               else PersistenceStorage.sqlite(tmp_path / "turn.db"))
    async with Persistence.open("tool-turn", tenant_id="tenant", storage=storage) as state:
        execution = replace(_execution(), revision=1, producer_generation=1, producer_claim_id="owner")
        await state.execution.executions.create_execution(execution)
        commands = StateCommands(state.execution.executions, namespace="tool-turn",
            events=state.execution.events, tools=state.checkpoint.tools, background_tasks=set())
        yield state, commands, execution


def _request(execution: ExecutionRecord, *, replay_safe: bool = False) -> ToolOperationAdmission:
    return replace(_admission(), producer_generation=execution.producer_generation,
                   owner=execution.producer_claim_id, replay_safe=replay_safe)


async def _takeover(state: Persistence, execution: ExecutionRecord, *, owner: str = "successor") -> ExecutionRecord:
    return await state.execution.executions.claim_next_agent_run(
        execution.execution_id, tenant_id="tenant", expected_revision=execution.revision,
        expected_agent_run_seq=execution.agent_run_seq, producer_claim_id=owner,
    )


async def _reconcile(commands: StateCommands, successor: ExecutionRecord) -> ToolOperationRecord:
    return await commands.reconcile_tool_turn_takeover("operation", execution_id="execution",
        producer_generation=successor.producer_generation, producer_claim_id=successor.producer_claim_id)


async def _finish(commands: StateCommands, observed: Observed[ToolOperationRecord], *, failed: bool = False) -> tuple[Observed[ToolOperationRecord], bool]:
    record = observed.value
    kwargs = ({"error_code": ErrorCode.TOOL_EXECUTION_FAILED.value,
               "error_payload": StoredPayload.inline_bytes(b'{"version":1,"kind":"tool_call_failed","message":"failed"}')}
              if failed else {"result_payload": StoredPayload.inline_bytes(b"actual-result")})
    return await commands.commit_tool_terminal(record.tool_operation_id, tenant_id="tenant",
        owner=record.owner, fence=record.fence, observed=observed, **kwargs)


@pytest.mark.asyncio
async def test_only_exact_current_execution_turn_can_admit_or_reconcile(turn: ToolTurn) -> None:
    state, commands, execution = turn
    request = _request(execution)
    assert decode_domain(encode_domain(request), ToolOperationAdmission) == request
    admitted = await commands.commit_tool_admission(request)
    assert admitted.value.producer_generation == execution.producer_generation
    assert await _reconcile(commands, execution) == admitted.value
    successor = await _takeover(state, execution)
    for stale in (request, replace(_request(successor), owner="another-owner")):
        with pytest.raises(AIError) as rejected:
            await commands.commit_tool_admission(stale)
        assert rejected.value.code is ErrorCode.STORAGE_CONFLICT
        assert rejected.value.retryable is False
    with pytest.raises(AIError) as rejected:
        await _reconcile(commands, execution)
    assert rejected.value.code is ErrorCode.STORAGE_CONFLICT
    assert await state.checkpoint.tools.get_operation("operation", tenant_id="tenant") == admitted.value
    unknown = await _reconcile(commands, successor)
    assert unknown.status is ToolOperationStatus.EFFECT_UNKNOWN
    assert (unknown.owner, unknown.fence, unknown.producer_generation) == ("owner", 1, 1)
    with pytest.raises(AIError) as rejected:
        await commands.commit_tool_admission(_request(successor))
    assert rejected.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", (False, True))
async def test_definite_old_outcome_can_settle_unresolved_unknown_with_stale_receipt(turn: ToolTurn, failed: bool) -> None:
    state, commands, execution = turn
    admitted = await commands.commit_tool_admission(_request(execution))
    successor = await _takeover(state, execution)
    unknown = await _reconcile(commands, successor)
    assert unknown.status is ToolOperationStatus.EFFECT_UNKNOWN
    settled, cancelled = await _finish(commands, admitted, failed=failed)
    assert settled.value.status is (ToolOperationStatus.FAILED if failed else ToolOperationStatus.COMPLETED)
    assert settled.value.owner == admitted.value.owner
    assert settled.value.fence == admitted.value.fence
    assert settled.storage_revision > admitted.storage_revision
    assert not cancelled
    assert await state.checkpoint.tools.get_operation("operation", tenant_id="tenant") == settled.value
    assert await _reconcile(commands, successor) == settled.value


@pytest.mark.asyncio
@pytest.mark.parametrize("successor_finished", (False, True))
async def test_replay_safe_retry_new_fence_rejects_old_result(turn: ToolTurn, successor_finished: bool) -> None:
    state, commands, execution = turn
    first = await commands.commit_tool_admission(_request(execution, replay_safe=True))
    successor = await _takeover(state, execution)
    pending = await _reconcile(commands, successor)
    assert pending.status is ToolOperationStatus.PENDING and pending.owner is None
    second = await commands.commit_tool_admission(_request(successor, replay_safe=True))
    assert second.value.fence == first.value.fence + 1
    assert second.value.owner == successor.producer_claim_id
    current = (await _finish(commands, second))[0] if successor_finished else second
    with pytest.raises(AIError) as rejected:
        await _finish(commands, first)
    assert rejected.value.code in {ErrorCode.TOOL_OPERATION_CONFLICT, ErrorCode.TOOL_RESULT_CONFLICT}
    assert await state.checkpoint.tools.get_operation("operation", tenant_id="tenant") == current.value


@pytest.mark.asyncio
@pytest.mark.parametrize("target", (ToolOperationStatus.PENDING, ToolOperationStatus.COMPLETED))
async def test_manual_resolution_prevents_late_worker_overwrite(turn: ToolTurn, target: ToolOperationStatus) -> None:
    state, commands, execution = turn
    first = await commands.commit_tool_admission(_request(execution))
    successor = await _takeover(state, execution)
    await _reconcile(commands, successor)
    resolution = replace(_resolution_operation(execution.updated_at), resource_id="operation", result_ref="operation")
    kwargs = {"result_payload": StoredPayload.inline_bytes(b"operator-result")} if target is ToolOperationStatus.COMPLETED else {}
    resolved = await _recovery(state).resolve_tool_effect(resolution, expected_fence=first.value.fence,
        target_status=target, **kwargs)
    with pytest.raises(AIError) as rejected:
        await _finish(commands, first)
    assert rejected.value.code in {ErrorCode.TOOL_OPERATION_CONFLICT, ErrorCode.TOOL_RESULT_CONFLICT}
    assert await state.checkpoint.tools.get_operation("operation", tenant_id="tenant") == resolved


@pytest.mark.asyncio
async def test_cancellation_is_observed_without_discarding_known_tool_result(turn: ToolTurn) -> None:
    state, commands, execution = turn
    first = await commands.commit_tool_admission(_request(execution))
    await state.execution.executions.compare_and_swap("execution", tenant_id="tenant", expected_revision=execution.revision,
        next_record=replace(execution, revision=execution.revision + 1, status=ExecutionStatus.CANCELLING))
    settled, cancellation_requested = await _finish(commands, first)
    assert cancellation_requested
    assert settled.value.result_payload == StoredPayload.inline_bytes(b"actual-result")
    with pytest.raises(AIError) as blocked:
        await commands.commit_tool_admission(replace(_request(execution), tool_operation_id="other", tool_call_id="other"))
    assert blocked.value.code in {ErrorCode.EXECUTION_CANCELLED, ErrorCode.STORAGE_CONFLICT}
