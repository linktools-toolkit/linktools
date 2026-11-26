#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A successor may claim the original pristine child reservation, never dispatch twice."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from linktools.ai.capability import ExecutionCapability, StepPersistenceCapability
from linktools.ai.core import ExecutionStatus, IdempotencyStatus, Principal, ToolOperationStatus, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionStartClaim
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._native_subagent import NativeSubagentAdmission

from ._runtime_test_helpers import _wait_for_committed
from .test_canonical_turn_subagents import _HostAuthorization, _model


@pytest.mark.asyncio
async def test_recovered_native_start_requires_retained_receipt_and_pristine_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_calls = []
    invocation = "original-delegate"

    async def parent(_messages, _info):
        return ModelResponse(parts=[ToolCallPart(
            "delegate_task", {"subagent_id": "child", "task": "Read the fixture"}, tool_call_id=invocation,
        )])

    async def child(messages, _info):
        child_calls.append(messages)
        return ModelResponse(parts=[TextPart("should not have dispatched")])

    async def stop_before_start(_proof, _commands, _claim):
        # The original reserve transaction has already committed; the child
        # first-start transaction is the precise failure point.
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)

    monkeypatch.setattr(NativeSubagentAdmission, "commit_start", stop_before_start)
    name = "native-reserved-child-start"
    features = RuntimeFeatures(name)
    features.model("parent", _model("parent", parent))
    features.model("child", _model("child", child))
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.capability(StepPersistenceCapability(), scope="runtime")
    features.agent("default", model="parent", allow_subagents=("child",), allow_tools=(), allow_skills=())
    features.agent("child", model="child", allow_subagents=(), allow_tools=(), allow_skills=())

    async with (
        Persistence.open(name, storage=PersistenceStorage.from_root(tmp_path)) as state,
        Runtime.open(persistence=state, features=features, authorization=_HostAuthorization(), auto_recover=False) as runtime,
    ):
        principal = Principal("owner", runtime.tenant_id)
        await runtime.sessions.create_turn_session("chat", principal=principal)
        turn = await runtime.agents.get().start(
            "Delegate once", session_id="chat", principal=principal, idempotency_key="original",
        )
        key = turn.key
        turns = state.session_turns

        async def original_delegate():
            return next((item for item in await state.checkpoint.tools.list_by_agent_run(
                key.turn_id, tenant_id=runtime.tenant_id, execution_id=key.turn_id,
            ) if item.tool_call_id == invocation), None)

        original = await _wait_for_committed(
            original_delegate, lambda row: row is not None and row.status is ToolOperationStatus.EFFECT_UNKNOWN,
        )
        children = await state.execution.executions.list_children(key.turn_id, tenant_id=runtime.tenant_id)
        [reserved] = children
        assert reserved.status is ExecutionStatus.PENDING_START and reserved.started_at is None
        request_key = canonical_sha256({
            "kind": "subagent", "parent_execution_id": key.turn_id, "invocation_id": invocation,
        })
        receipt = await state.execution.idempotency.get("execution.subagent", request_key, tenant_id=runtime.tenant_id)
        assert receipt is not None and receipt.status is IdempotencyStatus.RESERVED
        claim = ExecutionStartClaim(
            reserved.execution_id, reserved.revision, reserved.event_seq,
            receipt.scope, receipt.idempotency_key, receipt.request_digest, datetime.now(timezone.utc),
        )
        commands = runtime._execution_service.runtime_backend()._runtime_commands

        async def expire(connection):
            statement = turns._turns.update().filter_by(**turns._turn_values(key), generation=1, state="accepted")
            result = await connection.execute(statement.values(lease_until=turns._now - 1))
            assert result.rowcount == 1

        await turns._context.run_mutation(expire, domain="test.expire_parent")
        assert await turns.take_over(key, generation=1, lease_seconds=60)
        start = commands.commit_recovered_native_subagent_start

        with pytest.raises(AIError) as stale_generation:
            await start(turns, key, generation=1, principal=principal, original=original, claim=claim)
        assert stale_generation.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
        with pytest.raises(AIError) as stale_receipt:
            await start(turns, key, generation=2, principal=principal,
                        original=replace(original, fence=original.fence + 1), claim=claim)
        assert stale_receipt.value.code is ErrorCode.TOOL_OPERATION_CONFLICT
        assert await state.execution.executions.get(reserved.execution_id, tenant_id=runtime.tenant_id) == reserved

        started = await start(turns, key, generation=2, principal=principal, original=original, claim=claim)
        assert started.execution_id == reserved.execution_id and started.status is ExecutionStatus.STARTED
        assert started.agent_run_seq == started.producer_generation == 0 and started.started_at is not None
        with pytest.raises(AIError) as duplicate:
            await start(turns, key, generation=2, principal=principal, original=original, claim=claim)
        assert duplicate.value.code is ErrorCode.STORAGE_CONFLICT
        assert not child_calls and await original_delegate() == original
        assert await state.execution.executions.get(key.turn_id, tenant_id=runtime.tenant_id) is None
        await runtime.handoff()
