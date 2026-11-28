#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical deferred decisions settle the admitted turn and original leaf."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred
from pydantic_ai.messages import ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart

from linktools.ai.capability import AgentContext
from linktools.ai.core import ApprovalDecision, ApprovalStatus, ExternalCallStatus, ExecutionStatus, Principal, TenantAuthorizationPolicy, ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ApprovalDecisionRequest, ExternalCallSucceeded, ExternalSupplyRequest
from linktools.ai.persistence.turns import SessionTurnKey
from linktools.ai.runtime._session_deferred import SessionDeferredCommands

from ._runtime_test_helpers import _wait_for_committed
from .test_official_runtime_recovery import _runtime


class _SessionAuthorization(TenantAuthorizationPolicy):
    """An explicit host policy, not the implicit standalone-only default."""


@pytest.mark.asyncio
@pytest.mark.parametrize("resolution", ("denied", "external"))
async def test_canonical_deferred_outcome_uses_original_turn_and_pending_leaf(
    tmp_path: Path, resolution: str,
) -> None:
    entries = []
    returned = []

    async def checked(_context: AgentContext[None]) -> str:
        entries.append("called")
        assert len(entries) == 1, "a denied or externally settled effect must not be dispatched again"
        if resolution == "denied":
            raise ApprovalRequired()
        raise CallDeferred()

    async def model(messages, _info):
        answers = [
            part.content for message in messages for part in message.parts
            if isinstance(part, (ToolReturnPart, RetryPromptPart)) and part.tool_call_id == "original-call"
        ]
        if answers:
            returned.extend(answers)
            return ModelResponse(parts=[TextPart("finished")])
        return ModelResponse(parts=[ToolCallPart("checked", {}, tool_call_id="original-call")])

    database = tmp_path / "canonical-deferred.db"
    policy = _SessionAuthorization()
    async with _runtime(database, model, checked, authorization=policy) as (runtime, persistence, steps):
        principal = Principal("deferred-operator", runtime.tenant_id)
        await runtime.sessions.create_turn_session("chat", principal=principal)
        turn_id = (await runtime.sessions.start_turn(
            "chat", "question", agent="default", principal=principal, idempotency_key="original",
        )).key.turn_id
        key = SessionTurnKey("chat", turn_id)
        turns = persistence.session_turns
        async def read_pause():
            row = await turns.get_turn(key)
            if row is not None and row["state"] in ("failed", "cancelled", "succeeded"):
                pytest.fail(
                    f"deferred turn settled as {row['state']}: {row['result_json']!r}; "
                    f"tool invocations={len(entries)}"
                )
            return row

        paused = await _wait_for_committed(read_pause, lambda row: row is not None and row["state"] == "waiting")
        assert paused["generation"] == 1 and paused["lease_until"] is None
        assert (await turns.get_session("chat"))["active_turn_id"] == turn_id.encode()
        assert (await turns.get_session("chat"))["last_turn_no"] == 0
        assert await persistence.execution.executions.get(turn_id, tenant_id=runtime.tenant_id) is None
        operation, = await persistence.checkpoint.tools.list_by_agent_run(
            turn_id, tenant_id=runtime.tenant_id, execution_id=turn_id,
        )
        assert operation.tool_call_id == "original-call"
        assert operation.status is ToolOperationStatus.PENDING and operation.owner is None
        assert operation.producer_generation == 1
        frontier = await turns.get_waiting_turn(key)
        assert frontier is not None and frontier.key == key and frontier.generation == 1
        ids = frontier.approval_operation_ids if resolution == "denied" else frontier.external_operation_ids
        assert ids == (operation.tool_operation_id,)
        checkpoint = await steps.latest_snapshot(run_id=turn_id, include_interrupted=True)
        assert checkpoint is not None and checkpoint.state == "interrupted"
        assert checkpoint.idempotency_key == frontier.checkpoint_id
        assert checkpoint.conversation_id == "chat"
        await runtime.handoff()

    async with _runtime(database, model, checked, authorization=policy) as (runtime, persistence, _steps):
        principal = Principal("deferred-operator", runtime.tenant_id)
        assert await persistence.session_turns.get_waiting_turn(key) == frontier
        if resolution == "denied":
            pending, = await runtime.approvals.list(key, principal=principal)
            assert pending.status is ApprovalStatus.PENDING
            request = ApprovalDecisionRequest(
                principal, pending.approval_id, "settle", ApprovalDecision.DENY, message="denied by operator",
            )
            settle = runtime.approvals.decide
            conflict = ErrorCode.APPROVAL_CONFLICT
        else:
            pending, = await runtime.external_calls.list(key, principal=principal)
            assert pending.status == ExternalCallStatus.PENDING.value
            request = ExternalSupplyRequest(
                principal, pending.call_id, "settle", ExternalCallSucceeded("supplied outside the runtime"),
            )
            settle = runtime.external_calls.supply
            conflict = ErrorCode.EXTERNAL_RESULT_CONFLICT
        turn = await runtime.sessions.start_turn(
            "chat", "question", agent="default", principal=principal, idempotency_key="original",
        )
        assert turn.key == key
        assert (await turn.wait()).status is ExecutionStatus.WAITING_DEFERRED
        waiting = asyncio.create_task(turn.wait_final(timeout_seconds=10))
        try:
            await asyncio.sleep(0)
            assert not waiting.done()
            await settle(key, request)
            final = await waiting
        finally:
            if not waiting.done():
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)
        assert final.key == key and final.status is ExecutionStatus.SUCCEEDED
        assert final.view["output"] == {"text": "finished"}

        turns = persistence.session_turns
        terminal = await _wait_for_committed(
            lambda: turns.get_turn(key), lambda row: row is not None and row["state"] == "succeeded",
        )
        assert terminal["turn_id"] == turn_id.encode() and terminal["turn_no"] == 1
        assert terminal["generation"] == paused["generation"] + 1 and terminal["lease_until"] is None
        session = await turns.get_session("chat")
        assert session["last_turn_no"] == 1 and session["active_turn_id"] is None
        assert await turns.get_waiting_turn(key) is None
        assert await persistence.execution.executions.get(turn_id, tenant_id=runtime.tenant_id) is None
        settled, = await persistence.checkpoint.tools.list_by_agent_run(
            turn_id, tenant_id=runtime.tenant_id, execution_id=turn_id,
        )
        assert settled.tool_operation_id == operation.tool_operation_id
        assert (settled.owner, settled.fence, settled.producer_generation) == (
            operation.owner, operation.fence, operation.producer_generation,
        )
        assert settled.status is (ToolOperationStatus.FAILED if resolution == "denied" else ToolOperationStatus.COMPLETED)
        detail = await runtime.history.turn_result("chat", turn_id, principal=principal)
        assert detail["output"] == {"text": "finished"} and detail["conversation_committed"]

        # A late duplicate cannot claim another turn or rewrite the settled leaf.
        with pytest.raises(AIError) as caught:
            await settle(key, request)
        assert caught.value.code is conflict
        assert await persistence.checkpoint.tools.get_operation(operation.tool_operation_id, tenant_id=runtime.tenant_id) == settled
        assert await turns.get_turn(key) == terminal
    assert entries == ["called"] and len(returned) == 1
    assert ("denied by operator" if resolution == "denied" else "supplied outside the runtime") in str(returned[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("approval", "external", "ready"))
async def test_stop_parked_canonical_turn_withdraws_original_pending_effect(
    tmp_path: Path, kind: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_calls = []
    leaf_entries = []

    async def checked(_context: AgentContext[None]) -> str:
        # The tool adapter requests a decision before performing any effect.
        leaf_entries.append("before-effect")
        assert len(leaf_entries) == 1, "Stop must not enter the effect again"
        if kind != "external":
            raise ApprovalRequired()
        raise CallDeferred()

    async def model(messages, _info):
        model_calls.append(messages)
        assert len(model_calls) == 1, "Stop must not call the provider again"
        return ModelResponse(parts=[ToolCallPart("checked", {}, tool_call_id="parked-call")])

    database = tmp_path / "stop-parked-turn.db"
    policy = _SessionAuthorization()
    async with _runtime(database, model, checked, authorization=policy) as (runtime, persistence, steps):
        principal = Principal("deferred-operator", runtime.tenant_id)
        await runtime.sessions.create_turn_session("chat", principal=principal)
        turn = await runtime.sessions.start_turn(
            "chat", "question", agent="default", principal=principal, idempotency_key="parked",
        )
        key = turn.key
        turns = persistence.session_turns
        paused = await _wait_for_committed(
            lambda: turns.get_turn(key), lambda row: row is not None and row["state"] == "waiting",
        )
        frontier = await turns.get_waiting_turn(key)
        assert frontier is not None and frontier.generation == paused["generation"]
        original, = await persistence.checkpoint.tools.list_by_agent_run(
            key.turn_id, execution_id=key.turn_id, tenant_id=runtime.tenant_id,
        )
        assert original.status is ToolOperationStatus.PENDING and original.owner is None
        checkpoint = await steps.latest_snapshot(run_id=key.turn_id, include_interrupted=True)
        assert checkpoint is not None and checkpoint.idempotency_key == frontier.checkpoint_id
        if kind != "external":
            pending, = await runtime.approvals.list(key, principal=principal)
            assert pending.status is ApprovalStatus.PENDING
            receipt_id = pending.approval_id
        else:
            pending, = await runtime.external_calls.list(key, principal=principal)
            assert pending.status == ExternalCallStatus.PENDING.value
            receipt_id = pending.call_id
        await runtime.handoff()

    async with _runtime(database, model, checked, authorization=policy) as (runtime, persistence, _steps):
        principal = Principal("deferred-operator", runtime.tenant_id)
        turns = persistence.session_turns
        if kind == "ready":
            before_resume, allow_resume = asyncio.Event(), asyncio.Event()
            resume = SessionDeferredCommands.resume

            async def held_resume(commands, frontier, *, receipt_store):
                before_resume.set()
                await allow_resume.wait()
                return await resume(commands, frontier, receipt_store=receipt_store)

            monkeypatch.setattr(SessionDeferredCommands, "resume", held_resume)
            approving = asyncio.create_task(runtime.approvals.decide(
                key, ApprovalDecisionRequest(principal, receipt_id, "approve", ApprovalDecision.APPROVE),
            ))
            try:
                await asyncio.wait_for(before_resume.wait(), timeout=10)
                assert (await turns.get_turn(key))["state"] == "ready"
                assert await runtime.sessions.cancel_turn("chat", key.turn_id, principal=principal)
            finally:
                allow_resume.set()
                await approving
        else:
            assert await runtime.sessions.cancel_turn("chat", key.turn_id, principal=principal)
        terminal = await _wait_for_committed(
            lambda: turns.get_turn(key), lambda row: row is not None and row["state"] == "cancelled",
        )
        assert terminal["turn_id"] == key.turn_id.encode()
        assert terminal["generation"] == paused["generation"] + 1
        assert terminal["result_json"] == {"error_code": ErrorCode.EXECUTION_CANCELLED.value}
        assert await turns.get_waiting_turn(key) is None
        session = await turns.get_session("chat")
        assert session["active_turn_id"] is None and session["last_turn_no"] == 1
        assert await persistence.execution.executions.get(key.turn_id, tenant_id=runtime.tenant_id) is None
        settled = await persistence.checkpoint.tools.get_operation(original.tool_operation_id, tenant_id=runtime.tenant_id)
        assert settled is not None and settled.status is ToolOperationStatus.CANCELLED
        assert (settled.owner, settled.fence, settled.producer_generation, settled.arguments_payload) == (
            original.owner, original.fence, original.producer_generation, original.arguments_payload,
        )
        if kind != "external":
            approval = await persistence.checkpoint.approvals.get(receipt_id, tenant_id=runtime.tenant_id)
            assert approval is not None and approval.status is (
                ApprovalStatus.APPROVED if kind == "ready" else ApprovalStatus.CANCELLED
            )
            request = ApprovalDecisionRequest(principal, receipt_id, "late", ApprovalDecision.DENY)
            with pytest.raises(AIError) as caught:
                await runtime.approvals.decide(key, request)
            assert caught.value.code is ErrorCode.APPROVAL_CONFLICT
        else:
            call = await persistence.checkpoint.external_calls.get(receipt_id, tenant_id=runtime.tenant_id)
            assert call is not None and call.status is ExternalCallStatus.CANCELLED
            request = ExternalSupplyRequest(principal, receipt_id, "late", ExternalCallSucceeded("too late"))
            with pytest.raises(AIError) as caught:
                await runtime.external_calls.supply(key, request)
            assert caught.value.code is ErrorCode.EXTERNAL_RESULT_CONFLICT
        assert await turns.get_turn(key) == terminal
    assert len(model_calls) == len(leaf_entries) == 1
