#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Targeted cold recovery consumes only persisted deferred frontiers."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from linktools.ai.capability import AgentContext
from linktools.ai.core import (
    ApprovalDecision, AuthorizationAction, ExecutionStatus, OperationStatus, Principal,
    ResourceRef, TenantAuthorizationPolicy,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ApprovalDecisionRequest, ExternalCallSucceeded, ExternalSupplyRequest

from ._runtime_test_helpers import _wait_for_committed
from .test_official_runtime_recovery import _runtime


class _RecoveryAuthorization(TenantAuthorizationPolicy):
    async def authorize(self, principal: Principal, action: AuthorizationAction, resource: ResourceRef) -> None:
        if action is AuthorizationAction.EXECUTION_RECOVER and principal.principal_id == "recoverer":
            resource = replace(resource, owner_principal_id=None)
        await super().authorize(principal, action, resource)


@pytest.mark.asyncio
@pytest.mark.parametrize(("kind", "restart", "key"), (
    ("approval", True, "recover"), ("external", False, "recover"), ("denied", True, None),
))
async def test_targeted_deferred_recovery_waits_retries_and_preserves_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, restart: bool, key: str | None,
) -> None:
    owner = Principal("owner", "default", "user")
    actor = Principal("recoverer", "default", "service")
    entered = []
    returned = []

    async def checked(context: AgentContext[None]) -> str:
        entered.append(context.principal)
        if kind == "external":
            raise CallDeferred()
        if len(entered) <= 2:
            raise ApprovalRequired()
        return "approved result"

    async def model(messages, _info):
        values = [part.content for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if values:
            returned.extend(values)
            return ModelResponse(parts=[TextPart("finished")])
        return ModelResponse(parts=[ToolCallPart("checked", {}, tool_call_id="deferred")])

    database = tmp_path / "deferred.db"
    async with _runtime(database, model, checked, restart_checkpoints=restart,
                        authorization=_RecoveryAuthorization()) as (runtime, persistence, _steps):
        executions = []
        for session_id in ("selected", "unrelated"):
            session = await runtime.agents.get().create_session(session_id, principal=owner)
            execution = await session.start("question", principal=owner, idempotency_key=session_id)
            executions.append(execution)
            await _wait_for_committed(
                lambda: persistence.execution.executions.get(execution.execution_id, tenant_id="default"),
                lambda current: current is not None and current.status is ExecutionStatus.WAITING_DEFERRED,
            )
        if restart:
            await runtime.handoff()

    selected, unrelated = executions
    async with _runtime(database, model, checked, restart_checkpoints=restart,
                        authorization=_RecoveryAuthorization()) as (runtime, persistence, _steps):
        before = await persistence.execution.executions.get(selected.execution_id, tenant_id="default")
        other_before = await persistence.execution.executions.get(unrelated.execution_id, tenant_id="default")
        with pytest.raises(AIError) as denied:
            await runtime.executions.recover(selected.execution_id, principal=Principal("outsider", "other"), idempotency_key=key)
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        for _ in range(2):
            handle = await runtime.executions.recover(selected.execution_id, principal=actor, idempotency_key=key)
            assert handle.execution_id == selected.execution_id
            assert await persistence.execution.executions.get(selected.execution_id, tenant_id="default") == before
            assert len(entered) == 2 and returned == []
        if key is not None:
            assert await persistence.execution.operations.get(key, tenant_id="default") is None

        async def stop_before_reconciliation(*_args, **_kwargs):
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        # Acknowledged decisions survive an interruption before any resume claim.
        with monkeypatch.context() as fault:
            fault.setattr(runtime._execution_service.runtime_backend(), "reconcile_deferred", stop_before_reconciliation)
            for execution in executions:
                with pytest.raises(AIError) as interrupted:
                    if kind == "external":
                        pending = await runtime.external_calls.list(execution.execution_id, principal=actor)
                        await runtime.external_calls.supply(execution.execution_id, ExternalSupplyRequest(
                            actor, pending[0].call_id, "supply-" + execution.execution_id, ExternalCallSucceeded("saved external result"),
                        ))
                    else:
                        pending = await runtime.approvals.list(execution.execution_id, principal=actor)
                        await runtime.approvals.decide(execution.execution_id, ApprovalDecisionRequest(
                            actor, pending[0].approval_id, "decide-" + execution.execution_id,
                            ApprovalDecision.DENY if kind == "denied" else ApprovalDecision.APPROVE,
                        ))
                assert interrupted.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        if restart:
            await runtime.handoff()

    async with _runtime(database, model, checked, restart_checkpoints=restart,
                        authorization=_RecoveryAuthorization()) as (runtime, persistence, _steps):
        assert await persistence.execution.executions.get(selected.execution_id, tenant_id="default") == before
        await runtime.executions.recover(selected.execution_id, principal=actor, idempotency_key=key)
        result = (await runtime.executions.wait(selected.execution_id, principal=owner, timeout_seconds=5)).result
        assert result.status is ExecutionStatus.SUCCEEDED, result
        assert result.output == {"text": "finished"}
        current = await persistence.execution.executions.get(selected.execution_id, tenant_id="default")
        assert current.principal_id == owner.principal_id and current.principal_kind == owner.kind
        assert current.agent_run_seq == before.agent_run_seq + 1
        assert current.producer_generation > before.producer_generation
        assert await persistence.execution.executions.get(unrelated.execution_id, tenant_id="default") == other_before
        if key is not None:
            receipt = await persistence.execution.operations.get(key, tenant_id="default")
            assert receipt.status is OperationStatus.SUCCEEDED
            assert receipt.result_ref == current.producer_claim_id
            await runtime.executions.recover(selected.execution_id, principal=actor, idempotency_key=key)
            assert await persistence.execution.executions.get(selected.execution_id, tenant_id="default") == current
        assert entered == [owner] * (3 if kind == "approval" else 2)
        assert len(returned) == 1
        if kind == "external":
            assert returned == ["saved external result"]


@pytest.mark.asyncio
@pytest.mark.parametrize("same_key", (False, True))
async def test_concurrent_keyed_deferred_recovery_has_one_durable_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_key: bool,
) -> None:
    calls = []
    started = asyncio.Event()
    release = asyncio.Event()

    async def external(_context: AgentContext[None]) -> str:
        raise CallDeferred()

    async def model(messages, _info):
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            calls.append("resumed")
            started.set()
            await release.wait()
            return ModelResponse(parts=[TextPart("finished")])
        return ModelResponse(parts=[ToolCallPart("external", {}, tool_call_id="call")])

    database = tmp_path / "concurrent.db"
    async with _runtime(database, model, external) as (runtime, persistence, _steps):
        execution = await runtime.agents.get().start("question", idempotency_key="start")
        waiting = await _wait_for_committed(
            lambda: persistence.execution.executions.get(execution.execution_id, tenant_id="default"),
            lambda current: current is not None and current.status is ExecutionStatus.WAITING_DEFERRED,
        )
        await runtime.handoff()
        pending = await runtime.external_calls.list(execution.execution_id, principal=Principal("resolver", "default", "service"))
        async def stop_before_reconciliation(*_args, **_kwargs):
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        with monkeypatch.context() as fault:
            fault.setattr(runtime._execution_service.runtime_backend(), "reconcile_deferred", stop_before_reconciliation)
            with pytest.raises(AIError):
                await runtime.external_calls.supply(execution.execution_id, ExternalSupplyRequest(
                    Principal("resolver", "default", "service"), pending[0].call_id, "supply", ExternalCallSucceeded("saved"),
                ))

    async with AsyncExitStack() as stack:
        runtimes = [(await stack.enter_async_context(_runtime(database, model, external)))[0] for _ in range(2)]
        both_claiming = asyncio.Event()
        claims = 0
        for runtime in runtimes:
            backend = runtime._execution_service.runtime_backend()
            original = backend.claim_deferred_resume
            async def synchronized(current, *, recovery_operation=None, original=original):
                nonlocal claims
                claims += 1
                if claims == 2:
                    both_claiming.set()
                await asyncio.wait_for(both_claiming.wait(), 5)
                return await original(current, recovery_operation=recovery_operation)
            monkeypatch.setattr(backend, "claim_deferred_resume", synchronized)
        try:
            keys = ("resume", "resume") if same_key else ("resume-first", "resume-second")
            outcomes = await asyncio.gather(*(runtime.executions.recover(
                execution.execution_id, principal=runtime.default_principal, idempotency_key=key,
            ) for runtime, key in zip(runtimes, keys)), return_exceptions=True)
            assert any(not isinstance(outcome, BaseException) for outcome in outcomes), outcomes
            assert all(not isinstance(outcome, BaseException) or (
                isinstance(outcome, AIError) and outcome.code is (
                    ErrorCode.STORAGE_RECOVERY_REQUIRED if same_key else ErrorCode.STORAGE_CONFLICT
                )
            ) for outcome in outcomes), outcomes
            winning_key = next(key for key, outcome in zip(keys, outcomes) if not isinstance(outcome, BaseException))
            await asyncio.wait_for(started.wait(), 5)
            persistence = runtimes[0]._state
            current = await persistence.execution.executions.get(execution.execution_id, tenant_id="default")
            receipt = await persistence.execution.operations.get(winning_key, tenant_id="default")
            assert current.agent_run_seq == waiting.agent_run_seq + 1
            assert receipt.status is OperationStatus.SUCCEEDED
            assert receipt.result_ref == current.producer_claim_id
            for runtime in runtimes:
                await runtime.executions.recover(execution.execution_id, principal=runtime.default_principal, idempotency_key=winning_key)
            assert calls == ["resumed"]
        finally:
            release.set()
        result = await runtimes[0].executions.wait(execution.execution_id, principal=runtimes[0].default_principal, timeout_seconds=5)
        assert result.result.status is ExecutionStatus.SUCCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize("lost_ack", (False, True))
async def test_deferred_claim_atomically_replays_operation_and_rejects_stale_claim(
    monkeypatch: pytest.MonkeyPatch, lost_ack: bool,
) -> None:
    from datetime import datetime, timezone

    from linktools.ai.core import OperationKind, OperationLedgerInput, ResourceKind

    from .test_runtime_approval_recovery import _enter_waiting

    persistence, commands, execution, _, continuation = await _enter_waiting("targeted-claim")
    now = datetime.now(timezone.utc)
    operation = OperationLedgerInput(
        "recovery", "tenant", ResourceKind.EXECUTION,
        execution.execution_id, execution.execution_id, OperationKind.EXECUTION_RECOVER,
        OperationStatus.RUNNING, "a" * 64, "producer-token", None, None, False, now, now,
    )
    group = persistence.execution.executions.state_store.storage_group
    mutate = group.mutate
    injected = False

    async def lose_ack(*args, **kwargs):
        nonlocal injected
        result = await mutate(*args, **kwargs)
        if not injected:
            injected = True
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        return result

    async def claim(recovery_operation):
        return await commands.claim_deferred_resume_checkpoint(
            execution_id=execution.execution_id, tenant_id="tenant",
            expected_execution_revision=execution.revision + 1,
            expected_event_seq=execution.event_seq + 1,
            expected_agent_run_seq=execution.agent_run_seq,
            expected_pending_tools=continuation, recovery_operation=recovery_operation,
        )

    try:
        with monkeypatch.context() as fault:
            if lost_ack:
                fault.setattr(group, "mutate", lose_ack)
            resumed = await claim(operation)
        assert injected is lost_ack
        receipt = await persistence.execution.operations.get(operation.operation_id, tenant_id="tenant")
        assert receipt is not None and receipt.result_ref == resumed.producer_claim_id == operation.result_ref
        assert resumed.agent_run_seq == execution.agent_run_seq + 1
        assert await claim(replace(operation, result_ref="another-token")) == resumed
        with pytest.raises(AIError) as stale:
            await claim(replace(operation, operation_id="different", result_ref="another-token"))
        assert stale.value.code is ErrorCode.STORAGE_CONFLICT
        with pytest.raises(AIError) as collision:
            await claim(replace(operation, request_digest="b" * 64))
        assert collision.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        assert await persistence.execution.executions.get(execution.execution_id, tenant_id="tenant") == resumed
    finally:
        await persistence.close()


@pytest.mark.asyncio
async def test_targeted_deferred_recovery_observes_concurrent_cancellation() -> None:
    from unittest.mock import AsyncMock

    from linktools.ai.execution import PendingDeferredCall, PendingToolContinuation
    from linktools.ai.runtime._recovery_coordinator import _RecoveryCoordinator
    from linktools.ai.storage import StoredPayload

    from .test_recovery_turn_admission import _TurnAdmission, _port

    port = _port(_TurnAdmission())
    waiting = replace(
        port.execution, status=ExecutionStatus.WAITING_DEFERRED,
        recovery=replace(port.execution.recovery, pending_tools=PendingToolContinuation(
            "run", calls=(PendingDeferredCall("call", "tool", StoredPayload.inline_json({})),),
        )),
    )
    cancelling = replace(waiting, status=ExecutionStatus.CANCELLING, revision=waiting.revision + 1,
                         recovery=replace(waiting.recovery, pending_tools=None))
    port.load_execution.side_effect = (waiting, cancelling)
    port.load_recovery_execution.return_value = cancelling
    port.claim_deferred_resume = AsyncMock()
    port.launch = AsyncMock()
    with pytest.raises(AIError) as caught:
        await _RecoveryCoordinator(port, None).recover_execution(
            waiting.execution_id, tenant_id="tenant", expected_revision=waiting.revision,
        )
    assert caught.value.code is ErrorCode.STORAGE_CONFLICT
    port.claim_deferred_resume.assert_not_awaited()
    port.launch.assert_not_awaited()
