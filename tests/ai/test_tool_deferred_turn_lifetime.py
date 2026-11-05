#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Human approval and external results outlive the worker which deferred them."""
import asyncio
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path

import pytest
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from linktools.ai.capability import AgentContext
from linktools.ai.core import ApprovalDecision, ApprovalStatus, ExecutionStatus, ExternalCallStatus, Principal
from linktools.ai.execution import ApprovalDecisionRequest, ExternalCallSucceeded, ExternalSupplyRequest
from linktools.ai.persistence import _checkpoint_repositories, _sql
from linktools.ai.runtime import _approval, _external

from ._runtime_test_helpers import _wait_for_committed
from .test_official_runtime_recovery import _runtime


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("approval", "denied", "external"))
async def test_deferred_result_resumes_after_cold_worker_and_long_delay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    entries = []
    effects = []
    returned = []

    async def checked(_context: AgentContext[None]) -> str:
        entries.append("entered")
        if kind == "external":
            raise CallDeferred()
        if len(entries) == 1:
            raise ApprovalRequired()
        effects.append("approved-effect")
        return "approved-effect"

    async def model(messages, _info):
        results = [part.content for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if results:
            returned.extend(results)
            return ModelResponse(parts=[TextPart("finished")])
        return ModelResponse(parts=[ToolCallPart("checked", {}, tool_call_id="deferred-call")])

    database = tmp_path / "deferred.db"
    async with _runtime(database, model, checked) as (runtime, persistence, _steps):
        execution = await runtime.agents.get().start("check", idempotency_key="start")
        waiting = await _wait_for_committed(
            lambda: persistence.execution.executions.get(execution.execution_id, tenant_id="default"),
            lambda record: record is not None and record.status is ExecutionStatus.WAITING_DEFERRED,
        )
        await asyncio.wait_for(runtime.handoff(), 5)
        assert effects == [] and len(entries) == 1

    future = datetime.now(timezone.utc) + timedelta(days=7)

    class Later(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return future if tz is None else future.astimezone(tz)

    async def transaction_now(_clock: _sql._SqlTransactionClock) -> datetime:
        return future

    # Both durable transaction time and callback/tool time advance with no live worker.
    monkeypatch.setattr(_sql._SqlTransactionClock, "now", transaction_now)
    for module in (_checkpoint_repositories, _approval, _external):
        monkeypatch.setattr(module, "datetime", Later)
    async with _runtime(database, model, checked) as (runtime, persistence, _steps):
        backend = runtime._execution_service.runtime_backend()
        assert not backend.worker_installed(execution.execution_id)
        current = await persistence.execution.executions.get(execution.execution_id, tenant_id="default")
        assert current == waiting
        principal = Principal("resolver", "default", "service")
        if kind != "external":
            pending = await runtime.approvals.list(execution.execution_id, principal=principal)
            assert len(pending) == 1 and pending[0].status is ApprovalStatus.PENDING
            await runtime.approvals.decide(execution.execution_id, ApprovalDecisionRequest(
                principal, pending[0].approval_id, "resolve",
                ApprovalDecision.DENY if kind == "denied" else ApprovalDecision.APPROVE,
            ))
        else:
            pending = await runtime.external_calls.list(execution.execution_id, principal=principal)
            assert len(pending) == 1 and pending[0].status == ExternalCallStatus.PENDING.value
            await runtime.external_calls.supply(execution.execution_id, ExternalSupplyRequest(
                principal, pending[0].call_id, "resolve", ExternalCallSucceeded("external-effect"),
            ))
        outcome = await runtime.executions.wait(execution.execution_id, principal=runtime.default_principal,
                                                timeout_seconds=5)
        assert outcome.result.status is ExecutionStatus.SUCCEEDED, outcome.result
        assert outcome.result.output == {"text": "finished"}
        current = await persistence.execution.executions.get(execution.execution_id, tenant_id="default")
        assert current.producer_generation > waiting.producer_generation
        if kind == "approval":
            assert entries == ["entered", "entered"] and effects == ["approved-effect"]
            assert returned == ["approved-effect"]
            assert await runtime.approvals.list(execution.execution_id, principal=principal) == ()
        elif kind == "denied":
            assert entries == ["entered"] and effects == []
            assert len(returned) == 1 and "denied" in str(returned[0]).lower()
            assert await runtime.approvals.list(execution.execution_id, principal=principal) == ()
        else:
            assert entries == ["entered"] and effects == []
            assert returned == ["external-effect"]
            assert await runtime.external_calls.list(execution.execution_id, principal=principal) == ()
