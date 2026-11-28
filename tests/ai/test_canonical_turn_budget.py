#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical turn budgets fence effects and survive exact request replay."""

from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from linktools.ai.capability import AgentContext
from linktools.ai.core import RunBudget, TenantAuthorizationPolicy, ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence.turns import SessionTurnKey

from ._runtime_test_helpers import _wait_for_committed
from .test_official_runtime_recovery import _runtime


class _SessionAuthorization(TenantAuthorizationPolicy):
    """Use the explicit host permission boundary for SQL conversations."""


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", (0, 1))
async def test_canonical_budget_guards_effect_and_replayed_request_does_not_debit(tmp_path: Path, limit: int) -> None:
    invocations = []
    model_requests = []
    persistence = None

    async def work(context: AgentContext[None]) -> str:
        usage = await persistence.execution.budgets.read("turn:" + context.execution_id)
        assert usage.tool_calls == 1
        (operation,) = await persistence.checkpoint.tools.list_by_agent_run(
            context.execution_id, tenant_id=context.principal.tenant_id, execution_id=context.execution_id,
        )
        assert operation.status is ToolOperationStatus.CLAIMED
        invocations.append(context.execution_id)
        return "done"

    async def model(messages, _info):
        model_requests.append(None)
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart("complete")])
        return ModelResponse(parts=[ToolCallPart("work", {}, tool_call_id="budgeted-call")])

    database = tmp_path / "canonical-budget.db"
    limits = RunBudget(tool_calls=limit, model_requests=3)
    policy = _SessionAuthorization()
    async with _runtime(database, model, work, authorization=policy) as (runtime, persistence, _steps):
        principal = runtime.default_principal
        await runtime.sessions.create_turn_session("chat", principal=principal)
        turn_id = (await runtime.sessions.start_turn(
            "chat", "work", agent="default", principal=principal, idempotency_key="same-send", budget=limits,
        )).key.turn_id
        key = SessionTurnKey("chat", turn_id)
        terminal = await _wait_for_committed(
            lambda: persistence.session_turns.get_turn(key),
            lambda row: row is not None and row["state"] in ("failed", "succeeded", "cancelled"),
        )
        assert terminal["state"] == ("succeeded" if limit else "failed"), terminal["result_json"]
        if not limit:
            assert terminal["result_json"]["error_code"] == ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED.value
        usage = await runtime.sessions.budget_usage("chat", turn_id, principal=principal)
        assert usage is not None
        assert usage.limits == limits and usage.tool_calls == limit
        assert usage.model_requests == len(model_requests)
        assert invocations == ([turn_id] if limit else [])
        operations = await persistence.checkpoint.tools.list_by_agent_run(
            turn_id, tenant_id=principal.tenant_id, execution_id=turn_id,
        )
        assert [operation.status for operation in operations] == ([ToolOperationStatus.COMPLETED] if limit else [])
        assert await persistence.execution.executions.get(turn_id, tenant_id=principal.tenant_id) is None

    observed_requests = len(model_requests)
    async with _runtime(database, model, work, authorization=policy) as (runtime, persistence, _steps):
        principal = runtime.default_principal
        assert (await runtime.sessions.start_turn(
            "chat", "work", agent="default", principal=principal, idempotency_key="same-send", budget=limits,
        )).key.turn_id == turn_id
        for changed in (RunBudget(tool_calls=limit + 1, model_requests=3), None):
            with pytest.raises(AIError) as conflict:
                (await runtime.sessions.start_turn(
                    "chat", "work", agent="default", principal=principal, idempotency_key="same-send", budget=changed,
                )).key.turn_id
            assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        assert await runtime.sessions.budget_usage("chat", turn_id, principal=principal) == usage
        with pytest.raises(AIError) as denied:
            await runtime.sessions.budget_usage("other-chat", turn_id, principal=principal)
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert len(model_requests) == observed_requests
        assert invocations == ([turn_id] if limit else [])
