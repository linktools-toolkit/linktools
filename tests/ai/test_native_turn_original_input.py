#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A same-store native Session tool sees the detached admitted scalar input."""

import pytest
from pydantic_ai.exceptions import ApprovalRequired
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from linktools.ai.capability import AgentContext, SessionCapability
from linktools.ai.core import ApprovalDecision, ExecutionStatus, Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ApprovalDecisionRequest
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


class _HostAuthorization(TenantAuthorizationPolicy):
    """Authorize the synthetic host's native Session."""


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("fresh", "reopen", "empty"))
async def test_native_same_store_tool_sees_original_admitted_scalar(tmp_path, mode):
    scalar = '' if mode == "empty" else '{"request_id":"receipt","data":"  原文  "}'
    contexts = []

    async def inspect(context: AgentContext[None]):
        contexts.append(context)
        if mode == "reopen" and len(contexts) == 1:
            raise ApprovalRequired()
        return "validated"

    async def model(messages, _info):
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("inspect", {}, tool_call_id="inspect")])

    features = RuntimeFeatures("native-turn-original-input")
    features.capability(SessionCapability(checkpoints=True), scope="runtime")
    features.tool(inspect, effect_policy="replay_safe")
    features.agent("default", model="default", allow_tools=("inspect",))
    model_features(_Models(_UsageFunctionModel(model)), features)
    storage = PersistenceStorage.sqlite(tmp_path / "state.db")

    def open_host():
        return Runtime.open("native-turn-original-input", storage=storage,
                            features=features, authorization=_HostAuthorization())

    async with open_host() as runtime:
        principal = Principal("owner", runtime.tenant_id)
        await runtime.sessions.create_turn_session("chat", principal=principal)
        if not scalar:
            with pytest.raises(AIError) as rejected:
                await runtime.sessions.start_turn("chat", scalar, agent="default", principal=principal,
                                             idempotency_key="admit")
            assert rejected.value.code is ErrorCode.REQUEST_FIELD_INVALID
            assert not contexts
            return
        turn = await runtime.sessions.start_turn("chat", scalar, agent="default", principal=principal,
                                            idempotency_key="admit")
        if mode == "reopen":
            assert (await turn.wait()).status is ExecutionStatus.WAITING_DEFERRED
            await runtime.handoff()
        else:
            assert (await turn.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
    if mode == "reopen":
        async with open_host() as runtime:
            approval, = await runtime.approvals.list(turn.key, principal=principal)
            await runtime.approvals.decide(turn.key, ApprovalDecisionRequest(
                principal, approval.approval_id, "approve", ApprovalDecision.APPROVE,
            ))
            recovered = await runtime.sessions.start_turn("chat", scalar, agent="default", principal=principal,
                                                     idempotency_key="admit")
            assert recovered.key == turn.key
            assert (await recovered.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
    assert len(contexts) == (2 if mode == "reopen" else 1)
    for context in contexts:
        assert context.original_input == {
            "version": 1, "prompt": {"kind": "text", "text": scalar}, "files": [], "attachments": [],
        }
        assert (context.session_id, context.execution_id, context.principal) == ("chat", turn.key.turn_id, principal)
