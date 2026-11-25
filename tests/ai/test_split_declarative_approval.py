#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A declarative split approval is stoppable or explicitly resumed exactly once."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from linktools.ai.capability import AgentContext, CapabilityContribution, StepPersistenceCapability
from linktools.ai.core import ApprovalDecision, ApprovalStatus, Principal, TenantAuthorizationPolicy, ToolOperationStatus
from linktools.ai.execution import ApprovalDecisionRequest
from linktools.ai.persistence import Persistence, PersistencePlan, PersistenceStorage
from linktools.ai.persistence.contracts import deferred_resource_id
from linktools.ai.persistence.turns import SessionTurnKey
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import StorageRoute

from ._runtime_test_helpers import _Models, _UsageFunctionModel, _wait_for_committed, model_features


class _SessionPolicy(TenantAuthorizationPolicy):
    """Explicit host policy for this fixture's session."""


@pytest.mark.asyncio
@pytest.mark.parametrize("resolution", ("stop", "approve"))
async def test_split_declarative_approval_requires_a_trusted_resolution(tmp_path: Path, resolution: str) -> None:
    namespace = "split-declarative-approval"
    principal = Principal("reader", "default")
    calls: list[str] = []
    requests = 0

    def lookup(_context: AgentContext[None], value: str) -> str:
        calls.append(value)
        return value

    async def reply(messages, _info):
        nonlocal requests
        requests += 1
        if requests == 1:
            return ModelResponse(parts=[ToolCallPart("lookup", {"value": "fixture"}, "approve")])
        assert requests == 2 and resolution == "approve"
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        assert [(part.tool_call_id, part.content) for part in returns] == [("approve", "fixture")]
        return ModelResponse(parts=[TextPart("approved lookup finished")])

    features = RuntimeFeatures(namespace)
    tool = features.tool(lookup, effect_policy="none")
    tool.requires_approval = True
    features._register(CapabilityContribution.from_opaque("tool", "lookup", tool))
    features.capability(StepPersistenceCapability())
    features.agent("default", model="default", allow_tools=("lookup",))
    features = model_features(_Models(_UsageFunctionModel(reply)), features)
    step = StorageRoute.sqlite(tmp_path / "step.db")
    storage = PersistenceStorage(PersistencePlan(
        conversation=StorageRoute.sqlite(tmp_path / "conversation.db"), execution=step, checkpoint=step,
    ))
    async with Persistence.open(namespace, storage=storage) as state:
        async with Runtime.open(
            persistence=state, features=features, authorization=_SessionPolicy(principal.tenant_id),
        ) as runtime:
            await runtime.sessions.create_turn_session("chat", principal=principal)
            frames: asyncio.Queue[dict] = asyncio.Queue()
            detach = runtime.sessions.observe_turn_session("chat", frames.put_nowait)
            try:
                turn_id = await runtime.sessions.start_turn(
                    "chat", "look up a fixture", agent="default", principal=principal, idempotency_key="first",
                )
                async with asyncio.timeout(10):
                    while (frame := await frames.get())["type"] not in ("waiting", "terminal"):
                        pass
                    assert frame["type"] == "waiting" and frame["turn_id"] == turn_id, frame
                    await asyncio.gather(*tuple(runtime.sessions._turn_runner._tasks))
            finally:
                detach()

            key = SessionTurnKey("chat", turn_id)
            turn = await state.session_turns.get_turn(key)
            assert turn is not None and turn["state"] == "waiting" and turn["split_source_receipt"] is not None
            (operation,) = await state.checkpoint.tools.list_by_execution(turn_id, tenant_id=principal.tenant_id)
            assert operation.status is ToolOperationStatus.PENDING and operation.owner is None
            assert operation.tool_call_id == "approve" and operation.producer_generation == turn["generation"]
            approval_id = deferred_resource_id("approval-v1", principal.tenant_id, turn_id, turn_id, "approve")
            approval = await state.checkpoint.approvals.get(approval_id, tenant_id=principal.tenant_id)
            assert approval is not None and approval.status is ApprovalStatus.PENDING
            assert calls == []

            if resolution == "stop":
                assert await runtime.sessions.cancel_turn("chat", turn_id, principal=principal)
                stopped = await state.session_turns.get_turn(key)
                assert stopped is not None and stopped["state"] == "cancelled"
                assert requests == 1 and calls == []
                return

            pending, = await runtime.approvals.list(key, principal=principal)
            assert pending.approval_id == approval_id and pending.status is ApprovalStatus.PENDING
            await runtime.approvals.decide(key, ApprovalDecisionRequest(
                principal, approval_id, "approve-original", ApprovalDecision.APPROVE,
            ))
            ready = await state.session_turns.get_turn(key)
            assert ready is not None and ready["state"] == "ready" and requests == 1 and calls == []

            assert await runtime.sessions.recover_turn("chat", turn_id, principal=principal)
            terminal = await _wait_for_committed(
                lambda: state.session_turns.get_turn(key), lambda row: row is not None and row["state"] == "succeeded",
            )
            assert terminal["generation"] == turn["generation"] + 1
            settled, = await state.checkpoint.tools.list_by_execution(turn_id, tenant_id=principal.tenant_id)
            assert settled.tool_operation_id == operation.tool_operation_id
            assert settled.status is ToolOperationStatus.COMPLETED
            detail = await runtime.history.turn_result("chat", turn_id, principal=principal)
            assert detail["conversation_committed"] and detail["output"] == {"text": "approved lookup finished"}
            assert not await runtime.sessions.recover_turn("chat", turn_id, principal=principal)
            assert requests == 2 and calls == ["fixture"]
