#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deferred child continuations remain authorized by their originating parent scope."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from pydantic_ai.models.function import AgentInfo

from linktools.ai.capability import AgentContext, ExecutionCapability
from pydantic_ai_harness.step_persistence import StepPersistence
from tests.ai._runtime_test_helpers import native_checkpoint
from linktools.ai.core import ApprovalDecision, ExecutionStatus, Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ApprovalDecisionRequest, ExternalCallSucceeded, ExternalSupplyRequest
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import _Models, _UsageFunctionModel, _wait_for_committed
from .test_runtime_turn_lifecycle import _ObservedCache


pytestmark = pytest.mark.asyncio


class _Scenario:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.tool_entries = 0
        self.effects: list[str] = []
        self.child_requests = 0
        self.parent_requests = 0
        self.parent_returns: list[object] = []
        self.features = RuntimeFeatures("child-continuation")
        self.features.capability(ExecutionCapability(retain_history=True), scope="runtime")
        self.features.capability(native_checkpoint(), scope="runtime")
        self.features.model("parent", _Models(_UsageFunctionModel(self.parent)))
        self.features.model("child", _Models(_UsageFunctionModel(self.child)))
        self.features.tool(self.checked, effect_policy="none")
        self.features.agent("default", model="parent", allow_tools=(), allow_subagents=("child",))
        self.features.agent("child", model="child", allow_tools=("checked",))

    async def checked(self, context: AgentContext[object]) -> str:
        del context
        self.tool_entries += 1
        if self.kind == "external":
            raise CallDeferred()
        if self.tool_entries == 1:
            raise ApprovalRequired()
        self.effects.append("approved-effect")
        return "approved-effect"

    async def parent(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        self.parent_requests += 1
        self.parent_returns.extend(part.content for message in messages for part in message.parts
                                   if isinstance(part, ToolReturnPart))
        if any("child complete" in str(value) for value in self.parent_returns):
            return ModelResponse(parts=[TextPart("parent complete")])
        assert self.parent_requests <= 3, self.parent_returns
        return ModelResponse(parts=[ToolCallPart(
            "delegate_task", {"subagent_id": "child", "task": "perform checked work"},
            tool_call_id="child-invocation",
        )])

    async def child(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        self.child_requests += 1
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart("child complete")])
        return ModelResponse(parts=[ToolCallPart("checked", {}, tool_call_id="checked-call")])

    async def paused_child(self, runtime: Runtime, parent_id: str) -> str:
        children = await _wait_for_committed(
            lambda: runtime.executions.list_children(parent_id, principal=runtime.default_principal),
            lambda values: len(values) == 1,
        )
        child_id = children[0].execution_id
        await _wait_for_committed(
            lambda: runtime._state.execution.executions.get(child_id, tenant_id=runtime.tenant_id),
            lambda record: record is not None and record.status is ExecutionStatus.WAITING_DEFERRED,
        )
        backend = runtime._execution_service.runtime_backend()
        owner = backend._turn_owners.get(child_id)
        if owner is not None:
            await asyncio.wait_for(owner.closed.wait(), 5)
        assert not backend.worker_installed(child_id)
        return child_id

    async def resolve(self, runtime: Runtime, child_id: str) -> None:
        principal = Principal(runtime.default_principal.principal_id, runtime.tenant_id, "service")
        if self.kind == "approval":
            pending = await runtime.approvals.list(child_id, principal=principal)
            assert len(pending) == 1
            await runtime.approvals.decide(child_id, ApprovalDecisionRequest(
                principal=principal, approval_id=pending[0].approval_id,
                idempotency_key="resolve-child", decision=ApprovalDecision.APPROVE,
            ))
        else:
            pending = await runtime.external_calls.list(child_id, principal=principal)
            assert len(pending) == 1
            await runtime.external_calls.supply(child_id, ExternalSupplyRequest(
                principal=principal, call_id=pending[0].call_id,
                idempotency_key="resolve-child", resolution=ExternalCallSucceeded("external-effect"),
            ))

    def assert_effect_once(self) -> None:
        assert self.child_requests == 2
        if self.kind == "approval":
            assert self.effects == ["approved-effect"]
            assert self.tool_entries == 2
        else:
            assert self.effects == []
            assert self.tool_entries == 1


def _storage(tmp_path: Path) -> PersistenceStorage:
    cache = _ObservedCache()
    return PersistenceStorage.sqlite(tmp_path / "state.db", cache=cache, coordination=cache)


@pytest.mark.parametrize("kind", ("approval", "external"))
async def test_live_parent_authorizes_paused_child_continuation(tmp_path: Path, kind: str) -> None:
    scenario = _Scenario(kind)
    async with Persistence.open("child-continuation", storage=_storage(tmp_path)) as persistence:
        async with Runtime.attach(persistence=persistence, features=scenario.features) as runtime:
            parent = await runtime.agents.get().start("delegate", idempotency_key="parent")
            child_id = await scenario.paused_child(runtime, parent.execution_id)
            backend = runtime._execution_service.runtime_backend()
            assert backend.worker_installed(parent.execution_id)
            await scenario.resolve(runtime, child_id)
            assert (await parent.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
            children = await runtime.executions.list_children(parent.execution_id, principal=runtime.default_principal)
            assert [child.execution_id for child in children] == [child_id]
            assert children[0].status is ExecutionStatus.SUCCEEDED, scenario.parent_returns
            scenario.assert_effect_once()


@pytest.mark.parametrize("kind", ("approval", "external"))
async def test_cold_runtime_does_not_infer_child_continuation_authority(tmp_path: Path, kind: str) -> None:
    scenario = _Scenario(kind)
    async with Persistence.open("child-continuation", storage=_storage(tmp_path)) as persistence:
        async with Runtime.attach(persistence=persistence, features=scenario.features) as runtime:
            parent = await runtime.agents.get().start("delegate", idempotency_key="parent")
            child_id = await scenario.paused_child(runtime, parent.execution_id)
            await asyncio.wait_for(runtime.handoff(), 10)
        before = await persistence.execution.executions.get(child_id, tenant_id=runtime.tenant_id)
        async with Runtime.attach(persistence=persistence, features=scenario.features, auto_recover=False) as cold:
            with pytest.raises(AIError) as unavailable:
                await scenario.resolve(cold, child_id)
            assert unavailable.value.code is ErrorCode.STORAGE_CONFLICT
            assert unavailable.value.safe_details.get("phase") == "subagent_parent_unavailable"
            after = await persistence.execution.executions.get(child_id, tenant_id=cold.tenant_id)
            assert after == before
            assert scenario.effects == []
            assert scenario.tool_entries == 1 and scenario.child_requests == 1


@pytest.mark.parametrize("kind", ("approval", "external"))
async def test_explicit_parent_recovery_reattaches_original_paused_child(tmp_path: Path, kind: str) -> None:
    scenario = _Scenario(kind)
    async with Persistence.open("child-continuation", storage=_storage(tmp_path)) as persistence:
        async with Runtime.attach(persistence=persistence, features=scenario.features) as runtime:
            parent = await runtime.agents.get().start("delegate", idempotency_key="parent")
            child_id = await scenario.paused_child(runtime, parent.execution_id)
            await asyncio.wait_for(runtime.handoff(), 10)
        async with Runtime.attach(persistence=persistence, features=scenario.features, auto_recover=False) as recovered:
            with pytest.raises(AIError) as unavailable:
                await scenario.resolve(recovered, child_id)
            assert unavailable.value.safe_details.get("phase") == "subagent_parent_unavailable"
            await recovered.executions.recover(
                parent.execution_id, principal=recovered.default_principal,
                idempotency_key="recover-parent",
            )
            result = await recovered.executions.wait(
                parent.execution_id, principal=recovered.default_principal, timeout_seconds=10,
            )
            assert result.result.status is ExecutionStatus.SUCCEEDED
            children = await recovered.executions.list_children(
                parent.execution_id, principal=recovered.default_principal,
            )
            assert [child.execution_id for child in children] == [child_id]
            assert children[0].status is ExecutionStatus.SUCCEEDED, scenario.parent_returns
            scenario.assert_effect_once()
