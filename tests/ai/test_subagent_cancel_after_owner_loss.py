#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Business cancellation remains available after a child worker loses ownership."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from linktools.ai.capability import ExecutionCapability
from pydantic_ai_harness.step_persistence import StepPersistence
from tests.ai._runtime_test_helpers import native_checkpoint
from linktools.ai.core import (
    ExecutionStatus, OperationKind, OperationLedgerInput, OperationStatus, ResourceKind,
    canonical_sha256, principal_identity_payload,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import CancelExecutionRequest
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import Lease, TurnCoordination

from ._runtime_test_helpers import _Models, _UsageFunctionModel
from .test_runtime_turn_lifecycle import _ObservedCache


@pytest.mark.asyncio
async def test_explicit_parent_cancel_reaches_child_after_owner_loss(tmp_path: Path) -> None:
    entered = asyncio.Event()
    calls = {"parent": 0, "child": 0}

    async def parent(messages, info):
        calls["parent"] += 1
        return ModelResponse(parts=[ToolCallPart(
            "delegate_task", {"subagent_id": "child", "task": "work"}, tool_call_id="child-call",
        )])

    async def child(messages, info):
        calls["child"] += 1
        entered.set()
        await asyncio.Event().wait()

    class LostChild(_ObservedCache):
        lost: Lease | None = None

        async def renew_turn(self, lease: Lease, *, lease_seconds: float) -> TurnCoordination | None:
            if lease == self.lost:
                return None
            return await super().renew_turn(lease, lease_seconds=lease_seconds)

    features = RuntimeFeatures("child-loss-business-cancel")
    features.model("parent", _Models(_UsageFunctionModel(parent)))
    features.model("child", _Models(_UsageFunctionModel(child)))
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.capability(native_checkpoint(), scope="runtime")
    features.agent("default", model="parent", allow_tools=(), allow_subagents=("child",))
    features.agent("child", model="child", allow_tools=())
    cache = LostChild()
    storage = PersistenceStorage.sqlite(tmp_path / "state.db", cache=cache, coordination=cache)
    verified = False
    # Existing checkpoint cancellation cannot confirm a lost active producer's
    # effects. Keep its failure visible at close rather than erase the UNKNOWN.
    with pytest.raises(AIError) as shutdown:
        async with Persistence.open("child-loss-business-cancel", storage=storage) as persistence, Runtime.attach(
            persistence=persistence, features=features,
        ) as runtime:
            execution = await runtime.agents.get().start("delegate", idempotency_key="parent-request")
            await asyncio.wait_for(entered.wait(), 5)
            backend = runtime._execution_service.runtime_backend()
            parent_owner = backend._turn_owners[execution.execution_id]
            children = await runtime.executions.list_children(execution.execution_id,
                                                             principal=runtime.default_principal)
            assert len(children) == 1
            child_id = children[0].execution_id
            child_owner = backend._turn_owners[child_id]
            cache.lost = child_owner.coordination.lease
            await asyncio.wait_for(child_owner.closed.wait(), 5)
            await asyncio.wait_for(parent_owner.closed.wait(), 5)
            for execution_id in (execution.execution_id, child_id):
                record = await persistence.execution.executions.get(execution_id, tenant_id=persistence.tenant_id)
                assert record.status is ExecutionStatus.STARTED
            with pytest.raises(AIError) as observation:
                await runtime.executions.inspect(child_id, principal=runtime.default_principal)
            assert observation.value.code is ErrorCode.STORAGE_CONFLICT
            assert observation.value.safe_details.get("phase") == "turn_owner_lost"

            request = CancelExecutionRequest(runtime.default_principal, "explicit-parent-cancel", force=True)
            with pytest.raises(AIError):
                await asyncio.wait_for(runtime.executions.cancel(execution.execution_id, request), 5)
            for execution_id in (execution.execution_id, child_id):
                record = await persistence.execution.executions.get(execution_id, tenant_id=persistence.tenant_id)
                assert record.status is ExecutionStatus.CANCELLING
                assert not backend.worker_installed(execution_id)
            unknown = await persistence.execution.operations.list_pending(
                ResourceKind.EXECUTION, child_id, tenant_id=persistence.tenant_id,
                limit=100, states=frozenset({OperationStatus.EFFECT_UNKNOWN}),
            )
            assert len(unknown) == 1
            assert unknown[0].operation_kind is OperationKind.EXECUTION_CANCEL
            assert calls == {"parent": 1, "child": 1}
            verified = True
    assert verified
    assert shutdown.value.code is ErrorCode.STORAGE_CONFLICT


@pytest.mark.asyncio
async def test_terminal_child_cleanup_preserves_prior_failed_cancel_receipt(tmp_path: Path) -> None:
    entered, finish = asyncio.Event(), asyncio.Event()

    async def parent(messages, info):
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart("parent complete")])
        return ModelResponse(parts=[ToolCallPart(
            "delegate_task", {"subagent_id": "child", "task": "work"}, tool_call_id="child-call",
        )])

    async def child(messages, info):
        entered.set()
        await finish.wait()
        return ModelResponse(parts=[TextPart("child complete")])

    features = RuntimeFeatures("child-terminal-cancel")
    features.model("parent", _Models(_UsageFunctionModel(parent)))
    features.model("child", _Models(_UsageFunctionModel(child)))
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.agent("default", model="parent", allow_tools=(), allow_subagents=("child",))
    features.agent("child", model="child", allow_tools=())
    async with Persistence.open("child-terminal-cancel", storage=PersistenceStorage.sqlite(
        tmp_path / "terminal.db",
    )) as persistence, Runtime.attach(persistence=persistence, features=features) as runtime:
        execution = await runtime.agents.get().start("delegate", idempotency_key="parent-request")
        await asyncio.wait_for(entered.wait(), 5)
        children = await runtime.executions.list_children(execution.execution_id,
                                                         principal=runtime.default_principal)
        child_id = children[0].execution_id
        key = "subagent-cancel:" + canonical_sha256({
            "version": 1, "parent_execution_id": execution.execution_id, "child_execution_id": child_id,
        })
        request = CancelExecutionRequest(runtime.default_principal, key, force=True)
        now = datetime.now(timezone.utc)
        failed = await persistence.execution.operations.append(OperationLedgerInput(
            key, persistence.tenant_id, ResourceKind.EXECUTION,
            child_id, child_id, OperationKind.EXECUTION_CANCEL, OperationStatus.FAILED,
            canonical_sha256({"action": "execution.cancel", "principal": principal_identity_payload(request.principal),
                              "execution_id": child_id, "force": True}),
            None, None, ErrorCode.INTERNAL_ERROR.value, True, now, now,
        ))
        finish.set()
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        with pytest.raises(AIError) as replay:
            await runtime.executions.cancel(child_id, request)
        assert replay.value.code is ErrorCode.INTERNAL_ERROR
        backend = runtime._execution_service.runtime_backend()
        await backend._subagent_dispatcher.cancel_child(
            child_id, parent_execution_id=execution.execution_id, principal=runtime.default_principal,
        )
        assert await persistence.execution.operations.get(
            failed.operation_id, tenant_id=persistence.tenant_id,
        ) == failed
        assert (await runtime.executions.inspect(child_id, principal=runtime.default_principal)).status is ExecutionStatus.SUCCEEDED
