#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-call capability facts survive dynamic selection and durable effect recovery."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

from linktools.ai.capability import ToolCallFailed
from linktools.ai.core import ExecutionStatus, ToolOperationStatus, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.codec import decode_domain, encode_domain
from linktools.ai.persistence.contracts import ToolOperationAdmission
from linktools.ai.runtime._pending_turn import _PendingExecutionTurn
from linktools.ai.runtime._tool import RuntimeToolOperationBridge
from linktools.ai.spec import CapabilityRef
from linktools.ai.storage import InMemoryObjectStore, StoredPayload

from ._runtime_test_helpers import tool_run_context
from .test_budget_tool_admission_boundary import admission_state, _commands, _execution
from .test_tool_execution_boundary import _request


def _reference(revision: int = 1) -> CapabilityRef:
    return CapabilityRef("mcp", "server", revision)


def _bridge(state: Persistence, *, recovery: bool = False) -> RuntimeToolOperationBridge:
    return RuntimeToolOperationBridge(
        state.checkpoint.tools, InMemoryObjectStore(), namespace="admitted-capability",
        tenant_id="tenant", execution_id="execution", agent_run_id="resumed" if recovery else "run",
        binding_digest="b" * 64, owner="owner", background_tasks=set(),
         recovery_agent_run_id="run" if recovery else None,
    )


async def _begin(bridge, contract, *, replay_safe=False, args=None):
    args = {} if args is None else args
    return await bridge.begin(
        tool_run_context(), ToolCallPart("work", args, tool_call_id="call"),
        ToolDefinition(name="work"), args, replay_safe, capability_ref=contract,
    )


def test_tool_admission_wire_retains_reference_without_connection_configuration() -> None:
    request = replace(_request(), capability_ref=_reference())
    assert decode_domain(encode_domain(request), ToolOperationAdmission) == request
    payload = encode_domain(request)
    del payload["fields"]["capability_ref"]
    assert decode_domain(payload, ToolOperationAdmission).capability_ref is None
    malformed = encode_domain(request)
    malformed["fields"]["capability_ref"]["fields"]["revision"] = True
    with pytest.raises(AIError) as invalid:
        decode_domain(malformed, ToolOperationAdmission)
    assert invalid.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    incomplete = encode_domain(request)
    del incomplete["fields"]["capability_ref"]["fields"]["revision"]
    with pytest.raises(AIError) as invalid:
        decode_domain(incomplete, ToolOperationAdmission)
    assert invalid.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_capability_fact_is_atomic_with_control_budget_and_native_event(admission_state: Persistence) -> None:
    state = admission_state
    request = replace(_request(), capability_ref=_reference())
    execution = await _execution(state)

    async def fail_native(group):
        record = await state.checkpoint.tools.get_operation_in_transaction(
            group.transaction(state.checkpoint.tools.state_store), "operation", tenant_id="tenant",
        )
        assert record.capability_ref == request.capability_ref
        raise RuntimeError("native write failed")

    with pytest.raises((RuntimeError, AIError)):
        await _commands(state).commit_tool_admission_boundary(
            request, execution=execution, known_new=True, write=fail_native,
        )
    assert await state.checkpoint.tools.get_operation("operation", tenant_id="tenant") is None
    assert (await state.execution.budgets.read("shared")).tool_calls == 0
    assert await state.execution.executions.get_observed("execution", tenant_id="tenant") == execution
    receipt = await _commands(state).commit_tool_admission_boundary(
        request, execution=execution, known_new=True,
    )
    assert receipt.operation.value.capability_ref == request.capability_ref
    assert (await state.execution.budgets.read("shared")).tool_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ("claimed", "pending", "unknown"))
async def test_unresolved_call_rejects_changed_revision_and_keeps_original_facts(
    admission_state: Persistence, status: str,
) -> None:
    repository = admission_state.checkpoint.tools
    request = replace(_request(), capability_ref=_reference(), replay_safe=False)
    record = await repository.admit(request)
    if status == "pending":
        record = await repository.defer("operation", tenant_id="tenant", owner="worker", fence=record.fence)
    elif status == "unknown":
        record = await repository.mark_effect_unknown(
            "operation", tenant_id="tenant", owner="worker", fence=record.fence,
            error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value,
        )
    request = replace(request, agent_run_id="resumed", recovery_agent_run_id="run")
    with pytest.raises(AIError) as changed:
        await repository.admit(replace(request, capability_ref=_reference(2)))
    assert changed.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
    assert await repository.get_by_call("resumed", "call", tenant_id="tenant") is None
    if status == "unknown":
        with pytest.raises(AIError) as unknown:
            await repository.admit(replace(request, capability_ref=_reference()))
        assert unknown.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
    else:
        admitted = await repository.admit(replace(request, capability_ref=_reference()))
        assert admitted.capability_ref == record.capability_ref
    stored = await repository.get_operation("operation", tenant_id="tenant")
    assert stored.capability_ref == record.capability_ref
    assert stored.status is (ToolOperationStatus.EFFECT_UNKNOWN if status == "unknown" else ToolOperationStatus.CLAIMED)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("failed", (False, True))
async def test_warm_and_cold_bridge_keep_original_terminal_result_and_reject_changed_arguments(
    tmp_path: Path, backend: str, failed: bool,
) -> None:
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "state.db")
    async with Persistence.open("admitted-capability", tenant_id="tenant", storage=storage) as state:
        bridge = _bridge(state)
        original = _reference()
        decision = await _begin(bridge, original)
        assert decision.capability_ref == original
        if failed:
            await bridge.fail(decision, ToolCallFailed("original failure"))
        else:
            await bridge.complete(decision, "original result")
        for candidate in (bridge, _bridge(state), _bridge(state, recovery=True)):
            replayed = await _begin(candidate, _reference(2), replay_safe=True)
            assert replayed.capability_ref == original
            assert replayed.replay_safe is False
            if failed:
                assert replayed.cached_error.message == "original failure"
            else:
                assert replayed.has_cached_result and replayed.cached_result == "original result"
            with pytest.raises(AIError) as changed:
                await _begin(candidate, original, args={"different": True})
            assert changed.value.code is ErrorCode.IDEMPOTENCY_CONFLICT


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ("claimed", "pending", "unknown"))
async def test_bridge_never_reinterprets_an_unresolved_warm_call(status: str) -> None:
    async with Persistence.open("admitted-capability", tenant_id="tenant",
                                storage=PersistenceStorage.in_memory()) as state:
        bridge = _bridge(state)
        original = _reference()
        decision = await _begin(bridge, original)
        if status == "pending":
            await bridge.defer(decision)
        elif status == "unknown":
            with pytest.raises(AIError) as unknown:
                await bridge.unknown(decision, RuntimeError("effect outcome uncertain"))
            assert unknown.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
        with pytest.raises(AIError) as changed:
            await _begin(bridge, _reference(2))
        assert changed.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        same_identity = _reference()
        if status == "unknown":
            with pytest.raises(AIError) as unknown:
                await _begin(bridge, same_identity)
            assert unknown.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
        else:
            resumed = await _begin(bridge, same_identity)
            assert resumed.capability_ref == original


@pytest.mark.asyncio
async def test_staged_call_retains_original_contract_through_defer_and_terminal_receipt() -> None:
    request = replace(_request(), capability_ref=_reference())
    execution = SimpleNamespace(
        execution_id="execution", recovery=SimpleNamespace(agent_run_id="run"),
        binding_digest=request.binding_digest, producer_claim_id="worker", producer_generation=1,
        status=ExecutionStatus.STARTED,
    )
    pending = _PendingExecutionTurn("tenant", None, None, execution,
                                  coordination=SimpleNamespace(execution_id="execution", owner_id="worker"))
    original = await pending.stage_tool_operation(request)
    assert original.capability_ref == request.capability_ref
    for status in (ToolOperationStatus.CLAIMED, ToolOperationStatus.PENDING):
        if status is ToolOperationStatus.PENDING:
            await pending.settle_tool_operation("operation", owner="worker", fence=original.fence, status=status)
        with pytest.raises(AIError) as changed:
            await pending.stage_tool_operation(replace(request, capability_ref=_reference(2)))
        assert changed.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
    resumed = await pending.stage_tool_operation(replace(request, capability_ref=_reference()))
    assert resumed.capability_ref == original.capability_ref
    await pending.settle_tool_operation("operation", owner="worker", fence=resumed.fence,
                                        status=ToolOperationStatus.COMPLETED,
                                        result_payload=StoredPayload.inline_text("original"))
    terminal = await pending.stage_tool_operation(replace(request, capability_ref=_reference(2)))
    assert terminal.capability_ref == original.capability_ref
    assert terminal.result_payload.value == "original"
    with pytest.raises(AIError) as changed:
        await pending.stage_tool_operation(replace(request, arguments_digest=canonical_sha256({"different": True})))
    assert changed.value.code is ErrorCode.IDEMPOTENCY_CONFLICT


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ("pending", "completed"))
@pytest.mark.parametrize("recovery", (False, True))
async def test_boundary_keeps_prior_effect_when_current_tool_is_read_only(status: str, recovery: bool) -> None:
    from pydantic_ai.toolsets import FunctionToolset
    from linktools.ai.runtime._tool_boundary import BoundaryToolset, ManagedToolDescriptor

    calls = []

    def work() -> str:
        calls.append("new implementation")
        return "changed"

    async with Persistence.open("admitted-capability", tenant_id="tenant",
                                storage=PersistenceStorage.in_memory()) as state:
        original = _bridge(state)
        decision = await _begin(original, CapabilityRef("tool", "work", 1), replay_safe=True)
        if status == "pending":
            await original.defer(decision)
        else:
            await original.complete(decision, "original result")
        bridge = _bridge(state, recovery=True) if recovery else original
        boundary = BoundaryToolset((FunctionToolset([work]),), {
            "work": ManagedToolDescriptor("none", "none", "business",
                                          capability_ref=CapabilityRef("tool", "work", 2)),
        }, id="changed", tool_operations=bridge)
        ctx = tool_run_context()
        if recovery:
            ctx = replace(ctx, run_id="resumed")
        tool = (await boundary.get_tools(ctx))["work"]
        if status == "pending":
            with pytest.raises(AIError) as error:
                await boundary.call_tool("work", {}, ctx, tool)
            assert error.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        else:
            assert await boundary.call_tool("work", {}, ctx, tool) == "original result"
        assert calls == []


@pytest.mark.asyncio
async def test_fresh_read_only_call_does_not_query_receipts() -> None:
    repository = SimpleNamespace()
    bridge = RuntimeToolOperationBridge(
        repository, InMemoryObjectStore(), namespace="fresh", tenant_id="tenant", execution_id="execution",
        agent_run_id="run", binding_digest="b" * 64, owner="owner", background_tasks=set(),
         fresh_execution=True,
    )
    assert await bridge.existing_call_ids(("new-call",)) == frozenset()
