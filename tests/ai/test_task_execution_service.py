#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Callable execution settlement does not need an Agent host or model configuration."""

from pathlib import Path
from inspect import Parameter, signature

import pytest

from linktools.ai.core import ExecutionStatus, Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionService, TaskExecutionService
from linktools.ai.persistence import PersistenceDomain, Persistence, PersistenceStorage
from linktools.ai.persistence.commands import StoredExecutionManagementAdapter
from linktools.ai.task import TaskBindingContract


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
async def test_callable_settlement_round_trips_without_agent_host(tmp_path: Path, outcome: str) -> None:
    principal = Principal("worker", "tenant")
    binding = TaskBindingContract("callable", 1, "none", {"kind": "json"}, None, 1, 0)
    state = Persistence(PersistenceStorage.from_root(tmp_path, exclusive_writer=True))
    await state.initialize(namespace="independent-task", tenant_id="tenant")
    try:
        service = TaskExecutionService(
            state.execution, TenantAuthorizationPolicy(),
            port=StoredExecutionManagementAdapter(state.execution, state.object_store(PersistenceDomain.EXECUTION)),
        )
        started = await service.start_task(
            binding, principal=principal, input={"value": 3}, idempotency_key="start", correlation={},
        )
        attempt = await service.claim_task_attempt(started.execution_id, principal=principal)
        if outcome == "success":
            result = await service.complete_task(
                started.execution_id, principal=principal, output={"value": 6}, attempt=attempt,
            )
            assert result.output == {"value": 6}
            status = ExecutionStatus.SUCCEEDED
        elif outcome == "failure":
            await service.fail_task(
                started.execution_id, principal=principal,
                error=AIError(ErrorCode.TASK_NODE_FAILED), attempt=attempt,
            )
            status = ExecutionStatus.FAILED
        else:
            await service.cancel_task(started.execution_id, principal=principal)
            status = ExecutionStatus.CANCELLED
        assert (await service.inspect(started.execution_id, principal=principal)).status is status
    finally:
        await state.close()

    reopened = Persistence(PersistenceStorage.from_root(tmp_path, exclusive_writer=True))
    await reopened.initialize(namespace="independent-task", tenant_id="tenant")
    try:
        service = TaskExecutionService(
            reopened.execution, TenantAuthorizationPolicy(),
            port=StoredExecutionManagementAdapter(reopened.execution, reopened.object_store(PersistenceDomain.EXECUTION)),
        )
        replay = await service.start_task(
            binding, principal=principal, input={"value": 3}, idempotency_key="start", correlation={},
        )
        assert replay.execution_id == started.execution_id
        assert (await service.inspect(replay.execution_id, principal=principal)).status is status
        if outcome == "success":
            assert (await service.result(replay.execution_id, principal=principal)).output == {"value": 6}
        with pytest.raises(AIError) as denied:
            await service.inspect(replay.execution_id, principal=Principal("other", "other-tenant"))
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_callable_result_keeps_dependency_hold_until_handoff() -> None:
    principal = Principal("worker", "tenant")
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="task-handoff", tenant_id="tenant")
    released: set[str] = set()

    async def release(execution_id: str, *, tenant_id: str, terminal_commit: bool = False) -> bool:
        released.add(execution_id)
        return True

    try:
        service = TaskExecutionService(
            state.execution, TenantAuthorizationPolicy(),
            port=StoredExecutionManagementAdapter(state.execution, state.object_store(PersistenceDomain.EXECUTION)),
            release_terminal=release,
        )
        started = await service.start_task(
            TaskBindingContract("callable", 1, "none", {"kind": "json"}, None, 1, 0),
            principal=principal, input={}, idempotency_key="start", correlation={},
        )
        await service.acquire_dependency_hold(started.execution_id, tenant_id="tenant", hold_id="graph-node")
        await service.complete_task(started.execution_id, principal=principal, output="result")
        assert (await service.result(started.execution_id, principal=principal)).output == "result"
        assert started.execution_id not in released
        await service.release_dependency_hold(started.execution_id, tenant_id="tenant", hold_id="graph-node")
        assert started.execution_id in released
    finally:
        await state.close()


def test_execution_handoff_protocol_includes_existing_terminal_commit_keyword() -> None:
    for owner in (ExecutionService, TaskExecutionService):
        contract = signature(owner.request_terminal_handoff)
        parameter = contract.parameters["terminal_commit"]
        assert parameter.kind is Parameter.KEYWORD_ONLY
        assert parameter.default is False
        contract.bind(None, "execution", tenant_id="tenant", terminal_commit=True)
