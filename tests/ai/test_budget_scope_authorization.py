#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Existing budget scopes require authority over their durable owners."""

from collections.abc import AsyncIterator
from dataclasses import replace

import pytest
import pytest_asyncio

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import ExecutionStatus, Principal, RunBudget
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ExecutionRequest, Runtime, RuntimeStorage
from linktools.ai.runtime.state._codec import _decode_enveloped_domain, encode_domain, encode_envelope
from linktools.ai.runtime.state._contracts import ExecutionRecord
from linktools.ai.runtime.state._store import RecordQuery, StateTransaction
from linktools.ai.task import TaskBindingContract, TaskGraph

from ._runtime_test_helpers import RuntimeUsageModels


_BudgetRuntime = tuple[Runtime, RuntimeStorage, ExecutionRecord]
_TASK = TaskBindingContract("adapter.task", 1, "none", {}, None, 1, 0)


@pytest_asyncio.fixture
async def budget_runtime() -> AsyncIterator[_BudgetRuntime]:
    group = CapabilityGroup("scope-authorization")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    storage = RuntimeStorage.in_memory()
    async with Runtime.open("scope-authorization", models=RuntimeUsageModels(),
                            storage=storage, capabilities=(group,)) as runtime:
        root = await runtime.agents.get().start("root", budget=RunBudget(model_requests=20))
        assert (await root.wait()).result.status is ExecutionStatus.SUCCEEDED
        record = await storage.execution.executions.get(root.execution_id, tenant_id="default")
        assert record is not None
        graph = await runtime.tasks.bind().start(
            TaskGraph("owned", ()), idempotency_key="graph", budget=RunBudget(model_requests=20),
        )
        await graph.wait()
        await runtime.agents.get().create_session("session", principal=Principal("stranger", "default"))
        yield runtime, storage, record


async def _records(storage: RuntimeStorage) -> tuple[object, ...]:
    result = []
    for repository in (storage.execution.executions, storage.task.admissions, storage.conversation.sessions):
        result.append(await repository.state_store.read(lambda transaction: transaction.scan_records()))
    return tuple(result)


async def _attach(runtime: Runtime, record: ExecutionRecord, entry: str,
                  scope: str, principal: Principal, *, budget: RunBudget | None = None) -> object:
    service = runtime._execution_service
    request = ExecutionRequest("attached", principal, "attach", None, record.mode,
                               record.planning, record.thinking, budget=budget)
    if entry == "task":
        return await service.start_task(_TASK, principal=principal, input={},
                                        idempotency_key="attach", correlation={}, budget_scope_id=scope)
    if entry == "session":
        return await service.start_for_session("default", record.binding_digest, "session", request,
                                              binding_contract=record.binding, budget_scope_id=scope)
    operation = service.resolve_existing if entry == "resolve" else service.start
    return await operation(record.binding_digest, request, binding_contract=record.binding,
                           budget_scope_id=scope)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ("start", "session", "task", "resolve"))
@pytest.mark.parametrize("owner", ("execution", "graph"))
async def test_same_tenant_budget_owner_is_authorized_before_any_write(budget_runtime: _BudgetRuntime, entry: str, owner: str) -> None:
    runtime, storage, root = budget_runtime
    scope = root.budget_scope_id if owner == "execution" else "graph:owned"
    before = await _records(storage)
    with pytest.raises(AIError) as rejected:
        await _attach(runtime, root, entry, scope, Principal("stranger", "default"))
    assert rejected.value.code is ErrorCode.AUTHORIZATION_DENIED
    assert await _records(storage) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("graph:owned", "execution:foreign", "unattached", "graph:missing"))
async def test_cross_tenant_or_unowned_scope_is_not_attachment_authority(budget_runtime: _BudgetRuntime, scope: str) -> None:
    runtime, storage, root = budget_runtime
    await storage.execution.budgets.ensure(scope, RunBudget(model_requests=20))
    before = await _records(storage)
    principal = Principal("runtime", "other") if scope == "graph:owned" else runtime.default_principal
    with pytest.raises(AIError) as rejected:
        await _attach(runtime, root, "task", scope, principal)
    assert rejected.value.code is ErrorCode.AUTHORIZATION_DENIED
    assert await _records(storage) == before


@pytest.mark.asyncio
async def test_graph_owned_custom_task_adapter_attaches_matching_scope(budget_runtime: _BudgetRuntime) -> None:
    runtime, storage, root = budget_runtime
    before = await storage.execution.budgets.read("graph:owned")
    handle = await _attach(runtime, root, "task", "graph:owned", runtime.default_principal)
    attached = await storage.execution.executions.get(handle.execution_id, tenant_id="default")
    assert attached is not None and attached.budget_scope_id == "graph:owned"
    assert attached.principal_id == runtime.default_principal.principal_id
    assert await runtime._execution_service.budget_usage(handle.execution_id, principal=runtime.default_principal) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ("missing", "limits"))
async def test_graph_scope_missing_or_changed_limits_rejects_without_writes(budget_runtime: _BudgetRuntime, corruption: str) -> None:
    runtime, storage, root = budget_runtime
    usage = await storage.execution.budgets.read("graph:owned")

    async def corrupt(transaction: StateTransaction) -> None:
        records = await transaction.list_records(RecordQuery(kind="budget_scope"))
        for record in records:
            value = _decode_enveloped_domain(record.data, type(usage))
            if value.scope_id == "graph:owned":
                assert await transaction.delete_record(record.key_digest)
                if corruption == "limits":
                    payload = encode_domain(replace(value, limits=RunBudget(model_requests=21)))
                    await transaction.insert_record(replace(record, data=encode_envelope(payload)))
                return
        raise AssertionError("graph budget missing from fixture")

    await storage.execution.budgets.state_store.mutate(corrupt)
    before = await _records(storage)
    with pytest.raises(AIError) as rejected:
        await _attach(runtime, root, "task", "graph:owned", runtime.default_principal)
    assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert await _records(storage) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("", " ", 1))
async def test_resolve_existing_rejects_invalid_scope_ids(budget_runtime: _BudgetRuntime, scope: str) -> None:
    runtime, storage, root = budget_runtime
    before = await _records(storage)
    with pytest.raises(AIError) as rejected:
        await _attach(runtime, root, "resolve", scope, runtime.default_principal)
    assert rejected.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert await _records(storage) == before


@pytest.mark.asyncio
async def test_resolve_existing_rejects_fresh_budget_with_inherited_scope(budget_runtime: _BudgetRuntime) -> None:
    runtime, storage, root = budget_runtime
    before = await _records(storage)
    with pytest.raises(AIError) as rejected:
        await _attach(runtime, root, "resolve", "graph:owned", runtime.default_principal,
                      budget=RunBudget(model_requests=1))
    assert rejected.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert await _records(storage) == before


@pytest.mark.asyncio
async def test_retry_authorizes_retained_source_after_budget_root_is_retired(budget_runtime: _BudgetRuntime) -> None:
    runtime, storage, root = budget_runtime
    run = await runtime.executions.get(root.execution_id)
    retained = await run.retry("retained")
    assert (await retained.wait()).result.status is ExecutionStatus.SUCCEEDED

    async def retire_root(transaction: StateTransaction) -> None:
        for record in await transaction.list_records(RecordQuery(kind="execution")):
            execution = _decode_enveloped_domain(record.data, ExecutionRecord)
            if execution.execution_id == root.execution_id:
                assert await transaction.delete_record(record.key_digest)
                return
        raise AssertionError("root execution missing from fixture")

    await storage.execution.executions.state_store.mutate(retire_root)
    retried = await retained.retry("retry retained source")
    assert (await retried.wait()).result.status is ExecutionStatus.SUCCEEDED
    assert (await retried.budget_usage()).scope_id == root.budget_scope_id
    assert (await retried.budget_usage()).model_requests == 3
