#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mixed Task reads retain authorization and producer ownership with bounded IO."""

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import event

from linktools.ai.core import ExecutionStatus, Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import TaskExecutionService
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence.commands import StoredExecutionManagementAdapter
from linktools.ai.task import TaskBindingContract
from linktools.ai.task.engine._reads import TaskExecutionReads

from .test_incremental_projection_boundaries import _storage
from .test_runtime_persistence_evidence import _execution

pytestmark = pytest.mark.asyncio
_PRINCIPAL = Principal("principal", "tenant", "service")


def _service(state, *, authorization=None, release=None):
    return TaskExecutionService(
        state.execution, authorization or TenantAuthorizationPolicy(),
        port=StoredExecutionManagementAdapter(
            state.execution, state.object_store(PersistenceDomain.EXECUTION),
        ),
        release_terminal=release,
    )


async def _seed(state, *, agent=False):
    record = _execution()
    if not agent:
        record = replace(
            record, binding=TaskBindingContract("callable", 1, "none", {"kind": "json"}, None, 1, 0),
            mode=None, planning=None, thinking=None, agent_run_seq=0, conversation=False,
        )
    await state.execution.executions.create_execution(record)
    return record


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("registered", (False, True))
async def test_task_inspect_uses_one_authorized_storage_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, registered: bool,
) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="task-read", tenant_id="tenant")
    try:
        record = await _seed(state)
        repository = state.execution.executions
        get = repository.get_fresh
        reads = []
        statements = []
        transactions = {"begin": 0, "commit": 0}

        async def tracked_get(*args, **kwargs):
            reads.append((args, kwargs))
            return await get(*args, **kwargs)

        monkeypatch.setattr(repository, "get_fresh", tracked_get)
        engine = repository.state_store.storage_group.context.engine.sync_engine if backend == "sqlite" else None

        def capture(connection, cursor, statement, parameters, context, many):
            statements.append(statement)

        def begin(connection):
            transactions["begin"] += 1

        def commit(connection):
            transactions["commit"] += 1

        if engine is not None:
            event.listen(engine, "begin", begin)
            event.listen(engine, "commit", commit)
            event.listen(engine, "before_cursor_execute", capture)
        try:
            ports = {"unrelated-agent": SimpleNamespace(service=None)} if registered else {}
            view = await TaskExecutionReads(_service(state), ports).inspect("execution", principal=_PRINCIPAL)
        finally:
            if engine is not None:
                event.remove(engine, "before_cursor_execute", capture)
                event.remove(engine, "begin", begin)
                event.remove(engine, "commit", commit)
        assert view.execution_id == record.execution_id
        assert view.binding_kind == "task"
        assert view.status is ExecutionStatus.STARTED
        assert view.root_execution_id == record.root_execution_id
        selects = sum(sql.lstrip().upper().startswith("SELECT") for sql in statements)
        print({"backend": backend, "registered": registered, "get_fresh": len(reads),
               "select": selects, **transactions})
        assert len(reads) == 1
        assert selects <= 1
    finally:
        await state.close()


async def test_task_inspect_refreshes_state_and_ownership_between_requests() -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="task-read", tenant_id="tenant")
    try:
        record = await _seed(state)
        reads = TaskExecutionReads(_service(state), {"unrelated": SimpleNamespace(service=None)})
        assert (await reads.inspect("execution", principal=_PRINCIPAL)).status is ExecutionStatus.STARTED
        current = replace(record, revision=record.revision + 1, status=ExecutionStatus.WAITING_RETRY)
        assert await state.execution.executions.compare_and_swap(
            "execution", tenant_id="tenant", expected_revision=record.revision, next_record=current,
        )
        assert (await reads.inspect("execution", principal=_PRINCIPAL)).status is ExecutionStatus.WAITING_RETRY
        assert await state.execution.executions.compare_and_swap(
            "execution", tenant_id="tenant", expected_revision=current.revision,
            next_record=replace(current, revision=current.revision + 1, principal_id="other"),
        )
        with pytest.raises(AIError) as raised:
            await reads.inspect("execution", principal=_PRINCIPAL)
        assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        await state.close()


@pytest.mark.parametrize("derived", (False, True))
@pytest.mark.parametrize("change_owner", (False, True))
async def test_task_inspect_preserves_awaited_policy_freshness(derived: bool, change_owner: bool) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="task-read", tenant_id="tenant")
    try:
        record = await _seed(state)
        policy = TenantAuthorizationPolicy()
        changed = False

        class Authorization(TenantAuthorizationPolicy if derived else object):
            async def authorize(self, principal, action, resource):
                nonlocal changed
                await policy.authorize(principal, action, resource)
                await asyncio.sleep(0)
                if changed:
                    return
                changed = True
                await state.execution.executions.compare_and_swap(
                    "execution", tenant_id="tenant", expected_revision=record.revision,
                    next_record=replace(
                        record, revision=record.revision + 1, status=ExecutionStatus.WAITING_RETRY,
                        principal_id="other" if change_owner else record.principal_id,
                    ),
                )

        reads = TaskExecutionReads(_service(state, authorization=Authorization()), {"unrelated": object()})
        if change_owner:
            with pytest.raises(AIError) as raised:
                await reads.inspect("execution", principal=_PRINCIPAL)
            assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED
        else:
            assert (await reads.inspect("execution", principal=_PRINCIPAL)).status is ExecutionStatus.WAITING_RETRY
    finally:
        await state.close()


@pytest.mark.parametrize("route", ("direct", "root"))
@pytest.mark.parametrize("denial", ("none", "task", "producer", "missing"))
async def test_agent_inspect_requires_both_authorities_and_returns_producer_view(route: str, denial: str) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="task-read", tenant_id="tenant")
    try:
        record = replace(_execution(), root_execution_id="root")
        await state.execution.executions.create_execution(record)
        reached = []
        producer_view = object()

        class Authorization:
            async def authorize(self, principal, action, resource):
                reached.append("task")
                if denial == "task":
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                await TenantAuthorizationPolicy().authorize(principal, action, resource)

        class Producer:
            async def inspect(self, execution_id, *, principal):
                assert execution_id == "execution" and principal == _PRINCIPAL
                reached.append("producer")
                if denial == "producer":
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                return producer_view

        key = "execution" if route == "direct" else "root"
        ports = {key: SimpleNamespace(service=Producer())} if denial != "missing" else {"unrelated": object()}
        reads = TaskExecutionReads(_service(state, authorization=Authorization()), ports)
        if denial == "none":
            assert await reads.inspect("execution", principal=_PRINCIPAL) is producer_view
            assert reached == ["task", "producer"]
        else:
            with pytest.raises(AIError) as raised:
                await reads.inspect("execution", principal=_PRINCIPAL)
            assert raised.value.code is (ErrorCode.BINDING_NOT_REGISTERED if denial == "missing"
                                         else ErrorCode.AUTHORIZATION_DENIED)
            assert reached == (["task", "producer"] if denial == "producer" else ["task"])
    finally:
        await state.close()


@pytest.mark.parametrize("cancel", (False, True))
async def test_inspect_finishes_pending_handoff_without_requesting_a_new_one(cancel: bool) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="task-read", tenant_id="tenant")
    entered, proceed = asyncio.Event(), asyncio.Event()
    released = []
    query = None

    class Authorization:
        async def authorize(self, principal, action, resource):
            entered.set()
            await proceed.wait()
            await TenantAuthorizationPolicy().authorize(principal, action, resource)

    async def release(execution_id, *, tenant_id):
        released.append((execution_id, tenant_id))
        return True

    try:
        await _seed(state)
        service = _service(state, authorization=Authorization(), release=release)
        reads = TaskExecutionReads(service, {"unrelated": object()})
        proceed.set()
        await reads.inspect("execution", principal=_PRINCIPAL)
        assert released == []
        proceed.clear()
        entered.clear()
        query = asyncio.create_task(reads.inspect("execution", principal=_PRINCIPAL))
        await asyncio.wait_for(entered.wait(), 2)
        await service.request_terminal_handoff("execution", tenant_id="tenant")
        assert released == []
        if cancel:
            query.cancel()
            with pytest.raises(asyncio.CancelledError):
                await query
        else:
            proceed.set()
            assert (await query).status is ExecutionStatus.STARTED
        assert released == [("execution", "tenant")]
    finally:
        proceed.set()
        if query is not None and not query.done():
            query.cancel()
            await asyncio.gather(query, return_exceptions=True)
        await state.close()


@pytest.mark.parametrize("agent", (False, True))
@pytest.mark.parametrize("method", ("result", "wait", "result_payload_size", "list_children", "budget_usage"))
async def test_noninspect_reads_call_selected_service_operation(
    monkeypatch: pytest.MonkeyPatch, agent: bool, method: str,
) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="task-read", tenant_id="tenant")
    try:
        await _seed(state, agent=agent)
        service = _service(state)
        selected = SimpleNamespace() if agent else service
        marker = object()
        calls = []

        async def operation(execution_id, *, principal, **kwargs):
            calls.append((execution_id, principal, kwargs))
            return marker

        monkeypatch.setattr(selected, method, operation, raising=False)
        ports = {"execution": SimpleNamespace(service=selected)}
        reads = TaskExecutionReads(service, ports)
        kwargs = {"timeout_seconds": 0.25} if method == "wait" else {}
        assert await getattr(reads, method)("execution", principal=_PRINCIPAL, **kwargs) is marker
        assert calls == [("execution", _PRINCIPAL, kwargs)]

        async def rejected(*args, **kwargs):
            raise AIError(ErrorCode.EXECUTION_NOT_READY)

        monkeypatch.setattr(selected, method, rejected)
        with pytest.raises(AIError) as raised:
            await getattr(reads, method)("execution", principal=_PRINCIPAL, **kwargs)
        assert raised.value.code is ErrorCode.EXECUTION_NOT_READY
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_mixed_graph_results_preserve_callable_failure_and_agent_output(tmp_path: Path, backend: str) -> None:
    from pydantic_ai.messages import ModelResponse, TextPart

    from linktools.ai.core import TaskStatus
    from linktools.ai.inputs import AgentTaskInput
    from linktools.ai.runtime import Runtime, RuntimeFeatures
    from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext
    from linktools.ai.task.engine import TaskEngine

    from ._runtime_test_helpers import _Models, _UsageFunctionModel

    async def model(messages, info):
        return ModelResponse(parts=[TextPart("agent completed")])

    async def fail(context: TaskNodeContext[None]) -> str:
        raise AIError(
            ErrorCode.TASK_NODE_FAILED,
            retryable=False,
            safe_details={"reason": "callable failure detail"},
        )

    features = RuntimeFeatures("mixed-read-producer")
    features.model("default", _Models(_UsageFunctionModel(model)))
    features.agent("reader", allow_tools=(), allow_subagents=())
    async with Persistence.open("mixed-read", storage=_storage(backend, tmp_path)) as persistence:
        async with Runtime.attach(persistence=persistence, features=features) as runtime:
            agent = Task.from_agent("mixed-read.agent", runtime.agents.get("reader"))
            callable_task = Task("mixed-read.callable", fail, effect_policy="none")
            async with TaskEngine.open(agent, callable_task, persistence=persistence) as engine:
                graph = TaskGraph("mixed-read-graph", (
                    TaskNode("agent", task=agent, input=AgentTaskInput("finish")),
                    TaskNode("callable", ("agent",), task=callable_task),
                ))
                run = await engine.start(graph, idempotency_key="mixed-read")
                outcome = await run.wait(timeout_seconds=10)
                assert outcome.result.status is TaskStatus.FAILED
                page = await run.results(include_content=True)
                results = {item.node_id: item for item in page.items}
                successful = results["agent"]
                assert successful.status is TaskStatus.SUCCEEDED
                assert successful.content_included
                assert successful.output == {"text": "agent completed"}
                # Verify this public result read exercises a genuinely mixed producer registry.
                assert successful.execution_id in engine._agent_execution_ports
                failed = results["callable"]
                assert failed.status is TaskStatus.FAILED
                assert failed.execution_id is not None
                assert failed.error_code == ErrorCode.TASK_NODE_FAILED.value
                assert failed.safe_error_details == {"reason": "callable failure detail"}
