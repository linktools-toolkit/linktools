#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Domain entry points return actor-bound handles without changing producer ownership."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic_ai.models.function import FunctionModel

from linktools.ai.capability import ExecutionCapability, StepPersistenceCapability
from linktools.ai.core import (
    AuthorizationAction, ExecutionStatus, Principal, ResourceKind, ResourceRef,
    TaskStatus, TenantAuthorizationPolicy,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import (
    CreateSessionRequest, Execution, ExecutionHandle, ForkExecutionRequest,
    ForkSessionRequest, RetryExecutionRequest,
)
from linktools.ai.inputs import AgentTaskInput
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeExecutions, RuntimeSessions, Session
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext

from ._runtime_test_helpers import RuntimeUsageModels, _Models, model_features

pytestmark = pytest.mark.asyncio


async def test_session_and_execution_entry_points_return_bound_entities() -> None:
    features = model_features(RuntimeUsageModels())
    features.capability(ExecutionCapability(retain_history=True))
    actor = Principal("caller", "default", "service")
    async with Runtime.open("entity-handles", storage=PersistenceStorage.in_memory(), features=features) as runtime:
        assert runtime.agents.get().session("lazy").principal is runtime.default_principal
        session = await runtime.sessions.create("default", CreateSessionRequest(actor, "session", "create"))
        assert isinstance(session, Session)
        assert session.principal is actor
        assert (await session.inspect()).session_id == session.session_id
        assert (await runtime.sessions.get(session.session_id, principal=actor)).principal is actor
        execution = await session.start("first", idempotency_key="first")
        assert isinstance(execution, Execution) and execution.principal is actor
        outcome = await execution.wait(timeout_seconds=10)
        assert (await execution.result()) == outcome.result
        assert (await execution.inspect()).status is ExecutionStatus.SUCCEEDED
        obtained = await runtime.executions.get(execution.execution_id, principal=actor)
        assert obtained.principal is actor
        assert (await obtained.result()) == outcome.result
        forked_session = await runtime.sessions.fork(
            "default", session.session_id, ForkSessionRequest(actor, "forked", "fork-session"),
        )
        assert isinstance(forked_session, Session) and forked_session.principal is actor
        assert (await forked_session.inspect()).session_id == "forked"
        independent = await runtime.agents.get().start(
            "independent", principal=actor, idempotency_key="independent",
        )
        await independent.wait(timeout_seconds=10)
        retried = await runtime.executions.retry(
            independent.execution_id, RetryExecutionRequest("retry", actor, "retry"),
        )
        assert isinstance(retried, Execution) and retried.principal is actor
        await retried.wait(timeout_seconds=10)
        forked = await runtime.executions.fork(
            independent.execution_id, ForkExecutionRequest("fork", actor, "fork-execution"),
        )
        assert isinstance(forked, Execution) and forked.principal is actor
        await forked.wait(timeout_seconds=10)
        await forked_session.close(idempotency_key="close-fork")
        await session.close(idempotency_key="close-session")


@pytest.mark.parametrize("operation", ("retry", "fork", "recover"))
@pytest.mark.parametrize("unknown", (False, True))
async def test_admission_returns_handle_without_a_second_read(operation: str, unknown: bool) -> None:
    actor = Principal("controller", "tenant", "service")
    failure = AIError(ErrorCode.EXECUTION_START_UNKNOWN)
    admitted = AsyncMock(side_effect=failure if unknown else None, return_value=ExecutionHandle("accepted"))
    service = SimpleNamespace(**{operation: admitted})
    observe = AsyncMock(side_effect=AssertionError("admission must not start another observation"))
    facade = RuntimeExecutions(service, observe, bind_execution=lambda identity, principal: Execution(
        None, identity, principal, None,
    ))
    if operation == "recover":
        request = facade.recover("original", principal=actor, idempotency_key="operation")
    else:
        request_type = RetryExecutionRequest if operation == "retry" else ForkExecutionRequest
        request = getattr(facade, operation)("original", request_type("prompt", actor, "operation"))
    if unknown:
        with pytest.raises(AIError) as raised:
            await request
        assert raised.value is failure
    else:
        execution = await request
        assert isinstance(execution, Execution)
        assert execution.execution_id == "accepted" and execution.principal is actor
    observe.assert_not_awaited()


async def test_recovery_returns_handle_and_preserves_original_admitted_principal(tmp_path: Path) -> None:
    entered = asyncio.Event()

    async def blocked(messages, info):
        entered.set()
        await asyncio.Event().wait()
        yield "unreachable"

    features = model_features(_Models(FunctionModel(stream_function=blocked)))
    features.capability(StepPersistenceCapability())
    storage = PersistenceStorage.from_root(tmp_path)
    original = Principal("caller", "default", "service")
    async with Runtime.open("recover-handle", storage=storage, features=features) as runtime:
        execution = await runtime.agents.get().start("question", principal=original, idempotency_key="start")
        await asyncio.wait_for(entered.wait(), 10)
        await runtime.handoff()
    async def answer(messages, info):
        yield "recovered"

    features.model("default", _Models(FunctionModel(stream_function=answer)))
    actor = Principal("controller", "default", "service")

    class RecoveryAuthorization(TenantAuthorizationPolicy):
        async def authorize(self, principal: Principal, action: AuthorizationAction, resource: ResourceRef) -> None:
            if (principal == actor and resource.kind is ResourceKind.EXECUTION
                    and resource.id == execution.execution_id
                    and action in {AuthorizationAction.EXECUTION_RECOVER, AuthorizationAction.EXECUTION_READ}):
                principal = original
            await super().authorize(principal, action, resource)

    async with Runtime.open("recover-handle", storage=storage, features=features,
                            authorization=RecoveryAuthorization(), auto_recover=False) as runtime:
        recovered = await runtime.executions.recover(execution.execution_id, principal=actor, idempotency_key="recover")
        assert isinstance(recovered, Execution) and recovered.principal is actor
        assert recovered.execution_id == execution.execution_id
        assert (await recovered.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        record = await runtime._state.execution.executions.get(execution.execution_id, tenant_id="default")
        assert record.principal_id == original.principal_id
        assert record.principal_kind == original.kind


@pytest.mark.parametrize("agent_node", (False, True))
async def test_task_execution_handle_keeps_graph_cancellation(agent_node: bool) -> None:
    entered = asyncio.Event()

    async def blocked_model(messages, info):
        entered.set()
        await asyncio.Event().wait()
        yield "unreachable"

    async def blocked_task(context: TaskNodeContext[None]) -> None:
        entered.set()
        await asyncio.Event().wait()

    features = model_features(_Models(FunctionModel(stream_function=blocked_model))) if agent_node else None
    async with Runtime.open("task-handle", storage=PersistenceStorage.in_memory(), features=features) as runtime:
        actor = runtime.default_principal
        task = Task.from_agent("blocked", runtime.agents.get()) if agent_node else Task(
            "blocked", blocked_task, effect_policy="none",
        )
        engine = await runtime.tasks.bind(task)
        graph = await engine.start(TaskGraph("graph", (
            TaskNode("node", task=task, input=AgentTaskInput("wait") if agent_node else {}),
        )), principal=actor, idempotency_key="start")
        await asyncio.wait_for(entered.wait(), 10)
        async with asyncio.timeout(10):
            while not (await graph.state()).node_states[0].execution_id:
                await asyncio.sleep(0)
        execution = await graph.execution("node")
        assert isinstance(execution, Execution) and execution.principal is actor
        cancelled = await execution.cancel(idempotency_key="cancel-node")
        assert cancelled.cancelled
        assert (await execution.inspect()).status is ExecutionStatus.CANCELLED
        assert (await execution.result()).status is ExecutionStatus.CANCELLED
        await graph.wait(timeout_seconds=10)
        assert (await graph.state()).node_states[0].status is TaskStatus.CANCELLED
        if not agent_node:
            assert runtime.features is None
            with pytest.raises(AIError) as missing:
                runtime.agents.get()
            assert missing.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY


@pytest.mark.parametrize("operation", ("create", "fork"))
@pytest.mark.parametrize("unknown", (False, True))
async def test_session_mutation_binds_the_admitted_view_without_another_read(operation: str, unknown: bool) -> None:
    actor = Principal("controller", "tenant", "service")
    failure = AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
    view = SimpleNamespace(agent_id="default", session_id="accepted")
    service = SimpleNamespace(**{operation: AsyncMock(side_effect=failure if unknown else None, return_value=view)})
    observe = AsyncMock(side_effect=AssertionError("mutation must not start another observation"))
    facade = RuntimeSessions(service, observe, bind_session=lambda value, principal: Session(
        None, value.agent_id, None, value.session_id, principal,
    ))
    request = facade.create("default", CreateSessionRequest(actor, "accepted", "create")) if operation == "create" else (
        facade.fork("default", "source", ForkSessionRequest(actor, "accepted", "fork"))
    )
    if unknown:
        with pytest.raises(AIError) as raised:
            await request
        assert raised.value is failure
    else:
        session = await request
        assert isinstance(session, Session)
        assert session.session_id == view.session_id and session.principal is actor
    observe.assert_not_awaited()
