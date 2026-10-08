#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Invalid admission and closed Runtime requests fail before side effects."""

import ast
import inspect
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from linktools.ai.core import Page, Principal, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Agent, Execution, Runtime, RuntimeContext, RuntimeStorage, TaskGraphRun
from linktools.ai.runtime.service_api import (
    CancelExecutionRequest, CloseSessionRequest, CreateSessionRequest,
    ExecutionTreeEvent, ForkExecutionRequest, ForkSessionRequest,
    RetryExecutionRequest, UpdateSessionRequest,
)
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._memory import InMemoryStateStorageGroup, InMemoryStateStore
from linktools.ai.task import TaskEffectResolution, TaskGraph, TaskGraphInfo, TaskGraphState

from ._runtime_test_helpers import RuntimeUsageModels


async def _unused_watch(*args: object, **kwargs: object) -> AsyncIterator[ExecutionTreeEvent]:
    raise AssertionError("invalid requests must not subscribe")
    yield  # pragma: no cover


async def _reject_side_effect(*args: object, **kwargs: object) -> None:
    raise AssertionError("invalid requests must not activate or write")


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["", False], ids=["empty", "invalid-type"])
async def test_invalid_convenience_keys_never_activate_or_write(
    key: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = RuntimeStorage.in_memory()
    async with Runtime.open("admission-validation", models=RuntimeUsageModels(), storage=storage) as runtime:
        agent = runtime.agents.get()
        session = agent.session("session")
        principal = runtime.default_principal
        execution = Execution(runtime, "execution", principal, _unused_watch)
        task_execution = Execution(runtime, "task-execution", principal, _unused_watch,
            _task_cancel=lambda supplied_key, force: runtime._cancel_task_execution(
                "graph", "node", "task-execution", principal, supplied_key, force))
        engine = runtime.tasks.bind()
        graph = TaskGraphRun(runtime, runtime._graph_service, "graph", principal, _unused_watch, engine)
        operations = (
            lambda: agent.start("prompt", idempotency_key=key),
            lambda: agent.run("prompt", idempotency_key=key),
            lambda: agent.plan("prompt", idempotency_key=key),
            lambda: agent.start("prompt", session_id="session", idempotency_key=key),
            lambda: session.start("prompt", idempotency_key=key),
            lambda: session.run("prompt", idempotency_key=key),
            lambda: session.plan("prompt", idempotency_key=key),
            lambda: agent.create_session("new-session", idempotency_key=key),
            lambda: session.fork("forked-session", idempotency_key=key),
            lambda: session.update(expected_revision=1, metadata={}, idempotency_key=key),
            lambda: session.close(idempotency_key=key),
            lambda: execution.retry("prompt", idempotency_key=key),
            lambda: execution.fork("prompt", idempotency_key=key),
            lambda: execution.cancel(idempotency_key=key),
            lambda: task_execution.cancel(idempotency_key=key),
            lambda: graph.recover(idempotency_key=key),
            lambda: graph.cancel(idempotency_key=key),
            lambda: graph.resolve_effect("node", 1, TaskEffectResolution("not_applied"), idempotency_key=key),
            lambda: engine.start(TaskGraph("new-graph", ()), idempotency_key=key),
            lambda: engine.prepare_submission(TaskGraph("new-graph", ()), idempotency_key=key),
            lambda: engine.describe_submission(TaskGraph("new-graph", ()), idempotency_key=key),
        )
        with monkeypatch.context() as guarded:
            guarded.setattr(InMemoryStateStore, "mutate", _reject_side_effect)
            guarded.setattr(InMemoryStateStorageGroup, "mutate", _reject_side_effect)
            guarded.setattr(runtime, "_resolve_agent_binding", _reject_side_effect)
            guarded.setattr(engine, "_activate_graph", _reject_side_effect)
            guarded.setattr(runtime._task_node_runtime, "activate_graph", _reject_side_effect)
            for domain in RuntimeDomain:
                guarded.setattr(storage.object_store(domain), "put", _reject_side_effect)
            for operation in operations:
                with pytest.raises(AIError) as raised:
                    await operation()
                assert raised.value.code is ErrorCode.IDEMPOTENCY_KEY_INVALID


@pytest.mark.asyncio
async def test_omitted_keys_generate_independent_admissions_and_explicit_keys_replay() -> None:
    async with Runtime.open("generated-keys", models=RuntimeUsageModels(),
                            storage=RuntimeStorage.in_memory()) as runtime:
        agent = runtime.agents.get()
        first = await agent.start("prompt")
        await first.wait()
        second = await agent.start("prompt", idempotency_key=None)
        await second.wait()
        assert first.execution_id != second.execution_id
        original = await agent.start("prompt", idempotency_key="explicit-key")
        await original.wait()
        replay = await agent.start("prompt", idempotency_key="explicit-key")
        assert replay.execution_id == original.execution_id


def _readable_runtime() -> Runtime[None]:
    async def history(*args: object, **kwargs: object) -> Page:
        return Page(())

    service = SimpleNamespace(history=history)
    stub = object()
    return Runtime(stub, stub, service, service, stub,
        SimpleNamespace(_bind_observation=lambda *args: None), stub, stub, stub, stub, None,
        namespace="closed-runtime", context=RuntimeContext(None))


@pytest.mark.asyncio
@pytest.mark.parametrize("closing", [False, True], ids=["closed", "closing"])
async def test_closed_runtime_rejects_admission_control_and_observation_nonretryably(closing: bool) -> None:
    runtime = _readable_runtime()
    principal = runtime.default_principal
    agent = Agent(runtime, "default", 1)
    session = agent.session("session")
    execution = Execution(runtime, "execution", principal, _unused_watch)
    graph = TaskGraphRun(runtime, runtime._graph_service, "graph", principal, _unused_watch)
    if closing:
        runtime._closing = True
    else:
        await runtime.close()
    operations = (
        lambda: agent.start("prompt"),
        lambda: agent.create_session("new-session"),
        lambda: session.start("prompt"),
        lambda: session.fork("new-session"),
        lambda: session.update(expected_revision=1, metadata={}),
        lambda: session.close(),
        lambda: execution.cancel(),
        lambda: execution.recover(),
        lambda: execution.wait(),
        lambda: graph.cancel(),
        lambda: graph.recover(),
        lambda: graph.wait(),
        lambda: runtime.sessions.create("default", CreateSessionRequest(principal, "new-session", "create")),
        lambda: runtime.sessions.fork("default", "session", ForkSessionRequest(principal, "new-session", "fork")),
        lambda: runtime.sessions.update("default", "session", UpdateSessionRequest(principal, 1, "update", {})),
        lambda: runtime.sessions.close("session", CloseSessionRequest(principal, "close")),
        lambda: runtime.executions.retry("execution", RetryExecutionRequest("prompt", principal, "retry")),
        lambda: runtime.executions.fork("execution", ForkExecutionRequest("prompt", principal, "fork")),
        lambda: runtime.executions.cancel("execution", CancelExecutionRequest(principal, "cancel")),
        lambda: runtime.executions.recover("execution", principal=principal),
    )
    for operation in operations:
        with pytest.raises(AIError) as raised:
            await operation()
        assert raised.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
        assert raised.value.retryable is False
    for run in (execution, graph):
        with pytest.raises(AIError) as raised:
            run.watch()
        assert raised.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
        assert raised.value.retryable is False
    assert (await execution.history()).items == ()
    assert (await session.history()).items == ()


@pytest.mark.asyncio
async def test_graph_state_content_mode_selects_existing_public_types() -> None:
    runtime = _readable_runtime()
    expected = TaskGraphState("graph", TaskStatus.SUCCEEDED, (), (), 7)

    async def state(graph_id: str, *, principal: Principal) -> TaskGraphState:
        return expected

    run = TaskGraphRun(runtime, SimpleNamespace(state=state), "graph",
        runtime.default_principal, _unused_watch)
    assert isinstance(await run.state(), TaskGraphInfo)
    assert isinstance(await run.state(include_content=False), TaskGraphInfo)
    assert await run.state(include_content=True) is expected
    for include_content in (False, True):
        result = await run.state(include_content=include_content)
        assert isinstance(result, TaskGraphState if include_content else TaskGraphInfo)


def test_graph_state_overloads_preserve_content_selected_result_type() -> None:
    declaration = ast.parse(inspect.getsource(TaskGraphRun)).body[0]
    overloads = [
        node for node in declaration.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "state"
        and any(isinstance(decorator, ast.Name) and decorator.id == "overload" for decorator in node.decorator_list)
    ]
    assert len(overloads) == 3
    assert {
        ast.unparse(next(arg.annotation for arg in node.args.kwonlyargs if arg.arg == "include_content")):
        ast.unparse(node.returns)
        for node in overloads
    } == {
        "Literal[False]": "TaskGraphInfo",
        "Literal[True]": "TaskGraphState",
        "bool": "TaskGraphInfo | TaskGraphState",
    }
