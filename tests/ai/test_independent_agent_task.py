#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent tasks borrow an Agent producer without recreating its execution."""

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.task.engine import TaskEngine
from linktools.ai.task import Task, TaskGraph, TaskNode
from linktools.ai.inputs import AgentTaskInput
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.core import TaskStatus
from ._runtime_test_helpers import model_features, _UsageFunctionModel
from .test_live_history_readback_integration import _Models


@pytest.mark.asyncio
async def test_agent_task_binds_to_shared_persistence_without_runtime_task_owner() -> None:
    async def model(messages, info):
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("agent-task")
    features.agent("default", model="default", allow_tools=())
    async with Persistence.open("agent-task", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
            task = Task.from_agent("answer", runtime.agents.get())
            async with TaskEngine.open(task, persistence=persistence) as engine:
                graph = await engine.start(TaskGraph("answer", (
                    TaskNode("answer", task=task, input=AgentTaskInput("question")),
                )), idempotency_key="once")
                result = await graph.wait(timeout_seconds=10)
                assert result.result.status is TaskStatus.SUCCEEDED
                handle = await graph.execution("answer")
                assert (await handle.wait()).result.output == {"text": "answer"}
                record = await persistence.execution.executions.get(handle.execution_id, tenant_id=persistence.tenant_id)
                assert record.binding_kind == "agent"
                assert record.task_id is None
            assert (await (await runtime.agents.get().start("still alive")).wait()).result.output == {"text": "answer"}


@pytest.mark.asyncio
async def test_agent_task_rejects_an_unrelated_persistence_owner() -> None:
    from linktools.ai.errors import AIError, ErrorCode

    async def model(messages, info):
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("agent-task-scope")
    features.agent("default", model="default", allow_tools=())
    async with Persistence.open("first", storage=PersistenceStorage.in_memory()) as first:
        async with Persistence.open("second", storage=PersistenceStorage.in_memory()) as second:
            async with Runtime.attach(persistence=first, features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
                task = Task.from_agent("answer", runtime.agents.get())
                with pytest.raises(AIError) as mismatch:
                    async with TaskEngine.open(task, persistence=second):
                        pass
                assert mismatch.value.code is ErrorCode.RUNTIME_SERVICE_MISMATCH


@pytest.mark.asyncio
async def test_restored_agent_task_retains_the_captured_binding() -> None:
    async def model(messages, info):
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("agent-task-restore")
    features.agent("default", model="default", allow_tools=())
    async with Persistence.open("restore", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
            binding = runtime.agents.get().execution_port.binding_contract
            restored = runtime.agents.restore(binding)
            task = Task.from_agent("restore", restored)
            assert task.agent_execution.binding_contract == binding
            async with TaskEngine.open(task, persistence=persistence) as engine:
                graph = await engine.start(TaskGraph("restored", (
                    TaskNode("answer", task=task, input=AgentTaskInput("question")),
                )), idempotency_key="restore")
                assert (await graph.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
                execution = await graph.execution("answer")
                stored = await persistence.execution.executions.get(execution.execution_id, tenant_id=persistence.tenant_id)
                assert stored.binding_digest == binding.binding_digest


@pytest.mark.asyncio
async def test_mixed_task_cancellation_reaches_its_agent_producer() -> None:
    import asyncio

    entered, release = asyncio.Event(), asyncio.Event()

    async def model(messages, info):
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("agent-task-cancel")
    features.agent("default", model="default", allow_tools=())
    async with Persistence.open("cancel", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
            task = Task.from_agent("answer", runtime.agents.get())
            async with TaskEngine.open(task, persistence=persistence) as engine:
                graph = await engine.start(TaskGraph("cancel", (
                    TaskNode("answer", task=task, input=AgentTaskInput("question")),
                )), idempotency_key="start")
                await asyncio.wait_for(entered.wait(), 5)
                handle = await graph.execution("answer")
                try:
                    assert (await handle.cancel(idempotency_key="cancel")).cancelled
                    result = await graph.wait(timeout_seconds=10)
                    assert result.result.status is TaskStatus.CANCELLED
                finally:
                    release.set()


@pytest.mark.asyncio
async def test_task_hold_release_cleans_only_the_original_agent_staging(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    entered, release = asyncio.Event(), asyncio.Event()

    async def model(messages, info):
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("agent-task-hold")
    features.agent("default", model="default", allow_tools=())
    async with Persistence.open("hold", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
            released = []
            original_release = runtime._run_store.release_execution_staging

            async def release_staging(execution_id):
                released.append(execution_id)
                await original_release(execution_id)

            monkeypatch.setattr(runtime._run_store, "release_execution_staging", release_staging)
            task = Task.from_agent("answer", runtime.agents.get())
            async with TaskEngine.open(task, persistence=persistence) as engine:
                graph = await engine.start(TaskGraph("held", (
                    TaskNode("answer", task=task, input=AgentTaskInput("question")),
                )), idempotency_key="start")
                await asyncio.wait_for(entered.wait(), 5)
                handle = await graph.execution("answer")
                await engine._launcher._acquire_execution_hold(
                    handle.execution_id, tenant_id=persistence.tenant_id, hold_id="external-consumer",
                )
                release.set()
                assert (await graph.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
                assert handle.execution_id not in released
                await engine._launcher._release_execution_hold(
                    handle.execution_id, tenant_id=persistence.tenant_id, hold_id="external-consumer",
                )
                assert handle.execution_id in released
                assert not any(value[0] == handle.execution_id for value in runtime._run_store._execution_producers.values())
                assert persistence.ready


@pytest.mark.asyncio
async def test_task_admission_does_not_bypass_agent_execution_authorization() -> None:
    from linktools.ai.core import AuthorizationAction, TenantAuthorizationPolicy
    from linktools.ai.errors import AIError, ErrorCode

    class Policy:
        def __init__(self, denied=None):
            self.denied = denied
            self.actions = []

        async def authorize(self, principal, action, resource):
            self.actions.append(action)
            await TenantAuthorizationPolicy().authorize(principal, action, resource)
            if action is self.denied:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)

    calls = []

    async def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("answer")])

    agent_policy = Policy(AuthorizationAction.EXECUTION_RUN)
    task_policy = Policy()
    features = RuntimeFeatures("agent-task-authorization")
    features.agent("default", model="default", allow_tools=())
    async with Persistence.open("authorization", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence, authorization=agent_policy, features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
            task = Task.from_agent("answer", runtime.agents.get())
            async with TaskEngine.open(task, persistence=persistence, authorization=task_policy) as engine:
                graph = await engine.start(TaskGraph("denied", (
                    TaskNode("answer", task=task, input=AgentTaskInput("question")),
                )), idempotency_key="start")
                assert (await graph.wait(timeout_seconds=10)).result.status is TaskStatus.FAILED
                state = await graph.state(include_content=True)
                assert state.node_states[0].error_code == ErrorCode.AUTHORIZATION_DENIED.value
                assert not calls
                assert AuthorizationAction.TASK_RUN in task_policy.actions
                assert AuthorizationAction.EXECUTION_RUN in agent_policy.actions


@pytest.mark.asyncio
async def test_two_agent_producers_share_task_persistence_without_sharing_lifecycle() -> None:
    async def first_model(messages, info):
        return ModelResponse(parts=[TextPart("first")])

    async def second_model(messages, info):
        return ModelResponse(parts=[TextPart("second")])

    first_features = RuntimeFeatures("first-producer")
    second_features = RuntimeFeatures("second-producer")
    first_features.agent("default", model="default", allow_tools=())
    second_features.agent("default", model="default", allow_tools=())
    async with Persistence.open("two-agents", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence, features=model_features(_Models(_UsageFunctionModel(first_model)), first_features)) as first:
            async with Runtime.attach(persistence=persistence, features=model_features(_Models(_UsageFunctionModel(second_model)), second_features)) as second:
                first_task = Task.from_agent("first", first.agents.get())
                second_task = Task.from_agent("second", second.agents.get())
                async with TaskEngine.open(first_task, second_task, persistence=persistence) as engine:
                    first_graph = await engine.start(TaskGraph("first", (
                        TaskNode("answer", task=first_task, input=AgentTaskInput("question")),
                    )), idempotency_key="first")
                    assert (await first_graph.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
                    assert await first_graph.result("answer") == {"text": "first"}
                    await first.close()
                    second_graph = await engine.start(TaskGraph("second", (
                        TaskNode("answer", task=second_task, input=AgentTaskInput("question")),
                    )), idempotency_key="second")
                    assert (await second_graph.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
                    assert await second_graph.result("answer") == {"text": "second"}
                    assert persistence.ready
