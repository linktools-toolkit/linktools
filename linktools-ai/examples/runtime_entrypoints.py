#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline Agent, Session, Task and durable recovery entry points."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic_ai.models import Model
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from linktools.ai.capability import StepPersistenceCapability
from linktools.ai.core import ExecutionStatus, JsonValue, TaskStatus, service_principal
from linktools.ai.execution import CreateSessionRequest, ForkSessionRequest, RetryExecutionRequest
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext


class OfflineModel:
    provider = "test"
    model_identity = "test:runtime-entrypoints"
    contract: dict[str, JsonValue] = {"provider": "test", "model": "runtime-entrypoints"}

    def __init__(self, model: Model) -> None:
        self.model = model

    def materialize(self) -> Model:
        return self.model


async def agent_and_session() -> None:
    features = RuntimeFeatures()
    features.model("default", OfflineModel(TestModel(custom_output_text="hello")))
    actor = service_principal("default", "demo")
    async with Runtime.open("handles", storage=PersistenceStorage.in_memory(), features=features) as runtime:
        agent = runtime.agents.get()
        session = await runtime.sessions.create("default", CreateSessionRequest(actor, "chat", "create-chat"))
        execution = await session.start("Say hello", idempotency_key="hello")
        assert execution.principal == actor
        assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        print((await execution.result()).output)
        retrieved = await runtime.executions.get(execution.execution_id, principal=actor)
        assert (await retrieved.inspect()).execution_id == execution.execution_id
        retried = await runtime.executions.retry(
            execution.execution_id, RetryExecutionRequest("Say hello again", actor, "retry-hello"),
        )
        await retried.wait(timeout_seconds=10)
        forked = await runtime.sessions.fork("default", session.session_id, ForkSessionRequest(actor, "branch", "fork-chat"))
        print((await forked.run("Continue the conversation", idempotency_key="continue")).result.output)
        await forked.close(idempotency_key="close-branch")
        await session.close(idempotency_key="close-chat")
        print((await agent.run("A session is optional", principal=actor)).result.output)
    # Runtime closes its producer and its owned Persistence here.


async def echo(context: TaskNodeContext[None]) -> JsonValue:
    return context.input["message"]


async def task_only() -> None:
    definition = Task("echo", echo, effect_policy="none")
    async with Runtime.open("tasks", storage=PersistenceStorage.in_memory()) as runtime:
        engine = await runtime.tasks.bind(definition)
        graph = await engine.start(TaskGraph("echo-graph", (
            TaskNode("echo", task=definition, input={"message": "hello from a Task"}),
        )), idempotency_key="start-graph")
        assert (await graph.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
        print(await graph.result("echo"))
        execution = await graph.execution("echo")
        assert (await execution.result()).status is ExecutionStatus.SUCCEEDED
        assert (await execution.inspect()).binding_kind == "task"
        # Graph control remains with the Task producer; its node handle has the
        # same Execution API and retains graph-aware cancellation.
    # No Agent producer or model was created. Runtime closes the TaskEngine.


async def recovery_and_cancellation(root: Path) -> None:
    entered = asyncio.Event()

    async def blocked(messages, info):
        entered.set()
        await asyncio.Event().wait()
        yield "unreachable"

    async def answer(messages, info):
        yield "recovered"

    features = RuntimeFeatures()
    features.model("default", OfflineModel(FunctionModel(stream_function=blocked)))
    features.capability(StepPersistenceCapability())
    storage = PersistenceStorage.from_root(root)
    original_actor = service_principal("default", "demo")
    async with Runtime.open("recovery", storage=storage, features=features) as owner:
        execution = await owner.agents.get().start("Finish after handoff", principal=original_actor, idempotency_key="start")
        await asyncio.wait_for(entered.wait(), 10)
        execution_id = execution.execution_id
        await owner.handoff()
    # Handoff closes this producer without cancelling the admitted execution.
    # The local SQLite backend provisions its own schema explicitly on open.
    features.model("default", OfflineModel(FunctionModel(stream_function=answer)))
    async with Runtime.open("recovery", storage=storage, features=features, auto_recover=False) as successor:
        recovered = await successor.executions.recover(
            execution_id, principal=successor.default_principal, idempotency_key="take-over",
        )
        assert recovered.execution_id == execution_id
        assert recovered.principal == successor.default_principal
        assert (await recovered.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        print((await recovered.result()).output)
        # The recovery actor controls recovery; execution keeps original_actor.
        entered.clear()
        features.model("default", OfflineModel(FunctionModel(stream_function=blocked)))
        unfinished = await successor.agents.get().start("Cancel this request", idempotency_key="to-cancel")
        await asyncio.wait_for(entered.wait(), 10)
        assert (await unfinished.cancel(idempotency_key="cancel")).cancelled
        assert (await unfinished.wait(timeout_seconds=10)).result.status is ExecutionStatus.CANCELLED
    # Normal close drains the owned producer; borrowed resources are not closed.


async def main() -> None:
    await agent_and_session()
    await task_only()
    with TemporaryDirectory(prefix="linktools-entrypoints-") as directory:
        await recovery_and_cancellation(Path(directory))


if __name__ == "__main__":
    asyncio.run(main())
