#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Task test helpers and test-only legacy fixture adapters."""

from collections.abc import Callable, Mapping
from typing import TypeVar

from linktools.ai.capability import CapabilityGroup as _CapabilityGroup
from linktools.ai.core import Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import AgentTaskInput, Runtime, RuntimeStorage
from linktools.ai.task import (
    Task,
    TaskExpander,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphLimits,
    TaskGraphRequest,
    TaskGraphView,
    TaskNode,
    TaskNodeContext,
)

AppT = TypeVar("AppT")
_TASKS: dict[tuple[str, int], Task[object]] = {}
_EXPANDERS: dict[tuple[str, int], TaskExpander] = {}
_RUNS: dict[tuple[int, str], object] = {}


class TaskFunction(Task[AppT]):
    """Test adapter for migrating old focused runner fixtures to Task."""

    def __init__(
        self,
        id: str,
        revision: int,
        run: Callable[[TaskNodeContext[AppT]], object],
        *,
        effect_policy: str = "non_replay_safe",
        output_type: object | None = None,
        normalize: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
        cancel: Callable[[TaskNodeContext[AppT]], object] | None = None,
        reconcile: Callable[[TaskNodeContext[AppT]], object] | None = None,
    ) -> None:
        super().__init__(
            id,
            run,  # type: ignore[arg-type]
            revision=revision,
            effect_policy=effect_policy,
            output_type=output_type,  # type: ignore[arg-type]
            normalize=normalize,  # type: ignore[arg-type]
            cancel=cancel,  # type: ignore[arg-type]
            reconcile=reconcile,  # type: ignore[arg-type]
        )
        _TASKS[(self.id, self.revision)] = self

    def node(self, node_id: str, **kwargs: object) -> TaskNode:
        dependencies = kwargs.pop("dependencies", ())
        return TaskNode(
            node_id,
            dependencies,  # type: ignore[arg-type]
            task=self,
            **kwargs,  # type: ignore[arg-type]
        )


class CapabilityGroup(_CapabilityGroup[AppT]):
    """Capability fixture with out-of-band task definitions for legacy tests."""

    def task(
        self,
        task: TaskFunction[AppT],
        *,
        effect_policy: str | None = None,
        output_type: object | None = None,
        normalize: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
        cancel: Callable[[TaskNodeContext[AppT]], object] | None = None,
        reconcile: Callable[[TaskNodeContext[AppT]], object] | None = None,
    ) -> TaskFunction[AppT]:
        if not isinstance(task, TaskFunction):
            raise TypeError("test task must be TaskFunction")
        if any(value is not None for value in (effect_policy, output_type, normalize, cancel, reconcile)):
            function = task.function
            if function is None:
                raise TypeError("test task must have a function")
            task = TaskFunction(
                task.id,
                task.revision,
                function,
                effect_policy=task.effect_policy if effect_policy is None else effect_policy,
                output_type=task.output_type if output_type is None else output_type,
                normalize=task.normalizer if normalize is None else normalize,
                cancel=task.cancel_callback if cancel is None else cancel,
                reconcile=task.reconcile_callback if reconcile is None else reconcile,
            )
        _TASKS[(task.id, task.revision)] = task
        return task

    def task_expander(self, expander: object) -> TaskExpander:
        expander_id = getattr(expander, "id", None)
        revision = getattr(expander, "revision", None)
        callback = getattr(expander, "expand", None)
        if not isinstance(expander_id, str) or not isinstance(revision, int) or not callable(callback):
            raise AIError(ErrorCode.BINDING_CONFLICT)
        value = TaskExpander(expander_id, callback, revision=revision)
        _EXPANDERS[(value.id, value.revision)] = value
        return value


def clear_task_test_state() -> None:
    _TASKS.clear()
    _EXPANDERS.clear()
    _RUNS.clear()


def register_task_definition(task: Task[AppT]) -> Task[AppT]:
    _TASKS[(task.id, task.revision)] = task  # type: ignore[assignment]
    return task


def task_engine(runtime: Runtime[object]):
    tasks: list[Task[object]] = []
    for task in _TASKS.values():
        config = task.contract.get("config")
        if task.contract.get("type") != "agent" or not isinstance(config, Mapping):
            tasks.append(task)
            continue
        agent_id = config.get("agent_id")
        runner = task.runner
        if not isinstance(agent_id, str) or runner is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        build_input = getattr(runner, "_build_input", None)
        tasks.append(
            runtime.tasks.from_agent(
                task.id,
                runtime.agents.get(agent_id),
                revision=task.revision,
                build_input=build_input,  # type: ignore[arg-type]
            )
        )
    return runtime.tasks.bind(*tasks, *_EXPANDERS.values())


async def start_task_graph(runtime: Runtime[object], graph: TaskGraph, **kwargs: object):
    engine = task_engine(runtime)
    run = await engine.start(graph, **kwargs)  # type: ignore[arg-type]
    _RUNS[(id(runtime), graph.graph_id)] = run
    return run


async def run_task_graph(runtime: Runtime[object], graph: TaskGraph, **kwargs: object):
    timeout = kwargs.pop("timeout_seconds", None)
    run = await start_task_graph(runtime, graph, **kwargs)
    return (await run.wait(timeout_seconds=timeout)).result


async def get_task_graph_run(runtime: Runtime[object], graph_id: str):
    selected = _RUNS.get((id(runtime), graph_id))
    if selected is not None:
        return selected
    return await task_engine(runtime).get(graph_id)


def agent_task_node(runtime: Runtime[AppT], node_id: str, prompt: object, **kwargs: object) -> TaskNode:
    agent_id = str(kwargs.pop("agent_id", "default"))
    task_id = str(kwargs.pop("task_id", f"test.agent.{agent_id}"))
    revision = int(kwargs.pop("task_revision", 1))
    task = runtime.tasks.from_agent(
        task_id,
        runtime.agents.get(agent_id),
        revision=revision,
    )
    register_task_definition(task)
    dependencies = kwargs.pop("dependencies", ())
    input_refs = kwargs.pop("input_refs", None)
    files = kwargs.pop("files", ())
    session_id = kwargs.pop("session_id", None)
    memory_scope = kwargs.pop("memory_scope", None)
    planning = kwargs.pop("planning", None)
    thinking = kwargs.pop("thinking", None)
    return TaskNode(
        node_id,
        dependencies,  # type: ignore[arg-type]
        task=task,
        input=AgentTaskInput(
            prompt,  # type: ignore[arg-type]
            files=files,  # type: ignore[arg-type]
            session_id=session_id,  # type: ignore[arg-type]
            memory_scope=memory_scope,  # type: ignore[arg-type]
            planning=planning,  # type: ignore[arg-type]
            thinking=thinking,  # type: ignore[arg-type]
        ),
        input_refs=input_refs,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def agent_task_definition(runtime: Runtime[AppT], agent_id: str, task_id: str) -> Task[AppT]:
    task = runtime.tasks.from_agent(task_id, runtime.agents.get(agent_id))
    register_task_definition(task)
    return task


async def task_graph_state(runtime: Runtime[object], graph_id: str, **kwargs: object):
    del kwargs
    run = await get_task_graph_run(runtime, graph_id)
    return await run.state(include_content=True)


async def task_graph_wait(runtime: Runtime[object], graph_id: str, **kwargs: object):
    timeout = kwargs.get("timeout_seconds")
    run = await get_task_graph_run(runtime, graph_id)
    return (await run.wait(timeout_seconds=timeout)).result  # type: ignore[arg-type]


async def task_graph_resume(
    runtime: Runtime[object],
    graph_id: str,
    node_id: str,
    request: object,
):
    run = await get_task_graph_run(runtime, graph_id)
    return await run.resume(node_id, request)  # type: ignore[arg-type]


async def task_result(runtime: Runtime[object], graph_id: str, node_id: str):
    run = await get_task_graph_run(runtime, graph_id)
    return await run.result(node_id)


async def start_task_request(runtime: Runtime[object], request: TaskGraphRequest):
    run = await task_engine(runtime).start(
        request.graph,
        idempotency_key=request.idempotency_key,
        principal=request.principal,
        limits=request.limits,
    )
    _RUNS[(id(runtime), request.graph.graph_id)] = run
    return run


async def admit_graph(
    state: RuntimeStorage,
    graph: TaskGraph,
    *,
    tenant_id: str = "tenant",
) -> TaskGraphView:
    request = TaskGraphRequest(
        graph,
        Principal("task-test", tenant_id),
        f"test:{graph.graph_id}",
        TaskGraphLimits(),
    )
    return await state.task.admissions.admit(
        TaskGraphAdmission.from_request(request),
        graph,
    )
