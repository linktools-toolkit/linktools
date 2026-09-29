#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned Task definition views and execution entry points."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Generic, TypeVar

from ..core import Page, Principal
from ..errors import AIError, ErrorCode
from ..task import (
    Task,
    TaskExpander,
    TaskGraph,
    TaskGraphLimits,
    TaskGraphResult,
    TaskNode,
    TaskNodeInfo,
    TaskRef,
    TaskGraphService,
)
from ._task import TaskGraphRun
from ._agent_task_input import AgentTaskInputBuilder

if TYPE_CHECKING:
    from ._agent import Agent
    from ._runtime_service import Runtime

AppT = TypeVar("AppT")


class RuntimeTasks(Generic[AppT]):
    def __init__(
        self,
        runtime: "Runtime[AppT]",
        graph_service: TaskGraphService,
    ) -> None:
        self._runtime = runtime
        self._graph_service = graph_service

    def from_agent(
        self,
        id: str,
        agent: "Agent[AppT]",
        *,
        revision: int = 1,
        build_input: AgentTaskInputBuilder | None = None,
    ) -> Task[AppT]:
        self._runtime._ensure_open()
        return self._runtime._task_from_agent(
            id,
            agent,
            revision=revision,
            build_input=build_input,
        )

    def bind(self, *definitions: Task[AppT] | TaskExpander) -> "TaskEngine[AppT]":
        tasks: dict[tuple[str, int], Task[AppT]] = {}
        expanders: dict[tuple[str, int], TaskExpander] = {}
        for definition in definitions:
            if isinstance(definition, Task):
                identity = (definition.ref.id, definition.ref.revision)
                if identity in tasks:
                    raise AIError(ErrorCode.BINDING_CONFLICT)
                tasks[identity] = definition
            elif isinstance(definition, TaskExpander):
                identity = (definition.id, definition.revision)
                if identity in expanders:
                    raise AIError(ErrorCode.BINDING_CONFLICT)
                expanders[identity] = definition
            else:
                raise TypeError("definitions must be Task or TaskExpander")
        return TaskEngine(
            self._runtime,
            self._graph_service,
            tasks,
            expanders,
        )


class TaskEngine(Generic[AppT]):
    """An immutable definition view bound to one Runtime."""

    def __init__(
        self,
        runtime: "Runtime[AppT]",
        graph_service: TaskGraphService,
        tasks: Mapping[tuple[str, int], Task[AppT]],
        expanders: Mapping[tuple[str, int], TaskExpander],
    ) -> None:
        runtime._validate_task_bindings(tuple(tasks.values()))
        self._runtime = runtime
        self._graph_service = graph_service
        self._tasks = MappingProxyType(dict(sorted(tasks.items())))
        self._expanders = MappingProxyType(dict(sorted(expanders.items())))

    async def start(
        self,
        graph: TaskGraph,
        *,
        idempotency_key: str,
        principal: Principal | None = None,
        limits: TaskGraphLimits | None = None,
        correlation: Mapping[str, object] | None = None,
    ) -> TaskGraphRun[AppT]:
        runtime = self._runtime
        runtime._ensure_open()
        if not isinstance(graph, TaskGraph):
            raise TypeError("graph must be TaskGraph")
        request = await runtime._admit_graph(
            graph,
            principal=principal,
            idempotency_key=idempotency_key,
            limits=limits,
            correlation=correlation,
        )
        task_runtime = runtime._require_task_node_runtime()
        activation = await task_runtime.activate_graph(
            graph,
            tuple(self._tasks.values()),
            tuple(self._expanders.values()),
            track_pre_admission=True,
        )
        assert activation is not None
        try:
            await self._graph_service.start(request)
        except BaseException:
            await task_runtime.finish_graph_activation(
                graph.graph_id,
                request.principal.tenant_id,
                activation,
                admitted=False,
            )
            raise
        await task_runtime.finish_graph_activation(
            graph.graph_id,
            request.principal.tenant_id,
            activation,
            admitted=True,
        )
        return TaskGraphRun(
            runtime,
            self._graph_service,
            graph.graph_id,
            request.principal,
            runtime._watch_execution_tree,
            self,
        )

    async def get(
        self,
        graph_id: str,
        *,
        principal: Principal | None = None,
    ) -> TaskGraphRun[AppT]:
        runtime = self._runtime
        runtime._ensure_open()
        resolved_principal = runtime._resolve_principal(principal)
        await self._graph_service.inspect(graph_id, principal=resolved_principal)
        return TaskGraphRun(
            runtime,
            self._graph_service,
            graph_id,
            resolved_principal,
            runtime._watch_execution_tree,
            self,
        )

    async def recover_pending(
        self,
        *,
        cursor: str | None = None,
        limit: int = 100,
        principal: Principal | None = None,
    ) -> Page[TaskGraphResult]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 1000
            or cursor is not None
            and (not isinstance(cursor, str) or not cursor)
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return await self._runtime._recover_pending_tasks(
            self,
            cursor=cursor,
            limit=limit,
            principal=principal,
        )

    async def _activate_graph(
        self,
        graph_id: str,
        principal: Principal,
        *,
        recovery_nodes: tuple[TaskNodeInfo, ...] | None = None,
        recovery_principal: Principal | None = None,
    ) -> None:
        runtime = self._runtime
        runtime._ensure_open()
        admissions = runtime._task_admissions
        if admissions is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        authorization_principal = (
            principal if recovery_principal is None else recovery_principal
        )
        nodes = recovery_nodes
        if nodes is None:
            nodes = await self._graph_service.recovery_nodes(
                graph_id,
                principal=authorization_principal,
            )
        admission = await admissions.get(
            graph_id,
            tenant_id=principal.tenant_id,
        )
        if (
            admission is None
            or admission.graph_id != graph_id
            or admission.principal.tenant_id != principal.tenant_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        task_runtime = runtime._require_task_node_runtime()
        graph = TaskGraph(
            graph_id,
            tuple(
                TaskNode.from_resolved(
                    node.node_id,
                    node.dependencies,
                    task=node.task,
                    budget_cost=node.budget_cost,
                    expander=node.expander,
                    input_refs=node.input_refs,
                    timeout_seconds=node.timeout_seconds,
                    max_attempts=node.max_attempts,
                    retry_delay_seconds=node.retry_delay_seconds,
                    output_contract=node.output_contract,
                    effect_policy=node.effect_policy,
                    reconcile=node.reconcile,
                    dependency_policy=node.dependency_policy,
                )
                for node in nodes
            ),
        )
        await task_runtime.activate_graph(
            graph,
            tuple(self._tasks.values()),
            tuple(self._expanders.values()),
        )

    @property
    def _task_definitions(self) -> Mapping[tuple[str, int], Task[AppT]]:
        return self._tasks

    @property
    def _expander_definitions(self) -> Mapping[tuple[str, int], TaskExpander]:
        return self._expanders


__all__ = ["RuntimeTasks", "TaskEngine"]
