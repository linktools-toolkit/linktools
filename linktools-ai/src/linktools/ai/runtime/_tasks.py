#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned Task definition views and execution entry points."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Generic, TypeVar

from ..agent import AgentInputCaptureRef
from ..core import BudgetUsage, RunBudget, Page, Principal
from ..errors import AIError, ErrorCode
from ..task import (
    Task,
    TaskExpander,
    TaskExpanderRef,
    TaskGraph,
    TaskGraphCaptureRef,
    TaskGraphLimits,
    TaskGraphResult,
    TaskNode,
    TaskNodeInfo,
    TaskRef,
    TaskGraphService,
    TaskGraphSubmission,
    TaskSubmissionRef,
    TaskSubmissionResult,
    TaskSubmissionCancellation,
)
from ._task import TaskGraphRun
from ._input_capture import CaptureGraphRequest
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

    async def budget_usage(
        self, graph_id: str, *, principal: Principal | None = None,
    ) -> BudgetUsage | None:
        self._runtime._ensure_open()
        resolved = self._runtime._resolve_principal(principal)
        await self._graph_service.inspect(graph_id, principal=resolved)
        admissions = self._runtime._task_admissions
        if admissions is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        admission = await admissions.get(graph_id, tenant_id=resolved.tenant_id)
        if admission is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if admission.budget_scope_id is None:
            return None
        budgets = self._runtime._budgets
        if budgets is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        usage = await budgets.read(admission.budget_scope_id)
        if usage.limits != admission.budget:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return usage

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

    async def capture_graph(self, graph_id: str, request: CaptureGraphRequest) -> TaskGraphCaptureRef:
        self._runtime._ensure_open()
        return await self._runtime._input_captures.capture_graph(graph_id, request)

    async def from_agent_capture(self, id: str, capture: AgentInputCaptureRef, *,
                                 revision: int = 1, principal: Principal) -> Task[AppT]:
        self._runtime._ensure_open()
        value = await self._runtime._input_captures.read_agent(capture, principal=principal)
        if value.binding is None:
            raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE)
        return self._runtime._task_from_agent_capture(id, value.binding, revision=revision)

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

    @property
    def runtime(self) -> "Runtime[AppT]":
        return self._runtime

    @property
    def definitions(self) -> tuple[Task[AppT], ...]:
        return tuple(self._tasks.values())

    @property
    def expanders(self) -> tuple[TaskExpander, ...]:
        return tuple(self._expanders.values())

    def definition(self, ref: TaskRef) -> Task[AppT]:
        try:
            return self._tasks[(ref.id, ref.revision)]
        except KeyError as error:
            raise AIError(ErrorCode.BINDING_NOT_REGISTERED) from error

    def expander_definition(self, ref: TaskExpanderRef) -> TaskExpander:
        try:
            return self._expanders[(ref.id, ref.revision)]
        except KeyError as error:
            raise AIError(ErrorCode.BINDING_NOT_REGISTERED) from error

    def with_definitions(self, *definitions: Task[AppT]) -> "TaskEngine[AppT]":
        return self._runtime.tasks.bind(
            *self._tasks.values(), *self._expanders.values(), *definitions,
        )

    async def start(
        self,
        graph: TaskGraph,
        *,
        idempotency_key: str,
        principal: Principal | None = None,
        limits: TaskGraphLimits | None = None,
        budget: RunBudget | None = None,
        correlation: Mapping[str, object] | None = None,
    ) -> TaskGraphRun[AppT]:
        submission = await self.prepare_submission(
            graph, principal=principal, idempotency_key=idempotency_key,
            limits=limits, budget=budget, correlation=correlation,
        )
        result = await self.start_prepared(submission)
        if not result.admitted:
            raise AIError(ErrorCode.TASK_NOT_READY, "submission_cancelled")
        return TaskGraphRun(
            self._runtime, self._graph_service, graph.graph_id,
            submission.admission.principal, self._runtime._watch_execution_tree, self,
        )

    async def prepare_submission(
        self,
        graph: TaskGraph,
        *,
        idempotency_key: str,
        principal: Principal | None = None,
        limits: TaskGraphLimits | None = None,
        budget: RunBudget | None = None,
        correlation: Mapping[str, object] | None = None,
    ) -> TaskGraphSubmission:
        return await self._prepare_submission(graph, idempotency_key=idempotency_key,
            principal=principal, limits=limits, budget=budget, correlation=correlation, describe=False)

    async def describe_submission(
        self,
        graph: TaskGraph,
        *,
        idempotency_key: str,
        principal: Principal | None = None,
        limits: TaskGraphLimits | None = None,
        budget: RunBudget | None = None,
        correlation: Mapping[str, object] | None = None,
    ) -> TaskGraphSubmission:
        """Resolve submission identity without storing input or launching work."""
        return await self._prepare_submission(graph, idempotency_key=idempotency_key,
            principal=principal, limits=limits, budget=budget, correlation=correlation, describe=True)

    async def _prepare_submission(
        self,
        graph: TaskGraph,
        *,
        idempotency_key: str,
        principal: Principal | None = None,
        limits: TaskGraphLimits | None = None,
        budget: RunBudget | None = None,
        correlation: Mapping[str, object] | None = None,
        describe: bool,
    ) -> TaskGraphSubmission:
        runtime = self._runtime
        request = await runtime._admit_graph(
            graph, principal=principal, idempotency_key=idempotency_key,
            limits=limits, budget=budget, correlation=correlation,
        )
        task_runtime = runtime._require_task_node_runtime()
        activation = await task_runtime.activate_graph(
            graph, tuple(self._tasks.values()), tuple(self._expanders.values()),
            track_pre_admission=True,
        )
        assert activation is not None
        try:
            if describe:
                return await self._graph_service.describe_submission(request)
            return await self._graph_service.prepare_submission(request)
        finally:
            await task_runtime.finish_graph_activation(
                graph.graph_id, request.principal.tenant_id, activation,
                admitted=False,
            )

    async def _prepare_trial_submission(
        self, submission: TaskGraphSubmission,
    ) -> bool:
        runtime = self._runtime
        runtime._ensure_open()
        task_runtime = runtime._require_task_node_runtime()
        activation = await task_runtime.activate_graph(
            submission.graph, tuple(self._tasks.values()),
            tuple(self._expanders.values()), track_pre_admission=True,
        )
        assert activation is not None
        try:
            _, created = await self._graph_service.prepare_described_with_disposition(submission)
            return created
        finally:
            await task_runtime.finish_graph_activation(
                submission.graph.graph_id, submission.ref.tenant_id,
                activation, admitted=False,
            )

    async def start_prepared(
        self, submission: TaskGraphSubmission,
    ) -> TaskSubmissionResult:
        runtime = self._runtime
        runtime._ensure_open()
        task_runtime = runtime._require_task_node_runtime()
        activation = await task_runtime.activate_graph(
            submission.graph, tuple(self._tasks.values()),
            tuple(self._expanders.values()), track_pre_admission=True,
        )
        assert activation is not None
        admitted = False
        try:
            submission = await self._graph_service.prepare_described(submission)
            result = await self._graph_service.start_prepared(submission)
            admitted = result.admitted
            return result
        finally:
            await task_runtime.finish_graph_activation(
                submission.graph.graph_id, submission.ref.tenant_id,
                activation, admitted=admitted,
            )

    async def cancel_submission(
        self,
        submission: TaskSubmissionRef,
        *,
        principal: Principal | None = None,
        idempotency_key: str,
    ) -> TaskSubmissionCancellation:
        self._runtime._ensure_open()
        return await self._graph_service.cancel_submission(
            submission, principal=self._runtime._resolve_principal(principal),
            idempotency_key=idempotency_key,
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
                    input_capture=node.input_capture,
                    timeout_seconds=node.timeout_seconds,
                    max_attempts=node.max_attempts,
                    retry_delay_seconds=node.retry_delay_seconds,
                    output_contract=node.output_contract,
                    effect_policy=node.effect_policy,
                    reconcile=node.reconcile,
                    dependency_policy=node.dependency_policy,
                    failure_policy=node.failure_policy,
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
