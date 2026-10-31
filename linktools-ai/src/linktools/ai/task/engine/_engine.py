#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent Task definition views and execution entry points."""

from __future__ import annotations

import asyncio
import secrets
import uuid
from copy import copy
from contextlib import aclosing, asynccontextmanager
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
from ...execution.handle import ExecutionBinding
from ...core import (AuthorizationPolicy, CorrelationData, ExecutionContext, PrincipalKind, PromptLimits,
    TenantAuthorizationPolicy, canonical_sha256, overlay_correlation, principal_identity_payload, validate_idempotency_key)
from ...persistence import Persistence, PersistenceDomain
from ...observe import MetricRecorderProvider, MetricsControl, MetricFlushResult, MetricRecorderStatus
from ...storage import PayloadPolicy, StorageRetentionMode
from ...task import (
    DefaultTaskGraphService,
    LocalTaskGraphLauncher,
    TaskGraphRequest,
    CancelGraphRequest,
    RecoverGraphRequest,
)
from ...execution import TaskExecutionService
from ...persistence.commands import StoredExecutionManagementAdapter
from ...execution.events import DefaultEventService
from ...execution.tree import ExecutionTreeBroker, ExecutionTreeStreamer
from ...execution.coordinator import ExecutionStreamCoordinator
from ...inputs import ExecutionInputMaterializer
from ...observe.service import MetricsService, disabled_metric_status
from ...persistence.objects import ObjectKeyFactory
from ._runner import TaskNodeExecutor
from ...inputs import TaskGraphBindingCaptureStore
from ...inputs import InputCaptures
from ...observe.stream import ObservationSession, await_stream_cleanup
from ._run import _CancelSettlement
from ...execution import Execution
from ...execution import ExecutionView, ExecutionTreeEvent, CancelExecutionResult, CancelExecutionRequest
from linktools.core import environ

_logger = environ.get_logger("ai.task.engine")

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Generic, TypeVar

from ...agent import OutputBinding, restore_output
from pydantic import BaseModel
from ...core import JsonValue, ThinkingValue
from ...model import CanonicalUserInput
from ...execution import ExecutionInputContext, StoredUserInput
from ...task import TaskNodeInvocation, TaskGraphState
from ._agent import AgentTaskRunner
from ...core import BudgetUsage, RunBudget, Page, Principal
from ...errors import AIError, ErrorCode
from ...task import (
    Task,
    TaskExpander,
    TaskExpanderRef,
    TaskGraph,
    TaskGraphLimits,
    TaskGraphResult,
    TaskNode,
    TaskNodeInfo,
    TaskRef,
    TaskGraphSubmission,
    TaskSubmissionRef,
    TaskSubmissionResult,
    TaskSubmissionCancellation,
)
from ._run import TaskGraphRun
from ._history import TaskHistory
from ._reads import TaskExecutionReads

if TYPE_CHECKING:
    from ...runtime.agent_execution import AgentExecutionPort

AppT = TypeVar("AppT")
_ResultT = TypeVar("_ResultT")




class TaskEngine(Generic[AppT]):
    """An independent task-graph producer over borrowed persistence."""

    def __init__(
        self, *definitions: Task[AppT, object] | TaskExpander,
        persistence: Persistence,
        context: ExecutionContext[AppT] | None = None,
        authorization: AuthorizationPolicy | None = None,
        metrics: MetricRecorderProvider | None = None,
    ) -> None:
        if not isinstance(persistence, Persistence) or not persistence.ready:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if context is not None and not isinstance(context, ExecutionContext):
            raise TypeError("context must be ExecutionContext")
        if metrics is not None and not isinstance(metrics, MetricRecorderProvider):
            raise TypeError("metrics must be MetricRecorderProvider or None")
        self._persistence = persistence
        self._context = ExecutionContext(None) if context is None else context
        self._authorization = TenantAuthorizationPolicy() if authorization is None else authorization
        self._tasks, self._expanders = self._definition_maps(definitions)
        self._declarations = self._tasks
        for definition in self._declarations.values():
            port = definition.agent_execution
            if port is not None:
                if port.persistence is not persistence:
                    raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
                port.operations.ensure_open()
        self._owner = self
        self._closed = False
        self._notifying_closed = False
        self._close_callbacks: deque[Callable[[TaskEngine[AppT]], None]] = deque()
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None
        self._observation_sessions: set[ObservationSession] = set()
        self._cancel_settlements: set[_CancelSettlement] = set()
        self._default_principal = Principal("task-engine", self.tenant_id, PrincipalKind.LOCAL_TRUSTED.value)
        self._task_admissions = persistence.task.admissions
        self._budgets = persistence.execution.budgets
        self._history = None
        self._agent_execution_ports = {}
        self._metrics = None if metrics is None else MetricsService(metrics)
        if self._metrics is not None:
            self._metrics.configure_runtime_dimensions(self._context.metric_dimensions)
        self._metrics_control = MetricsControl(self._metric_status, self._flush_metrics)
        key_factory = ObjectKeyFactory(self.namespace)
        payload_policy = PayloadPolicy()
        self._execution_service = TaskExecutionService(
            persistence.execution, self._authorization,
            port=StoredExecutionManagementAdapter(
                persistence.execution, persistence.object_store(PersistenceDomain.EXECUTION),
                object_key_factory=key_factory, object_reader=persistence.read_object,
            ),
            sessions=persistence.conversation.sessions, task_admissions=persistence.task.admissions,
            release_terminal=persistence.retention.release_execution_handoff,
            payload_policy=payload_policy,
        )
        self._execution_reads = TaskExecutionReads(self._execution_service, self._agent_execution_ports)
        self._history = (TaskHistory(self._execution_service, self._agent_execution_ports)
                         if any(task.agent_execution is not None for task in self._declarations.values()) else None)
        self._input_materializer = ExecutionInputMaterializer(
            None, PromptLimits(), object_store=persistence.object_store(PersistenceDomain.TASK),
            object_key_factory=key_factory, payload_policy=PayloadPolicy(inline_limit_bytes=0),
            object_domain=PersistenceDomain.TASK,
        )
        self._input_captures = InputCaptures.from_persistence(
            persistence, authorization=self._authorization, execution=self._execution_reads,
            materializer=self._input_materializer,
        )
        self._tasks = MappingProxyType({key: self._bind_agent_task(value) for key, value in self._tasks.items()})
        self._task_node_runtime = TaskNodeExecutor(
            self._execution_service, namespace=self.namespace, app=self.context.app,
            execution_reads=self._execution_reads,
            authorization=self._authorization, task_state=persistence.task.tasks,
            task_admissions=persistence.task.admissions, execution_state=persistence.execution.executions,
            budgets=persistence.execution.budgets, task_objects=persistence.object_store(PersistenceDomain.TASK),
            artifact_state=persistence.artifact, artifact_objects=persistence.object_store(PersistenceDomain.ARTIFACT),
            object_key_factory=key_factory,
            binding_captures=TaskGraphBindingCaptureStore(self.namespace, persistence.object_store(PersistenceDomain.TASK)),
            tasks=tuple(self._tasks.values()), expanders=tuple(self._expanders.values()),
            input_captures=self._input_captures, input_materializer=self._input_materializer,
            task_durable=persistence.plan.route(PersistenceDomain.TASK).retention is StorageRetentionMode.DURABLE,
            execution_durable=persistence.plan.route(PersistenceDomain.EXECUTION).retention is StorageRetentionMode.DURABLE,
            recovery_durable=persistence.plan.route(PersistenceDomain.RECOVERY).retention is StorageRetentionMode.DURABLE,
        )
        self._launcher = LocalTaskGraphLauncher(
            persistence.task.tasks, self._task_node_runtime,
            owner=f"task-engine:{self.tenant_id}:{uuid.uuid4().hex}",
            acquire_execution_hold=self._execution_reads.acquire_dependency_hold,
            release_execution_hold=self._execution_reads.release_dependency_hold,
        )
        self._graph_service = DefaultTaskGraphService(
            persistence.task, self._authorization, self._launcher,
            local_waiter=self._launcher, preflight=self._task_node_runtime,
            bound_execution_recovery=self._task_node_runtime,
            metric_recorder=self._metrics,
            metric_source_namespace=self.namespace if self._metrics is not None else None,
        )
        event = DefaultEventService(
            persistence.execution.executions, persistence.execution.events, self._authorization,
            lambda *args, **kwargs: None,
        )
        self._tree_streamer = ExecutionTreeStreamer(
            self._execution_reads, ExecutionStreamCoordinator(self._execution_reads, event), ExecutionTreeBroker(),
        )

        self._execution_binding = ExecutionBinding(
            namespace=self.namespace, service=self._execution_service,
            ensure_open=self._ensure_open, register_observation=self._register_observation,
            release_observation=self._release_observation,
            capture_tree=self._capture_execution_tree, replay_tree=self._replay_execution_tree,
        )

    @property
    def metrics(self) -> MetricsControl:
        return self._metrics_control

    def _metric_status(self) -> MetricRecorderStatus:
        return disabled_metric_status() if self._metrics is None else self._metrics.status()

    async def _flush_metrics(self, *, timeout_seconds: float = 5.0) -> MetricFlushResult:
        if self._metrics is None:
            return MetricFlushResult(True, self._metric_status())
        return await self._metrics.flush(timeout_seconds=timeout_seconds)

    def _bind_agent_task(self, task: Task[AppT]) -> Task[AppT]:
        port = task.agent_execution
        if port is None:
            if isinstance(task.runner, AgentTaskRunner):
                task.runner.validate_binding(self.persistence, task)
            elif task.contract.get("type") == "agent":
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            return task
        if port.persistence is not self.persistence:
            raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        port.operations.ensure_open()
        task_id, revision, build_input = task.id, task.revision, task.build_input
        compiled, binding_contract = port.compiled, port.binding_contract
        input_mode = "projected" if build_input is not None else "literal"

        async def start_execution(
            invocation: TaskNodeInvocation,
            prompt: CanonicalUserInput,
            *,
            files: Sequence[str],
            session_id: str | None,
            memory_scope: str | None,
            planning: bool,
            thinking: ThinkingValue,
            idempotency_key: str,
            dependency_hold_id: str,
        ) -> Execution[AppT]:
            output_type: type[BaseModel] | OutputBinding | None = (
                invocation.node.output_type
            )
            if output_type is None and invocation.node.output_contract is not None:
                output_contract = invocation.node.output_contract
                mode = output_contract.get("mode")
                schema = output_contract.get("schema")
                if mode is not None and schema is not None:
                    output_type = restore_output(mode, schema)
            if not isinstance(output_type, OutputBinding) and (
                not isinstance(output_type, type)
                or not issubclass(output_type, BaseModel)
            ):
                output_type = restore_output(binding_contract.output_mode, binding_contract.output_schema)
            admissions = self._task_admissions
            if admissions is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            admission = await admissions.get(
                invocation.graph_id, tenant_id=invocation.principal.tenant_id,
            )
            if admission is None or admission.graph_id != invocation.graph_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if admission.budget_scope_id is not None:
                if self._budgets is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                usage = await self._budgets.read(admission.budget_scope_id)
                if usage.limits != admission.budget:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            execution = await port.start(
                prompt,
                files=files,
                output=output_type,
                principal=invocation.principal,
                session_id=session_id,
                idempotency_key=idempotency_key,
                memory_scope=memory_scope,
                mode="run",
                planning=planning,
                thinking=thinking,
                correlation=invocation.correlation,
                dependency_hold_id=dependency_hold_id,
                budget_scope_id=admission.budget_scope_id,
                requires_task_invocation_capture=True,
                input_context=None if invocation.node.input.get("capture_context") is None else ExecutionInputContext.from_payload(invocation.node.input["capture_context"]),
            )
            self._agent_execution_ports[execution.execution_id] = port
            return execution

        async def record_invocation(execution_id: str, invocation: TaskNodeInvocation) -> None:
            if port.captures is not None:
                await port.captures.record_invocation(execution_id, invocation)

        async def get_execution(
            execution_id: str,
            principal: Principal,
        ) -> Execution[AppT]:
            self._agent_execution_ports[execution_id] = port
            return await port.get(execution_id, principal)

        async def acquire_execution_hold(
            execution_id: str,
            principal: Principal,
            hold_id: str,
        ) -> None:
            await port.service.acquire_dependency_hold(
                execution_id,
                tenant_id=principal.tenant_id,
                hold_id=hold_id,
            )

        async def release_execution_hold(
            execution_id: str,
            principal: Principal,
            hold_id: str,
        ) -> None:
            await port.service.release_dependency_hold(
                execution_id,
                tenant_id=principal.tenant_id,
                hold_id=hold_id,
            )

        async def store_prepared_prompt(
            value: CanonicalUserInput, *, files: Sequence[str], tenant_id: str,
        ) -> StoredUserInput:
            content = await port.materialize_input(value, files)
            return await self._input_materializer.store(content, tenant_id=tenant_id)

        runner = AgentTaskRunner[AppT](
            id=task_id,
            revision=revision,
            persistence_owner=self.persistence,
            execution_port=port,
            agent_id=compiled.spec.id,
            agent_revision=compiled.spec.revision,
            input_mode=input_mode,
            planning_default=compiled.spec.planning,
            thinking_default=compiled.spec.thinking,
            binding_contract=binding_contract.to_payload(),
            build_input=build_input,
            start_execution=start_execution,
            record_invocation=record_invocation,
            inspect_execution=lambda execution_id, principal: port.service.inspect(
                execution_id, principal=principal,
            ),
            get_execution=get_execution,
            acquire_execution_hold=acquire_execution_hold,
            release_execution_hold=release_execution_hold,
            result_reader=lambda *args, **kwargs: self._task_node_runtime.read_input_result(*args, **kwargs),
            result_ref_reader=lambda *args, **kwargs: self._task_node_runtime.read_input_result_ref(*args, **kwargs),
            get_prepared_input=lambda *args, **kwargs: self._task_node_runtime.get_prepared_agent_input(*args, **kwargs),
            publish_prepared_input=lambda *args, **kwargs: self._task_node_runtime.publish_prepared_agent_input(*args, **kwargs),
            store_prepared_prompt=store_prepared_prompt,
            restore_prepared_prompt=lambda *args, **kwargs: self._task_node_runtime.restore_prepared_agent_prompt(*args, **kwargs),
        )
        contract: dict[str, JsonValue] = {
            "version": 1,
            "type": "agent",
            "effect_policy": "none",
            "output_contract": {"kind": "json"},
            "reconcile": False,
            "config": {
                "agent_id": compiled.spec.id,
                "agent_revision": compiled.spec.revision,
                "binding_contract": binding_contract.to_payload(),
                "input_mode": input_mode,
            },
        }
        if dict(task.contract) != contract:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return Task.from_runner(
            task_id,
            runner,
            revision=revision,
            contract=contract,
        )


    @classmethod
    @asynccontextmanager
    async def open(
        cls, *definitions: Task[AppT, object] | TaskExpander,
        persistence: Persistence,
        context: ExecutionContext[AppT] | None = None,
        authorization: AuthorizationPolicy | None = None,
        metrics: MetricRecorderProvider | None = None,
    ) -> AsyncIterator["TaskEngine[AppT]"]:
        engine = cls(*definitions, persistence=persistence, context=context, authorization=authorization, metrics=metrics)
        try:
            yield engine
        finally:
            await await_stream_cleanup(engine.close(), None)

    @classmethod
    def _definition_maps(
        cls, definitions: Sequence[Task[AppT, object] | TaskExpander],
    ) -> tuple[Mapping[tuple[str, int], Task[AppT, object]], Mapping[tuple[str, int], TaskExpander]]:
        tasks: dict[tuple[str, int], Task[AppT, object]] = {}
        expanders: dict[tuple[str, int], TaskExpander] = {}
        for definition in definitions:
            if isinstance(definition, Task):
                identity = definition.ref.id, definition.ref.revision
                if identity in tasks:
                    raise AIError(ErrorCode.BINDING_CONFLICT)
                tasks[identity] = definition
            elif isinstance(definition, TaskExpander):
                identity = definition.id, definition.revision
                if identity in expanders:
                    raise AIError(ErrorCode.BINDING_CONFLICT)
                expanders[identity] = definition
            else:
                raise TypeError("definitions must be Task or TaskExpander")
        return MappingProxyType(dict(sorted(tasks.items()))), MappingProxyType(dict(sorted(expanders.items())))

    def shares_execution_owner(self, other: TaskEngine[object]) -> bool:
        """Whether two definition views borrow the same task execution lifetime."""
        return self._owner is other._owner

    @property
    def authorization(self) -> AuthorizationPolicy:
        return self._authorization

    @property
    def inputs(self) -> InputCaptures:
        return self._input_captures

    @property
    def persistence(self) -> Persistence:
        return self._persistence

    @property
    def context(self) -> ExecutionContext[AppT]:
        return self._context

    @property
    def namespace(self) -> str:
        return self._persistence.namespace

    @property
    def tenant_id(self) -> str:
        return self._persistence.tenant_id

    @property
    def correlation(self) -> CorrelationData:
        return self._context.correlation

    def _ensure_open(self) -> None:
        if self._closed or self._owner._closed or self._owner._closing or not self._persistence.ready:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, retryable=False)

    def _resolve_principal(self, principal: Principal | None) -> Principal:
        selected = self._default_principal if principal is None else principal
        if selected.tenant_id != self.tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return selected

    def _require_task_node_runtime(self) -> TaskNodeExecutor[AppT]:
        return self._task_node_runtime

    async def _inspect_execution(self, execution_id: str, *, principal: Principal) -> ExecutionView:
        return await self._execution_reads.inspect(execution_id, principal=principal)

    async def _watch_execution_tree(self, execution_id: str, *, principal: Principal,
        after_event_seqs: Mapping[str, int] | None = None, include_content: bool = False,
        ready: asyncio.Event | None = None,
    ) -> AsyncIterator[ExecutionTreeEvent]:
        view = await self._execution_reads.inspect(execution_id, principal=principal)
        port = None if view.binding_kind == "task" else (
            self._agent_execution_ports.get(execution_id)
            or self._agent_execution_ports.get(view.root_execution_id)
        )
        watch = self._tree_streamer.stream if port is None else port.watch_tree
        async with aclosing(watch(execution_id, principal=principal,
            after_event_seqs=after_event_seqs, include_content=include_content, ready=ready)) as stream:
            async for event in stream:
                yield event

    async def _capture_execution_tree(self, execution_id: str, *, principal: Principal,
        after_event_seqs: Mapping[str, int] | None = None,
    ) -> tuple[tuple[ExecutionView, int, int], ...]:
        return await self._tree_streamer.capture(execution_id, principal=principal, after_event_seqs=after_event_seqs)

    def _replay_execution_tree(self, captured: tuple[tuple[ExecutionView, int, int], ...], *,
        principal: Principal, after_event_seqs: Mapping[str, int] | None = None, include_content: bool = False,
    ) -> AsyncIterator[ExecutionTreeEvent]:
        return self._tree_streamer.replay(captured, principal=principal,
            after_event_seqs=after_event_seqs, include_content=include_content)

    def _register_observation(self, session: ObservationSession) -> None:
        self._ensure_open()
        self._observation_sessions.add(session)

    def _release_observation(self, session: ObservationSession) -> None:
        self._observation_sessions.discard(session)

    def add_close_callback(self, callback: Callable[[TaskEngine[AppT]], None]) -> None:
        """Notify an owner after successful cleanup; definition views share that lifetime.

        Callbacks run synchronously and are removed only after returning. A failed
        callback is retried by close without repeating successful resource cleanup.
        """
        owner = self._owner
        owner._close_callbacks.append(callback)
        if owner._closed:
            owner._notify_closed()

    def _notify_closed(self) -> None:
        if self._notifying_closed:
            return
        self._notifying_closed = True
        try:
            while self._close_callbacks:
                self._close_callbacks[0](self)
                self._close_callbacks.popleft()
        finally:
            self._notifying_closed = False

    async def close(self) -> None:
        if self._owner is not self:
            self._closed = True
            return
        if self._closed:
            self._notify_closed()
            return
        self._closing = True
        if self._close_task is None or (self._close_task.done() and (self._close_task.cancelled() or self._close_task.exception() is not None)):
            self._close_task = asyncio.create_task(self._cleanup(), name="linktools-task-engine-close")
        await asyncio.shield(self._close_task)

    async def _cleanup(self) -> None:
        sessions = tuple(self._observation_sessions)
        for session in sessions:
            session.stop()
        deadline = asyncio.get_running_loop().time() + max((session.close_timeout for session in sessions), default=0.0)
        for session in sessions:
            await session.close(deadline=deadline)
        for settlement in tuple(self._cancel_settlements):
            await settlement.close()
        await self._graph_service.drain_owned_finalizers()
        await self._graph_service.preflight_close()
        await self._launcher.shutdown()
        await self._graph_service.drain_metric_projector()
        await self._input_materializer.close()
        if self._metrics is not None:
            await self._metrics.close()
        self._closed = True
        self._notify_closed()

    @property
    def definitions(self) -> tuple[Task[AppT, object], ...]:
        return tuple(self._declarations.values())

    @property
    def expanders(self) -> tuple[TaskExpander, ...]:
        return tuple(self._expanders.values())

    def definition(self, ref: TaskRef[_ResultT]) -> Task[AppT, _ResultT]:
        try:
            return self._tasks[(ref.id, ref.revision)]
        except KeyError as error:
            raise AIError(ErrorCode.BINDING_NOT_REGISTERED) from error

    def expander_definition(self, ref: TaskExpanderRef) -> TaskExpander:
        try:
            return self._expanders[(ref.id, ref.revision)]
        except KeyError as error:
            raise AIError(ErrorCode.BINDING_NOT_REGISTERED) from error

    def with_definitions(self, *definitions: Task[AppT, object]) -> "TaskEngine[AppT]":
        self._ensure_open()
        tasks, expanders = self._definition_maps((*self._declarations.values(), *self._expanders.values(), *definitions))
        declarations = tasks
        tasks = MappingProxyType({key: self._bind_agent_task(value) for key, value in tasks.items()})
        view = copy(self)
        view._tasks, view._expanders = tasks, expanders
        view._declarations = declarations
        view._history = (TaskHistory(self._execution_service, self._agent_execution_ports)
                         if any(task.agent_execution is not None for task in declarations.values()) else None)
        view._owner = self._owner
        return view

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
            self, self._graph_service, graph.graph_id,
            submission.admission.principal, self._watch_execution_tree,
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
        runtime = self
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

    async def prepare_described_submission(
        self, submission: TaskGraphSubmission,
    ) -> bool:
        runtime = self
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
        runtime = self
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
        self._ensure_open()
        return await self._graph_service.cancel_submission(
            submission, principal=self._resolve_principal(principal),
            idempotency_key=idempotency_key,
        )

    async def get(
        self,
        graph_id: str,
        *,
        principal: Principal | None = None,
    ) -> TaskGraphRun[AppT]:
        runtime = self
        runtime._ensure_open()
        resolved_principal = runtime._resolve_principal(principal)
        await self._graph_service.inspect(graph_id, principal=resolved_principal)
        return TaskGraphRun(
            runtime,
            self._graph_service,
            graph_id,
            resolved_principal,
            runtime._watch_execution_tree,
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
        return await self._recover_pending_tasks(
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
        runtime = self
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


    async def _task_execution(
        self,
        graph_id: str,
        node_id: str,
        execution_id: str,
        principal: Principal,
    ) -> Execution[AppT]:
        self._ensure_open()
        port = await self._agent_port_for_node(graph_id, node_id, principal)
        binding = self._execution_binding if port is None else port.operations
        watch = self._watch_execution_tree if port is None else port.watch_tree
        return Execution(binding,
            execution_id,
            principal,
            watch,
            None,
            lambda idempotency_key, force: self._cancel_task_execution(
                graph_id,
                node_id,
                execution_id,
                principal,
                idempotency_key,
                force,
            ),
        )


    def _bind_graph_history(self, state: TaskGraphState) -> None:
        nodes = {node.node_id: node for node in state.nodes}
        for node_state in state.node_states:
            node = nodes[node_state.node_id]
            if node_state.execution_id is None or node.task is None:
                continue
            definition = self._declarations.get((node.task.id, node.task.revision))
            if definition is not None and definition.agent_execution is not None:
                self._agent_execution_ports[node_state.execution_id] = definition.agent_execution

    async def _agent_port_for_node(self, graph_id: str, node_id: str, principal: Principal) -> "AgentExecutionPort[AppT] | None":
        state = await self._graph_service.state(graph_id, principal=principal)
        node = next((node for node in state.nodes if node.node_id == node_id), None)
        if node is None or node.task is None:
            return None
        definition = self._declarations.get((node.task.id, node.task.revision))
        if definition is None:
            raise AIError(ErrorCode.BINDING_NOT_REGISTERED)
        return definition.agent_execution

    async def _cancel_task_execution(
        self,
        graph_id: str,
        node_id: str,
        execution_id: str,
        principal: Principal,
        idempotency_key: str | None,
        force: bool,
    ) -> CancelExecutionResult:
        self._ensure_open()
        key = secrets.token_urlsafe(32) if idempotency_key is None else idempotency_key
        request = CancelGraphRequest(principal, key, force)
        view = await self._inspect_execution(
            execution_id,
            principal=principal,
        )
        if view.binding_kind == "task":
            await self._graph_service.settle_execution_cancellation(
                graph_id,
                node_id,
                execution_id,
                request,
                cancel_confirmed=None,
            )
        try:
            if view.binding_kind == "task":
                cancelled = await self._execution_service.cancel_task(
                    execution_id,
                    principal=principal,
                )
            else:
                port = await self._agent_port_for_node(graph_id, node_id, principal)
                if port is None:
                    raise AIError(ErrorCode.BINDING_NOT_REGISTERED)
                cancelled = await port.service.cancel(
                    execution_id,
                    CancelExecutionRequest(
                        principal,
                        key,
                        force,
                    ),
                )
        except AIError as error:
            if error.code is ErrorCode.TASK_EFFECT_UNKNOWN:
                await self._graph_service.settle_execution_cancellation(
                    graph_id,
                    node_id,
                    execution_id,
                    request,
                    cancel_confirmed=None,
                )
            raise

        if cancelled.cancelled:
            await self._graph_service.settle_execution_cancellation(
                graph_id,
                node_id,
                execution_id,
                request,
                cancel_confirmed=True,
            )
        else:
            await self._graph_service.settle_execution_cancellation(
                graph_id,
                node_id,
                execution_id,
                request,
                cancel_confirmed=False,
            )
        return cancelled


    async def _admit_graph(
        self,
        graph: TaskGraph,
        *,
        principal: "Principal | None",
        idempotency_key: str,
        limits: "TaskGraphLimits | None",
        budget: RunBudget | None = None,
        correlation: "Mapping[str, object] | None" = None,
    ) -> TaskGraphRequest:
        self._ensure_open()
        resolved_principal = self._resolve_principal(principal)
        effective_correlation = overlay_correlation(
            self.correlation,
            correlation,
        )
        selected_limits = limits or TaskGraphLimits()
        validate_idempotency_key(idempotency_key)
        graph.validate_limits(selected_limits)
        _logger.info(
            "task graph request prepared: graph=%s tenant=%s nodes=%s",
            graph.graph_id,
            resolved_principal.tenant_id,
            len(graph.nodes),
        )
        return TaskGraphRequest(
            graph,
            resolved_principal,
            idempotency_key,
            selected_limits,
            effective_correlation,
            budget,
        )


    async def _recover_pending_tasks(
        self,
        engine: TaskEngine[AppT],
        *,
        cursor: str | None,
        limit: int,
        principal: Principal | None,
    ) -> Page[TaskGraphResult]:
        self._ensure_open()
        admissions = self._task_admissions
        if admissions is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        resolved_principal = self._resolve_principal(principal)
        if resolved_principal.tenant_id != self.tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        page = await admissions.list_recoverable_page(
            cursor=cursor,
            limit=limit,
        )
        launches = tuple(page.items)
        for launch in launches:
            if launch.principal.tenant_id != resolved_principal.tenant_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            recovery_nodes = await self._graph_service.recovery_nodes(
                launch.graph_id,
                principal=resolved_principal,
            )
            await engine._activate_graph(
                launch.graph_id,
                launch.principal,
                recovery_nodes=recovery_nodes,
                recovery_principal=resolved_principal,
            )
        recovered: list[TaskGraphResult] = []
        for launch in launches:
            idempotency_key = "recover-pending-" + canonical_sha256(
                {
                    "namespace": self.namespace,
                    "principal": principal_identity_payload(resolved_principal),
                    "graph_id": launch.graph_id,
                }
            )
            recovered.append(
                await self._graph_service.recover(
                    launch.graph_id,
                    RecoverGraphRequest(resolved_principal, idempotency_key),
                )
            )
        return Page(tuple(recovered), page.next_cursor)


    async def budget_usage(
        self, graph_id: str, *, principal: Principal | None = None,
    ) -> BudgetUsage | None:
        self._ensure_open()
        resolved = self._resolve_principal(principal)
        await self._graph_service.inspect(graph_id, principal=resolved)
        admissions = self._task_admissions
        if admissions is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        admission = await admissions.get(graph_id, tenant_id=resolved.tenant_id)
        if admission is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if admission.budget_scope_id is None:
            return None
        budgets = self._budgets
        if budgets is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        usage = await budgets.read(admission.budget_scope_id)
        if usage.limits != admission.budget:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return usage


__all__ = ["TaskEngine"]
