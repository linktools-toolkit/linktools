#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public Runtime composition boundary and runtime-bound convenience behavior."""

import asyncio
import math
import secrets
from dataclasses import replace
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Generic, Protocol, TypedDict, TypeVar, overload

from linktools.core import environ
from pydantic import BaseModel

from ..agent import (
    AgentBinding,
    AgentBindingContract,
    AgentCatalog,
    AgentCompiler,
    CompiledAgent,
    OutputBinding,
)
from ._features import RuntimeFeatures, RuntimeFeaturesCapture
from ..execution.handle import ExecutionBinding
from ..core import (
    RunBudget,
    CorrelationData,
    JsonValue,
    Principal,
    PrincipalKind,
    normalize_correlation,
    AuthorizationPolicy,
    TenantAuthorizationPolicy,
    validate_persistence_namespace,
    validate_tenant_id,
    validate_agent_id,
    validate_resource_id,
)
from ..errors import AIError, ErrorCode


if TYPE_CHECKING:
    from ..inputs import InputCaptures
    from ..history import History
    from ..asset import AssetStoreReader
    from ._factory import _AgentLifecycle
from .agent_execution import AgentExecutionPort
from ._agent import Agent, Session
from ..execution import Execution
from ._agent_binding_resolver import _AgentBindingResolver
from ..observe.stream import ObservationSession, await_stream_cleanup
from ._domains import RuntimeAgents, RuntimeExecutions, RuntimeMetrics, RuntimeSessions
from ..core import ExecutionContext
from ..model import normalize_input_files
from ..inputs import CanonicalUserInput
from ..observe import (MetricRecorderProvider, MetricRecorderStatus, MetricFlushResult, MetricsControl,
                       RuntimeMetricsStatus, RuntimeMetricsFlushResult)
from ..observe.service import disabled_metric_status
from ..execution import (
    ExecutionView,
    ApprovalService,
    ArtifactService,
    CloseSessionRequest,
    CreateSessionRequest,
    EventService,
    ExternalService,
    ExecutionService,
    ForkExecutionRequest,
    ForkSessionRequest,
    RetryExecutionRequest,
    SessionService,
    SessionView,
    UpdateSessionRequest,
    ExecutionTreeEvent,
)
from ..persistence import Persistence, PersistenceStatus, PersistenceStorage
from ..persistence.runs import RuntimeAgentRunStore
from ..task import Task, TaskExpander
from ..task.engine import TaskEngine
from ..evaluation.engine import Evaluations
from ._composition_domains import RuntimeTasks, RuntimeEvaluations
from ._lifecycle import _RuntimeCloseCoordinator

_logger = environ.get_logger("ai.runtime")
AppT = TypeVar("AppT")




class _MetricControl(Protocol):
    def status(self) -> MetricRecorderStatus: ...

    async def flush(
        self,
        *,
        timeout_seconds: float = 5.0,
    ) -> MetricFlushResult: ...


class _ExecutionTreeStreamer(Protocol):
    async def capture(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_event_seqs: Mapping[str, int] | None = None,
    ) -> tuple[tuple[ExecutionView, int, int], ...]: ...

    def replay(
        self,
        captured: tuple[tuple[ExecutionView, int, int], ...],
        *,
        principal: Principal,
        after_event_seqs: Mapping[str, int] | None = None,
        include_content: bool = False,
    ) -> AsyncIterator[ExecutionTreeEvent]: ...

    def stream(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_event_seqs: Mapping[str, int] | None = None,
        include_content: bool = False,
        ready: asyncio.Event | None = None,
    ) -> AsyncIterator[ExecutionTreeEvent]: ...


def _request_correlation(value: "Mapping[str, object] | None") -> CorrelationData:
    try:
        return normalize_correlation(value)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error




class RuntimeStatus(TypedDict):
    namespace: str
    tenant_id: str
    storage: tuple[PersistenceStatus, ...] | None
    metrics: RuntimeMetricsStatus


class Runtime(Generic[AppT]):
    """Immutable Runtime composition and service graph."""

    def __init__(
        self, *, persistence: Persistence, namespace: str, tenant_id: str,
        owns_persistence: bool, context: ExecutionContext[AppT] | None = None,
        features: RuntimeFeatures[AppT] | RuntimeFeaturesCapture[AppT] | None = None,
        metrics: MetricRecorderProvider | None = None, authorization: AuthorizationPolicy | None = None,
        auto_recover: bool = True,
    ) -> None:
        if not isinstance(persistence, Persistence):
            raise TypeError("persistence must be Persistence")
        if context is not None and not isinstance(context, ExecutionContext):
            raise TypeError("context must be ExecutionContext")
        if features is not None and not isinstance(features, (RuntimeFeatures, RuntimeFeaturesCapture)):
            raise TypeError("features must be RuntimeFeatures or RuntimeFeaturesCapture")
        if metrics is not None and not isinstance(metrics, MetricRecorderProvider):
            raise TypeError("metrics must be MetricRecorderProvider or None")
        if not isinstance(auto_recover, bool):
            raise TypeError("auto_recover must be bool")
        self._namespace = validate_persistence_namespace(namespace)
        self._tenant_id = validate_tenant_id(tenant_id)
        self._state = persistence
        self._owns_persistence = owns_persistence
        self._persistence_cleanup: Callable[[], Awaitable[None]] | None = None
        self._context = ExecutionContext(None) if context is None else context
        self._features = features
        self._metrics_dataset = metrics
        self._authorization = TenantAuthorizationPolicy(self._tenant_id) if authorization is None else authorization
        self._auto_recover = auto_recover
        self._default_principal = Principal("runtime", self._tenant_id, PrincipalKind.LOCAL_TRUSTED.value)
        self._entered = False
        self._ready = False
        self._closed = False
        self._closing = False
        self._agent_ready = False
        self._handoff_requested = False
        self._close_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._close_coordinator: _RuntimeCloseCoordinator | None = None
        self._agent_lifecycle: _AgentLifecycle | None = None
        self._close_callback: Callable[[], Awaitable[None]] | None = None
        self._handoff_callback: Callable[[], Awaitable[None]] | None = None
        self._metric_control: _MetricControl | None = None
        self._agent_metric_control = MetricsControl(self._metric_status, self._flush_metrics)
        self._observation_sessions: set[ObservationSession] = set()
        self._task_engines: dict[TaskEngine[AppT], None] = {}
        self._evaluations: dict[Evaluations, None] = {}
        self.agents = RuntimeAgents(self._get_agent, self._restore_agent)
        self.tasks = RuntimeTasks(self._bind_tasks)
        self.evaluations = RuntimeEvaluations(self._bind_evaluations)
        self.metrics = RuntimeMetrics(self._metric_controls)

    @classmethod
    def open(
        cls, namespace: str, *, storage: PersistenceStorage, tenant_id: str = "default",
        context: ExecutionContext[AppT] | None = None,
        features: RuntimeFeatures[AppT] | RuntimeFeaturesCapture[AppT] | None = None,
        metrics: MetricRecorderProvider | None = None, authorization: AuthorizationPolicy | None = None,
        auto_recover: bool = True,
    ) -> "Runtime[AppT]":
        """Own persistence and only the producer domains explicitly composed here.

        Keep the returned Runtime before entering when the host needs to retry
        cleanup after an initialization failure. Features opt into the Agent
        domain; omitting them does not start an Agent producer.
        """
        return cls(
            persistence=Persistence(storage), namespace=namespace, tenant_id=tenant_id,
            owns_persistence=True, context=context, features=features, metrics=metrics,
            authorization=authorization, auto_recover=auto_recover,
        )

    @classmethod
    def attach(
        cls, *, persistence: Persistence, context: ExecutionContext[AppT] | None = None,
        features: RuntimeFeatures[AppT] | RuntimeFeaturesCapture[AppT] | None = None,
        metrics: MetricRecorderProvider | None = None, authorization: AuthorizationPolicy | None = None,
        auto_recover: bool = True,
    ) -> "Runtime[AppT]":
        """Borrow an open Persistence; its provider retains its close responsibility."""
        if not isinstance(persistence, Persistence) or not persistence.ready or persistence.read_only:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "persistence must be open and writable")
        return cls(
            persistence=persistence, namespace=persistence.namespace, tenant_id=persistence.tenant_id,
            owns_persistence=False, context=context, features=features, metrics=metrics,
            authorization=authorization, auto_recover=auto_recover,
        )

    async def __aenter__(self) -> "Runtime[AppT]":
        initializing = False
        try:
            async with self._close_lock:
                if self._entered or self._closing or self._closed:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "Runtime cannot be entered twice")
                initializing = True
                self._entered = True
                if self._owns_persistence:
                    await self._state.initialize(namespace=self.namespace, tenant_id=self.tenant_id)
                elif not self._state.ready or self._state.read_only:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                self._persistence_cleanup = await self._state.bind_runtime(
                    execution_producer=self._features is not None,
                    owns_persistence=self._owns_persistence,
                )
                if self._features is not None:
                    from ._factory import _AgentLifecycle, compose_runtime_components

                    self._agent_lifecycle = _AgentLifecycle()
                    components = await compose_runtime_components(
                        persistence=self._state, app=self.app, features=self._features,
                        metrics=self._metrics_dataset, authorization=self._authorization,
                        auto_recover=self._auto_recover, lifecycle=self._agent_lifecycle,
                    )
                    if components.metric_control is not None:
                        components.metric_control.configure_runtime_dimensions(self.context.metric_dimensions)
                    self._bind_agent_components(
                        components.catalog, components.compiler, components.execution,
                        components.session, components.approval, components.external,
                        components.event, components.artifact, components.history,
                        persistence=self._state, run_store=components.run_store, context=self.context,
                        close_callback=components.close_callback, handoff_callback=components.handoff_callback,
                        tree_streamer=components.tree_streamer, metric_control=components.metric_control,
                        _binding_resolver=components.binding_resolver, input_captures=components.input_captures,
                        memory_scope=components.memory_scope,
                    )
                self._ready = True
            return self
        except BaseException as primary:
            if not initializing:
                raise
            try:
                await await_stream_cleanup(self.close(), None)
            except BaseException as cleanup_error:
                _log_secondary_cleanup("runtime.initialize", cleanup_error)
                raise primary from cleanup_error
            raise

    async def __aexit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> None:
        try:
            await await_stream_cleanup(self.close(), None)
        except BaseException as cleanup_error:
            if exc is not None:
                _log_secondary_cleanup("runtime.body", cleanup_error)
                raise exc from cleanup_error
            raise

    async def _bind_tasks(self, *definitions: Task[AppT, object] | TaskExpander) -> TaskEngine[AppT]:
        async with self._close_lock:
            self._ensure_open()
            engine = TaskEngine(
                *definitions, persistence=self._state, context=self.context,
                authorization=self._authorization, metrics=self._metrics_dataset,
            )
            self._task_engines[engine] = None
            engine.add_close_callback(self._release_task_engine)
            return engine

    async def _bind_evaluations(
        self, engine: TaskEngine[AppT], *, assets: Sequence["AssetStoreReader"] = (),
    ) -> Evaluations:
        async with self._close_lock:
            self._ensure_open()
            evaluations = await Evaluations.create(engine, assets=assets)
            self._evaluations[evaluations] = None
            evaluations.add_close_callback(self._release_evaluations)
            return evaluations

    def _release_task_engine(self, engine: TaskEngine[AppT]) -> None:
        self._task_engines.pop(engine, None)

    def _release_evaluations(self, evaluations: Evaluations) -> None:
        self._evaluations.pop(evaluations, None)

    def _metric_controls(self) -> tuple[MetricsControl, ...]:
        controls = (() if not self._agent_ready else (self._agent_metric_control,))
        return (*controls, *(engine.metrics for engine in self._task_engines))

    def _require_agents(self) -> None:
        if not self._agent_ready:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                          safe_details={"reason": "agent_domain_not_configured"})

    def _bind_agent_components(
        self,
        catalog: AgentCatalog,
        compiler: AgentCompiler,
        execution: ExecutionService,
        session: SessionService,
        approval: ApprovalService,
        external: ExternalService,
        event: EventService,
        artifact: ArtifactService,
        history: "History | None",
        *,
        persistence: Persistence,
        run_store: RuntimeAgentRunStore,
        context: ExecutionContext[AppT],
        close_callback: "Callable[[], Awaitable[None]] | None" = None,
        handoff_callback: "Callable[[], Awaitable[None]] | None" = None,
        tree_streamer: "_ExecutionTreeStreamer | None" = None,
        metric_control: "_MetricControl | None" = None,
        _binding_resolver: "_AgentBindingResolver | None" = None,
        input_captures: "InputCaptures | None" = None,
        memory_scope: str | None = None,
    ) -> None:
        if any(
            value is None
            for value in (
                catalog,
                compiler,
                execution,
                session,
                approval,
                external,
                event,
                artifact,
            )
        ):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if not isinstance(context, ExecutionContext):
            raise TypeError("context must be ExecutionContext")
        self._catalog = catalog
        self._compiler = compiler
        self._execution_service = execution
        self._input_captures = input_captures
        self._memory_scope = memory_scope
        self._executions = RuntimeExecutions(
            execution, self._get_execution, input_captures, ensure_open=self._ensure_open,
        )
        self._session_service = session
        self._sessions = RuntimeSessions(session, self._get_session, ensure_open=self._ensure_open)
        self._approvals = approval
        self._external_calls = external
        self._events = event
        self._artifacts = artifact
        self._history = history
        self._close_callback = close_callback
        self._handoff_callback = handoff_callback
        self._tree_streamer = tree_streamer
        self._metric_control = metric_control
        self._binding_resolver = _binding_resolver
        self._state = persistence
        self._run_store = run_store
        self._execution_binding = ExecutionBinding(
            namespace=self.namespace, service=self._execution_service,
            ensure_open=self._ensure_open, register_observation=self._register_observation,
            release_observation=self._release_observation,
            capture_tree=self._capture_execution_tree, replay_tree=self._replay_execution_tree,
            history=history, list_events=None if history is None else history.list_execution_events,
            retry=self._retry_execution, fork=self._fork_execution,
        )
        self._agent_ready = True

    @property
    def executions(self) -> RuntimeExecutions[AppT]:
        self._require_agents()
        return self._executions

    @property
    def sessions(self) -> RuntimeSessions[AppT]:
        self._require_agents()
        return self._sessions

    @property
    def approvals(self) -> ApprovalService:
        self._require_agents()
        return self._approvals

    @property
    def external_calls(self) -> ExternalService:
        self._require_agents()
        return self._external_calls

    @property
    def events(self) -> EventService:
        self._require_agents()
        return self._events

    @property
    def artifacts(self) -> ArtifactService:
        self._require_agents()
        return self._artifacts

    @property
    def history(self) -> "History | None":
        self._require_agents()
        return self._history

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def default_principal(self) -> Principal:
        return self._default_principal

    @property
    def namespace(self) -> str:
        return self._namespace

    def status(self) -> RuntimeStatus:
        """Read current composition and recorder state without a health probe."""
        self._ensure_open()
        return {
            "namespace": self.namespace,
            "tenant_id": self.tenant_id,
            "storage": self._state.status() if self._state is not None else None,
            "metrics": self.metrics.status(),
        }

    def _watch_execution_tree(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_event_seqs: Mapping[str, int] | None = None,
        include_content: bool = False,
        ready: asyncio.Event | None = None,
    ) -> AsyncIterator[ExecutionTreeEvent]:
        if self._tree_streamer is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._tree_streamer.stream(
            execution_id,
            principal=principal,
            after_event_seqs=after_event_seqs,
            include_content=include_content, ready=ready,
        )

    async def _capture_execution_tree(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_event_seqs: Mapping[str, int] | None = None,
    ) -> tuple[tuple[ExecutionView, int, int], ...]:
        if self._tree_streamer is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await self._tree_streamer.capture(
            execution_id, principal=principal, after_event_seqs=after_event_seqs,
        )

    def _replay_execution_tree(
        self,
        captured: tuple[tuple[ExecutionView, int, int], ...],
        *,
        principal: Principal,
        after_event_seqs: Mapping[str, int] | None = None,
        include_content: bool = False,
    ) -> AsyncIterator[ExecutionTreeEvent]:
        if self._tree_streamer is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._tree_streamer.replay(
            captured, principal=principal, after_event_seqs=after_event_seqs,
            include_content=include_content,
        )

    @property
    def app(self) -> AppT:
        return self._context.app

    @property
    def context(self) -> ExecutionContext[AppT]:
        return self._context

    @property
    def correlation(self) -> CorrelationData:
        return self._context.correlation

    def _metric_status(self) -> MetricRecorderStatus:
        control = self._metric_control
        return disabled_metric_status() if control is None else control.status()

    async def _flush_metrics(
        self,
        *,
        timeout_seconds: float = 5.0,
    ) -> MetricFlushResult:
        self._ensure_open()
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds < 0
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        control = self._metric_control
        if control is None:
            return MetricFlushResult(True, disabled_metric_status())
        return await control.flush(timeout_seconds=timeout_seconds)

    def _get_agent(self, agent_id: str = "default") -> "Agent[AppT]":
        self._ensure_open()
        self._require_agents()
        validate_agent_id(agent_id)
        root = self._catalog.root_agent(agent_id)
        return Agent(self, root.spec.id, root.spec.revision)

    def _restore_agent(self, binding: AgentBindingContract) -> Agent[AppT]:
        self._ensure_open()
        self._require_agents()
        restored = self._compiler.restore(binding)
        compiled = restored.compiled_agent
        return Agent(self, compiled.spec.id, compiled.spec.revision, compiled, binding)

    async def _get_execution(
        self,
        execution_id: str,
        principal: Principal | None,
    ) -> Execution[AppT]:
        self._ensure_open()
        validate_resource_id(execution_id)
        resolved_principal = self._resolve_principal(principal)
        await self.executions.inspect(execution_id, principal=resolved_principal)
        return Execution(self._execution_binding,
            execution_id,
            resolved_principal,
            self._watch_execution_tree,
        )

    async def _get_session(
        self,
        session_id: str,
        principal: Principal | None,
    ) -> Session[AppT]:
        self._ensure_open()
        validate_resource_id(session_id)
        resolved_principal = self._resolve_principal(principal)
        view = await self._session_service.get(
            session_id,
            principal=resolved_principal,
        )
        return Session(
            self,
            view.agent_id,
            None,
            view.session_id,
            resolved_principal,
        )

    def _derive_agent(
        self,
        agent: Agent[AppT],
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        instructions: Sequence[str] | None = None,
        allow_tools: Sequence[str] | None = None,
        allow_skills: Sequence[str] | None = None,
    ) -> "Agent[AppT]":
        self._ensure_open()
        if not isinstance(agent, Agent) or agent.runtime is not self:
            raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        root = self._compiled_agent(agent.id, agent.revision, agent.compiled)
        if all(
            value is None
            for value in (
                model,
                system_prompt,
                instructions,
                allow_tools,
                allow_skills,
            )
        ):
            return agent
        changes: dict[str, object] = {}
        if model is not None:
            changes["model"] = model
        if system_prompt is not None:
            changes["system_prompt"] = system_prompt
        if instructions is not None:
            changes["instructions"] = tuple(instructions)
        if allow_tools is not None:
            changes["allow_tools"] = tuple(allow_tools)
        if allow_skills is not None:
            changes["allow_skills"] = tuple(allow_skills)
        try:
            spec = replace(root.spec, **changes)
            derived = self._compiler.compile(spec)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
        root_tools = {
            (item.kind, item.id)
            for item in (*root.selected_tools, *root.selected_mcp)
        }
        derived_tools = {
            (item.kind, item.id)
            for item in (*derived.selected_tools, *derived.selected_mcp)
        }
        if not derived_tools.issubset(root_tools):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if not {
            item.id for item in derived.selected_skills
        }.issubset({item.id for item in root.selected_skills}):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        return Agent(self, derived.spec.id, derived.spec.revision, derived)

    def _compiled_agent(
        self,
        agent_id: str,
        agent_revision: int,
        compiled_agent: "CompiledAgent | None" = None,
    ) -> CompiledAgent:
        self._ensure_open()
        if compiled_agent is None:
            compiled_agent = self._catalog.root_agent(agent_id)
        if (
            compiled_agent.spec.id != agent_id
            or compiled_agent.spec.revision != agent_revision
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return compiled_agent

    async def _resolve_agent_binding(self, binding: AgentBinding) -> AgentBinding:
        if self._binding_resolver is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await self._binding_resolver.resolve(binding)

    def _agent_execution_port(
        self, agent_id: str, revision: int, compiled: CompiledAgent | None = None,
        binding_contract: AgentBindingContract | None = None,
    ) -> AgentExecutionPort[AppT]:
        self._ensure_open()
        selected = self._compiled_agent(agent_id, revision, compiled)
        resolver = self._binding_resolver
        if resolver is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return AgentExecutionPort(
            self._state, selected, binding_contract,
            self._compiler, self._execution_service, self._session_service, self._execution_binding,
            self._watch_execution_tree, resolver.resolve, self._default_principal, self.correlation,
            self._input_captures, self._memory_scope, binding_contract is not None,
        )

    async def _retry_execution(
        self,
        execution_id: str,
        user_prompt: CanonicalUserInput,
        *,
        files: Sequence[str],
        principal: Principal,
        idempotency_key: "str | None",
        correlation: "Mapping[str, object] | None" = None,
    ) -> "Execution[AppT]":
        self._ensure_open()
        request = RetryExecutionRequest(
            user_prompt=user_prompt,
            principal=principal,
            idempotency_key=secrets.token_urlsafe(32) if idempotency_key is None else idempotency_key,
            correlation=_request_correlation(correlation),
            files=normalize_input_files(files),
        )
        handle = await self.executions.retry(execution_id, request)
        return Execution(self._execution_binding,
            handle.execution_id,
            principal,
            self._watch_execution_tree,
        )

    async def _fork_execution(
        self,
        execution_id: str,
        user_prompt: CanonicalUserInput,
        *,
        files: Sequence[str],
        budget: RunBudget | None = None,
        principal: Principal,
        idempotency_key: "str | None",
        correlation: "Mapping[str, object] | None" = None,
    ) -> "Execution[AppT]":
        self._ensure_open()
        request = ForkExecutionRequest(
            user_prompt=user_prompt,
            principal=principal,
            idempotency_key=secrets.token_urlsafe(32) if idempotency_key is None else idempotency_key,
            correlation=_request_correlation(correlation),
            files=normalize_input_files(files),
            budget=budget,
        )
        handle = await self.executions.fork(execution_id, request)
        return Execution(self._execution_binding,
            handle.execution_id,
            principal,
            self._watch_execution_tree,
        )

    async def _create_session_for_agent(
        self,
        agent_id: str,
        session_id: str,
        *,
        principal: "Principal | None",
        cwd: "str | None",
        metadata: "Mapping[str, JsonValue] | None",
        idempotency_key: "str | None",
    ) -> SessionView:
        resolved_principal = self._resolve_principal(principal)
        validate_agent_id(agent_id)
        values = dict(metadata or {})
        if any(key.startswith("linktools.ai.") for key in values):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return await self.sessions.create(
            agent_id,
            CreateSessionRequest(
                resolved_principal,
                session_id,
                secrets.token_urlsafe(32) if idempotency_key is None else idempotency_key,
                cwd,
                values,
            ),
        )

    async def _fork_session(
        self,
        agent_id: str,
        session_id: str,
        new_session_id: str,
        *,
        principal: "Principal | None",
        idempotency_key: "str | None",
        cwd: "str | None",
    ) -> None:
        resolved_principal = self._resolve_principal(principal)
        await self.sessions.fork(
            agent_id,
            session_id,
            ForkSessionRequest(
                resolved_principal,
                new_session_id,
                secrets.token_urlsafe(32) if idempotency_key is None else idempotency_key,
                cwd,
            ),
        )

    async def _update_session(
        self,
        agent_id: str,
        session_id: str,
        *,
        expected_revision: int,
        metadata: Mapping[str, JsonValue],
        principal: "Principal | None",
        idempotency_key: "str | None",
        cwd: "str | None",
    ) -> SessionView:
        resolved_principal = self._resolve_principal(principal)
        return await self.sessions.update(
            agent_id,
            session_id,
            UpdateSessionRequest(
                resolved_principal,
                expected_revision,
                secrets.token_urlsafe(32) if idempotency_key is None else idempotency_key,
                metadata,
                cwd,
            ),
        )

    async def _close_session(
        self,
        session_id: str,
        *,
        principal: "Principal | None",
        idempotency_key: "str | None",
        force: bool,
        wait_timeout_seconds: int,
    ) -> SessionView:
        resolved_principal = self._resolve_principal(principal)
        return await self.sessions.close(
            session_id,
            CloseSessionRequest(
                resolved_principal,
                secrets.token_urlsafe(32) if idempotency_key is None else idempotency_key,
                force,
                wait_timeout_seconds,
            ),
        )








    def _ensure_open(self) -> None:
        if not self._ready or self._closed or self._closing or not self._state.ready:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, retryable=False)


    def _resolve_principal(self, principal: "Principal | None") -> Principal:
        if principal is not None:
            return principal
        return self._default_principal



    async def handoff(self) -> None:
        """Close this Runtime without cancelling its checkpointed executions.

        A fresh Runtime must explicitly recover the original execution IDs.
        Requires durable synchronous admission and CheckpointCapability.
        Unknown external effects still require explicit recovery resolution.
        """
        await self._close(handoff=True)

    async def close(self) -> None:
        """Close owned resources, cancelling unfinished work by default."""
        await self._close(handoff=False)

    async def _close(self, *, handoff: bool) -> None:
        async with self._close_lock:
            if handoff and self._closing and not self._handoff_requested:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                    safe_details={"phase": "runtime_handoff", "reason": "close_already_started"})
            if self._closed:
                return
            if not self._closing:
                if handoff:
                    if self._handoff_callback is None:
                        raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING,
                            safe_details={"capability": "durable_runtime_handoff"})
                    for domain in self._evaluations:
                        domain.preflight_handoff()
                    self._handoff_requested = True
                for domain in self._evaluations:
                    domain.stop_admission()
                self._closing = True
                for session in tuple(self._observation_sessions):
                    session.stop()
                _logger.info("runtime close started: tenant=%s", self.tenant_id)
            task = self._close_task
            retry = task is None
            if task is not None and task.done():
                try:
                    task.result()
                except (asyncio.CancelledError, Exception):
                    retry = True
                else:
                    if self._closed:
                        return
                    retry = True
            if retry:
                task = asyncio.create_task(
                    self._cleanup(),
                    name="linktools-runtime-close",
                )
                task.add_done_callback(self._consume_close_result)
                self._close_task = task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                try:
                    task.result()
                except asyncio.CancelledError:
                    pass
                except BaseException as error:  # noqa: BLE001
                    _log_secondary_cleanup("runtime.close", error)
            raise
        except BaseException:
            async with self._close_lock:
                if self._close_task is task:
                    self._close_task = None
            raise

    def _consume_close_result(self, task: "asyncio.Task[None]") -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except BaseException as error:  # noqa: BLE001
            _log_secondary_cleanup("runtime.close", error)

    def _register_observation(self, session: ObservationSession) -> None:
        self._ensure_open()
        self._observation_sessions.add(session)

    def _release_observation(self, session: ObservationSession) -> None:
        self._observation_sessions.discard(session)

    async def _cleanup(self) -> None:
        if self._close_coordinator is None:
            actions: list[Callable[[], Awaitable[None]]] = [
                *(domain.close for domain in reversed(self._evaluations)),
                *(engine.close for engine in reversed(self._task_engines)),
                self._close_agent,
            ]
            if self._persistence_cleanup is not None:
                actions.append(self._persistence_cleanup)
            elif self._owns_persistence:
                actions.append(self._state.close)
            self._close_coordinator = _RuntimeCloseCoordinator(tuple(actions))
            del actions
        await self._close_coordinator.close()
        async with self._close_lock:
            self._closed = True
            self._ready = False
            _logger.info("runtime close completed: tenant=%s", self.tenant_id)

    async def _close_agent(self) -> None:
        sessions = tuple(self._observation_sessions)
        for session in sessions:
            session.stop()
        deadline = asyncio.get_running_loop().time() + max(
            (session.close_timeout for session in sessions), default=0.0,
        )
        for session in sessions:
            await session.close(deadline=deadline)
        if self._agent_lifecycle is not None:
            if self._handoff_requested:
                await self._agent_lifecycle.handoff()
            else:
                await self._agent_lifecycle.close()


def _log_secondary_cleanup(phase: str, error: BaseException) -> None:
    code = error.code.value if isinstance(error, AIError) else None
    _logger.error(
        "secondary cleanup failed: phase=%s code=%s exception_type=%s",
        phase,
        code,
        type(error).__name__,
    )


__all__ = ["Runtime", "RuntimeStatus"]
