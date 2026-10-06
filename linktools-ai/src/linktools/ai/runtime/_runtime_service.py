#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public Runtime composition boundary and runtime-bound convenience behavior."""

import asyncio
import inspect
import secrets
from dataclasses import replace
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, overload

from linktools.core import environ
from pydantic import BaseModel

from ..agent import (
    AgentBinding,
    AgentBindingContract,
    AgentCatalog,
    AgentCompiler,
    CompiledAgent,
    OutputBinding,
    restore_output,
)
from ..capability import CapabilityGroup, CapabilityGroupCapture
from ..core import (
    CorrelationData,
    ExecutionMode,
    JsonValue,
    Principal,
    PrincipalKind,
    SessionStatus,
    TaskStatus,
    ThinkingValue,
    PromptLimits,
    Page,
    normalize_correlation,
    normalize_execution_mode,
    normalize_thinking,
    overlay_correlation,
    canonical_sha256,
    principal_identity_payload,
    validate_agent_id,
    validate_idempotency_key,
    validate_memory_scope,
    validate_persistence_namespace,
    validate_resource_id,
)
from ..errors import AIError, ErrorCode
from ..model import ModelRegistry

if TYPE_CHECKING:
    from ._evaluation import RuntimeEvaluations
    from ._input_capture import RuntimeInputCaptures
    from ..observe import Metrics
    from ..task import TaskResultRecord
    from ._runtime_history import RuntimeHistory
    from .state._contracts import (
        StoredUserInput,
        TaskAdmissionRepository,
        TaskPreparedInputRecord,
    )
from ..task import (
    CancelGraphRequest,
    Task,
    TaskExpander,
    TaskGraphAdmission,
    TaskGraph,
    TaskGraphLimits,
    TaskGraphRequest,
    TaskGraphResult,
    TaskGraphService,
    RecoverGraphRequest,
    TaskResultRef,
    TaskNode,
    TaskNodeInvocation,
    TaskExpanderRef,
)
from ._agent import Agent, Execution, Session
from ._agent_binding_resolver import _AgentBindingResolver
from ._task import TaskGraphRun
from ._task_observation import _ObservationSession
from ._tasks import RuntimeTasks, TaskEngine
from ._domains import RuntimeAgents, RuntimeExecutions, RuntimeMetrics, RuntimeSessions
from ._agent_task import RuntimeAgentTaskRunner
from ._agent_task_input import AgentTaskInputBuilder
from ._context import RuntimeContext
from ._execution_context import ExecutionInputContext
from ._input_contract import normalize_input_files
from ._input import CanonicalUserInput
from ._metrics import (
    MetricFlushResult,
    MetricBufferStatus,
    _disabled_metric_status,
)
from .service_api import (
    ApprovalService,
    ArtifactService,
    CancelExecutionRequest,
    CancelExecutionResult,
    CloseSessionRequest,
    CreateSessionRequest,
    EventService,
    ExternalService,
    ExecutionRequest,
    ExecutionResult,
    ExecutionService,
    ForkExecutionRequest,
    ForkSessionRequest,
    ResumeSessionRequest,
    RetryExecutionRequest,
    SessionService,
    SessionView,
    UpdateSessionRequest,
    ExecutionTreeEvent,
    TaskGraphRunEvent,
)
from .state import RuntimeStorage

_logger = environ.get_logger("ai.runtime")
AppT = TypeVar("AppT")


class _TaskNodeRuntimePort(Protocol):
    async def activate_graph(
        self,
        graph: TaskGraph,
        tasks: Sequence[Task[AppT]],
        expanders: Sequence[TaskExpander],
        *,
        track_pre_admission: bool = False,
    ) -> object | None: ...

    async def finish_graph_activation(
        self,
        graph_id: str,
        tenant_id: str,
        activation: object,
        *,
        admitted: bool,
    ) -> None: ...

    async def load_admission(self, admission: TaskGraphAdmission) -> None: ...

    def admit_node(self, node: "TaskNode") -> "TaskNode": ...

    async def get_result_record(
        self,
        graph_id: str,
        node_id: str,
        *,
        tenant_id: str,
    ) -> "TaskResultRecord | None": ...

    async def get_result_records(
        self,
        graph_id: str,
        node_ids: tuple[str, ...],
        *,
        tenant_id: str,
    ) -> "Mapping[str, TaskResultRecord]": ...

    async def read_result_record(
        self,
        record: "TaskResultRecord",
        *,
        principal: "Principal | None" = None,
    ) -> JsonValue: ...

    async def result_payload_size(
        self,
        record: "TaskResultRecord",
        *,
        principal: "Principal",
    ) -> int: ...

    async def read_execution_failure(
        self,
        execution_id: str,
        *,
        principal: "Principal",
    ) -> "ExecutionResult | None": ...

    async def read_input_result(
        self,
        invocation: "TaskNodeInvocation",
        name: str,
    ) -> JsonValue: ...

    async def read_input_result_ref(
        self,
        invocation: "TaskNodeInvocation",
        name: str,
    ) -> TaskResultRef: ...

    async def restore_prepared_agent_prompt(
        self,
        value: "StoredUserInput",
    ) -> CanonicalUserInput: ...

    async def store_prepared_agent_prompt(
        self,
        value: CanonicalUserInput,
        *,
        files: Sequence[str],
        tenant_id: str,
    ) -> "StoredUserInput": ...

    async def get_prepared_agent_input(
        self,
        invocation: "TaskNodeInvocation",
    ) -> "TaskPreparedInputRecord | None": ...

    async def publish_prepared_agent_input(
        self,
        invocation: "TaskNodeInvocation",
        *,
        input_identity: str,
        source_refs: tuple[tuple[str, TaskResultRef], ...],
        stored_user_input: "StoredUserInput",
        final_input_digest: str,
        request_identity: str,
    ) -> "TaskPreparedInputRecord": ...


class _MetricControl(Protocol):
    def status(self) -> MetricBufferStatus: ...

    async def flush(
        self,
        *,
        timeout_seconds: float = 5.0,
    ) -> MetricFlushResult: ...


class _ExecutionTreeStreamer(Protocol):
    def stream(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_sequences: Mapping[str, int] | None = None,
        include_content: bool = False,
    ) -> AsyncIterator[ExecutionTreeEvent]: ...


def _request_correlation(value: "Mapping[str, object] | None") -> CorrelationData:
    try:
        return normalize_correlation(value)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error


def _overlay_request_correlation(
    base: Mapping[str, object],
    overlay: "Mapping[str, object] | None",
) -> CorrelationData:
    try:
        return overlay_correlation(base, overlay)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error


class Runtime(Generic[AppT]):
    """Immutable Runtime composition and service graph."""

    def __init__(
        self,
        catalog: AgentCatalog,
        compiler: AgentCompiler,
        execution: ExecutionService,
        session: SessionService,
        graph: TaskGraphService,
        evaluation: "RuntimeEvaluations",
        approval: ApprovalService,
        external: ExternalService,
        event: EventService,
        artifact: ArtifactService,
        history: "RuntimeHistory | None",
        *,
        namespace: str,
        context: RuntimeContext[AppT],
        close_callback: "Callable[[], Awaitable[None]] | None" = None,
        task_node_runtime: "_TaskNodeRuntimePort | None" = None,
        task_admissions: "TaskAdmissionRepository | None" = None,
        tree_streamer: "_ExecutionTreeStreamer | None" = None,
        metric_control: "_MetricControl | None" = None,
        _binding_resolver: "_AgentBindingResolver | None" = None,
        input_captures: "RuntimeInputCaptures | None" = None,
    ) -> None:
        if any(
            value is None
            for value in (
                catalog,
                compiler,
                execution,
                session,
                graph,
                evaluation,
                approval,
                external,
                event,
                artifact,
            )
        ):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if not isinstance(context, RuntimeContext):
            raise TypeError("context must be RuntimeContext")
        self._catalog = catalog
        self._compiler = compiler
        self._task_owner_token = object()
        self._execution_service = execution
        self._input_captures = input_captures
        self.executions = RuntimeExecutions(execution, self._get_execution, input_captures)
        self._session_service = session
        self.sessions = RuntimeSessions(session, self._get_session)
        self._graph_service = graph
        self.evaluations = evaluation
        self.approvals = approval
        self.external_calls = external
        self.events = event
        self.artifacts = artifact
        self.history = history
        self._namespace = validate_persistence_namespace(namespace)
        self._context = context
        self._default_principal = Principal(
            principal_id="runtime",
            tenant_id=self._context.tenant_id,
            kind=PrincipalKind.LOCAL_TRUSTED.value,
        )
        self._close_callback = close_callback
        self._task_node_runtime = task_node_runtime
        self._task_admissions = task_admissions
        self.tasks = RuntimeTasks(self, graph)
        self.agents = RuntimeAgents(self._get_agent)
        self.metrics = RuntimeMetrics(self._metric_status, self._flush_metrics)
        self._tree_streamer = tree_streamer
        self._metric_control = metric_control
        self._binding_resolver = _binding_resolver
        self._closed = False
        self._closing = False
        self._observation_sessions: set[_ObservationSession] = set()
        self._close_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None

    @classmethod
    @overload
    def open(
        cls,
        namespace: str,
        *,
        models: ModelRegistry,
        storage: RuntimeStorage,
        context: None = None,
        capabilities: "Sequence[CapabilityGroup[None] | CapabilityGroupCapture[None]]" = (),
        metrics: "Metrics | None" = None,
        limits: "PromptLimits | None" = None,
    ) -> "AbstractAsyncContextManager[Runtime[None]]": ...

    @classmethod
    @overload
    def open(
        cls,
        namespace: str,
        *,
        models: ModelRegistry,
        storage: RuntimeStorage,
        context: RuntimeContext[AppT],
        capabilities: "Sequence[CapabilityGroup[AppT] | CapabilityGroupCapture[AppT]]" = (),
        metrics: "Metrics | None" = None,
        limits: "PromptLimits | None" = None,
    ) -> "AbstractAsyncContextManager[Runtime[AppT]]": ...

    @classmethod
    def open(
        cls,
        namespace: str,
        *,
        models: ModelRegistry,
        storage: RuntimeStorage,
        context: "RuntimeContext[object] | None" = None,
        capabilities: "Sequence[CapabilityGroup[object] | CapabilityGroupCapture[object]]" = (),
        metrics: "Metrics | None" = None,
        limits: "PromptLimits | None" = None,
    ) -> "AbstractAsyncContextManager[Runtime[object]]":
        resolved_namespace = validate_persistence_namespace(namespace)
        root_context = RuntimeContext(None) if context is None else context
        if not isinstance(root_context, RuntimeContext):
            raise TypeError("context must be RuntimeContext")
        selected_limits = PromptLimits() if limits is None else limits
        return _open_runtime(
            resolved_namespace,
            context=root_context,
            models=models,
            storage=storage,
            capabilities=capabilities,
            metrics=metrics,
            limits=selected_limits,
        )

    @property
    def tenant_id(self) -> str:
        return self._context.tenant_id

    @property
    def default_principal(self) -> Principal:
        return self._default_principal

    @property
    def namespace(self) -> str:
        return self._namespace

    def _watch_execution_tree(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_sequences: Mapping[str, int] | None = None,
        include_content: bool = False,
    ) -> AsyncIterator[ExecutionTreeEvent]:
        if self._tree_streamer is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._tree_streamer.stream(
            execution_id,
            principal=principal,
            after_sequences=after_sequences,
            include_content=include_content,
        )

    @property
    def app(self) -> AppT:
        return self._context.app

    @property
    def context(self) -> RuntimeContext[AppT]:
        return self._context

    @property
    def correlation(self) -> CorrelationData:
        return self._context.correlation

    def _metric_status(self) -> MetricBufferStatus:
        control = self._metric_control
        return _disabled_metric_status() if control is None else control.status()

    async def _flush_metrics(
        self,
        *,
        timeout_seconds: float = 5.0,
    ) -> MetricFlushResult:
        self._ensure_open()
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds < 0
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        control = self._metric_control
        if control is None:
            return MetricFlushResult(True, _disabled_metric_status())
        return await control.flush(timeout_seconds=timeout_seconds)

    def _get_agent(self, agent_id: str = "default") -> "Agent[AppT]":
        self._ensure_open()
        validate_agent_id(agent_id)
        root = self._catalog.root_agent(agent_id)
        return Agent(self, root.spec.id, root.spec.revision)

    async def _get_execution(
        self,
        execution_id: str,
        principal: Principal | None,
    ) -> Execution[AppT]:
        self._ensure_open()
        validate_resource_id(execution_id)
        resolved_principal = self._resolve_principal(principal)
        await self.executions.inspect(execution_id, principal=resolved_principal)
        return Execution(
            self,
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

    def _bind_agent(
        self,
        agent_id: str,
        agent_revision: int,
        *,
        output: "type[BaseModel] | None" = None,
        compiled_agent: "CompiledAgent | None" = None,
    ) -> AgentBinding:
        resolved = self._compiled_agent(agent_id, agent_revision, compiled_agent)
        return self._compiler.bind(resolved, output=output)

    async def _resolve_agent_binding(self, binding: AgentBinding) -> AgentBinding:
        if self._binding_resolver is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await self._binding_resolver.resolve(binding)

    async def _start_for_agent(
        self,
        agent_id: str,
        agent_revision: int,
        user_prompt: CanonicalUserInput,
        *,
        files: Sequence[str],
        output: "type[BaseModel] | OutputBinding | None",
        principal: "Principal | None",
        session_id: "str | None",
        idempotency_key: "str | None",
        memory_scope: "str | None",
        mode: ExecutionMode,
        planning: "bool | None",
        thinking: "ThinkingValue | None",
        correlation: "Mapping[str, object] | None" = None,
        compiled_agent: "CompiledAgent | None" = None,
        dependency_hold_id: "str | None" = None,
        requires_task_invocation_capture: bool = False,
        input_context: ExecutionInputContext | None = None,
    ) -> "Execution[AppT]":
        self._ensure_open()
        if input_context is not None and session_id is not None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "imported_context_requires_new_execution"})
        resolved_principal = self._resolve_principal(principal)
        effective_correlation = _overlay_request_correlation(
            self.correlation,
            correlation,
        )
        resolved_files = normalize_input_files(files)
        compiled_agent = self._compiled_agent(agent_id, agent_revision, compiled_agent)
        resolved_mode, resolved_planning, resolved_thinking = _execution_policy(
            compiled_agent,
            mode=mode,
            planning=planning,
            thinking=thinking,
        )
        binding = await self._resolve_agent_binding(
            self._compiler.bind(compiled_agent, output=output)
        )
        request = ExecutionRequest(
            user_prompt=user_prompt,
            principal=resolved_principal,
            idempotency_key=idempotency_key or secrets.token_urlsafe(32),
            memory_scope=_validate_memory_scope(memory_scope),
            mode=resolved_mode,
            planning=resolved_planning,
            thinking=resolved_thinking,
            correlation=effective_correlation,
            files=resolved_files,
            input_context=input_context,
        )
        if session_id is None:
            handle = await self._execution_service.start(
                binding.binding_digest,
                request,
                dependency_hold_id=dependency_hold_id,
                requires_task_invocation_capture=requires_task_invocation_capture,
                binding_contract=binding.binding_contract,
            )
        else:
            if not isinstance(session_id, str) or not session_id.strip():
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            await self._ensure_session(compiled_agent, session_id, resolved_principal)
            resume_request = ResumeSessionRequest(
                principal=resolved_principal,
                user_prompt=request.user_prompt,
                idempotency_key=request.idempotency_key,
                memory_scope=request.memory_scope,
                mode=request.mode,
                planning=request.planning,
                thinking=request.thinking,
                correlation=effective_correlation,
                files=request.files,
            )
            handle = await self._session_service.resume(
                compiled_agent.spec.id,
                binding.binding_digest,
                session_id,
                resume_request,
                binding_contract=binding.binding_contract,
                dependency_hold_id=dependency_hold_id,
                requires_task_invocation_capture=requires_task_invocation_capture,
            )
        _logger.info(
            "runtime execution admitted: execution=%s agent=%s session=%s "
            "mode=%s planning=%s thinking=%s",
            handle.execution_id,
            compiled_agent.spec.id,
            session_id,
            resolved_mode,
            resolved_planning,
            resolved_thinking,
        )
        return Execution(
            self,
            handle.execution_id,
            resolved_principal,
            self._watch_execution_tree,
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
            idempotency_key=idempotency_key or secrets.token_urlsafe(32),
            correlation=_request_correlation(correlation),
            files=normalize_input_files(files),
        )
        handle = await self.executions.retry(execution_id, request)
        return Execution(
            self,
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
        principal: Principal,
        idempotency_key: "str | None",
        correlation: "Mapping[str, object] | None" = None,
    ) -> "Execution[AppT]":
        self._ensure_open()
        request = ForkExecutionRequest(
            user_prompt=user_prompt,
            principal=principal,
            idempotency_key=idempotency_key or secrets.token_urlsafe(32),
            correlation=_request_correlation(correlation),
            files=normalize_input_files(files),
        )
        handle = await self.executions.fork(execution_id, request)
        return Execution(
            self,
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
                idempotency_key or secrets.token_urlsafe(32),
                cwd,
                values,
            ),
        )

    async def _fork_session(
        self,
        agent_id: str,
        agent_revision: int,
        session_id: str,
        new_session_id: str,
        *,
        principal: "Principal | None",
        idempotency_key: "str | None",
        cwd: "str | None",
        compiled_agent: "CompiledAgent | None" = None,
    ) -> "Session[AppT]":
        resolved_principal = self._resolve_principal(principal)
        await self.sessions.fork(
            agent_id,
            session_id,
            ForkSessionRequest(
                resolved_principal,
                new_session_id,
                idempotency_key or secrets.token_urlsafe(32),
                cwd,
            ),
        )
        return Session(
            self,
            agent_id,
            agent_revision,
            new_session_id,
            resolved_principal,
            compiled_agent,
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
                idempotency_key or secrets.token_urlsafe(32),
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
                idempotency_key or secrets.token_urlsafe(32),
                force,
                wait_timeout_seconds,
            ),
        )

    def _task_from_agent_capture(self, task_id: str, binding: AgentBindingContract, *, revision: int) -> Task[AppT]:
        restored = self._compiler.restore(binding)
        compiled = restored.compiled_agent
        agent = Agent(self, compiled.spec.id, compiled.spec.revision, compiled)
        return self._task_from_agent(task_id, agent, revision=revision, build_input=None, binding_contract=binding)

    def _task_from_agent(
        self,
        task_id: str,
        agent: Agent[AppT],
        *,
        revision: int,
        build_input: AgentTaskInputBuilder | None,
        binding_contract: AgentBindingContract | None = None,
    ) -> Task[AppT]:
        self._ensure_open()
        if not isinstance(agent, Agent) or agent.runtime is not self:
            raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        if build_input is not None and not inspect.iscoroutinefunction(build_input):
            raise TypeError("build_input must be async")
        compiled = self._compiled_agent(agent.id, agent.revision, agent.compiled)
        binding_contract = binding_contract or self._compiler.bind(compiled).binding_contract
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
            execution = await self._start_for_agent(
                agent.id,
                agent.revision,
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
                compiled_agent=compiled,
                dependency_hold_id=dependency_hold_id,
                requires_task_invocation_capture=True,
                input_context=None if invocation.node.input.get("capture_context") is None else ExecutionInputContext.from_payload(invocation.node.input["capture_context"]),
            )
            return execution

        async def record_invocation(execution_id: str, invocation: TaskNodeInvocation) -> None:
            if self._input_captures is not None:
                await self._input_captures.record_invocation(execution_id, invocation)

        async def get_execution(
            execution_id: str,
            principal: Principal,
        ) -> Execution[AppT]:
            return Execution(self, execution_id, principal, self._watch_execution_tree)

        async def acquire_execution_hold(
            execution_id: str,
            principal: Principal,
            hold_id: str,
        ) -> None:
            await self._execution_service.acquire_dependency_hold(
                execution_id,
                tenant_id=principal.tenant_id,
                hold_id=hold_id,
            )

        async def release_execution_hold(
            execution_id: str,
            principal: Principal,
            hold_id: str,
        ) -> None:
            await self._execution_service.release_dependency_hold(
                execution_id,
                tenant_id=principal.tenant_id,
                hold_id=hold_id,
            )

        node_runtime = self._require_task_node_runtime()
        runner = RuntimeAgentTaskRunner[AppT](
            id=task_id,
            revision=revision,
            runtime_owner=self._task_owner_token,
            agent_id=compiled.spec.id,
            agent_revision=compiled.spec.revision,
            input_mode=input_mode,
            planning_default=compiled.spec.planning,
            thinking_default=compiled.spec.thinking,
            binding_contract=binding_contract.to_payload(),
            build_input=build_input,
            start_execution=start_execution,
            record_invocation=record_invocation,
            get_execution=get_execution,
            acquire_execution_hold=acquire_execution_hold,
            release_execution_hold=release_execution_hold,
            result_reader=node_runtime.read_input_result,
            result_ref_reader=node_runtime.read_input_result_ref,
            get_prepared_input=node_runtime.get_prepared_agent_input,
            publish_prepared_input=node_runtime.publish_prepared_agent_input,
            store_prepared_prompt=node_runtime.store_prepared_agent_prompt,
            restore_prepared_prompt=node_runtime.restore_prepared_agent_prompt,
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
        return Task.from_runner(
            task_id,
            runner,
            revision=revision,
            contract=contract,
        )

    def _task_execution(
        self,
        graph_id: str,
        node_id: str,
        execution_id: str,
        principal: Principal,
    ) -> Execution[AppT]:
        self._ensure_open()
        return Execution(
            self,
            execution_id,
            principal,
            self._watch_execution_tree,
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

    async def _cancel_task_execution(
        self,
        graph_id: str,
        node_id: str,
        execution_id: str,
        principal: Principal,
        idempotency_key: str | None,
        force: bool,
    ) -> CancelExecutionResult:
        view = await self.executions.inspect(
            execution_id,
            principal=principal,
        )
        key = idempotency_key or secrets.token_urlsafe(32)
        request = CancelGraphRequest(principal, key, force)
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
                cancelled = await self.executions.cancel(
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
        correlation: "Mapping[str, object] | None" = None,
    ) -> TaskGraphRequest:
        self._ensure_open()
        resolved_principal = self._resolve_principal(principal)
        effective_correlation = _overlay_request_correlation(
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
        )

    async def _ensure_session(
        self,
        compiled_agent: CompiledAgent,
        session_id: str,
        principal: Principal,
    ) -> None:
        try:
            session = await self._session_service.get(
                session_id,
                principal=principal,
            )
        except AIError as error:
            if error.code not in {
                ErrorCode.SESSION_NOT_FOUND,
                ErrorCode.AUTHORIZATION_DENIED,
            }:
                raise
            try:
                await self.sessions.create(
                    compiled_agent.spec.id,
                    CreateSessionRequest(
                        principal,
                        session_id,
                        secrets.token_urlsafe(32),
                        None,
                        {},
                    ),
                )
                return
            except AIError as create_error:
                if create_error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
            session = await self._session_service.get(
                session_id,
                principal=principal,
            )
        if session.status is not SessionStatus.OPEN:
            raise AIError(ErrorCode.SESSION_CONFLICT)
        if session.agent_id != compiled_agent.spec.id:
            raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)

    def _ensure_open(self) -> None:
        if self._closed or self._closing:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    def _validate_task_bindings(self, tasks: Sequence[Task[AppT]]) -> None:
        self._ensure_open()
        for task in tasks:
            runner = task.runner
            if isinstance(runner, RuntimeAgentTaskRunner):
                runner.validate_binding(self._task_owner_token, task)
            elif task.contract.get("type") == "agent":
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

    def _resolve_principal(self, principal: "Principal | None") -> Principal:
        if principal is not None:
            return principal
        return self._default_principal

    def _require_task_node_runtime(self) -> _TaskNodeRuntimePort:
        if self._task_node_runtime is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._task_node_runtime

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

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            if not self._closing:
                self._closing = True
                for session in tuple(self._observation_sessions):
                    session.seal()
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

    def _register_observation(self, session: _ObservationSession) -> None:
        self._ensure_open()
        self._observation_sessions.add(session)

    def _release_observation(self, session: _ObservationSession) -> None:
        self._observation_sessions.discard(session)

    async def _cleanup(self) -> None:
        sessions = tuple(self._observation_sessions)
        for session in sessions:
            session.seal()
        deadline = asyncio.get_running_loop().time() + max(
            (session.close_timeout for session in sessions), default=0.0,
        )
        for session in sessions:
            await session.close(deadline=deadline)
        if self._close_callback is not None:
            await self._close_callback()
        async with self._close_lock:
            self._closed = True
            _logger.info("runtime close completed: tenant=%s", self.tenant_id)


def _execution_policy(
    compiled_agent: CompiledAgent,
    *,
    mode: ExecutionMode,
    planning: "bool | None",
    thinking: "ThinkingValue | None",
) -> "tuple[ExecutionMode, bool, ThinkingValue]":
    resolved_mode = normalize_execution_mode(mode)
    if planning is not None and not isinstance(planning, bool):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    resolved_planning = (
        compiled_agent.spec.planning
        if planning is None
        else planning
    )
    if resolved_mode == "plan":
        resolved_planning = True
    resolved_thinking = (
        compiled_agent.spec.thinking if thinking is None else normalize_thinking(thinking)
    )
    return resolved_mode, resolved_planning, resolved_thinking


def _validate_memory_scope(value: "str | None") -> "str | None":
    if value is not None:
        validate_memory_scope(value)
    return value


@asynccontextmanager
async def _open_runtime(
    namespace: str,
    *,
    context: RuntimeContext[object],
    models: ModelRegistry,
    storage: RuntimeStorage,
    capabilities: "Sequence[CapabilityGroup[object] | CapabilityGroupCapture[object]]",
    metrics: "Metrics | None",
    limits: PromptLimits,
):
    from ._factory import compose_runtime_components

    components = await compose_runtime_components(
        namespace,
        app=context.app,
        tenant_id=context.tenant_id,
        models=models,
        storage=storage,
        capabilities=capabilities,
        metrics=metrics,
        limits=limits,
    )
    try:
        if components.metric_control is not None:
            components.metric_control.configure_runtime_dimensions(
                context.metric_dimensions
            )
        runtime = Runtime(
            components.catalog,
            components.compiler,
            components.execution,
            components.session,
            components.graph,
            components.evaluation,
            components.approval,
            components.external,
            components.event,
            components.artifact,
            getattr(components, "history", None),
            namespace=namespace,
            context=context,
            close_callback=components.close_callback,
            task_node_runtime=components.task_node_runtime,
            task_admissions=components.task_admissions,
            tree_streamer=components.tree_streamer,
            metric_control=components.metric_control,
            _binding_resolver=components.binding_resolver,
            input_captures=components.input_captures,
        )
    except BaseException:
        try:
            await components.close_callback()
        except BaseException as error:
            _log_secondary_cleanup("runtime.construct", error)
        raise
    try:
        yield runtime
    except BaseException:
        try:
            await runtime.close()
        except BaseException as error:
            _log_secondary_cleanup("runtime.body", error)
        raise
    else:
        await runtime.close()

def _log_secondary_cleanup(phase: str, error: BaseException) -> None:
    code = error.code.value if isinstance(error, AIError) else None
    _logger.error(
        "secondary cleanup failed: phase=%s code=%s exception_type=%s",
        phase,
        code,
        type(error).__name__,
    )


__all__ = ["Runtime"]
