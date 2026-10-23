#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime composition and local service graph construction."""

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from copy import copy
from pathlib import Path
from typing import TypeVar

from linktools.core import environ
from pydantic_ai.capabilities import AbstractCapability, WrapperCapability

from ..asset import AssetStoreReader
from ..agent import AgentCatalog, AgentCompiler
from ..capability import (
    ToolResponsesCapability,
    SessionCapability,
    ExecutionCapability,
    AgentContext,
    AssetCapability,
    AssetSkillSource,
    CapabilityContribution,
    SkillSourceRegistry,
)
from ._features import (
    RuntimeFeatures,
    RuntimeFeaturesCapture,
)
from ..core import (
    CommitPolicy,
    AuthorizationAction,
    AuthorizationPolicy,
    HmacCursorSigner,
    PromptLimits,
    TenantAuthorizationPolicy,
    validate_persistence_namespace,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode
from ..session import SessionPendingExecutionLookup
from ..observe import Metrics
from ..spec import (
    AgentSpec,
    RuleInstructionResolver,
    RepositoryInstructionResolver,
    RepositoryInstructions,
)
from ..storage import BatchWriteScope, CacheBackend, ObjectStore, PayloadPolicy, StreamBackend, TurnCoordinationBackend
from ..task import DefaultTaskGraphService, LocalTaskGraphLauncher
from ..workspace import (
    WorkspaceInstructionResolver,
    LocalSandbox,
    Sandbox,
    Workspace,
    WorkspaceAccess,
)
from ._agent_executor import AgentExecutor
from ._approval import DefaultApprovalService
from ._agent_binding_resolver import _AgentBindingResolver
from ._artifact import DefaultArtifactService
from ._coordinator import _LocalRuntimeCoordinator
from ._evaluation import RuntimeEvaluations
from ._event import DefaultEventService, LiveExecutionEventBroker
from ._external import DefaultExternalService
from ._execution import DefaultExecutionService, _ExecutionRuntimeBridge
from .state._contracts import ExecutionRecord
from ._shared_stream import SharedStreamPublisher
from ._execution_tree import ExecutionTreeBroker, ExecutionTreeStreamer
from ._history_projection import StepExecutionHistoryReader, StepSessionHistoryReader
from ._history_service import DefaultExecutionHistoryService
from ._input import ExecutionInputMaterializer
from ._input_capture import RuntimeInputCaptures
from ._context_archive import ExecutionContextArchive
from ._local import LocalExecutionBackend
from ._pending_coordination import _PendingTurnCoordination
from ._memory import MemoryStore, RuntimeMemoryStore
from ._object import RuntimeObjectKeyFactory
from ._planner import RuntimeTaskNodeRunner
from ._capabilities import runtime_feature_leaves
from ..storage import StorageRoute
from .persistence import CapabilityPersistence
from ._capability_writes import RuntimeCapabilityWrites
from ._metrics_service import MetricsService
from ._persistence_configuration import persistence_storage_configuration, capability_storage_configuration
from linktools.ai.capability import TraceCapability, CheckpointCapability
from ._runtime_history import RuntimeHistory
from ._task_graph_binding_capture import TaskGraphBindingCaptureStore
from ._runtime_identity import token_seed
from ._session import RuntimeSessionService
from ._subagent import SubagentDispatcher
from ._transient_history import TransientExecutionHistoryStore
from ..capability import ToolResponseFixture
from .service_api import CancelExecutionRequest, ExecutionHistoryReader, SessionHistoryReader
from ..storage import StorageRetentionMode
from .state import RuntimeDomain, RuntimeStorage, RuntimeState
from .state._contracts import BudgetRepository, TaskAdmissionRepository
from .state._step_archive import StagingAgentRunStore

AppT = TypeVar("AppT")
_logger = environ.get_logger("ai.runtime.factory")


@dataclass(frozen=True, slots=True)
class _RuntimeComponents:
    state: RuntimeState
    catalog: AgentCatalog
    compiler: AgentCompiler
    execution: DefaultExecutionService
    session: RuntimeSessionService
    graph: DefaultTaskGraphService
    evaluation: RuntimeEvaluations
    approval: DefaultApprovalService
    external: DefaultExternalService
    event: DefaultEventService
    artifact: DefaultArtifactService
    tenant_id: str
    close_callback: Callable[[], Awaitable[None]]
    handoff_callback: Callable[[], Awaitable[None]] | None
    task_node_runtime: RuntimeTaskNodeRunner[object]
    tree_streamer: ExecutionTreeStreamer
    metric_control: MetricsService | None
    binding_resolver: _AgentBindingResolver
    history: object
    task_admissions: TaskAdmissionRepository
    budgets: BudgetRepository
    input_captures: RuntimeInputCaptures


async def compose_runtime_components(
    namespace: str,
    *,
    app: "AppT | None" = None,
    tenant_id: "str | None" = None,
    storage: RuntimeStorage,
    metrics: Metrics | None = None,
    features: "RuntimeFeatures[AppT] | RuntimeFeaturesCapture[AppT] | None" = None,
    auto_recover: bool = True,
) -> _RuntimeComponents:
    """Capture declarations and build Runtime-private services."""
    resolved_namespace = validate_persistence_namespace(namespace)
    if not isinstance(storage, RuntimeStorage):
        raise TypeError("storage must be RuntimeStorage")
    if features is not None and not isinstance(features, (RuntimeFeatures, RuntimeFeaturesCapture)):
        raise TypeError("features must be RuntimeFeatures or RuntimeFeaturesCapture")
    captured = await features.capture() if isinstance(features, RuntimeFeatures) else features
    if captured is not None:
        await captured.verify_source_revision()
    selected_limits = PromptLimits() if captured is None else captured.limits
    authorization = None if captured is None else captured.authorization
    execution_capability = ExecutionCapability()
    execution_declared = False
    session_capability = SessionCapability()
    session_declared = False
    tool_responses: ToolResponseFixture | None = None
    workspace = None if captured is None else captured.workspace
    sandbox = LocalSandbox() if captured is None or captured.sandbox is None else captured.sandbox
    if workspace is not None:
        workspace.policy.validate()

    selected_state: RuntimeState | None = None
    workspace_access: WorkspaceAccess | None = None
    input_materializer: ExecutionInputMaterializer | None = None
    initialized = False
    ownership_transferred = False
    try:
        candidates: list[CapabilityContribution[object]] = []
        asset_sources: dict[str, AssetStoreReader] = {}
        if captured is not None:
            candidates.extend(captured.contributions)
            if captured.asset_reader is not None:
                asset_sources[captured.group_id] = captured.asset_reader
        _validate_candidate_uniqueness(candidates)
        def require_runtime_feature_scope(capability: AbstractCapability[object]) -> None:
            if isinstance(capability, CapabilityPersistence):
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID, "Persistent capabilities require runtime scope")
            if isinstance(capability, (
                CheckpointCapability,
                TraceCapability,
                ToolResponsesCapability,
                SessionCapability,
                ExecutionCapability,
            )):
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID, "Runtime feature capabilities require Runtime scope")

        for candidate in candidates:
            if candidate.kind == "capability" and candidate.scope != "runtime":
                for leaf in runtime_feature_leaves(candidate.value):
                    require_runtime_feature_scope(leaf)
        declared_ids: set[str] = set()
        for candidate in candidates:
            if candidate.scope != "runtime":
                continue
            for leaf in runtime_feature_leaves(candidate.value):
                if leaf.id is None:
                    continue
                if leaf.id in declared_ids:
                    raise AIError(ErrorCode.CAPABILITY_CONFLICT)
                declared_ids.add(leaf.id)

        def consume_runtime_declaration(capability: AbstractCapability[AgentContext[AppT]]) -> AbstractCapability[AgentContext[AppT]] | None:
            nonlocal tool_responses, session_capability, session_declared, execution_capability, execution_declared
            if isinstance(capability, ExecutionCapability):
                if execution_declared:
                    raise AIError(ErrorCode.CAPABILITY_CONFLICT)
                execution_capability, execution_declared = capability, True
                return capability
            if isinstance(capability, SessionCapability):
                if session_declared:
                    raise AIError(ErrorCode.CAPABILITY_CONFLICT)
                session_capability, session_declared = capability, True
                return None
            if isinstance(capability, ToolResponsesCapability):
                if tool_responses is not None:
                    raise AIError(ErrorCode.CAPABILITY_CONFLICT)
                if not isinstance(capability.fixture, ToolResponseFixture):
                    raise TypeError("fixture must be ToolResponseFixture")
                tool_responses = capability.fixture
                return None
            if isinstance(capability, AssetCapability):
                return None
            if isinstance(capability, WrapperCapability):
                wrapped = capability.wrapped.visit_and_replace(consume_runtime_declaration)
                if wrapped is None:
                    return None
                rebound = copy(capability)
                rebound.wrapped = wrapped
                return rebound
            return capability

        capability_routes: dict[str, StorageRoute] = {}
        capability_revisions: dict[str, int] = {}
        runtime_capability_values = []
        for candidate in candidates:
            if candidate.scope != "runtime":
                continue

            def capture_persistence(capability: AbstractCapability[AgentContext[AppT]]) -> AbstractCapability[AgentContext[AppT]] | None:
                if isinstance(capability, WrapperCapability):
                    wrapped = capability.wrapped.visit_and_replace(capture_persistence)
                    if wrapped is None:
                        return None
                    rebound = copy(capability)
                    rebound.wrapped = wrapped
                    return rebound
                selected = consume_runtime_declaration(capability)
                if selected is None or not isinstance(selected, CapabilityPersistence):
                    return selected
                identity = selected.id
                if identity is None:
                    if capability is not candidate.value:
                        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID, "Persistent capability leaves require a registered identity")
                    identity = candidate.id
                if identity in capability_routes:
                    raise AIError(ErrorCode.CAPABILITY_CONFLICT)
                if selected.storage is not None and not isinstance(selected.storage, StorageRoute):
                    raise TypeError("Persistent capability storage must use StorageRoute")
                capability_routes[identity] = selected.storage or storage.plan.route(RuntimeDomain.EXECUTION)
                capability_revisions[identity] = candidate.revision
                bound = copy(selected)
                bound.id = identity
                return bound

            selected = candidate.value.visit_and_replace(capture_persistence)
            if selected is not None:
                runtime_capability_values.append(selected)
        if not execution_declared:
            runtime_capability_values.append(execution_capability)
        runtime_capabilities = tuple(runtime_capability_values)
        runtime_ids: set[str] = set()

        def register_runtime_id(capability: AbstractCapability[AgentContext[AppT]]) -> None:
            if capability.id is not None:
                if capability.id in runtime_ids:
                    raise AIError(ErrorCode.CAPABILITY_CONFLICT)
                runtime_ids.add(capability.id)

        for capability in runtime_capabilities:
            capability.apply(register_runtime_id)
        module_routes = persistence_storage_configuration(runtime_capabilities)
        capability_routes = capability_storage_configuration(
            tuple((capability.id or "", capability) for capability in runtime_capabilities),
            default_route=storage.plan.route(RuntimeDomain.EXECUTION),
        )
        candidates = [candidate for candidate in candidates if candidate.scope == "agent"]
        skill_sources = SkillSourceRegistry(
            tuple(AssetSkillSource(group_id, reader) for group_id, reader in asset_sources.items())
        )
        rules = RepositoryInstructions(() if captured is None else captured.instructions.documents)
        agents = {
            candidate.id: candidate.value
            for candidate in candidates
            if candidate.kind == "agent"
        }
        models = {} if captured is None else captured.models
        if "default" not in agents and "default" in models:
            agents["default"] = AgentSpec("default")
        compiler = AgentCompiler(
            defaults=None if captured is None else captured.defaults,
            models=models,
            candidates=tuple(
                candidate
                for candidate in candidates
                if candidate.kind != "agent"
            ),
            agents=agents,
            capability_filter=None if captured is None else captured.filter,
        )
        root_agents = {
            agent_id: compiler.compile(agents[agent_id])
            for agent_id in sorted(agents)
        }
        catalog = AgentCatalog(root_agents)

        effective_tenant_id = (
            "default" if tenant_id is None else validate_tenant_id(tenant_id)
        )
        selected_state = RuntimeState(storage)
        await selected_state.initialize(
            namespace=resolved_namespace,
            tenant_id=effective_tenant_id,
            module_routes=module_routes,
            capability_routes=capability_routes,
        )

        if selected_state.capability_stores:
            selected_state.run_store.capability_writes = RuntimeCapabilityWrites(
                selected_state.capability_stores, namespace=resolved_namespace, tenant_id=effective_tenant_id,
                durable_capability_ids=selected_state.durable_capability_ids,
                require_active_run=selected_state.run_store.require_active_capability_run,
                has_active_execution=selected_state.run_store.has_active_execution_producer,
                marker_store=selected_state.capability_marker_store(),
            )
        selected_state.run_store.capability_revisions = capability_revisions
        initialized = True
        if captured is not None:
            await captured.verify_source_revision()
        if workspace is None:
            instruction_resolver: RepositoryInstructionResolver | None = (
                RuleInstructionResolver(rules)
                if rules.documents
                else None
            )
            workspace_access = None
            execution_cwd = _capture_host_cwd()
        else:
            instruction_resolver = WorkspaceInstructionResolver(
                workspace.root,
                workspace.policy,
                rules,
            )
            workspace_access = WorkspaceAccess.for_workspace(
                workspace,
                sandbox=sandbox,
            )
            execution_cwd = str(workspace.root)
        object_key_factory = RuntimeObjectKeyFactory(resolved_namespace)
        payload_policy = PayloadPolicy()
        input_materializer = ExecutionInputMaterializer(
            workspace_access,
            selected_limits,
            object_store=selected_state.object_store(RuntimeDomain.EXECUTION),
            object_key_factory=object_key_factory,
            payload_policy=payload_policy,
        )
        runtime_token_seed = token_seed(resolved_namespace)
        history_reader = _execution_history_reader(
            resolved_namespace,
            selected_state,
            runtime_token_seed,
        )
        session_history_reader = StepSessionHistoryReader(
            store=selected_state.run_store.read_store(RuntimeDomain.CONVERSATION),
            cursor_signer=HmacCursorSigner("session-history", runtime_token_seed),
            sessions=selected_state.conversation.sessions,
        )
        memory_store_factory = _memory_store_factory(
            resolved_namespace,
            selected_state,
        )
        selected_authorization = (
            TenantAuthorizationPolicy(effective_tenant_id)
            if authorization is None else authorization
        )
        ownership_transferred = True
        components = await _build_local_components(
            state=selected_state,
            metrics=metrics,
            commit=execution_capability.commit,
            coordination=execution_capability.coordination,
            catalog=catalog,
            compiler=compiler,
            authorization=selected_authorization,
            tenant_id=effective_tenant_id,
            namespace=resolved_namespace,
            workspace=workspace,
            sandbox=sandbox,
            limits=selected_limits,
            execution_cwd=execution_cwd,
            app=app,
            history_reader=history_reader,
            session_history_reader=session_history_reader,
            memory_store_factory=memory_store_factory,
            skill_sources=skill_sources,
            runtime_capabilities=runtime_capabilities,
            asset_sources=asset_sources,
            runtime_token_seed=runtime_token_seed,
            instruction_resolver=instruction_resolver,
            object_key_factory=object_key_factory,
            payload_policy=payload_policy,
            input_materializer=input_materializer,
            session_execution_ready=session_capability.enabled,
            auto_recover=auto_recover,
            tool_responses=tool_responses,
        )
        try:
            if captured is not None:
                await captured.verify_source_revision()
        except BaseException as primary_error:
            try:
                await components.close_callback()
            except BaseException as cleanup_error:
                raise primary_error from cleanup_error
            raise
        _logger.info("runtime capability capture admitted: group=%s", None if captured is None else captured.group_id)
        return components
    except BaseException:
        if not ownership_transferred:
            await _cleanup_compose_resources(
                selected_state=selected_state,
                initialized=initialized,
                input_materializer=input_materializer,
                workspace_access=workspace_access,
            )
        raise


def _runtime_close_actions(
    *,
    graph_service: DefaultTaskGraphService | None,
    task_launcher: LocalTaskGraphLauncher | None,
    execution: DefaultExecutionService,
    backend: LocalExecutionBackend | None,
    input_materializer: ExecutionInputMaterializer,
    metric_buffer: MetricsService | None,
    state: RuntimeState,
    session: RuntimeSessionService | None = None,
    handoff: bool = False,
) -> tuple[tuple[str, Callable[[], Awaitable[None]]], ...]:
    actions: list[tuple[str, Callable[[], Awaitable[None]]]] = []
    if session is not None:
        actions.append(("runtime.session.display", session.drain_display_reads))
    if graph_service is not None:
        actions.extend(
            (
                ("runtime.graph.finalizers", graph_service.drain_owned_finalizers),
                ("runtime.graph.preflight", graph_service.preflight_close),
            )
        )
    if task_launcher is not None:
        actions.append(("runtime.task_launcher", task_launcher.shutdown))
    if graph_service is not None:
        actions.append(
            ("runtime.graph.metrics", graph_service.drain_metric_projector)
        )
    actions.append(("runtime.execution.preflight", execution.preflight_close))
    if backend is not None:
        actions.append(("runtime.backend", backend.handoff if handoff else backend.close))
    actions.append(("runtime.input", input_materializer.close))
    if metric_buffer is not None:
        actions.append(("runtime.metrics", metric_buffer.close))
    actions.append(("runtime.state", state.close))
    return tuple(actions)


async def _run_cleanup_actions(
    actions: Sequence[tuple[str, Callable[[], Awaitable[None]]]],
    *,
    stop_on_error: bool = True,
) -> None:
    for phase, action in actions:
        try:
            await action()
        except BaseException as error:
            _log_secondary_cleanup(phase, error)
            if stop_on_error:
                return


async def _cleanup_compose_resources(
    *,
    selected_state: RuntimeState | None,
    initialized: bool,
    input_materializer: ExecutionInputMaterializer | None,
    workspace_access: WorkspaceAccess | None,
) -> None:
    actions: list[tuple[str, Callable[[], Awaitable[None]]]] = []
    if input_materializer is not None:
        actions.append(("runtime.compose.input", input_materializer.close))
    elif workspace_access is not None:
        actions.append(("runtime.compose.workspace_access", workspace_access.close))
    if initialized and selected_state is not None:
        actions.append(("runtime.compose.state", selected_state.close))
    await _run_cleanup_actions(actions, stop_on_error=False)


def _log_secondary_cleanup(phase: str, error: BaseException) -> None:
    code = error.code.value if isinstance(error, AIError) else None
    _logger.error(
        "secondary cleanup failed: phase=%s code=%s exception_type=%s",
        phase,
        code,
        type(error).__name__,
    )


def _validate_candidate_uniqueness(
    candidates: Sequence[CapabilityContribution[object]],
) -> None:
    identities = tuple((candidate.kind, candidate.id) for candidate in candidates)
    if len(identities) != len(set(identities)):
        raise AIError(ErrorCode.CAPABILITY_CONFLICT)


def _execution_history_reader(
    namespace: str,
    state: RuntimeState,
    runtime_token_seed: bytes,
) -> StepExecutionHistoryReader:
    local_store = state.run_store.read_store(RuntimeDomain.TRACE, durable=False)
    if not isinstance(local_store, StagingAgentRunStore):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    transient_store = TransientExecutionHistoryStore(local_store, state.run_store)
    history_store = state.run_store.read_store(RuntimeDomain.TRACE)
    if state.plan.route(RuntimeDomain.TRACE).retention is StorageRetentionMode.TRANSIENT:
        if not isinstance(history_store, StagingAgentRunStore):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        history_store = TransientExecutionHistoryStore(history_store, state.run_store)
    return StepExecutionHistoryReader(
        namespace=namespace,
        executions=state.execution.executions,
        store=history_store,
        transient_store=transient_store,
        cursor_signer=HmacCursorSigner("execution-history", runtime_token_seed),
        tool_operations=state.recovery.tools,
        notifications=state.run_store,
        durable_history_available=state.run_store.model_interaction_history_available,
    )


def _memory_store_factory(
    namespace: str,
    state: RuntimeState,
) -> "Callable[[str, str, str, ObjectStore, bool], MemoryStore]":
    def build(
        tenant_id: str,
        execution_id: str,
        memory_scope: str,
        object_store: ObjectStore,
        transient: bool,
    ) -> MemoryStore:
        return RuntimeMemoryStore(
            state.memory,
            object_store=object_store,
            namespace=namespace,
            tenant_id=tenant_id,
            execution_id=execution_id,
            memory_scope=memory_scope,
            transient=transient,
        )

    return build


def _capture_host_cwd() -> "str | None":
    try:
        return str(Path.cwd().resolve())
    except (OSError, RuntimeError):
        return None


def _require_storage_identity(
    state: RuntimeState,
    *,
    namespace: str,
    tenant_id: str,
) -> None:
    if state.namespace != namespace or state.tenant_id != tenant_id:
        raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)


async def _build_local_components(
    *,
    state: RuntimeState,
    metrics: Metrics | None,
    commit: CommitPolicy,
    coordination: CacheBackend | TurnCoordinationBackend | StreamBackend | None,
    catalog: AgentCatalog,
    compiler: AgentCompiler,
    authorization: AuthorizationPolicy,
    tenant_id: str,
    namespace: str,
    workspace: "Workspace | None",
    sandbox: "Sandbox | None",
    limits: PromptLimits,
    execution_cwd: "str | None",
    app: AppT,
    history_reader: ExecutionHistoryReader,
    session_history_reader: SessionHistoryReader,
    memory_store_factory: "Callable[[str, str, str, ObjectStore, bool], MemoryStore] | None",
    skill_sources: SkillSourceRegistry,
    runtime_capabilities: Sequence[AbstractCapability[AgentContext[AppT]]],
    asset_sources: Mapping[str, AssetStoreReader],
    runtime_token_seed: bytes,
    instruction_resolver: "RepositoryInstructionResolver | None",
    object_key_factory: RuntimeObjectKeyFactory,
    payload_policy: PayloadPolicy,
    input_materializer: ExecutionInputMaterializer,
    session_execution_ready: bool,
    auto_recover: bool,
    tool_responses: ToolResponseFixture | None,
) -> _RuntimeComponents:
    metric_buffer: MetricsService | None = None
    metric_source_namespace: str | None = None
    backend: LocalExecutionBackend | None = None

    async def release_execution_handoff(
        execution_id: str,
        *,
        tenant_id: str,
        terminal_commit: bool = False,
    ) -> bool:
        if backend is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        await backend.release_runtime_execution(
            execution_id,
            tenant_id=tenant_id,
        )
        return await state.retention.release_execution_handoff(
            execution_id, tenant_id=tenant_id, terminal_commit=terminal_commit,
        )

    try:
        if not state.ready:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        _require_storage_identity(state, namespace=namespace, tenant_id=tenant_id)
        runtime_bridge = _ExecutionRuntimeBridge()
        if isinstance(history_reader, StepExecutionHistoryReader):
            history_reader.bind_pending_execution_reader(runtime_bridge.pending_turns.read_execution)
        persistence_capabilities: list[AbstractCapability[object]] = []
        for capability in runtime_capabilities:
            persistence_capabilities.extend(runtime_feature_leaves(capability))
        restart_checkpoints = any(
            isinstance(value, CheckpointCapability) for value in persistence_capabilities
        )
        durable_trace = any(
            isinstance(value, TraceCapability) for value in persistence_capabilities
        )
        conversation = any(
            isinstance(value, ExecutionCapability) and value.retain_history
            for value in persistence_capabilities
        )
        batch_scopes = BatchWriteScope.scoped_factory()
        state.run_store.trace_persistence.bind_batch_scopes(batch_scopes)
        metric_buffer = None if metrics is None else MetricsService(metrics, batch_scopes=batch_scopes)
        metric_source_namespace = None if metric_buffer is None else namespace
        if coordination is not None:
            runtime_bridge.pending_turns.bind_coordination(_PendingTurnCoordination.select(
                coordination, namespace=namespace, tenant_id=tenant_id,
            ))
        stream_cache = (
            coordination
            if isinstance(coordination, StreamBackend)
            and state.plan.route(RuntimeDomain.EXECUTION).retention is StorageRetentionMode.DURABLE
            else None
        )
        live_broker = LiveExecutionEventBroker(shared=(
            None if stream_cache is None else SharedStreamPublisher(
                stream_cache, namespace, tenant_id,
            )
        ))
        pending_observer = runtime_bridge.pending_turns.coordination
        history_service = DefaultExecutionHistoryService(
            state.execution.executions,
            authorization,
            history_reader,
            HmacCursorSigner("execution", runtime_token_seed),
            pending_execution=None if pending_observer is None else pending_observer.read_execution,
        )
        binding_resolver = _AgentBindingResolver(
            catalog,
            compiler,
            sandbox=sandbox,
        )
        execution = DefaultExecutionService(
            state.execution,
            state.object_store(RuntimeDomain.EXECUTION),
            authorization,
            sessions=state.conversation.sessions,
            task_admissions=state.task.admissions,
            catalog=catalog,
            compiler=compiler,
            runtime_bridge=runtime_bridge,
            live_broker=live_broker,
            history_reader=history_reader,
            history_service=history_service,
            release_terminal=release_execution_handoff,
            instruction_resolver=instruction_resolver,
            object_key_factory=object_key_factory,
            payload_policy=payload_policy,
            input_materializer=input_materializer,
            session_execution_ready=session_execution_ready,
            tool_responses=tool_responses,
            object_reader=state.read_object,
            restart_checkpoints=restart_checkpoints,
            durable_trace=durable_trace,
            conversation=conversation,
            terminal_commit=commit.mode == "terminal",
            context_archive=ExecutionContextArchive(
                state.execution.executions.state_store,
                state.archive_object_store(RuntimeDomain.EXECUTION),
                namespace=namespace, tenant_id=tenant_id,
                object_reader=lambda reference: state.read_archive_object(RuntimeDomain.EXECUTION, reference),
            ),
        )
        execution_tree_broker = ExecutionTreeBroker()
        dispatcher = SubagentDispatcher(
            catalog,
            compiler,
            execution,
            child_observer=execution_tree_broker,
        )
        executor = AgentExecutor(
            skill_sources,
            runtime_capabilities=runtime_capabilities,
            asset_sources=asset_sources,
            metrics=metric_buffer,
            sandbox=sandbox,
            tool_responses=tool_responses,
        )
    except BaseException:
        actions: list[tuple[str, Callable[[], Awaitable[None]]]] = [
            ("runtime.build.input", input_materializer.close),
        ]
        if metric_buffer is not None:
            actions.append(("runtime.build.metrics", metric_buffer.close))
        actions.append(("runtime.build.state", state.close))
        await _run_cleanup_actions(actions, stop_on_error=False)
        raise

    def build_memory_store(
        memory_tenant: str,
        execution_id: str,
        memory_scope: str,
    ) -> MemoryStore:
        if memory_store_factory is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        route = state.plan.route(RuntimeDomain.MEMORY)
        transient = route.retention is StorageRetentionMode.TRANSIENT
        store = (
            state.working_object_store(
                RuntimeDomain.MEMORY,
                owner_scope=f"execution:{execution_id}",
            )
            if transient
            else state.object_store(RuntimeDomain.MEMORY)
        )
        return memory_store_factory(
            memory_tenant,
            execution_id,
            memory_scope,
            store,
            transient,
        )

    session: RuntimeSessionService | None = None
    task_launcher: LocalTaskGraphLauncher | None = None
    graph_service: DefaultTaskGraphService | None = None
    coordinator: _RuntimeCloseCoordinator | None = None
    close_actions: tuple[tuple[str, Callable[[], Awaitable[None]]], ...] | None = None

    async def worker_completed(execution_id: str) -> None:
        await execution.resume_terminal_handoff(execution_id, tenant_id=tenant_id)

    exclusive_execution_writer = state.plan.route(RuntimeDomain.EXECUTION).kind == "filesystem"
    try:
        backend = LocalExecutionBackend(
            state.conversation,
            state.execution,
            state.recovery,
            state.object_store(RuntimeDomain.EXECUTION),
            state.object_store(RuntimeDomain.RECOVERY),
            namespace,
            state.run_store,
            executor,
            catalog,
            archive_objects=state.archive_object_store(RuntimeDomain.EXECUTION),
            archive_object_reader=lambda reference: state.read_archive_object(RuntimeDomain.EXECUTION, reference),
            restore_binding=compiler.restore,
            tenant_id=tenant_id,
            workspace=workspace,
            limits=limits,
            execution_cwd=execution_cwd,
            instruction_resolver=instruction_resolver,
            app=app,
            run_stores={
                domain: state.run_store.read_store(domain)
                for domain in (
                    RuntimeDomain.CONVERSATION,
                    RuntimeDomain.TRACE,
                    RuntimeDomain.CHECKPOINT,
                )
            },
            agent_run_lifecycle=state.run_store,
            memory_store_factory=build_memory_store,
            conversation_durable=(
                state.plan.route(RuntimeDomain.CONVERSATION).retention
                is StorageRetentionMode.DURABLE
            ),
            input_materializer=input_materializer,
            subagent_dispatcher=dispatcher,
            live_broker=live_broker,
            payload_policy=payload_policy,
            execution_objects_durable=(
                state.plan.route(RuntimeDomain.EXECUTION).retention
                is StorageRetentionMode.DURABLE
            ),
            tool_operations=state.recovery.tools,
            metric_recorder=metric_buffer,
            worker_completed=worker_completed,
            exclusive_execution_writer=exclusive_execution_writer,
            pending_turns=runtime_bridge.pending_turns,
            pending_cancel=lambda execution_id, principal: execution.cancel(
                execution_id, CancelExecutionRequest(principal, f"pending-cancel:{execution_id}"),
            ),
        )
        runtime_bridge.bind(backend)
        runtime_bridge.pending_start = backend.start_pending
        runtime_bridge.pending_promote = backend.promote_pending_execution
        runtime_bridge.pending_cancel_promote = lambda execution_id: backend.promote_pending_execution(
            execution_id, allow_cancelled=True,
        )
        state.run_store.bind_pending_producer_guard(backend.require_pending_producer)
        session = RuntimeSessionService(
            state.conversation,
            state.execution.executions,
            authorization,
            execution,
            HmacCursorSigner("session", runtime_token_seed),
            history_reader=session_history_reader,
            enabled=session_execution_ready,
            cache=(state.plan.route(RuntimeDomain.CONVERSATION).cache if all(
                state.plan.route(domain).retention is StorageRetentionMode.DURABLE
                for domain in (RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION)
            ) else None),
            transcript_store=state.run_store.conversation_persistence,
            release_terminal=state.retention.release_session,
            pending_session_execution=runtime_bridge.pending_turns.read_session_execution,
            workspace_access=input_materializer.access,
        )
        async def prepare_pending_capture(execution_id: str, principal) -> None:
            await execution.load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_CAPTURE_INPUT)
            await execution.load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
            await backend.promote_pending_execution(execution_id)

        input_captures = RuntimeInputCaptures(
            namespace, authorization, execution, input_materializer,
            tenant_id=tenant_id,
            execution_repositories=state.execution,
            task_repositories=state.task,
            object_stores={domain: state.object_store(domain) for domain in RuntimeDomain},
            execution_retention=state.plan.route(RuntimeDomain.EXECUTION).retention,
            task_retention=state.plan.route(RuntimeDomain.TASK).retention,
            context_archive=ExecutionContextArchive(
                state.execution.executions.state_store,
                state.archive_object_store(RuntimeDomain.EXECUTION),
                namespace=namespace, tenant_id=tenant_id,
                object_reader=lambda reference: state.read_archive_object(RuntimeDomain.EXECUTION, reference),
            ),
            prepare_pending_capture=prepare_pending_capture,
        )
        task_graph_binding_captures = TaskGraphBindingCaptureStore(
            namespace,
            state.object_store(RuntimeDomain.TASK),
        )
        task_runner = RuntimeTaskNodeRunner(
            execution,
            namespace=namespace,
            app=app,
            authorization=authorization,
            task_state=state.task.tasks,
            input_captures=input_captures,
            task_admissions=state.task.admissions,
            execution_state=state.execution.executions,
            recovery_backend=backend,
            resolve_agent_binding=binding_resolver.resolve_contract,
            budgets=state.execution.budgets,
            task_objects=state.object_store(RuntimeDomain.TASK),
            artifact_state=state.artifact,
            artifact_objects=state.object_store(RuntimeDomain.ARTIFACT),
            object_key_factory=object_key_factory,
            binding_captures=task_graph_binding_captures,
            input_materializer=ExecutionInputMaterializer(
                input_materializer.access,
                limits,
                object_store=state.object_store(RuntimeDomain.TASK),
                object_key_factory=object_key_factory,
                payload_policy=PayloadPolicy(inline_limit_bytes=0),
                object_domain=RuntimeDomain.TASK,
            ),
            task_durable=(
                state.plan.route(RuntimeDomain.TASK).retention
                is StorageRetentionMode.DURABLE
            ),
            execution_durable=(
                state.plan.route(RuntimeDomain.EXECUTION).retention
                is StorageRetentionMode.DURABLE
            ),
            recovery_durable=(
                state.plan.route(RuntimeDomain.RECOVERY).retention
                is StorageRetentionMode.DURABLE
            ),
        )
        backend.bind_recovery_relaunch_guard(task_runner.allow_execution_relaunch)
        task_launcher = LocalTaskGraphLauncher(
            state.task.tasks,
            task_runner,
            owner=f"runtime:{tenant_id}:{uuid.uuid4().hex}",
            acquire_execution_hold=execution.acquire_dependency_hold,
            release_execution_hold=execution.release_dependency_hold,
        )
        graph_service = DefaultTaskGraphService(
            state.task,
            authorization,
            task_launcher,
            local_waiter=task_launcher,
            preflight=task_runner,
            bound_execution_recovery=task_runner,
            metric_recorder=metric_buffer,
            metric_source_namespace=metric_source_namespace,
        )
        approval = DefaultApprovalService(
            state.recovery.approvals,
            state.execution.executions,
            state.recovery.checkpoints,
            authorization,
            objects=state.object_store(RuntimeDomain.RECOVERY),
            continuation=backend,
        )
        external = DefaultExternalService(
            state.recovery.external_calls,
            state.execution.executions,
            state.recovery.checkpoints,
            authorization,
            objects=state.object_store(RuntimeDomain.RECOVERY),
            object_key_factory=object_key_factory,
            payload_policy=payload_policy,
            continuation=backend,
        )
        event = DefaultEventService(
            state.execution.executions,
            state.execution.events,
            authorization,
            backend.worker_failure,
            live_broker,
            pending_execution=backend.pending_execution,
            pending_stream=runtime_bridge.pending_turns.read_stream,
        )
        artifact = DefaultArtifactService(
            state.artifact,
            authorization,
            token_seed=runtime_token_seed,
            cursor_signer=HmacCursorSigner("artifact", runtime_token_seed),
        )
        history = _borrowed_runtime_history(
            history_service, tenant_id=tenant_id, state=state,
            authorization=authorization, artifact=artifact,
            pending_session_execution=(
                None if pending_observer is None else pending_observer.read_session_execution
            ),
            pending_execution=None if pending_observer is None else pending_observer.read_execution,
        )
        evaluation = RuntimeEvaluations(
            namespace, authorization, execution, graph_service,
            input_captures, history,
            tenant_id=tenant_id, plan=state.plan,
            evaluation_repositories=state.evaluation, task_repositories=state.task,
            object_stores={domain: state.object_store(domain) for domain in RuntimeDomain},
            maintenance=state.snapshot_operations(),
            cursor_signer=HmacCursorSigner("evaluation", runtime_token_seed),
            asset_readers=tuple(asset_sources.values()),
        )
        local_coordinator = _LocalRuntimeCoordinator(execution, event)
        tree_streamer = ExecutionTreeStreamer(
            execution,
            local_coordinator,
            execution_tree_broker,
        )
        close_actions = _runtime_close_actions(
            graph_service=graph_service,
            task_launcher=task_launcher,
            execution=execution,
            backend=backend,
            input_materializer=input_materializer,
            metric_buffer=metric_buffer,
            state=state,
            session=session,
        )
        close_actions = (("runtime.evaluation", evaluation.close), *close_actions)
        handoff_actions = _runtime_close_actions(
            graph_service=graph_service, task_launcher=task_launcher,
            execution=execution, backend=backend, input_materializer=input_materializer,
            metric_buffer=metric_buffer, state=state, session=session, handoff=True,
        )
        coordinator = _RuntimeCloseCoordinator(
            tuple(action for _, action in close_actions),
            (evaluation.close, *(action for _, action in handoff_actions)),
        )
        await state.run_store.prepare_archives()
        if auto_recover and RuntimeDomain.RECOVERY in state.plan.durable_domains:
            await backend.reconcile(
                allow_active_recovery=exclusive_execution_writer,
            )
    except BaseException:
        abort_actions = (
            close_actions
            if close_actions is not None
            else _runtime_close_actions(
                graph_service=graph_service,
                task_launcher=task_launcher,
                execution=execution,
                backend=backend,
                input_materializer=input_materializer,
                metric_buffer=metric_buffer,
                state=state,
                session=session,
            )
        )
        await _run_cleanup_actions(abort_actions)
        raise
    return _RuntimeComponents(
        state=state,
        catalog=catalog,
        compiler=compiler,
        execution=execution,
        session=session,
        graph=graph_service,
        evaluation=evaluation,
        approval=approval,
        external=external,
        event=event,
        artifact=artifact,
        tenant_id=tenant_id,
        close_callback=coordinator.close,
        handoff_callback=(coordinator.handoff if restart_checkpoints and commit.mode == "sync"
            and all(state.plan.route(domain).retention is StorageRetentionMode.DURABLE
                    for domain in (RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY,
                                   RuntimeDomain.CHECKPOINT)) else None),
        task_node_runtime=task_runner,
        tree_streamer=tree_streamer,
        metric_control=metric_buffer,
        binding_resolver=binding_resolver,
        history=history,
        task_admissions=state.task.admissions,
        budgets=state.execution.budgets,
        input_captures=input_captures,
    )


def _borrowed_runtime_history(
    service: DefaultExecutionHistoryService,
    *,
    tenant_id: str,
    state: RuntimeState,
    authorization: object,
    artifact: DefaultArtifactService,
    pending_session_execution: SessionPendingExecutionLookup[ExecutionRecord] | None = None,
    pending_execution: Callable[..., Awaitable[ExecutionRecord | None]] | None = None,
) -> "RuntimeHistory":
    return RuntimeHistory(
        service,
        tenant_id=tenant_id,
        executions=state.execution.executions,
        events=state.execution.events,
        sessions=state.conversation.sessions,
        conversation=state.conversation,
        session_transcript_store=state.run_store.conversation_persistence,
        pending_session_execution=pending_session_execution,
        pending_execution=pending_execution,
        cache=(state.plan.route(RuntimeDomain.CONVERSATION).cache if all(
            state.plan.route(domain).retention is StorageRetentionMode.DURABLE
            for domain in (RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION)
        ) else None),
        tasks=state.task.tasks,
        authorization=authorization,
        namespace=state.namespace,
        execution_objects=state.object_store(RuntimeDomain.EXECUTION),
        task_objects=state.object_store(RuntimeDomain.TASK),
        object_reader=state.read_object,
        artifacts=artifact,
        cursor_signer=HmacCursorSigner(
            "runtime-history",
            token_seed(state.namespace),
        ),
    )


class _RuntimeCloseCoordinator:
    def __init__(
        self, actions: tuple[Callable[[], Awaitable[None]], ...],
        handoff_actions: tuple[Callable[[], Awaitable[None]], ...] = (),
    ) -> None:
        self._actions = actions
        self._handoff_actions = handoff_actions
        self._cursor = 0
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def handoff(self) -> None:
        if self._task is None and self._cursor == 0:
            self._actions = self._handoff_actions
        await self.close()

    async def close(self) -> None:
        async with self._lock:
            if self._cursor >= len(self._actions):
                return
            task = self._task
            if task is None or task.done():
                task = asyncio.create_task(self._run(), name="linktools-runtime-close")
                self._task = task
        await asyncio.shield(task)

    async def _run(self) -> None:
        while self._cursor < len(self._actions):
            await self._actions[self._cursor]()
            self._cursor += 1


__all__ = []
