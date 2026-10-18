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
    AgentContext,
    AssetCapability,
    AssetSkillSource,
    CapabilityContribution,
    RuntimeFeatures,
    RuntimeFeaturesCapture,
    SkillSourceRegistry,
)
from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    HmacCursorSigner,
    PromptLimits,
    TenantAuthorizationPolicy,
    validate_persistence_namespace,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode
from ..model import ModelRegistry
from ..spec import (
    AgentSpec,
    RuleInstructionResolver,
    RepositoryInstructionResolver,
    RepositoryInstructions,
)
from ..storage import ObjectStore, PayloadPolicy, StreamBackend
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
from ._shared_stream import SharedStreamPublisher
from ._execution_tree import ExecutionTreeBroker, ExecutionTreeStreamer
from ._history_projection import StepExecutionHistoryReader, StepSessionHistoryReader
from ._history_service import DefaultExecutionHistoryService
from ._input import ExecutionInputMaterializer
from ._input_capture import RuntimeInputCaptures
from ._local import LocalExecutionBackend
from ._pending_coordination import _PendingTurnCoordination
from ._memory import MemoryStore, RuntimeMemoryStore
from ._object import RuntimeObjectKeyFactory
from ._planner import RuntimeTaskNodeRunner
from ._capabilities import runtime_feature_leaves
from ._conversation_capability import ConversationCapability
from ._metrics_capability import MetricsCapability, MetricsRecordingBinding
from ._persistence_configuration import persistence_storage_configuration
from ._persistence_capabilities import (
    TraceCapability,
    CheckpointCapability,
)
from ._runtime_history import RuntimeHistory
from ._task_graph_binding_capture import TaskGraphBindingCaptureStore
from ._runtime_identity import token_seed
from ._session import DefaultSessionService
from ._subagent import SubagentDispatcher
from ._transient_history import TransientExecutionHistoryStore
from ._tool_response_fixture import ToolResponseFixture
from .service_api import CancelExecutionRequest, ExecutionHistoryReader, SessionHistoryReader
from .state import RuntimeDomain, RuntimeRetentionMode, RuntimeStorage
from .state._contracts import BudgetRepository, TaskAdmissionRepository
from .state._step_archive import StagingAgentRunStore

AppT = TypeVar("AppT")
_logger = environ.get_logger("ai.runtime.factory")


@dataclass(frozen=True, slots=True)
class _RuntimeComponents:
    catalog: AgentCatalog
    compiler: AgentCompiler
    execution: DefaultExecutionService
    session: DefaultSessionService
    graph: DefaultTaskGraphService
    evaluation: RuntimeEvaluations
    approval: DefaultApprovalService
    external: DefaultExternalService
    event: DefaultEventService
    artifact: DefaultArtifactService
    tenant_id: str
    close_callback: Callable[[], Awaitable[None]]
    task_node_runtime: RuntimeTaskNodeRunner[object]
    tree_streamer: ExecutionTreeStreamer
    metric_control: MetricsRecordingBinding | None
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
    models: ModelRegistry,
    storage: RuntimeStorage,
    features: "RuntimeFeatures[AppT] | RuntimeFeaturesCapture[AppT] | None" = None,
    auto_recover: bool = True,
) -> _RuntimeComponents:
    """Capture declarations and build Runtime-private services."""
    resolved_namespace = validate_persistence_namespace(namespace)
    if not isinstance(storage, RuntimeStorage):
        raise TypeError("storage must be RuntimeStorage")
    sources = () if features is None else (features,)
    if any(
        not isinstance(source, (RuntimeFeatures, RuntimeFeaturesCapture))
        for source in sources
    ):
        raise TypeError(
            "features must be RuntimeFeatures or RuntimeFeaturesCapture"
        )
    captures: list[RuntimeFeaturesCapture[AppT]] = []
    for source in sources:
        capture = (
            await source.capture()
            if isinstance(source, RuntimeFeatures)
            else source
        )
        await capture.verify_source_revision()
        captures.append(capture)
    groups = tuple(captures)
    selected_limits = PromptLimits() if not groups else groups[0].limits
    authorization = None if not groups else groups[0].authorization
    tool_responses: ToolResponseFixture | None = None
    group_ids = tuple(group.group_id for group in groups)
    if len(group_ids) != len(set(group_ids)):
        raise AIError(ErrorCode.CAPABILITY_CONFLICT)
    workspace_groups = tuple(group for group in groups if group.workspace is not None)
    if len(workspace_groups) > 1:
        raise AIError(ErrorCode.CAPABILITY_CONFLICT)
    sandbox_groups = tuple(group for group in groups if group.sandbox is not None)
    if len(sandbox_groups) > 1:
        raise AIError(ErrorCode.CAPABILITY_CONFLICT)
    workspace = None if not workspace_groups else workspace_groups[0].workspace
    sandbox = sandbox_groups[0].sandbox if sandbox_groups else LocalSandbox()
    if workspace is not None:
        workspace.policy.validate()

    selected_storage: RuntimeStorage | None = None
    workspace_access: WorkspaceAccess | None = None
    input_materializer: ExecutionInputMaterializer | None = None
    initialized = False
    ownership_transferred = False
    try:
        candidates: list[CapabilityContribution[object]] = []
        asset_sources: dict[str, AssetStoreReader] = {}
        for group in groups:
            candidates.extend(group.contributions)
            if group.asset_reader is not None:
                asset_sources[group.group_id] = group.asset_reader
        _validate_candidate_uniqueness(candidates)
        def require_runtime_feature_scope(capability: AbstractCapability[object]) -> None:
            if isinstance(capability, (
                CheckpointCapability, TraceCapability, ConversationCapability, MetricsCapability, ToolResponsesCapability,
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

        def consume_assets(capability: AbstractCapability[AgentContext[AppT]]) -> AbstractCapability[AgentContext[AppT]] | None:
            nonlocal tool_responses
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
                wrapped = capability.wrapped.visit_and_replace(consume_assets)
                if wrapped is None:
                    return None
                rebound = copy(capability)
                rebound.wrapped = wrapped
                return rebound
            return capability

        runtime_capabilities = tuple(
            selected for candidate in candidates if candidate.scope == "runtime"
            if (selected := candidate.value.visit_and_replace(consume_assets)) is not None
        )
        runtime_ids: set[str] = set()

        def register_runtime_id(capability: AbstractCapability[AgentContext[AppT]]) -> None:
            if capability.id is not None:
                if capability.id in runtime_ids:
                    raise AIError(ErrorCode.CAPABILITY_CONFLICT)
                runtime_ids.add(capability.id)

        for capability in runtime_capabilities:
            capability.apply(register_runtime_id)
        candidates = [candidate for candidate in candidates if candidate.scope == "agent"]
        skill_sources = SkillSourceRegistry(
            tuple(
                AssetSkillSource(group.group_id, reader)
                for group in groups
                if (reader := group.asset_reader) is not None
            )
        )
        instruction_documents = tuple(
            document
            for group in groups
            for document in group.instructions.documents
        )
        rules = RepositoryInstructions(instruction_documents)
        agents = {
            candidate.id: candidate.value
            for candidate in candidates
            if candidate.kind == "agent"
        }
        resolver = models.capture()
        if "default" not in agents:
            try:
                resolver.resolve("default")
            except AIError as error:
                if error.code is not ErrorCode.MODEL_CONNECTION_NOT_FOUND:
                    raise
            else:
                agents["default"] = AgentSpec("default")
        compiler = AgentCompiler(
            defaults=(captures[0].defaults if captures else None),
            model_resolver=resolver,
            candidates=tuple(
                candidate
                for candidate in candidates
                if candidate.kind != "agent"
            ),
            agents=agents,
            capability_filter=captures[0].filter if captures else None,
        )
        root_agents = {
            agent_id: compiler.compile(agents[agent_id])
            for agent_id in sorted(agents)
        }
        catalog = AgentCatalog(root_agents)

        effective_tenant_id = (
            "default" if tenant_id is None else validate_tenant_id(tenant_id)
        )
        selected_storage = storage
        module_routes, module_caches = persistence_storage_configuration(runtime_capabilities)
        await selected_storage.initialize(
            namespace=resolved_namespace,
            tenant_id=effective_tenant_id,
            module_routes=module_routes,
            module_caches=module_caches,
        )

        initialized = True
        for group in groups:
            await group.verify_source_revision()
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
            object_store=selected_storage.object_store(RuntimeDomain.EXECUTION),
            object_key_factory=object_key_factory,
            payload_policy=payload_policy,
        )
        runtime_token_seed = token_seed(resolved_namespace)
        history_reader = _execution_history_reader(
            resolved_namespace,
            selected_storage,
            runtime_token_seed,
        )
        session_history_reader = StepSessionHistoryReader(
            store=selected_storage.run_store.read_store(RuntimeDomain.CONVERSATION),
            cursor_signer=HmacCursorSigner("session-history", runtime_token_seed),
            sessions=selected_storage.conversation.sessions,
        )
        memory_store_factory = _memory_store_factory(
            resolved_namespace,
            selected_storage,
        )
        selected_authorization = (
            TenantAuthorizationPolicy(effective_tenant_id)
            if authorization is None else authorization
        )
        ownership_transferred = True
        components = await _build_local_components(
            storage=selected_storage,
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
            session_execution_ready=True,
            auto_recover=auto_recover,
            tool_responses=tool_responses,
        )
        try:
            for capture in groups:
                await capture.verify_source_revision()
        except BaseException as primary_error:
            try:
                await components.close_callback()
            except BaseException as cleanup_error:
                raise primary_error from cleanup_error
            raise
        _logger.info("runtime capability captures admitted: groups=%s", group_ids)
        return components
    except BaseException:
        if not ownership_transferred:
            await _cleanup_compose_resources(
                selected_storage=selected_storage,
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
    metric_buffer: MetricsRecordingBinding | None,
    storage: RuntimeStorage,
    session: DefaultSessionService | None = None,
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
        actions.append(("runtime.backend", backend.close))
    actions.append(("runtime.input", input_materializer.close))
    if metric_buffer is not None:
        actions.append(("runtime.metrics", metric_buffer.close))
    actions.append(("runtime.storage", storage.close))
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
    selected_storage: RuntimeStorage | None,
    initialized: bool,
    input_materializer: ExecutionInputMaterializer | None,
    workspace_access: WorkspaceAccess | None,
) -> None:
    actions: list[tuple[str, Callable[[], Awaitable[None]]]] = []
    if input_materializer is not None:
        actions.append(("runtime.compose.input", input_materializer.close))
    elif workspace_access is not None:
        actions.append(("runtime.compose.workspace_access", workspace_access.close))
    if initialized and selected_storage is not None:
        actions.append(("runtime.compose.storage", selected_storage.close))
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
    storage: RuntimeStorage,
    runtime_token_seed: bytes,
) -> StepExecutionHistoryReader:
    local_store = storage.run_store.read_store(RuntimeDomain.TRACE, durable=False)
    if not isinstance(local_store, StagingAgentRunStore):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    transient_store = TransientExecutionHistoryStore(local_store, storage.run_store)
    history_store = storage.run_store.read_store(RuntimeDomain.TRACE)
    if storage.plan.route(RuntimeDomain.TRACE).retention is RuntimeRetentionMode.TRANSIENT:
        if not isinstance(history_store, StagingAgentRunStore):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        history_store = TransientExecutionHistoryStore(history_store, storage.run_store)
    return StepExecutionHistoryReader(
        namespace=namespace,
        executions=storage.execution.executions,
        store=history_store,
        transient_store=transient_store,
        cursor_signer=HmacCursorSigner("execution-history", runtime_token_seed),
        tool_operations=storage.recovery.tools,
        notifications=storage.run_store,
        durable_history_available=storage.run_store.model_interaction_history_available,
    )


def _memory_store_factory(
    namespace: str,
    storage: RuntimeStorage,
) -> "Callable[[str, str, str, ObjectStore, bool], MemoryStore]":
    def build(
        tenant_id: str,
        execution_id: str,
        memory_scope: str,
        object_store: ObjectStore,
        transient: bool,
    ) -> MemoryStore:
        return RuntimeMemoryStore(
            storage.memory,
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
    storage: RuntimeStorage,
    *,
    namespace: str,
    tenant_id: str,
) -> None:
    if storage.namespace != namespace or storage.tenant_id != tenant_id:
        raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)


async def _build_local_components(
    *,
    storage: RuntimeStorage,
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
    metric_buffer: MetricsRecordingBinding | None = None
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
        return await storage.retention.release_execution_handoff(
            execution_id, tenant_id=tenant_id, terminal_commit=terminal_commit,
        )

    try:
        if not storage.ready:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        _require_storage_identity(storage, namespace=namespace, tenant_id=tenant_id)
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
            isinstance(value, ConversationCapability) for value in persistence_capabilities
        )
        metric_capabilities = tuple(
            value for value in persistence_capabilities if isinstance(value, MetricsCapability)
        )
        if len(metric_capabilities) > 1:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT, "one Metrics dataset is allowed per Runtime")
        automatic_metrics = bool(metric_capabilities)
        metric_buffer = metric_capabilities[0].open_recording() if metric_capabilities else None
        metric_source_namespace = None if metric_buffer is None else namespace
        agent_metric_buffer = metric_buffer.recorder if automatic_metrics and metric_buffer is not None else None
        if metric_buffer is not None:
            def bind_metrics(capability: AbstractCapability[AgentContext[AppT]]) -> AbstractCapability[AgentContext[AppT]]:
                if isinstance(capability, WrapperCapability):
                    rebound = copy(capability)
                    rebound.wrapped = capability.wrapped.visit_and_replace(bind_metrics)
                    return rebound
                if isinstance(capability, MetricsCapability):
                    return metric_buffer.bind(capability)
                return capability

            runtime_capabilities = tuple(
                bound for template in runtime_capabilities
                if (bound := template.visit_and_replace(bind_metrics)) is not None
            )
        if storage.cache is not None:
            runtime_bridge.pending_turns.bind_coordination(_PendingTurnCoordination.select(
                storage.cache, namespace=namespace, tenant_id=tenant_id,
            ))
        stream_cache = (
            storage.cache
            if isinstance(storage.cache, StreamBackend)
            and storage.plan.route(RuntimeDomain.EXECUTION).retention is RuntimeRetentionMode.DURABLE
            else None
        )
        live_broker = LiveExecutionEventBroker(shared=(
            None if stream_cache is None else SharedStreamPublisher(
                stream_cache, namespace, tenant_id,
            )
        ))
        history_service = DefaultExecutionHistoryService(
            storage.execution.executions,
            authorization,
            history_reader,
            HmacCursorSigner("execution", runtime_token_seed),
            pending_execution=runtime_bridge.pending_turns.read_execution,
        )
        binding_resolver = _AgentBindingResolver(
            catalog,
            compiler,
            sandbox=sandbox,
        )
        execution = DefaultExecutionService(
            storage.execution,
            storage.object_store(RuntimeDomain.EXECUTION),
            authorization,
            sessions=storage.conversation.sessions,
            task_admissions=storage.task.admissions,
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
            object_reader=storage.read_object,
            restart_checkpoints=restart_checkpoints,
            durable_trace=durable_trace,
            conversation=conversation,
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
            metrics=agent_metric_buffer,
            sandbox=sandbox,
            tool_responses=tool_responses,
        )
    except BaseException:
        actions: list[tuple[str, Callable[[], Awaitable[None]]]] = [
            ("runtime.build.input", input_materializer.close),
        ]
        if metric_buffer is not None:
            actions.append(("runtime.build.metrics", metric_buffer.close))
        actions.append(("runtime.build.storage", storage.close))
        await _run_cleanup_actions(actions, stop_on_error=False)
        raise

    def build_memory_store(
        memory_tenant: str,
        execution_id: str,
        memory_scope: str,
    ) -> MemoryStore:
        if memory_store_factory is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        route = storage.plan.route(RuntimeDomain.MEMORY)
        transient = route.retention is RuntimeRetentionMode.TRANSIENT
        store = (
            storage.working_object_store(
                RuntimeDomain.MEMORY,
                owner_scope=f"execution:{execution_id}",
            )
            if transient
            else storage.object_store(RuntimeDomain.MEMORY)
        )
        return memory_store_factory(
            memory_tenant,
            execution_id,
            memory_scope,
            store,
            transient,
        )

    session: DefaultSessionService | None = None
    task_launcher: LocalTaskGraphLauncher | None = None
    graph_service: DefaultTaskGraphService | None = None
    coordinator: _RuntimeCloseCoordinator | None = None
    close_actions: tuple[tuple[str, Callable[[], Awaitable[None]]], ...] | None = None

    async def worker_completed(execution_id: str) -> None:
        await execution.resume_terminal_handoff(execution_id, tenant_id=tenant_id)

    exclusive_execution_writer = storage.plan.route(RuntimeDomain.EXECUTION).kind == "filesystem"
    try:
        backend = LocalExecutionBackend(
            storage.conversation,
            storage.execution,
            storage.recovery,
            storage.object_store(RuntimeDomain.EXECUTION),
            storage.object_store(RuntimeDomain.RECOVERY),
            namespace,
            storage.run_store,
            executor,
            catalog,
            restore_binding=compiler.restore,
            tenant_id=tenant_id,
            workspace=workspace,
            limits=limits,
            execution_cwd=execution_cwd,
            instruction_resolver=instruction_resolver,
            app=app,
            run_stores={
                domain: storage.run_store.read_store(domain)
                for domain in (
                    RuntimeDomain.CONVERSATION,
                    RuntimeDomain.TRACE,
                    RuntimeDomain.CHECKPOINT,
                )
            },
            agent_run_lifecycle=storage.run_store,
            memory_store_factory=build_memory_store,
            conversation_durable=(
                storage.plan.route(RuntimeDomain.CONVERSATION).retention
                is RuntimeRetentionMode.DURABLE
            ),
            input_materializer=input_materializer,
            subagent_dispatcher=dispatcher,
            live_broker=live_broker,
            payload_policy=payload_policy,
            execution_objects_durable=(
                storage.plan.route(RuntimeDomain.EXECUTION).retention
                is RuntimeRetentionMode.DURABLE
            ),
            tool_operations=storage.recovery.tools,
            metric_recorder=agent_metric_buffer,
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
        storage.run_store.bind_pending_producer_guard(backend.require_pending_producer)
        session = DefaultSessionService(
            storage.conversation,
            storage.execution.executions,
            authorization,
            execution,
            HmacCursorSigner("session", runtime_token_seed),
            conversation_enabled=conversation,
            history_reader=session_history_reader,
            cache=(
                next((feature.cache for feature in persistence_capabilities if isinstance(feature, ConversationCapability)), None)
                if all(storage.plan.route(domain).retention is RuntimeRetentionMode.DURABLE
                       for domain in (RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION))
                else None
            ),
            transcript_store=storage.run_store.conversation_persistence,
            release_terminal=storage.retention.release_session,
            pending_session_execution=runtime_bridge.pending_turns.read_session_execution,
            workspace_access=input_materializer.access,
        )
        async def prepare_pending_capture(execution_id: str, principal) -> None:
            await execution.load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_CAPTURE_INPUT)
            await execution.load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
            await backend.promote_pending_execution(execution_id)

        input_captures = RuntimeInputCaptures(
            namespace, storage, authorization, execution, input_materializer,
            prepare_pending_capture=prepare_pending_capture,
        )
        task_graph_binding_captures = TaskGraphBindingCaptureStore(
            namespace,
            storage.object_store(RuntimeDomain.TASK),
        )
        task_runner = RuntimeTaskNodeRunner(
            execution,
            namespace=namespace,
            app=app,
            authorization=authorization,
            task_state=storage.task.tasks,
            input_captures=input_captures,
            task_admissions=storage.task.admissions,
            execution_state=storage.execution.executions,
            recovery_backend=backend,
            resolve_agent_binding=binding_resolver.resolve_contract,
            budgets=storage.execution.budgets,
            task_objects=storage.object_store(RuntimeDomain.TASK),
            artifact_state=storage.artifact,
            artifact_objects=storage.object_store(RuntimeDomain.ARTIFACT),
            object_key_factory=object_key_factory,
            binding_captures=task_graph_binding_captures,
            input_materializer=ExecutionInputMaterializer(
                input_materializer.access,
                limits,
                object_store=storage.object_store(RuntimeDomain.TASK),
                object_key_factory=object_key_factory,
                payload_policy=PayloadPolicy(inline_limit_bytes=0),
                object_domain=RuntimeDomain.TASK,
            ),
            task_durable=(
                storage.plan.route(RuntimeDomain.TASK).retention
                is RuntimeRetentionMode.DURABLE
            ),
            execution_durable=(
                storage.plan.route(RuntimeDomain.EXECUTION).retention
                is RuntimeRetentionMode.DURABLE
            ),
            recovery_durable=(
                storage.plan.route(RuntimeDomain.RECOVERY).retention
                is RuntimeRetentionMode.DURABLE
            ),
        )
        backend.bind_recovery_relaunch_guard(task_runner.allow_execution_relaunch)
        task_launcher = LocalTaskGraphLauncher(
            storage.task.tasks,
            task_runner,
            owner=f"runtime:{tenant_id}:{uuid.uuid4().hex}",
            acquire_execution_hold=execution.acquire_dependency_hold,
            release_execution_hold=execution.release_dependency_hold,
        )
        graph_service = DefaultTaskGraphService(
            storage.task,
            authorization,
            task_launcher,
            local_waiter=task_launcher,
            preflight=task_runner,
            bound_execution_recovery=task_runner,
            metric_recorder=None if metric_buffer is None else metric_buffer.recorder,
            metric_source_namespace=metric_source_namespace,
        )
        approval = DefaultApprovalService(
            storage.recovery.approvals,
            storage.execution.executions,
            storage.recovery.checkpoints,
            authorization,
            objects=storage.object_store(RuntimeDomain.RECOVERY),
            continuation=backend,
        )
        external = DefaultExternalService(
            storage.recovery.external_calls,
            storage.execution.executions,
            storage.recovery.checkpoints,
            authorization,
            objects=storage.object_store(RuntimeDomain.RECOVERY),
            object_key_factory=object_key_factory,
            payload_policy=payload_policy,
            continuation=backend,
        )
        event = DefaultEventService(
            storage.execution.executions,
            storage.execution.events,
            authorization,
            backend.worker_failure,
            live_broker,
            pending_execution=backend.pending_execution,
            pending_stream=runtime_bridge.pending_turns.read_stream,
        )
        artifact = DefaultArtifactService(
            storage.artifact,
            authorization,
            token_seed=runtime_token_seed,
            cursor_signer=HmacCursorSigner("artifact", runtime_token_seed),
        )
        history = _borrowed_runtime_history(
            history_service, tenant_id=tenant_id, storage=storage,
            authorization=authorization, artifact=artifact,
        )
        evaluation = RuntimeEvaluations(
            namespace, storage, authorization, execution, graph_service,
            input_captures, history,
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
            storage=storage,
            session=session,
        )
        close_actions = (("runtime.evaluation", evaluation.close), *close_actions)
        coordinator = _RuntimeCloseCoordinator(
            tuple(action for _, action in close_actions)
        )
        if auto_recover and RuntimeDomain.RECOVERY in storage.plan.durable_domains:
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
                storage=storage,
                session=session,
            )
        )
        await _run_cleanup_actions(abort_actions)
        raise
    return _RuntimeComponents(
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
        task_node_runtime=task_runner,
        tree_streamer=tree_streamer,
        metric_control=metric_buffer,
        binding_resolver=binding_resolver,
        history=history,
        task_admissions=storage.task.admissions,
        budgets=storage.execution.budgets,
        input_captures=input_captures,
    )


def _borrowed_runtime_history(
    service: DefaultExecutionHistoryService,
    *,
    tenant_id: str,
    storage: RuntimeStorage,
    authorization: object,
    artifact: DefaultArtifactService,
) -> "RuntimeHistory":
    return RuntimeHistory(
        service,
        tenant_id=tenant_id,
        executions=storage.execution.executions,
        events=storage.execution.events,
        sessions=storage.conversation.sessions,
        conversation=storage.conversation,
        session_transcript_store=storage.run_store.conversation_persistence,
        tasks=storage.task.tasks,
        authorization=authorization,
        namespace=storage.namespace,
        execution_objects=storage.object_store(RuntimeDomain.EXECUTION),
        task_objects=storage.object_store(RuntimeDomain.TASK),
        object_reader=storage.read_object,
        artifacts=artifact,
        cursor_signer=HmacCursorSigner(
            "runtime-history",
            token_seed(storage.namespace),
        ),
    )


class _RuntimeCloseCoordinator:
    def __init__(self, actions: tuple[Callable[[], Awaitable[None]], ...]) -> None:
        self._actions = actions
        self._cursor = 0
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

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
