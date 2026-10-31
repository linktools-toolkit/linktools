#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime composition and local service graph construction."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from copy import copy
from pathlib import Path
from typing import TypeVar

from linktools.core import environ
from pydantic_ai.capabilities import AbstractCapability, WrapperCapability
from pydantic_ai.workspaces import WorkspaceBackend

from ..asset import AssetStoreReader
from ..agent import AgentBinding, AgentBindingContract, AgentCatalog, AgentCompiler
from ..capability import (
    ToolResponsesCapability,
    SessionCapability,
    MemoryCapability,
    ExecutionCapability,
    AgentContext,
    AssetCapability,
    AssetSkillSource,
    CapabilityContribution,
    SkillSourceRegistry,
    tool_class_from_metadata,
)
from ._lifecycle import _RuntimeCloseCoordinator
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
)
from ..errors import AIError, ErrorCode
from ..session import SessionPendingExecutionLookup
from ..observe import MetricRecorderProvider
from ..model import ModelBinding
from ..spec import (
    AgentSpec,
    RuleInstructionResolver,
    RepositoryInstructionResolver,
    RepositoryInstructions,
)
from ..storage import CacheBackend, ObjectStore, PayloadPolicy, StreamBackend, TurnCoordinationBackend
from ..workspace import (
    WorkspaceInstructionResolver,
    LocalWorkspaceBackend,
    SupportsClose,
    Workspace,
    WorkspaceAccess,
)
from ._agent_executor import AgentExecutor
from ._approval import DefaultApprovalService
from ._agent_binding_resolver import _AgentBindingResolver
from ..history import DefaultArtifactService
from ..execution.coordinator import ExecutionStreamCoordinator
from ..execution.events import DefaultEventService, LiveExecutionEventBroker
from ._external import DefaultExternalService
from ._execution import DefaultExecutionService, _ExecutionRuntimeBridge
from ..execution import ExecutionRecord
from ..execution.stream import SharedStreamPublisher
from ..execution.tree import ExecutionTreeBroker, ExecutionTreeStreamer
from ..history import StepExecutionHistoryReader, StepSessionHistoryReader
from ..history import DefaultExecutionHistoryService
from ..inputs import ExecutionInputMaterializer
from ..inputs import InputCaptures
from ..inputs.recovery import TaskRecoveryGuard
from ..inputs import ExecutionContextArchive
from ._local import LocalExecutionBackend
from ..persistence.coordination import PendingTurnCoordination
from ._memory import MemoryStore, RuntimeMemoryStore
from ..persistence.objects import ObjectKeyFactory
from ._capabilities import runtime_feature_leaves
from ..persistence.capabilities import CapabilityPersistence
from ..persistence.capabilities import RuntimeCapabilityWrites
from ..observe.service import MetricsService
from ._persistence_configuration import validate_persistence_capabilities, capability_persistence_ids
from ..capability import TraceCapability, CheckpointCapability
from ..history import History
from ..core import token_seed
from ._session import RuntimeSessionService
from ._subagent import SubagentDispatcher
from ..history import TransientExecutionHistoryStore
from ..capability import ToolResponseFixture
from ..execution import CancelExecutionRequest, ExecutionHistoryReader, SessionHistoryReader
from ..storage import StorageRetentionMode
from ..persistence import PersistenceDomain, Persistence
from ..persistence.runs import RuntimeAgentRunStore
from ..persistence.runs import StagingAgentRunStore

AppT = TypeVar("AppT")
_logger = environ.get_logger("ai.runtime.factory")


@dataclass(frozen=True, slots=True)
class _RuntimeComponents:
    state: Persistence
    run_store: RuntimeAgentRunStore
    catalog: AgentCatalog
    compiler: AgentCompiler
    execution: DefaultExecutionService
    session: RuntimeSessionService
    approval: DefaultApprovalService
    external: DefaultExternalService
    event: DefaultEventService
    artifact: DefaultArtifactService
    tenant_id: str
    close_callback: Callable[[], Awaitable[None]]
    handoff_callback: Callable[[], Awaitable[None]] | None
    tree_streamer: ExecutionTreeStreamer
    metric_control: MetricsService | None
    binding_resolver: _AgentBindingResolver
    history: object
    input_captures: InputCaptures
    memory_scope: str | None


class _AgentLifecycle:
    """Keep partial Agent resources reachable before composition can fail."""

    def __init__(self) -> None:
        self.run_store: RuntimeAgentRunStore | None = None
        self.workspace_access: WorkspaceAccess | None = None
        self.input_materializer: ExecutionInputMaterializer | None = None
        self.metric_buffer: MetricsService | None = None
        self.execution: DefaultExecutionService | None = None
        self.backend: LocalExecutionBackend | None = None
        self.session: RuntimeSessionService | None = None
        self._coordinator: _RuntimeCloseCoordinator | None = None

    def configure(
        self,
        actions: Sequence[tuple[str, Callable[[], Awaitable[None]]]],
        handoff_actions: Sequence[tuple[str, Callable[[], Awaitable[None]]]],
    ) -> None:
        self._coordinator = _RuntimeCloseCoordinator(
            tuple(action for _, action in actions),
            tuple(action for _, action in handoff_actions),
        )

    async def close(self) -> None:
        if self._coordinator is None:
            actions = _runtime_close_actions(
                execution=self.execution, backend=self.backend,
                input_materializer=None, metric_buffer=None, run_store=None,
                session=self.session,
            )
            if self.input_materializer is not None:
                actions = (*actions, ("runtime.input", self.input_materializer.close))
            elif self.workspace_access is not None:
                actions = (*actions, ("runtime.workspace", self.workspace_access.close))
            if self.run_store is not None:
                actions = (*actions, ("runtime.runs.preflight", self.run_store.preflight_close),
                           ("runtime.runs", self.run_store.close))
            if self.metric_buffer is not None:
                actions = (*actions, ("runtime.metrics", self.metric_buffer.close))
            self.configure(actions, ())
        await self._coordinator.close()

    async def handoff(self) -> None:
        if self._coordinator is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        await self._coordinator.handoff()


async def compose_runtime_components(
    *,
    persistence: Persistence,
    app: "AppT | None" = None,
    metrics: MetricRecorderProvider | None = None,
    features: "RuntimeFeatures[AppT] | RuntimeFeaturesCapture[AppT] | None" = None,
    auto_recover: bool = True,
    authorization: AuthorizationPolicy | None = None,
    lifecycle: "_AgentLifecycle | None" = None,
) -> _RuntimeComponents:
    """Capture declarations and build Runtime-private services."""
    if not isinstance(persistence, Persistence) or not persistence.ready:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "persistence must be open")
    if persistence.read_only:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "persistence is read-only")
    resolved_namespace = persistence.namespace
    effective_tenant_id = persistence.tenant_id
    if features is not None and not isinstance(features, (RuntimeFeatures, RuntimeFeaturesCapture)):
        raise TypeError("features must be RuntimeFeatures or RuntimeFeaturesCapture")
    captured = await features.capture() if isinstance(features, RuntimeFeatures) else features
    if captured is not None:
        await captured.verify_source_revision()
    selected_limits = PromptLimits() if captured is None else captured.limits
    execution_capability = ExecutionCapability()
    execution_declared = False
    session_capability = SessionCapability()
    session_declared = False
    memory_scope: str | None = None
    tool_responses: ToolResponseFixture | None = None
    workspace = None if captured is None else captured.workspace
    workspace_factory = LocalWorkspaceBackend if captured is None else captured.workspace_factory
    if workspace is not None:
        workspace.policy.validate()
    execution_cwd = _capture_host_cwd() if workspace is None else str(workspace.root)
    input_backend = (None if workspace_factory is None or execution_cwd is None
                     else workspace_factory(Path(execution_cwd), resources=()))
    if input_backend is not None and not isinstance(input_backend, SupportsClose):
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE,
                      safe_details={"reason": "workspace_cleanup_required"})

    selected_state: Persistence | None = None
    run_store: RuntimeAgentRunStore | None = None
    workspace_access: WorkspaceAccess | None = None
    input_materializer: ExecutionInputMaterializer | None = None
    owns_lifecycle = lifecycle is None
    lifecycle = _AgentLifecycle() if lifecycle is None else lifecycle
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
                MemoryCapability,
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
            nonlocal tool_responses, session_capability, session_declared, execution_capability, execution_declared, memory_scope
            if isinstance(capability, ExecutionCapability):
                if execution_declared:
                    raise AIError(ErrorCode.CAPABILITY_CONFLICT)
                execution_capability, execution_declared = capability, True
                return capability
            if isinstance(capability, MemoryCapability):
                if memory_scope is not None:
                    raise AIError(ErrorCode.CAPABILITY_CONFLICT)
                memory_scope = capability.scope
                return None
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

        captured_persistence_ids: set[str] = set()
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
                if identity in captured_persistence_ids:
                    raise AIError(ErrorCode.CAPABILITY_CONFLICT)
                captured_persistence_ids.add(identity)
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
        validate_persistence_capabilities(runtime_capabilities)
        capability_ids = capability_persistence_ids(
            tuple((capability.id or "", capability) for capability in runtime_capabilities),
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
        compile_candidates = tuple(candidate for candidate in candidates if candidate.kind != "agent")
        compiler_options = dict(defaults=None if captured is None else captured.defaults,
                                models=models, agents=agents, capability_filter=None if captured is None else captured.filter)

        class _RuntimeAgentCompiler(AgentCompiler):
            def restore(self, binding_contract: AgentBindingContract, *, model: ModelBinding | None = None) -> AgentBinding:
                memory_ids = {pin.id for pin in binding_contract.selected
                              if pin.kind == "tool" and tool_class_from_metadata(pin.contract.get("metadata")) == "memory"}
                if not memory_ids:
                    return super().restore(binding_contract, model=model)
                # Current admission selects host contributions. Admitted Memory pins
                # retain their native implementation even when the host disables it.
                memory_candidates = tuple(CapabilityContribution.from_opaque("tool", tool.name, tool)
                                          for tool in MemoryCapability().tool_definitions() if tool.name in memory_ids)
                memory_restore_compiler = AgentCompiler(
                    candidates=tuple(candidate for candidate in compile_candidates
                                     if candidate.kind != "tool" or candidate.id not in memory_ids) + memory_candidates,
                    **compiler_options,
                )
                return memory_restore_compiler.restore(binding_contract, model=model)

        compiler = _RuntimeAgentCompiler(candidates=compile_candidates, **compiler_options)
        root_agents = {
            agent_id: compiler.compile(agents[agent_id])
            for agent_id in sorted(agents)
        }
        catalog = AgentCatalog(root_agents)

        selected_state = persistence
        for identity in capability_ids:
            await persistence.capability_store(identity)

        run_store = await persistence.create_agent_run_store()
        lifecycle.run_store = run_store
        if selected_state.capability_stores:
            run_store.capability_writes = RuntimeCapabilityWrites(
                selected_state.capability_stores, namespace=resolved_namespace, tenant_id=effective_tenant_id,
                durable_capability_ids=selected_state.durable_capability_ids,
                require_active_run=run_store.require_active_capability_run,
                has_active_execution=run_store.has_active_execution_producer,
                marker_store=selected_state.capability_marker_store(),
            )
        run_store.capability_revisions = capability_revisions
        if captured is not None:
            await captured.verify_source_revision()
        if workspace is None:
            instruction_resolver: RepositoryInstructionResolver | None = (
                RuleInstructionResolver(rules)
                if rules.documents
                else None
            )
            workspace_access = None
        else:
            instruction_resolver = WorkspaceInstructionResolver(
                workspace.root,
                workspace.policy,
                rules,
            )
            workspace_access = WorkspaceAccess.for_workspace(
                workspace, backend=input_backend, owns_backend=True,
            )
            lifecycle.workspace_access = workspace_access
            execution_cwd = str(workspace.root)
        object_key_factory = ObjectKeyFactory(resolved_namespace)
        payload_policy = PayloadPolicy()
        input_materializer = ExecutionInputMaterializer(
            workspace_access,
            selected_limits,
            object_store=selected_state.object_store(PersistenceDomain.EXECUTION),
            object_key_factory=object_key_factory,
            payload_policy=payload_policy,
        )
        lifecycle.input_materializer = input_materializer
        runtime_token_seed = token_seed(resolved_namespace)
        history_reader = _execution_history_reader(
            resolved_namespace,
            selected_state,
            runtime_token_seed,
            run_store,
        )
        session_history_reader = StepSessionHistoryReader(
            store=run_store.read_store(PersistenceDomain.CONVERSATION),
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
        components = await _build_local_components(
            state=selected_state,
            lifecycle=lifecycle,
            run_store=run_store,
            metrics=metrics,
            commit=execution_capability.commit,
            coordination=persistence.coordination_backend,
            catalog=catalog,
            compiler=compiler,
            authorization=selected_authorization,
            tenant_id=effective_tenant_id,
            namespace=resolved_namespace,
            workspace=workspace,
            workspace_factory=workspace_factory,
            workspace_backend=input_backend,
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
            memory_scope=memory_scope,
            session_execution_ready=session_capability.enabled,
            auto_recover=auto_recover,
            tool_responses=tool_responses,
        )
        if captured is not None:
            await captured.verify_source_revision()
        _logger.info("runtime capability capture admitted: group=%s", None if captured is None else captured.group_id)
        return components
    except BaseException as primary_error:
        if owns_lifecycle:
            try:
                await lifecycle.close()
            except BaseException as cleanup_error:
                raise primary_error from cleanup_error
        raise


def _runtime_close_actions(
    *,
    execution: DefaultExecutionService | None,
    backend: LocalExecutionBackend | None,
    input_materializer: ExecutionInputMaterializer | None,
    metric_buffer: MetricsService | None,
    run_store: RuntimeAgentRunStore | None,
    session: RuntimeSessionService | None = None,
    handoff: bool = False,
) -> tuple[tuple[str, Callable[[], Awaitable[None]]], ...]:
    actions: list[tuple[str, Callable[[], Awaitable[None]]]] = []
    if session is not None:
        actions.append(("runtime.session.display", session.drain_display_reads))
    if execution is not None:
        actions.append(("runtime.execution.preflight", execution.preflight_close))
    if backend is not None:
        actions.append(("runtime.backend", backend.handoff if handoff else backend.close))
    if run_store is not None:
        actions.append(("runtime.runs.preflight", run_store.preflight_close))
        actions.append(("runtime.runs", run_store.close))
    if input_materializer is not None:
        actions.append(("runtime.input", input_materializer.close))
    if metric_buffer is not None:
        actions.append(("runtime.metrics", metric_buffer.close))
    return tuple(actions)


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
    state: Persistence,
    runtime_token_seed: bytes,
    run_store: RuntimeAgentRunStore,
) -> StepExecutionHistoryReader:
    local_store = run_store.read_store(PersistenceDomain.TRACE, durable=False)
    if not isinstance(local_store, StagingAgentRunStore):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    transient_store = TransientExecutionHistoryStore(local_store, run_store)
    history_store = run_store.read_store(PersistenceDomain.TRACE)
    if state.plan.route(PersistenceDomain.TRACE).retention is StorageRetentionMode.TRANSIENT:
        if not isinstance(history_store, StagingAgentRunStore):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        history_store = TransientExecutionHistoryStore(history_store, run_store)
    return StepExecutionHistoryReader(
        namespace=namespace,
        executions=state.execution.executions,
        store=history_store,
        transient_store=transient_store,
        cursor_signer=HmacCursorSigner("execution-history", runtime_token_seed),
        tool_operations=state.recovery.tools,
        notifications=run_store,
        durable_history_available=run_store.model_interaction_history_available,
    )


def _memory_store_factory(
    namespace: str,
    state: Persistence,
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
    state: Persistence,
    *,
    namespace: str,
    tenant_id: str,
) -> None:
    if state.namespace != namespace or state.tenant_id != tenant_id:
        raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)


async def _build_local_components(
    *,
    state: Persistence,
    lifecycle: "_AgentLifecycle",
    run_store: RuntimeAgentRunStore,
    metrics: MetricRecorderProvider | None,
    commit: CommitPolicy,
    coordination: CacheBackend | TurnCoordinationBackend | StreamBackend | None,
    catalog: AgentCatalog,
    compiler: AgentCompiler,
    authorization: AuthorizationPolicy,
    tenant_id: str,
    namespace: str,
    workspace: "Workspace | None",
    workspace_factory: "Callable[..., WorkspaceBackend] | None",
    workspace_backend: "WorkspaceBackend | None",
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
    object_key_factory: ObjectKeyFactory,
    payload_policy: PayloadPolicy,
    input_materializer: ExecutionInputMaterializer,
    memory_scope: str | None,
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
        released = await state.retention.release_execution_handoff(
            execution_id, tenant_id=tenant_id, terminal_commit=terminal_commit,
        )
        if released:
            await run_store.release_execution_staging(execution_id)
        return released

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
        metric_buffer = None if metrics is None else MetricsService(metrics)
        lifecycle.metric_buffer = metric_buffer
        metric_source_namespace = None if metric_buffer is None else namespace
        if coordination is not None:
            runtime_bridge.pending_turns.bind_coordination(PendingTurnCoordination.select(
                coordination, namespace=namespace, tenant_id=tenant_id,
            ))
        stream_cache = (
            coordination
            if isinstance(coordination, StreamBackend)
            and state.plan.route(PersistenceDomain.EXECUTION).retention is StorageRetentionMode.DURABLE
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
            workspace_backend=workspace_backend,
        )
        execution = DefaultExecutionService(
            state.execution,
            state.object_store(PersistenceDomain.EXECUTION),
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
            commit=commit,
            context_archive=ExecutionContextArchive(
                state.execution.executions.state_store,
                state.archive_object_store(PersistenceDomain.EXECUTION),
                namespace=namespace, tenant_id=tenant_id,
                object_reader=lambda reference: state.read_archive_object(PersistenceDomain.EXECUTION, reference),
            ),
        )
        lifecycle.execution = execution
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
            workspace_factory=workspace_factory,
            tool_responses=tool_responses,
        )
    except BaseException:
        raise

    def build_memory_store(
        memory_tenant: str,
        execution_id: str,
        memory_scope: str,
    ) -> MemoryStore:
        if memory_store_factory is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        route = state.plan.route(PersistenceDomain.MEMORY)
        transient = route.retention is StorageRetentionMode.TRANSIENT
        store = (
            state.working_object_store(
                PersistenceDomain.MEMORY,
                owner_scope=f"execution:{execution_id}",
            )
            if transient
            else state.object_store(PersistenceDomain.MEMORY)
        )
        return memory_store_factory(
            memory_tenant,
            execution_id,
            memory_scope,
            store,
            transient,
        )

    session: RuntimeSessionService | None = None
    coordinator: _RuntimeCloseCoordinator | None = None
    close_actions: tuple[tuple[str, Callable[[], Awaitable[None]]], ...] | None = None

    async def worker_completed(execution_id: str) -> None:
        await execution.resume_terminal_handoff(execution_id, tenant_id=tenant_id)

    exclusive_execution_writer = state.exclusive_execution_writer
    try:
        backend = LocalExecutionBackend(
            state.conversation,
            state.execution,
            state.recovery,
            state.object_store(PersistenceDomain.EXECUTION),
            state.object_store(PersistenceDomain.RECOVERY),
            namespace,
            run_store,
            executor,
            catalog,
            archive_objects=state.archive_object_store(PersistenceDomain.EXECUTION),
            archive_object_reader=lambda reference: state.read_archive_object(PersistenceDomain.EXECUTION, reference),
            restore_binding=compiler.restore,
            tenant_id=tenant_id,
            workspace=workspace,
            limits=limits,
            execution_cwd=execution_cwd,
            instruction_resolver=instruction_resolver,
            app=app,
            run_stores={
                domain: run_store.read_store(domain)
                for domain in (
                    PersistenceDomain.CONVERSATION,
                    PersistenceDomain.TRACE,
                    PersistenceDomain.CHECKPOINT,
                )
            },
            agent_run_lifecycle=run_store,
            memory_store_factory=build_memory_store,
            conversation_durable=(
                state.plan.route(PersistenceDomain.CONVERSATION).retention
                is StorageRetentionMode.DURABLE
            ),
            input_materializer=input_materializer,
            subagent_dispatcher=dispatcher,
            live_broker=live_broker,
            payload_policy=payload_policy,
            execution_objects_durable=(
                state.plan.route(PersistenceDomain.EXECUTION).retention
                is StorageRetentionMode.DURABLE
            ),
            tool_operations=state.recovery.tools,
            metric_recorder=metric_buffer,
            worker_completed=worker_completed,
            exclusive_execution_writer=exclusive_execution_writer,
            commit=commit,
            trace_commit=next((value.commit for value in persistence_capabilities
                               if isinstance(value, TraceCapability)), None),
            pending_turns=runtime_bridge.pending_turns,
            pending_cancel=lambda execution_id, principal: execution.cancel(
                execution_id, CancelExecutionRequest(principal, f"pending-cancel:{execution_id}"),
            ),
        )
        runtime_bridge.bind(backend)
        runtime_bridge.pending_start = backend.start_pending
        runtime_bridge.pending_promote = backend.promote_pending_execution
        lifecycle.backend = backend
        runtime_bridge.pending_cancel_promote = lambda execution_id: backend.promote_pending_execution(
            execution_id, allow_cancelled=True,
        )
        run_store.bind_pending_producer_guard(backend.require_pending_producer)
        session = RuntimeSessionService(
            state.conversation,
            state.execution.executions,
            authorization,
            execution,
            HmacCursorSigner("session", runtime_token_seed),
            history_reader=session_history_reader,
            enabled=session_execution_ready,
            cache=(state.plan.route(PersistenceDomain.CONVERSATION).cache if all(
                state.plan.route(domain).retention is StorageRetentionMode.DURABLE
                for domain in (PersistenceDomain.CONVERSATION, PersistenceDomain.EXECUTION)
            ) else None),
            transcript_store=run_store.conversation_persistence,
            release_terminal=state.retention.release_session,
            pending_session_execution=runtime_bridge.pending_turns.read_session_execution,
            observe_local_session=runtime_bridge.pending_turns.observe_session,
            read_session_owner=(None if runtime_bridge.pending_turns.coordination is None else
                                runtime_bridge.pending_turns.coordination.read_session_owner),
            local_waiter=backend,
            workspace_access=input_materializer.access,
        )
        lifecycle.session = session

        async def prepare_pending_capture(execution_id: str, principal) -> None:
            await execution.load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_CAPTURE_INPUT)
            await execution.load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
            await backend.promote_pending_execution(execution_id)

        input_captures = InputCaptures(
            namespace, authorization, execution, input_materializer,
            tenant_id=tenant_id,
            execution_repositories=state.execution,
            task_repositories=state.task,
            object_stores={domain: state.object_store(domain) for domain in PersistenceDomain},
            execution_retention=state.plan.route(PersistenceDomain.EXECUTION).retention,
            task_retention=state.plan.route(PersistenceDomain.TASK).retention,
            context_archive=ExecutionContextArchive(
                state.execution.executions.state_store,
                state.archive_object_store(PersistenceDomain.EXECUTION),
                namespace=namespace, tenant_id=tenant_id,
                object_reader=lambda reference: state.read_archive_object(PersistenceDomain.EXECUTION, reference),
            ),
            prepare_pending_capture=prepare_pending_capture,
        )
        backend.bind_recovery_relaunch_guard(TaskRecoveryGuard(state, input_captures))
        approval = DefaultApprovalService(
            state.recovery.approvals,
            state.execution.executions,
            state.recovery.checkpoints,
            authorization,
            objects=state.object_store(PersistenceDomain.RECOVERY),
            continuation=backend,
        )
        external = DefaultExternalService(
            state.recovery.external_calls,
            state.execution.executions,
            state.recovery.checkpoints,
            authorization,
            objects=state.object_store(PersistenceDomain.RECOVERY),
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
            history_service, tenant_id=tenant_id, state=state, run_store=run_store,
            authorization=authorization, artifact=artifact,
            pending_session_execution=(
                None if pending_observer is None else pending_observer.read_session_execution
            ),
            pending_execution=None if pending_observer is None else pending_observer.read_execution,
        )
        local_coordinator = ExecutionStreamCoordinator(execution, event)
        tree_streamer = ExecutionTreeStreamer(
            execution,
            local_coordinator,
            execution_tree_broker,
        )
        close_actions = _runtime_close_actions(
            execution=execution,
            backend=backend,
            input_materializer=input_materializer,
            metric_buffer=metric_buffer,
            run_store=run_store,
            session=session,
        )
        handoff_actions = _runtime_close_actions(
            execution=execution, backend=backend, input_materializer=input_materializer,
            metric_buffer=metric_buffer, run_store=run_store, session=session, handoff=True,
        )
        lifecycle.configure(close_actions, handoff_actions)
        await run_store.prepare_archives()
        if auto_recover and PersistenceDomain.RECOVERY in state.plan.durable_domains:
            await backend.reconcile(
                allow_active_recovery=exclusive_execution_writer,
            )
    except BaseException:
        raise
    return _RuntimeComponents(
        state=state,
        run_store=run_store,
        catalog=catalog,
        compiler=compiler,
        execution=execution,
        session=session,
        approval=approval,
        external=external,
        event=event,
        artifact=artifact,
        tenant_id=tenant_id,
        close_callback=lifecycle.close,
        handoff_callback=(lifecycle.handoff if restart_checkpoints and commit.mode == "sync"
            and all(state.plan.route(domain).retention is StorageRetentionMode.DURABLE
                    for domain in (PersistenceDomain.EXECUTION, PersistenceDomain.RECOVERY,
                                   PersistenceDomain.CHECKPOINT)) else None),
        tree_streamer=tree_streamer,
        metric_control=metric_buffer,
        binding_resolver=binding_resolver,
        history=history,
        input_captures=input_captures,
        memory_scope=memory_scope,
    )


def _borrowed_runtime_history(
    service: DefaultExecutionHistoryService,
    *,
    tenant_id: str,
    state: Persistence,
    run_store: RuntimeAgentRunStore,
    authorization: object,
    artifact: DefaultArtifactService,
    pending_session_execution: SessionPendingExecutionLookup[ExecutionRecord] | None = None,
    pending_execution: Callable[..., Awaitable[ExecutionRecord | None]] | None = None,
) -> "History":
    return History(
        service,
        tenant_id=tenant_id,
        executions=state.execution.executions,
        events=state.execution.events,
        sessions=state.conversation.sessions,
        conversation=state.conversation,
        session_transcript_store=run_store.conversation_persistence,
        pending_session_execution=pending_session_execution,
        pending_execution=pending_execution,
        cache=(state.plan.route(PersistenceDomain.CONVERSATION).cache if all(
            state.plan.route(domain).retention is StorageRetentionMode.DURABLE
            for domain in (PersistenceDomain.CONVERSATION, PersistenceDomain.EXECUTION)
        ) else None),
        tasks=state.task.tasks,
        authorization=authorization,
        namespace=state.namespace,
        execution_objects=state.object_store(PersistenceDomain.EXECUTION),
        task_objects=state.object_store(PersistenceDomain.TASK),
        object_reader=state.read_object,
        artifacts=artifact,
        cursor_signer=HmacCursorSigner(
            "runtime-history",
            token_seed(state.namespace),
        ),
    )




__all__ = []
