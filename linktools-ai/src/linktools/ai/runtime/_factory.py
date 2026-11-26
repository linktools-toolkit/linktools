#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime composition and local service graph construction."""

import asyncio
import secrets
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from copy import copy
from pathlib import Path
from typing import TypeVar

from linktools.core import environ
from pydantic_ai_harness.step_persistence import StepPersistence
from pydantic_ai.capabilities import AbstractCapability, WrapperCapability
from pydantic_ai.workspaces import WorkspaceBackend

from ..asset import AssetStoreReader
from ..agent import AgentBinding, AgentBindingContract, AgentCatalog, AgentCompiler, CompiledAgent
from ..steps import NativeStepSource
from ..capability import (
    ToolResponsesCapability,
    SessionCapability,
    MemoryCapability,
    StepPersistenceCapability,
    AgentContext,
    AssetCapability,
    AssetSkillSource,
    CapabilityContribution,
    SkillSourceRegistry,
    SkillDefinition,
    tool_class_from_metadata,
)
from ._lifecycle import _RuntimeCloseCoordinator
from ._native_session import _NativeTurnRead
from ._feature_policy import _RuntimeFeaturePolicy, _runtime_feature_policy
from ._features import (
    RuntimeFeatures,
    RuntimeFeaturesSource,
    RuntimeFeaturesResolution,
)
from ..core import (
    CommitPolicy,
    ExecutionStatus,
    AuthorizationAction,
    AuthorizationPolicy,
    HmacCursorSigner,
    PromptLimits,
    Principal,
    ResourceKind,
    ResourceRef,
    agent_conversation_id,
    TenantAuthorizationPolicy,
)
from ..errors import AIError, ErrorCode
from ..session import SessionPendingExecutionLookup
from ..observe import MetricRecorderProvider
from ..model import ModelBinding
from ..spec import (
    AgentSpec,
    CapabilityRef,
    RuleInstructionResolver,
    RepositoryInstructionResolver,
    RepositoryInstructions,
)
from ..storage import CacheBackend, ObjectStore
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
from ..execution import CancelExecutionRequest, ExecutionRecord
from ..execution.stream import SharedStreamPublisher
from ..execution.tree import ExecutionTreeBroker, ExecutionTreeStreamer
from ..history import StepExecutionHistoryReader, StepSessionHistoryReader
from ..history import DefaultExecutionHistoryService
from ..inputs import ExecutionInputMaterializer
from ..inputs import InputCaptures
from ..inputs.recovery import TaskRecoveryGuard
from ..inputs import ExecutionContextArchive
from ._local import LocalExecutionBackend
from pydantic_ai_harness.memory import MemoryStore
from ..memory import PersistentMemoryStore
from ..persistence.objects import ObjectKeyFactory
from ._capabilities import runtime_feature_leaves
from ..persistence.capabilities import CapabilityPersistence
from ..persistence.capabilities import RuntimeCapabilityWrites
from ..observe.service import MetricsService
from ._persistence_configuration import validate_persistence_capabilities, capability_persistence_ids
from ..capability import TraceCapability
from ..history import History
from ..core import token_seed
from ._regenerable_turn import _RegenerableTurnRunner
from ._session import RuntimeSessionService
from ._subagent import SubagentDispatcher
from ..history import TransientExecutionHistoryStore
from ..capability import ToolResponseFixture
from ..execution import CancelExecutionRequest, ExecutionHistoryReader
from ..storage import StorageRetentionMode
from ..persistence import PersistenceDomain, Persistence
from ..persistence.commands import graph_task_repository
from ..persistence.repositories import BudgetRepositoryImpl, ExecutionRepositoryImpl, ToolRepositoryImpl
from ..persistence.runs import RuntimeAgentRunStore
from ..persistence.runs import StagingAgentRunStore, StateStepArchive

AppT = TypeVar("AppT")
_logger = environ.get_logger("ai.runtime.factory")


@dataclass(frozen=True, slots=True)
class _RuntimeComponents:
    state: Persistence
    run_store: RuntimeAgentRunStore
    catalog: AgentCatalog
    compiler: AgentCompiler
    execution: DefaultExecutionService
    session: RuntimeSessionService | None
    turn_runner: _RegenerableTurnRunner | None
    require_session_available: Callable[[], Awaitable[None]]
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
    native_turn_read: _NativeTurnRead | None
    input_captures: InputCaptures
    memory_scope: str | None
    asset_sources: Mapping[str, AssetStoreReader]


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
        self.turn_runner: _RegenerableTurnRunner | None = None
        self.step_store_cleanup: Callable[[], Awaitable[None]] | None = None
        self._coordinator: _RuntimeCloseCoordinator | None = None

    def configure(
        self,
        actions: Sequence[tuple[str, Callable[[], Awaitable[None]]]],
        handoff_actions: Sequence[tuple[str, Callable[[], Awaitable[None]]]],
    ) -> None:
        resource_actions = (self.close_step_store,)
        self._coordinator = _RuntimeCloseCoordinator(
            (*tuple(action for _, action in actions), *resource_actions),
            (*tuple(action for _, action in handoff_actions), *resource_actions),
        )

    async def close_step_store(self) -> None:
        if self.step_store_cleanup is not None:
            await self.step_store_cleanup()

    async def close(self) -> None:
        if self._coordinator is None:
            actions = _runtime_close_actions(
                execution=self.execution, backend=self.backend,
                input_materializer=None, metric_buffer=None, run_store=None,
                session=self.session, turn_runner=self.turn_runner,
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


def _resolution_asset_readers(resolution: RuntimeFeaturesResolution | None) -> dict[str, AssetStoreReader]:
    if resolution is None:
        return {}
    return {
        resolution.group_id: leaf.reader
        for contribution in resolution.contributions if contribution.kind == "capability"
        for leaf in runtime_feature_leaves(contribution.value) if isinstance(leaf, AssetCapability)
    }


async def compose_runtime_components(
    *,
    persistence: Persistence,
    app: "AppT | None" = None,
    metrics: MetricRecorderProvider | None = None,
    features: "RuntimeFeatures[AppT] | RuntimeFeaturesSource[AppT] | None" = None,
    auto_recover: bool = False,
    authorization: AuthorizationPolicy | None = None,
    lifecycle: "_AgentLifecycle | None" = None,
    wake_graph: Callable[[str, str], Awaitable[None]] | None = None,
) -> _RuntimeComponents:
    """Open host resources and resolve declarations when they are used."""
    if not isinstance(persistence, Persistence) or not persistence.ready:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "persistence must be open")
    if persistence.read_only:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "persistence is read-only")
    resolved_namespace = persistence.namespace
    effective_tenant_id = persistence.tenant_id
    if features is not None and not isinstance(features, (RuntimeFeatures, RuntimeFeaturesSource)):
        raise TypeError("features must be RuntimeFeatures or RuntimeFeaturesSource")
    source = features.borrow() if isinstance(features, RuntimeFeatures) else features
    captured = None if source is None else await source.resolve()
    selected_limits = PromptLimits() if captured is None else captured.limits
    workspace = None if captured is None else captured.workspace
    workspace_factory = LocalWorkspaceBackend if captured is None else captured.workspace_factory
    if workspace is not None:
        workspace.policy.validate()
    execution_cwd = _capture_host_cwd() if workspace is None else str(workspace.root)
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
            asset_sources.update(_resolution_asset_readers(captured))
        _validate_candidate_uniqueness(candidates)
        policy = _runtime_feature_policy(candidates)
        runtime_capabilities = policy.capabilities
        execution_capability, session_capability = policy.execution, policy.session
        memory_scope, tool_responses = policy.memory_scope, policy.tool_responses
        capability_revisions, capability_ids = policy.capability_revisions, policy.capability_ids
        skill_sources = SkillSourceRegistry(tuple(
            AssetSkillSource(group_id, reader) for group_id, reader in asset_sources.items()
        ))
        rules = RepositoryInstructions(() if captured is None else captured.instructions.documents)

        def build_compiler(resolution) -> AgentCompiler:
            current = () if resolution is None else resolution.contributions
            _validate_candidate_uniqueness(current)
            agents = {item.id: item.value for item in current if item.kind == "agent"}
            models = {} if resolution is None else resolution.models
            if "default" not in agents and "default" in models:
                agents["default"] = AgentSpec("default")
            return AgentCompiler(
                candidates=tuple(item for item in current if item.kind != "agent"),
                defaults=None if resolution is None else resolution.defaults,
                models=models, agents=agents,
                repository_instructions=RepositoryInstructions(()) if resolution is None else resolution.instructions,
                capability_filter=None if resolution is None else resolution.filter,
            )

        compiler = build_compiler(captured)
        root_agents = {agent_id: compiler.compile_registered(agent_id) for agent_id in compiler.agent_ids}
        catalog = AgentCatalog(root_agents)

        async def current_resolution():
            current = None if source is None else await source.resolve()
            if current is not None:
                selected_workspace = current.workspace
                if (selected_workspace is not None and (workspace is None
                        or selected_workspace.root != workspace.root or selected_workspace.policy != workspace.policy)
                        or current.workspace_factory is not workspace_factory
                        or any(asset_sources.get(key) is not reader for key, reader in _resolution_asset_readers(current).items())
                        or _runtime_feature_policy(current.contributions).execution.commit.buffer is not policy.execution.commit.buffer):
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                                  safe_details={"reason": "runtime_resource_replacement"})
            return current

        async def current_compiler() -> AgentCompiler:
            current = await current_resolution()
            await prepare_runtime_policy(_runtime_feature_policy(() if current is None else current.contributions))
            return build_compiler(current)

        async def current_policy():
            current = await current_resolution()
            selected_policy = _runtime_feature_policy(() if current is None else current.contributions)
            await prepare_runtime_policy(selected_policy)
            return selected_policy

        async def current_agent(compiled: CompiledAgent) -> CompiledAgent:
            current = build_compiler(await current_resolution())
            return current.compile_current(compiled.spec, model=compiled.model)

        selected_state = persistence
        default_step_store = None
        resource_lock = asyncio.Lock()

        async def prepare_runtime_policy(selected_policy: _RuntimeFeaturePolicy):
            nonlocal default_step_store
            values = selected_policy.capabilities
            async with resource_lock:
                for identity in selected_policy.capability_ids:
                    store = await persistence.capability_store(identity)
                    if run_store is not None and run_store.capability_writes is not None:
                        run_store.capability_writes.register_store(
                            identity, store, durable=identity in persistence.durable_capability_ids,
                        )
                declaration = next((leaf for feature in values for leaf in runtime_feature_leaves(feature)
                                    if isinstance(leaf, StepPersistenceCapability)), None)
                if declaration is not None:
                    store = declaration.store
                    if store is None:
                        if default_step_store is None:
                            default_step_store = await persistence.create_step_store()
                            lifecycle.step_store_cleanup = default_step_store.close
                        store = default_step_store
                    native = declaration.bind(store)

                    def bind_steps(value):
                        if value is declaration:
                            return native
                        if isinstance(value, WrapperCapability):
                            bound = copy(value)
                            bound.wrapped = value.wrapped.visit_and_replace(bind_steps)
                            return bound
                        return value

                    values = tuple(value.visit_and_replace(bind_steps) for value in values)
                if run_store is not None:
                    for feature in values:
                        for leaf in runtime_feature_leaves(feature):
                            if isinstance(leaf, StepPersistence):
                                run_store.bind_step_store(leaf.store)
            return values

        async def prepare_runtime_capabilities(compiled: CompiledAgent):
            return await prepare_runtime_policy(_runtime_feature_policy(compiled.runtime_contributions))

        # Source inspection does not initialize resources. Runtime owns the
        # default adapters, including one first requested by a later admission.
        if root_agents:
            runtime_capabilities = await prepare_runtime_capabilities(next(iter(root_agents.values())))
        step_features = [leaf for feature in runtime_capabilities for leaf in runtime_feature_leaves(feature)
                         if isinstance(leaf, StepPersistence)]
        run_store = await persistence.create_agent_run_store(
            step_store=step_features[0].store if step_features else None,
        )
        lifecycle.run_store = run_store
        run_store.capability_writes = RuntimeCapabilityWrites(
            selected_state.capability_stores, namespace=resolved_namespace, tenant_id=effective_tenant_id,
            durable_capability_ids=selected_state.durable_capability_ids,
            require_active_run=run_store.require_active_capability_run,
            has_active_execution=run_store.has_active_execution_producer,
            marker_store=selected_state.capability_marker_store(),
        )
        run_store.capability_revisions = capability_revisions
        def resolver_for(rules: RepositoryInstructions) -> RepositoryInstructionResolver:
            return (RuleInstructionResolver(rules) if workspace is None else
                    WorkspaceInstructionResolver(workspace.root, workspace.policy, rules))

        class _CurrentInstructionResolver:
            def validate(self, instructions: RepositoryInstructions) -> None:
                resolver_for(RepositoryInstructions(())).validate(instructions)

            async def resolve(
                self, path: str | Path = ".", *, exclude_sources: frozenset[str] = frozenset(),
            ) -> RepositoryInstructions:
                current = await current_resolution()
                current_rules = RepositoryInstructions(()) if current is None else current.instructions
                return await resolver_for(current_rules).resolve(path, exclude_sources=exclude_sources)

        instruction_resolver = _CurrentInstructionResolver()
        if workspace is None:
            workspace_access = None
        else:
            input_backend = (None if workspace_factory is None
                             else workspace_factory(workspace.root, resources=()))
            if input_backend is not None and not isinstance(input_backend, SupportsClose):
                raise AIError(ErrorCode.SANDBOX_UNAVAILABLE,
                              safe_details={"reason": "workspace_cleanup_required"})
            workspace_access = WorkspaceAccess.for_workspace(
                workspace, backend=input_backend, owns_backend=True,
            )
            lifecycle.workspace_access = workspace_access
            execution_cwd = str(workspace.root)
        object_key_factory = ObjectKeyFactory(resolved_namespace)
        input_materializer = ExecutionInputMaterializer(
            workspace_access,
            selected_limits,
            object_store=selected_state.object_store(PersistenceDomain.EXECUTION),
            object_key_factory=object_key_factory,
        )
        lifecycle.input_materializer = input_materializer
        runtime_token_seed = token_seed(resolved_namespace)
        history_reader = _execution_history_reader(
            resolved_namespace,
            selected_state,
            runtime_token_seed,
            run_store,
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
            catalog=catalog,
            compiler=compiler,
            authorization=selected_authorization,
            enable_turn_sessions=authorization is not None and type(authorization) is not TenantAuthorizationPolicy,
            tenant_id=effective_tenant_id,
            namespace=resolved_namespace,
            workspace=workspace,
            workspace_factory=workspace_factory,
            limits=selected_limits,
            execution_cwd=execution_cwd,
            app=app,
            history_reader=history_reader,
            memory_store_factory=memory_store_factory,
            skill_sources=skill_sources,
            current_agent=current_agent,
            current_compiler=current_compiler,
            prepare_runtime_capabilities=prepare_runtime_capabilities,
            current_policy=current_policy,
            runtime_capabilities=runtime_capabilities,
            asset_sources=asset_sources,
            runtime_token_seed=runtime_token_seed,
            instruction_resolver=instruction_resolver,
            instruction_resolver_for=resolver_for,
            object_key_factory=object_key_factory,
            input_materializer=input_materializer,
            memory_scope=memory_scope,
            session_execution_ready=session_capability.enabled,
            auto_recover=auto_recover,
            tool_responses=tool_responses,
            wake_graph=wake_graph,
        )
        _logger.info("runtime feature source opened: group=%s", None if captured is None else captured.group_id)
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
    turn_runner: _RegenerableTurnRunner | None = None,
    handoff: bool = False,
) -> tuple[tuple[str, Callable[[], Awaitable[None]]], ...]:
    actions: list[tuple[str, Callable[[], Awaitable[None]]]] = []
    if turn_runner is not None:
        actions.append(("runtime.turns", turn_runner.close))
    if session is not None:
        actions.append(("runtime.session.watch", session.close_watches))
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
        tool_operations=state.checkpoint.tools,
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
        return PersistentMemoryStore(
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
    catalog: AgentCatalog,
    compiler: AgentCompiler,
    authorization: AuthorizationPolicy,
    enable_turn_sessions: bool,
    tenant_id: str,
    namespace: str,
    workspace: "Workspace | None",
    workspace_factory: "Callable[..., WorkspaceBackend] | None",
    limits: PromptLimits,
    execution_cwd: "str | None",
    app: AppT,
    history_reader: ExecutionHistoryReader,
    memory_store_factory: "Callable[[str, str, str, ObjectStore, bool], MemoryStore] | None",
    skill_sources: SkillSourceRegistry,
    current_agent: Callable[[CompiledAgent], Awaitable[CompiledAgent]],
    current_compiler: Callable[[], Awaitable[AgentCompiler]],
    prepare_runtime_capabilities: Callable[[CompiledAgent], Awaitable[Sequence[AbstractCapability[AgentContext[AppT]]]]],
    current_policy: Callable[[], Awaitable[_RuntimeFeaturePolicy]],
    runtime_capabilities: Sequence[AbstractCapability[AgentContext[AppT]]],
    asset_sources: Mapping[str, AssetStoreReader],
    runtime_token_seed: bytes,
    instruction_resolver: "RepositoryInstructionResolver | None",
    instruction_resolver_for: Callable[[RepositoryInstructions], RepositoryInstructionResolver],
    object_key_factory: ObjectKeyFactory,
    input_materializer: ExecutionInputMaterializer,
    memory_scope: str | None,
    session_execution_ready: bool,
    auto_recover: bool,
    tool_responses: ToolResponseFixture | None,
    wake_graph: Callable[[str, str], Awaitable[None]] | None,
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
        metric_buffer = None if metrics is None else MetricsService(metrics)
        lifecycle.metric_buffer = metric_buffer
        metric_source_namespace = None if metric_buffer is None else namespace
        runtime_bridge.pending_turns.bind_coordination(state.pending_turn_coordination)
        stream_cache = (state.turn_stream
                        if state.plan.route(PersistenceDomain.EXECUTION).retention is StorageRetentionMode.DURABLE
                        else None)
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
            current_compiler=current_compiler,
        )
        async def native_step_source(binding: AgentBinding) -> NativeStepSource | None:
            await prepare_runtime_capabilities(binding.compiled_agent)
            return run_store.native_step_source

        execution = DefaultExecutionService(
            state.execution,
            state.object_store(PersistenceDomain.EXECUTION),
            authorization,
            sessions=state.conversation.sessions,
            resolve_task_admissions=lambda: state.task.admissions,
            catalog=catalog,
            compiler=compiler,
            restore_binding=binding_resolver.restore,
            native_step_source=native_step_source,
            runtime_bridge=runtime_bridge,
            live_broker=live_broker,
            history_reader=history_reader,
            history_service=history_service,
            release_terminal=release_execution_handoff,
            retained_turn_confirmed=state.retention.terminal_publication_confirmed,
            retained_publication=state.retention.terminal_publication,
            instruction_resolver=instruction_resolver,
            instruction_resolver_for=instruction_resolver_for,
            object_key_factory=object_key_factory,
            input_materializer=input_materializer,
            object_reader=state.read_object,
            context_archive=ExecutionContextArchive(
                state.execution.executions.state_store,
                state.archive_object_store(PersistenceDomain.EXECUTION),
                namespace=namespace, tenant_id=tenant_id,
                asset_sources=asset_sources,
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
            restore_binding=binding_resolver.restore,
        )
        turn_state = state.conversation.histories.state_store
        conversation_group = turn_state.storage_group
        canonical_memory_store = (
            build_memory_store
            if state.session_turns is not None
            and memory_store_factory is not None
            and state.plan.route(PersistenceDomain.MEMORY).retention is StorageRetentionMode.DURABLE
            and state.memory.records.state_store.storage_group is conversation_group
            and state.memory.operations.state_store.storage_group is conversation_group
            else None
        )
        executor = AgentExecutor(
            skill_sources,
            current_agent=current_agent,
            prepare_runtime_capabilities=prepare_runtime_capabilities,
            runtime_capabilities=runtime_capabilities,
            asset_sources=asset_sources,
            metrics=metric_buffer,
            workspace_factory=workspace_factory,
            tool_responses=tool_responses,
            canonical_memory_store=canonical_memory_store,
        )
    except BaseException:
        raise

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
            state.checkpoint,
            state.object_store(PersistenceDomain.EXECUTION),
            state.object_store(PersistenceDomain.CHECKPOINT),
            namespace,
            run_store,
            executor,
            catalog,
            archive_objects=state.archive_object_store(PersistenceDomain.EXECUTION),
            archive_object_reader=lambda reference: state.read_archive_object(PersistenceDomain.EXECUTION, reference),
            asset_sources=asset_sources,
            restore_binding=binding_resolver.restore,
            native_step_source=native_step_source,
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
            execution_objects_durable=(
                state.plan.route(PersistenceDomain.EXECUTION).retention
                is StorageRetentionMode.DURABLE
            ),
            tool_operations=state.checkpoint.tools,
            metric_recorder=metric_buffer,
            worker_completed=worker_completed,
            terminal_publication=state.retention.confirm_terminal_publication,
            retained_publication=state.retention.terminal_publication,
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
        if live_broker.shared is not None and type(state.execution.executions) is ExecutionRepositoryImpl:
            runtime_bridge.child_topology = live_broker.shared.child
        runtime_bridge.prepare_reserved_start = backend.prepare_reserved_start
        runtime_bridge.pending_start = backend.start_pending
        runtime_bridge.run_turn = backend.run_turn
        runtime_bridge.can_group_start = backend.can_group_start
        runtime_bridge.run_existing_turn = backend.run_existing_turn
        runtime_bridge.resume_subagent = backend.resume_subagent
        runtime_bridge.pending_promote = backend.promote_pending_execution
        lifecycle.backend = backend
        runtime_bridge.pending_cancel_promote = lambda execution_id: backend.promote_pending_execution(
            execution_id, allow_cancelled=True,
        )
        run_store.bind_pending_producer_guard(backend.require_pending_producer)
        split_sql_source = state._configured_split_step_source() if enable_turn_sessions else None

        async def read_split_effects(turn_id):
            return await state.checkpoint.tools.list_by_execution(turn_id, tenant_id=tenant_id)

        async def open_split_recovery_source():
            from ._canonical_split_step import _CanonicalSplitStep

            source, native, _, _ = await state._open_split_step_archive()
            return _CanonicalSplitStep(
                state.session_turns, source, native, read_effects=read_split_effects,
                stop_effects=(state.checkpoint.tools, tenant_id),
            )

        async def prepare_split_wait_resume(source, key, generation, binding_payload, resolver, *, ready):
            from ._tool import RuntimeToolOperationBridge

            if binding_payload is None:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            # This unbound bridge only decodes the persisted native leaf. It
            # has no canonical write boundary or ability to dispatch.
            receipt_reader = RuntimeToolOperationBridge(
                state.checkpoint.tools, state.object_store(PersistenceDomain.CHECKPOINT),
                namespace=namespace, tenant_id=tenant_id, execution_id=key.turn_id, agent_run_id=key.turn_id,
                binding_digest=AgentBindingContract.from_payload(binding_payload).binding_digest,
                owner=secrets.token_hex(16), background_tasks=set(), fresh_execution=False,
            )
            return await source.prepare_waiting_recovery(
                key, generation=generation, checkpoint=state.checkpoint, tenant_id=tenant_id,
                binding_payload=binding_payload, resolver=resolver, read_payload=backend.read_deferred_payload,
                tools=receipt_reader, ready=ready,
            )

        async def prepare_canonical_step(binding, key, generation, *, settled_split=False):
            from ..capability.steps import AtomicStepPersistence
            from ..persistence import SqlStepStore
            from ._step_store import RuntimeStepStore
            from ._tool import RuntimeToolOperationBridge

            policy = _runtime_feature_policy(binding.compiled_agent.runtime_contributions)
            if not policy.restart_checkpoints:
                return None
            values = await prepare_runtime_capabilities(binding.compiled_agent)
            steps = tuple(leaf for value in values for leaf in runtime_feature_leaves(value)
                          if isinstance(leaf, StepPersistence))
            native = AtomicStepPersistence.adapt(steps[0]) if len(steps) == 1 else None
            if (type(native) is not AtomicStepPersistence or type(native.store) is not SqlStepStore
                    or native.store.store_key != SqlStepStore.scope_key(namespace=namespace, tenant_id=tenant_id)):
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY, "Canonical Step requires native SQL persistence")
            turn_store = state.session_turns
            selected_store = turn_state
            split_source = None
            split_approval_ref = None
            if native.store.storage_context.engine is not selected_store.storage_group.context.engine:
                if split_sql_source is None:
                    raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING,
                                  "Cross-store Step requires provisioned separate Conversation and Execution/Checkpoint databases")

                from ..capability import ExecutionCapability
                from ._canonical_split_step import _CanonicalSplitStep

                compiled = binding.compiled_agent
                candidates = compiled.selected_tools
                business = False
                tool_kind = None
                if len(candidates) == 1 and native.capture_frontier:
                    candidate = candidates[0]
                    contract = candidate.contract
                    metadata = contract.get("metadata")
                    business = (tuple(compiled.spec.allow_tools) == (candidate.id,)
                                and isinstance(metadata, Mapping)
                                and tool_class_from_metadata(metadata) == "business"
                                and not contract.get("defer_loading", False)
                                and candidate.materialize_tool().prepare is None)
                    tool_kind = contract.get("kind", "function")
                approval_profile = business and tool_kind == "unapproved"
                settled_profile = business and settled_split and generation >= 2 and tool_kind in (
                    "function", "unapproved",
                )
                if ((candidates and not (approval_profile or settled_profile))
                        or (settled_split and not settled_profile)
                        or compiled.selected_skills or compiled.selected_mcp
                        or compiled.selected_subagents or compiled.selected_capabilities
                        or binding.output_binding.mode != "text"
                        or policy.durable_trace or policy.memory_scope is not None
                        or any(type(leaf) not in (StepPersistence, StepPersistenceCapability, ExecutionCapability)
                               for item in policy.capabilities for leaf in runtime_feature_leaves(item))):
                    raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING,
                                  "Cross-store Step requires a text agent with no tools or one fixed approval tool")
                if approval_profile:
                    split_approval_ref = CapabilityRef("tool", candidate.id, candidate.revision)
                selected_store = split_sql_source
                if (native.store.storage_context is not selected_store.context
                        or not native.store.backup_supported):
                    raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                split_source = _CanonicalSplitStep(
                    turn_store, selected_store, native.store, read_effects=read_split_effects,
                )
            tools = RuntimeToolOperationBridge(
                state.checkpoint.tools, state.object_store(PersistenceDomain.CHECKPOINT),
                namespace=namespace, tenant_id=tenant_id, execution_id=key.turn_id, agent_run_id=key.turn_id,
                binding_digest=binding.binding_digest, owner=secrets.token_hex(16),
                background_tasks=set(), fresh_execution=generation == 1,
            )
            tools._split_approval_ref = split_approval_ref
            store = RuntimeStepStore.for_canonical_turn(
                native.store, tools, turn_store=turn_store, state_store=selected_store,
                key=key, generation=generation, run_id=key.turn_id, split_source=split_source,
            )
            return replace(native, store=store, run_id=key.turn_id), tools

        def prepare_canonical_plan(key, generation):
            from ..capability.planning import PlanningStore
            from ..persistence.planning import PlanStateStore

            async def guard(transaction):
                await state.session_turns.guard_step_in_transaction(transaction, key, generation=generation)

            return PlanningStore(PlanStateStore(
                turn_state,
                namespace=namespace, tenant_id=tenant_id,
                owner_kind="session", owner_id=key.session_id, _before_edit=guard,
            ))

        budget_repository = state.execution.budgets

        def prepare_canonical_budget(key, generation):
            from ._budget import RunBudgetContext
            scope = "turn:" + key.turn_id

            async def guard(transaction):
                await state.session_turns.guard_step_in_transaction(transaction, key, generation=generation)

            authority = turn_state, guard
            return RunBudgetContext(
                budget_repository, scope, key.turn_id, key.turn_id, _turn_generation=generation,
                _guarded_admit_model=lambda request_id: budget_repository._admit_model(
                    scope, request_id, turn_authority=authority,
                ),
                _guarded_admit_tool=lambda call_id: budget_repository._admit_tool(
                    scope, call_id, turn_authority=authority,
                ),
            )

        async def cancel_native_subagents(key, admitted: Principal, actor: Principal) -> frozenset[str]:
            repository = state.execution.executions
            if type(repository) is not ExecutionRepositoryImpl:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            return await dispatcher.cancel_native_children(repository, key, admitted, actor)

        native_subagent_topology = (
            type(state.checkpoint.tools) is ToolRepositoryImpl
            and type(state.execution.executions) is ExecutionRepositoryImpl
            and state.checkpoint.tools.state_store.storage_group is turn_state.storage_group
            and state.execution.executions.state_store.storage_group is turn_state.storage_group
        )

        @asynccontextmanager
        async def prepare_native_subagents(
            turn, binding: AgentBinding, principal: Principal,
            instructions: Callable[[], RepositoryInstructions | None],
        ):
            turns = state.session_turns
            tools = state.checkpoint.tools
            executions = state.execution.executions
            if (turns is None or turn.native_step is None or turn.split_source is not None
                    or type(tools) is not ToolRepositoryImpl or turn.native_step[1]._repository is not tools
                    or type(executions) is not ExecutionRepositoryImpl
                    or tools.state_store.storage_group is not turn_state.storage_group
                    or executions.state_store.storage_group is not turn_state.storage_group):
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            async def stop_children() -> frozenset[str] | None:
                row = await turns.get_turn(turn.key)
                if row is None or row["generation"] != turn.generation or row["state"] != "cancelling":
                    return None
                # Handoff also cancels the coroutine, but only a committed
                # Stop authorizes durable child cancellation.
                return await dispatcher.cancel_native_children(executions, turn.key, principal, principal)

            async with backend.canonical_subagent_parent(
                turn.key, turn.generation, principal, stop_children=stop_children,
            ):
                yield lambda capability: dispatcher.claimed_native_leaf(
                    turns=turns, key=turn.key, generation=turn.generation, tools=tools, principal=principal,
                    binding=binding.binding_contract, capability=capability, mode=turn.mode,
                    instructions=instructions,
                )

        async def recover_native_subagent(key, generation, principal, binding, mode, tools, original, args, start_reserved):
            reserved = None
            if start_reserved:
                from ._native_subagent import NativeSubagentAdmission
                from ._tool import ToolOperationDecision

                if original.owner is None:
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                reserved = NativeSubagentAdmission(
                    state.session_turns, key, generation, state.checkpoint.tools,
                    ToolOperationDecision(original.tool_operation_id, original.owner, original.fence, False),
                    original.tool_call_id, principal, original=original,
                )
            child, result = await dispatcher.recover_native_result(
                key=key, original=original, args=args, principal=principal,
                binding=binding.binding_contract, mode=mode, timeout_seconds=0.0, reserved=reserved,
            )
            result_payload, error_code, error_payload = await tools.native_delegate_payload(original, child, result)
            return await backend._runtime_commands.settle_native_subagent_result(
                state.session_turns, key, generation=generation, principal=principal, original=original,
                child=child, result_payload=result_payload, error_code=error_code, error_payload=error_payload,
            )

        turn_runner = (
            None if state.session_turns is None or not enable_turn_sessions else _RegenerableTurnRunner(
                state.session_turns, executor, workspace=workspace,
                # Borrow the existing workspace access; its original materializer owns teardown.
                input_materializer=ExecutionInputMaterializer(
                    input_materializer.access,
                    replace(limits, max_binary_input_bytes=min(limits.max_binary_input_bytes, limits.max_turn_attachment_bytes)),
                    object_store=state.object_store(PersistenceDomain.EXECUTION),
                    object_key_factory=object_key_factory,
                ),
                instruction_resolver=instruction_resolver, instruction_resolver_for=instruction_resolver_for,
                prepare_step=prepare_canonical_step, prepare_plan=prepare_canonical_plan,
                abandon_child_repository=(state.execution.executions if native_subagent_topology else None),
                cancel_subagents=cancel_native_subagents,
                begin_child_stop=backend.begin_native_child_stop if native_subagent_topology else None,
                prepare_subagents=prepare_native_subagents if native_subagent_topology else None,
                recover_subagent=recover_native_subagent,
                prepare_budget=(prepare_canonical_budget
                                if type(budget_repository) is BudgetRepositoryImpl
                                and budget_repository.state_store.storage_group is turn_state.storage_group else None),
            )
        )
        lifecycle.turn_runner = turn_runner
        if turn_runner is not None:
            if split_sql_source is not None:
                turn_runner._split_recovery_source = open_split_recovery_source
                turn_runner._prepare_split_wait_resume = prepare_split_wait_resume
            from ..persistence.runs import ModelInteractionStateStepArchive
            from ._canonical_deferred_publication import bind_deferred_publication

            trace_archive = run_store.trace_persistence.archive
            if (isinstance(trace_archive, ModelInteractionStateStepArchive)
                    and state.plan.route(PersistenceDomain.TRACE).retention is StorageRetentionMode.DURABLE
                    and all(member.storage_group is conversation_group for member in trace_archive.transaction_stores)):
                turn_runner._trace_archive = trace_archive

            bind_deferred_publication(turn_runner, state.session_turns, state.checkpoint)

            async def start_attached_graph(
                request, reservation, session_id, session_row, task_admission, task_invocation, dependency_hold_id,
            ):

                task_repository = graph_task_repository(state.task.tasks)
                if task_repository is None:
                    raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
                return await backend._start_attached_graph(
                    request, reservation, session_id=session_id, session_row=session_row, runner=turn_runner,
                    authorization=authorization, task_repository=task_repository,
                    task_admission=task_admission, task_invocation=task_invocation,
                    dependency_hold_id=dependency_hold_id, record_task_invocation=input_captures.record_invocation,
                )

            execution._start_attached_graph = start_attached_graph

            async def cancel_graph_turn(key, principal: Principal) -> bool:
                result = await execution.cancel(
                    key.turn_id, CancelExecutionRequest(principal, secrets.token_hex(16)),
                )
                return result.cancelled

            turn_runner._cancel_graph_turn = cancel_graph_turn
        if state.session_turns is not None:
            async def canonical_graph_session(session_id: str, principal: Principal):
                if not enable_turn_sessions:
                    raise AIError(
                        ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                        safe_details={"reason": "turn_session_authorization_not_configured"},
                    )
                await authorization.authorize(
                    principal, AuthorizationAction.SESSION_READ,
                    ResourceRef(ResourceKind.SESSION, session_id, tenant_id),
                )
                return await state.session_turns.get_session(session_id)

            execution._canonical_graph_session = canonical_graph_session

            async def resolve_graph_cancel_link(record: ExecutionRecord, principal: Principal | None):
                if not record.requires_task_invocation_capture or record.session_id is not None:
                    return None
                linked = await state.session_turns._linked_graph_turn(record.execution_id)
                if linked is None:
                    return None
                key, generation = linked
                if principal is None:
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                await authorization.authorize(
                    principal, AuthorizationAction.SESSION_READ,
                    ResourceRef(ResourceKind.SESSION, key.session_id, tenant_id),
                )
                return state.session_turns, key, generation

            backend._resolve_graph_cancel_link = resolve_graph_cancel_link

            async def has_graph_execution_link(execution_id: str) -> bool:
                return await state.session_turns._linked_graph_turn(execution_id) is not None

            backend._has_graph_execution_link = has_graph_execution_link

            async def is_linked_graph_turn(record: ExecutionRecord, principal: Principal) -> bool:
                return await resolve_graph_cancel_link(record, principal) is not None

            execution._is_linked_graph_turn = is_linked_graph_turn

            async def settle_cold_graph_cancel(record, node_id, operation, principal) -> bool:

                linked = await resolve_graph_cancel_link(record, principal)
                if linked is None:
                    return False
                turns, key, generation = linked
                if record.status is ExecutionStatus.CANCELLED:
                    return True
                if record.status is not ExecutionStatus.CANCELLING or backend.worker_installed(record.execution_id):
                    return False
                turn = await turns.get_turn(key)
                if turn is None or turn["generation"] != generation or turn["state"] != "cancelling":
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                task_repository = graph_task_repository(state.task.tasks)
                if task_repository is None:
                    raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
                finished = await backend._settle_cold_graph_cancel(
                    record, turns=turns, key=key, generation=generation, turn_no=turn["turn_no"],
                    task_repository=task_repository, node_id=node_id, operation=operation,
                )
                if finished.status is not ExecutionStatus.CANCELLED:
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                if turn_runner is not None:
                    turn_runner._gap(key)
                    session_row = await turns.get_session(key.session_id)
                    if session_row is not None:
                        turn_runner._notice_lifecycle(key, "terminal", session_row["revision"])
                return True

            execution._settle_cold_graph_cancel = settle_cold_graph_cancel

            async def recover_undispatched_graph_turn(record, invocation, operation, principal) -> bool:

                linked = await resolve_graph_cancel_link(record, principal)
                if linked is None:
                    return False
                if turn_runner is None:
                    raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
                task_repository = graph_task_repository(state.task.tasks)
                if task_repository is None:
                    raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
                handle = await backend._recover_undispatched_attached_graph(
                    record, runner=turn_runner, authorization=authorization, task_repository=task_repository,
                    task_invocation=invocation, recovery_operation=operation,
                    record_task_invocation=input_captures.record_invocation,
                )
                if handle is None or handle.execution_id != record.execution_id:
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                return True

            execution._recover_undispatched_graph_turn = recover_undispatched_graph_turn
        async def require_session_available() -> None:
            if not (await current_policy()).session.enabled:
                raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING, "Session management is disabled")

        if (repository := state.conversation.sessions) is not None:
            session = RuntimeSessionService(
                state.conversation,
                state.execution.executions,
                authorization,
                execution,
                HmacCursorSigner("session", runtime_token_seed),
                history_reader=StepSessionHistoryReader(
                    store=run_store.read_store(PersistenceDomain.CONVERSATION),
                    cursor_signer=HmacCursorSigner("session-history", runtime_token_seed),
                    sessions=repository,
                ),
                enabled=session_execution_ready,
                current_policy=current_policy,
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
                capture_terminal_publication=state.retention.capture_terminal_publication,
            )
            session.bind_turn_observer(backend.observe_turns)
            lifecycle.session = session
            runtime_bridge.pending_turns.session_changed = repository.notify_session_change

        async def prepare_pending_capture(execution_id: str, principal) -> None:
            await execution.load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_CAPTURE_INPUT)
            await execution.load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
            await backend.promote_pending_execution(execution_id)

        input_captures = InputCaptures.from_persistence(
            state, authorization=authorization, execution=execution,
            materializer=input_materializer, asset_sources=asset_sources,
            prepare_pending_capture=prepare_pending_capture,
        )
        backend.bind_recovery_relaunch_guard(TaskRecoveryGuard(state, input_captures))
        if state.session_turns is not None:
            def graph_wait_proof(record, turns, key, generation, turn, lease):
                from ..persistence.graph_turns import GraphDeferredWait
                from ..persistence import SqlStepStore
                from ._regenerable_turn import _TURN_BYTES

                task_repository, step_store = graph_task_repository(state.task.tasks), run_store.step_store
                if task_repository is None or type(step_store) is not SqlStepStore:
                    raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
                return GraphDeferredWait(
                    turns, key, generation, step_store,
                    agent_conversation_id(namespace=namespace, tenant_id=tenant_id, execution_id=record.execution_id),
                    turn["result_json"]["checkpoint_id"], task_repository, lease, _TURN_BYTES,
                )

            async def queue_deferred_graph(record: ExecutionRecord) -> bool:
                from ..task import TaskStatus

                if not record.requires_task_invocation_capture or record.session_id is not None:
                    return False
                turns = state.session_turns
                link = await turns._linked_graph_turn(record.execution_id)
                if link is None:
                    return False
                key, generation = link
                turn = await turns.get_turn(key)
                if turn is None or turn["generation"] != generation:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                if turn["state"] not in {"waiting", "ready"}:
                    return True
                principal = Principal(record.principal_id, tenant_id, record.principal_kind)
                captured = await input_captures.read_invocation_node(record.execution_id, principal=principal)
                if captured is None:
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                graph_id, node = captured
                if turn["state"] == "ready":
                    if wake_graph is not None:
                        await wake_graph(graph_id, tenant_id)
                    return True
                proof = graph_wait_proof(record, turns, key, generation, turn, None)
                node_states = await proof.task_repository.get_node_states(graph_id, (node.node_id,), tenant_id=tenant_id)
                if len(node_states) != 1 or node_states[0].execution_id != record.execution_id:
                    raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                task = node_states[0]
                if task.status is not TaskStatus.WAITING:
                    return True
                queued = await backend._runtime_commands._mark_canonical_graph_deferred_ready(
                    record, graph=proof, _queued_task=(graph_id, node.node_id, task.fence),
                )
                if queued and wake_graph is not None:
                    await wake_graph(graph_id, tenant_id)
                return True

            backend._queue_deferred_graph = execution._queue_deferred_graph = queue_deferred_graph
            if turn_runner is not None:
                async def resume_bound_graph_turn(record, invocation):
                    if record.status is not ExecutionStatus.WAITING_DEFERRED:
                        return None
                    linked = await resolve_graph_cancel_link(record, invocation.principal)
                    if linked is None:
                        return None
                    turns, key, generation = linked
                    turn = await turns.get_turn(key)
                    if turn is None or turn["generation"] != generation:
                        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                    if turn["state"] == "waiting":
                        return "waiting"
                    if turn["state"] != "ready":
                        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                    if invocation.task_lease is None:
                        raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                    proof = graph_wait_proof(record, turns, key, generation, turn, invocation.task_lease)
                    handle = await backend._resume_ready_attached_graph(
                        record, runner=turn_runner, authorization=authorization, graph=proof,
                        task_invocation=invocation, recovery_operation=None,
                    )
                    if handle is None or handle.execution_id != record.execution_id:
                        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                    return "resumed"

                execution._resume_bound_graph_turn = resume_bound_graph_turn
        session_deferred = None
        if turn_runner is not None:
            from ._canonical_deferred_resume import canonical_deferred_commands

            async def ignore_turn_emission(_event) -> None:
                pass

            async def resume_deferred(frontier, continuation, results) -> None:
                await turn_runner.resume_deferred(
                    frontier, continuation, results, namespace=namespace, tenant_id=tenant_id,
                    app=app, authorization=authorization, resolver=binding_resolver, limits=limits,
                    event_sink=ignore_turn_emission,
                )

            session_deferred = canonical_deferred_commands(
                state.session_turns, state.checkpoint, authorization, namespace=namespace, tenant_id=tenant_id,
                objects=state.object_store(PersistenceDomain.CHECKPOINT), read_payload=backend.read_deferred_payload,
                resume=resume_deferred, turn_store=turn_state,
                split_source=None if split_sql_source is None else open_split_recovery_source,
            )

        approval = DefaultApprovalService(
            state.checkpoint.approvals,
            state.execution.executions,
            authorization,
            objects=state.object_store(PersistenceDomain.CHECKPOINT),
            continuation=backend,
            session_deferred=session_deferred,
        )
        external = DefaultExternalService(
            state.checkpoint.external_calls,
            state.execution.executions,
            authorization,
            objects=state.object_store(PersistenceDomain.CHECKPOINT),
            object_key_factory=object_key_factory,
            continuation=backend,
            session_deferred=session_deferred,
        )
        event = DefaultEventService(
            state.execution.executions,
            state.execution.events,
            authorization,
            backend.worker_failure,
            live_broker,
            pending_execution=backend.pending_execution,
            pending_stream=runtime_bridge.pending_turns.read_stream,
            local_admission=backend.local_admission,
        )
        artifact = DefaultArtifactService(
            state.artifact,
            authorization,
            token_seed=runtime_token_seed,
            cursor_signer=HmacCursorSigner("artifact", runtime_token_seed),
        )
        native_turn_read: _NativeTurnRead | None = None

        def bind_native_turn_read(read: _NativeTurnRead) -> None:
            nonlocal native_turn_read
            native_turn_read = read

        history = _borrowed_runtime_history(
            history_service, tenant_id=tenant_id, state=state, run_store=run_store,
            authorization=authorization, artifact=artifact,
            enable_turn_sessions=enable_turn_sessions,
            bind_native_turn_read=bind_native_turn_read,
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
            session=session, turn_runner=turn_runner,
        )
        handoff_actions = _runtime_close_actions(
            execution=execution, backend=backend, input_materializer=input_materializer,
            metric_buffer=metric_buffer, run_store=run_store, session=session, turn_runner=turn_runner, handoff=True,
        )
        lifecycle.configure(close_actions, handoff_actions)
        await run_store.prepare_archives()
        if auto_recover and PersistenceDomain.EXECUTION in state.plan.durable_domains:
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
        turn_runner=turn_runner,
        require_session_available=require_session_available,
        approval=approval,
        external=external,
        event=event,
        artifact=artifact,
        tenant_id=tenant_id,
        close_callback=lifecycle.close,
        handoff_callback=(lifecycle.handoff if all(state.plan.route(domain).retention is StorageRetentionMode.DURABLE
                    for domain in (PersistenceDomain.EXECUTION, PersistenceDomain.CHECKPOINT)) else None),
        tree_streamer=tree_streamer,
        metric_control=metric_buffer,
        binding_resolver=binding_resolver,
        history=history,
        native_turn_read=native_turn_read,
        input_captures=input_captures,
        memory_scope=memory_scope,
        asset_sources=dict(asset_sources),
    )


def _borrowed_runtime_history(
    service: DefaultExecutionHistoryService,
    *,
    tenant_id: str,
    state: Persistence,
    run_store: RuntimeAgentRunStore,
    authorization: object,
    artifact: DefaultArtifactService,
    enable_turn_sessions: bool,
    bind_native_turn_read: Callable[[_NativeTurnRead], None],
    pending_session_execution: SessionPendingExecutionLookup[ExecutionRecord] | None = None,
    pending_execution: Callable[..., Awaitable[ExecutionRecord | None]] | None = None,
) -> "History":
    conversation = state.conversation if state.session_turns is None else None
    trace_archive = run_store.trace_persistence.archive if enable_turn_sessions else None
    return History(
        service,
        tenant_id=tenant_id,
        executions=state.execution.executions,
        events=state.execution.events,
        sessions=None if conversation is None else conversation.sessions,
        session_turn_reader=state.session_turn_reader if enable_turn_sessions else None,
        session_trace_source=trace_archive if isinstance(trace_archive, StateStepArchive) else None,
        _bind_runtime_turn_read=bind_native_turn_read,
        conversation=conversation,
        session_transcript_store=None if conversation is None else run_store.conversation_persistence,
        pending_session_execution=None if conversation is None else pending_session_execution,
        pending_execution=pending_execution,
        cache=(state.plan.route(PersistenceDomain.CONVERSATION).cache if conversation is not None and all(
            state.plan.route(domain).retention is StorageRetentionMode.DURABLE
            for domain in (PersistenceDomain.CONVERSATION, PersistenceDomain.EXECUTION)
        ) else None),
        _task_source=lambda: state.task.tasks,
        authorization=authorization,
        namespace=state.namespace,
        execution_objects=state.object_store(PersistenceDomain.EXECUTION),
        object_reader=state.read_object,
        artifacts=artifact,
        cursor_signer=HmacCursorSigner(
            "runtime-history",
            token_seed(state.namespace),
        ),
    )




__all__ = []
