#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime composition and local service graph construction."""

import asyncio
import hashlib
import os
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, cast

from linktools.core import environ
from pydantic_ai_harness.memory import SearchableMemoryStore

from ..agent import AgentCatalog, AgentCompiler, AgentDefinition
from ..asset import AssetKey, AssetPathAdapter, AssetStore, DirectoryAssetBackend, PrefixAssetPathAdapter
from ..capability import (
    CapabilityContribution,
    CapabilityGroup,
    LocalSkillResourceSource,
    SkillSourceRegistry,
    workspace_tool_contributions,
)
from ..core import DEFAULT_DISCOVERY_POLICY, HmacCursorSigner, TenantAuthorizationPolicy, validate_tenant_id
from ..errors import AIError, ErrorCode
from ..model import ModelRegistry
from ..observe import MiddlewarePipeline
if TYPE_CHECKING:
    from ..observe import Middleware
from ..spec import AgentSpec, AgentSpecCodec
from ..storage import ObjectStore, PayloadPolicy, StorageOverlay
from ..task import LocalTaskGraphLauncher
from ..workspace import (
    LocalRepositoryInstructionResolver,
    LocalRuleCatalog,
    Workspace,
)
from ._agent_executor import AgentExecutor
from ._approval import DefaultApprovalService
from ._artifact import DefaultArtifactService
from ._coordinator import _LocalRuntimeCoordinator
from ._evaluation import DefaultEvaluationService
from ._event import DefaultEventService, LiveExecutionEventBroker
from ._execution import DefaultExecutionService
from ._history import StepExecutionHistoryReader, StepSessionHistoryReader
from ._memory import RuntimeMemoryStore
from ._object import RuntimeObjectKeyFactory
from ._local import LocalExecutionBackend
from ._planner import DefaultTaskService, RuntimeTaskNodeRunner
from ._session import DefaultSessionService
from ._subagent import SubagentDispatcher
from .service_api import ExecutionHistoryReader, SessionHistoryReader
from .state import (
    ExecutionReadModelRepository,
    RecoveryCheckpointState,
    RuntimeDomain,
    RuntimeRetentionMode,
    RuntimeState,
    RuntimeStatePlan,
    RuntimeStateRoute,
    StateStepArchive,
)

AppT = TypeVar("AppT")
_logger = environ.get_logger("ai.runtime.factory")


@dataclass(frozen=True, slots=True)
class _RuntimeComponents:
    catalog: AgentCatalog
    compiler: AgentCompiler
    execution: DefaultExecutionService
    session: DefaultSessionService
    task: DefaultTaskService
    evaluation: DefaultEvaluationService
    approval: DefaultApprovalService
    event: DefaultEventService
    artifact: DefaultArtifactService
    tenant_id: str
    close_callback: Callable[[], Awaitable[None]]
    local_coordinator: _LocalRuntimeCoordinator


async def compose_runtime_components(
    workspace: Workspace,
    *,
    app: "AppT | None" = None,
    tenant_id: "str | None" = None,
    models: "ModelRegistry | None" = None,
    state: "RuntimeState | None" = None,
    capabilities: "Sequence[CapabilityGroup[AppT]]" = (),
    middleware: "Sequence[Middleware]" = (),
) -> _RuntimeComponents:
    """Freeze declarations and build Runtime-private services without constructing Runtime."""
    if not isinstance(workspace, Workspace):
        raise TypeError("workspace must be Workspace")
    workspace.policy.validate()
    if isinstance(middleware, (str, bytes, bytearray)) or not isinstance(
        middleware, Sequence
    ):
        raise TypeError("middleware must be a sequence")
    middleware_values = tuple(middleware)
    for item in middleware_values:
        try:
            mutating = item.mutating
        except AttributeError as error:
            raise TypeError("middleware must define a bool mutating attribute") from error
        if not isinstance(mutating, bool):
            raise TypeError("middleware mutating attribute must be bool")
        if mutating:
            raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
    groups = tuple(capabilities)
    if any(not isinstance(group, CapabilityGroup) for group in groups):
        raise TypeError("capabilities must contain CapabilityGroup values")
    workspace_groups = tuple(group for group in groups if group.id == "workspace")
    if len(workspace_groups) > 1:
        raise AIError(ErrorCode.CAPABILITY_CONFLICT)

    owned_workspace_assets: tuple[AssetStore, DirectoryAssetBackend] | None = None
    selected_state: RuntimeState | None = None
    initialized = False
    try:
        if workspace_groups:
            effective_groups: tuple[CapabilityGroup[object], ...] = cast(
                "tuple[CapabilityGroup[object], ...]", groups
            )
        else:
            owned_workspace_assets = _default_workspace_store(workspace)
            owned_store, _owned_backend = owned_workspace_assets
            await owned_store.initialize()
            workspace_group: CapabilityGroup[object] = CapabilityGroup.from_store(
                "workspace",
                owned_store,
                skill_source=LocalSkillResourceSource(
                    "workspace",
                    workspace.storage_root / "skills",
                ),
            )
            effective_groups = (workspace_group, *cast("tuple[CapabilityGroup[object], ...]", groups))

        group_ids = tuple(group.id for group in effective_groups)
        if len(group_ids) != len(set(group_ids)):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)

        frozen: list[CapabilityContribution[object]] = list(workspace_tool_contributions(workspace))
        for group in effective_groups:
            frozen.extend(await group.freeze())
        _validate_candidate_uniqueness(frozen)
        skill_sources = SkillSourceRegistry(
            tuple(
                source
                for group in effective_groups
                if (source := group.skill_source) is not None
            )
        )

        agents = {
            candidate.id: cast(AgentSpec, candidate.value)
            for candidate in frozen
            if candidate.kind == "agent"
        }
        if "default" not in agents:
            agents["default"] = AgentSpec("default")
        model_registry = models or _build_default_models(workspace)
        resolver = model_registry.snapshot()
        compiler = AgentCompiler(
            model_resolver=resolver,
            candidates=tuple(candidate for candidate in frozen if candidate.kind != "agent"),
            agents=agents,
        )
        catalog = AgentCatalog({agent_id: compiler.compile(agents[agent_id]) for agent_id in sorted(agents)})

        effective_tenant_id = "default" if tenant_id is None else validate_tenant_id(tenant_id)
        selected_state = state or _default_runtime_state(workspace)
        await selected_state.initialize(namespace=workspace.workspace_id, tenant_id=effective_tenant_id)
        initialized = True
        if workspace.policy.tool_permissions.requires_approval:
            if (
                selected_state.plan.route(RuntimeDomain.EXECUTION).retention
                is not RuntimeRetentionMode.DURABLE
                or selected_state.plan.route(RuntimeDomain.RECOVERY).retention
                is not RuntimeRetentionMode.DURABLE
            ):
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            recovery_steps = selected_state.steps.read_store(RuntimeDomain.RECOVERY)
            if not isinstance(recovery_steps, StateStepArchive):
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            approval_group = (
                selected_state.execution.executions.state_store.storage_group
            )
            if (
                selected_state.recovery.checkpoints.state_store.storage_group
                is not approval_group
                or selected_state.recovery.approvals.state_store.storage_group
                is not approval_group
                or recovery_steps.state_store.storage_group is not approval_group
            ):
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        rules = await LocalRuleCatalog.load(workspace.root, workspace.policy)
        instruction_resolver = LocalRepositoryInstructionResolver(
            workspace.root, workspace.policy, rules
        )
        object_key_factory = RuntimeObjectKeyFactory(workspace.workspace_id)
        payload_policy = PayloadPolicy()
        session_execution_ready = (
            not workspace.policy.tool_permissions.requires_approval
            or selected_state.plan.route(RuntimeDomain.CONVERSATION).retention
            is RuntimeRetentionMode.DURABLE
        )
        owned_workspace_close = (
            None
            if owned_workspace_assets is None
            else partial(_close_owned_workspace_assets, *owned_workspace_assets)
        )
        return await _build_local_components(
            state=selected_state,
            catalog=catalog,
            compiler=compiler,
            authorization=TenantAuthorizationPolicy(effective_tenant_id),
            tenant_id=effective_tenant_id,
            namespace=workspace.workspace_id,
            workspace=workspace,
            app=app,
            history_reader=_execution_history_reader(workspace, selected_state, effective_tenant_id),
            session_history_reader=StepSessionHistoryReader(
                store=selected_state.steps.read_store(RuntimeDomain.CONVERSATION),
                cursor_signer=HmacCursorSigner("session-history", _grant_key(workspace)),
            ),
            memory_store_factory=_memory_store_factory(workspace, selected_state),
            skill_sources=skill_sources,
            grant_key=_grant_key(workspace),
            instruction_resolver=instruction_resolver,
            object_key_factory=object_key_factory,
            payload_policy=payload_policy,
            session_execution_ready=session_execution_ready,
            middleware=middleware_values,
            owned_workspace_close=owned_workspace_close,
        )
    except BaseException:
        try:
            if initialized and selected_state is not None:
                await selected_state.close()
        finally:
            if owned_workspace_assets is not None:
                await _close_owned_workspace_assets(*owned_workspace_assets)
        raise


class _WorkspaceDeclarationPathAdapter:
    def __init__(self) -> None:
        self._delegate = PrefixAssetPathAdapter(
            {
                "agent": "agents",
                "skill": "skills",
                "mcp": "mcp",
            }
        )

    def validate(self, kinds: Sequence[str]) -> None:
        self._delegate.validate(kinds)

    def root_path(self, kind: str) -> str:
        return self._delegate.root_path(kind)

    def to_path(self, key: AssetKey) -> str:
        return self._delegate.to_path(key)

    def from_path(self, path: str) -> "AssetKey | None":
        key = self._delegate.from_path(path)
        if key is None or key.kind != "skill":
            return key
        if "/" not in key.id or key.id.endswith("/SKILL.md"):
            return key
        return None


def _default_workspace_store(
    workspace: Workspace,
) -> tuple[AssetStore, DirectoryAssetBackend]:
    adapter: AssetPathAdapter = _WorkspaceDeclarationPathAdapter()
    source = DirectoryAssetBackend(
        str(workspace.storage_root),
        path_adapter=adapter,
        kinds=("agent", "skill", "mcp"),
        follow_external_symlinks=True,
        ignore_paths=DEFAULT_DISCOVERY_POLICY.ignores,
    )
    return AssetStore(StorageOverlay(source)), source


async def _close_owned_workspace_assets(
    store: AssetStore,
    backend: DirectoryAssetBackend,
) -> None:
    await store.close()
    await backend.close()


def _validate_candidate_uniqueness(
    candidates: Sequence[CapabilityContribution[object]],
) -> None:
    identities = tuple((candidate.kind, candidate.id) for candidate in candidates)
    if len(identities) != len(set(identities)):
        raise AIError(ErrorCode.CAPABILITY_CONFLICT)


def _build_default_models(workspace: Workspace) -> ModelRegistry:
    configured = workspace.config.get("model")
    model = (
        configured.strip()
        if isinstance(configured, str) and configured.strip()
        else os.getenv("OPENAI_MODEL", "").strip()
    )
    if not model:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "model is required")
    return ModelRegistry.openai(
        model=model,
        base_url=os.getenv("OPENAI_BASE_URL", "").strip() or None,
        api_key=os.getenv("OPENAI_API_KEY", "").strip() or None,
    )


def _default_runtime_state(workspace: Workspace) -> RuntimeState:
    runtime_root = workspace.storage_root / "runtime"
    return RuntimeState.from_plan(
        RuntimeStatePlan(
            conversation=RuntimeStateRoute.filesystem(
                runtime_root / "conversation",
                transaction_root=runtime_root,
            ),
            execution=RuntimeStateRoute.filesystem(
                runtime_root / "execution",
                transaction_root=runtime_root,
            ),
            recovery=RuntimeStateRoute.filesystem(
                runtime_root / "recovery",
                transaction_root=runtime_root,
            ),
            task=RuntimeStateRoute.filesystem(
                runtime_root / "task",
                transaction_root=runtime_root,
            ),
        )
    )


def _execution_history_reader(
    workspace: Workspace,
    state: RuntimeState,
    tenant_id: str,
) -> StepExecutionHistoryReader:
    return StepExecutionHistoryReader(
        namespace=workspace.workspace_id,
        executions=state.execution.executions,
        store=state.steps.read_store(RuntimeDomain.EXECUTION),
        cursor_signer=HmacCursorSigner("execution-history", _grant_key(workspace)),
        read_model=ExecutionReadModelRepository(
            state.execution.executions.state_store,
            namespace=workspace.workspace_id,
            tenant_id=tenant_id,
        ),
    )


def _memory_store_factory(
    workspace: Workspace,
    state: RuntimeState,
) -> "Callable[[str, str, str, ObjectStore, bool], SearchableMemoryStore]":
    def build(
        tenant_id: str,
        execution_id: str,
        memory_scope: str,
        object_store: ObjectStore,
        transient: bool,
    ) -> SearchableMemoryStore:
        return RuntimeMemoryStore(
            state.memory,
            object_store=object_store,
            namespace=workspace.workspace_id,
            tenant_id=tenant_id,
            execution_id=execution_id,
            memory_scope=memory_scope,
            transient=transient,
        )

    return build


def _grant_key(workspace: Workspace) -> bytes:
    return hashlib.sha256(f"workspace:{workspace.workspace_id}".encode()).digest()


def _require_state_identity(
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
    catalog: AgentCatalog,
    compiler: AgentCompiler,
    authorization: TenantAuthorizationPolicy,
    tenant_id: str,
    namespace: str,
    workspace: Workspace,
    app: AppT,
    history_reader: ExecutionHistoryReader,
    session_history_reader: SessionHistoryReader,
    memory_store_factory: "Callable[[str, str, str, ObjectStore, bool], SearchableMemoryStore] | None",
    skill_sources: SkillSourceRegistry,
    grant_key: bytes,
    instruction_resolver: LocalRepositoryInstructionResolver,
    object_key_factory: RuntimeObjectKeyFactory,
    payload_policy: PayloadPolicy,
    session_execution_ready: bool,
    middleware: "Sequence[Middleware]",
    owned_workspace_close: "Callable[[], Awaitable[None]] | None" = None,
) -> _RuntimeComponents:
    if not state.ready:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    _require_state_identity(state, namespace=namespace, tenant_id=tenant_id)
    execution = DefaultExecutionService(
        state.execution,
        state.object_store(RuntimeDomain.EXECUTION),
        authorization,
        sessions=state.conversation.sessions,
        catalog=catalog,
        compiler=compiler,
        history_reader=history_reader,
        release_terminal=state.retention.release_execution_handoff,
        instruction_resolver=instruction_resolver,
        object_key_factory=object_key_factory,
        payload_policy=payload_policy,
        session_execution_ready=session_execution_ready,
    )
    dispatcher = SubagentDispatcher(catalog, compiler, execution)
    middleware_pipeline = MiddlewarePipeline(middleware)
    executor = AgentExecutor(
        skill_sources,
        instruction_resolver=instruction_resolver,
        middleware=middleware_pipeline,
    )

    def build_memory_store(
        memory_tenant: str,
        execution_id: str,
        memory_scope: str,
    ) -> SearchableMemoryStore:
        if memory_store_factory is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        route = state.plan.route(RuntimeDomain.MEMORY)
        transient = route.retention is RuntimeRetentionMode.TRANSIENT
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

    backend: LocalExecutionBackend | None = None
    task_launcher: LocalTaskGraphLauncher | None = None
    live_broker = LiveExecutionEventBroker()
    try:
        backend = LocalExecutionBackend(
            state.conversation,
            state.execution,
            state.recovery,
            state.object_store(RuntimeDomain.EXECUTION),
            state.object_store(RuntimeDomain.RECOVERY),
            state.metrics,
            namespace,
            state.steps,
            executor,
            catalog,
            tenant_id=tenant_id,
            workspace=workspace,
            app=app,
            step_reads={
                domain: state.steps.read_store(domain)
                for domain in (
                    RuntimeDomain.CONVERSATION,
                    RuntimeDomain.EXECUTION,
                    RuntimeDomain.RECOVERY,
                )
            },
            step_lifecycle=state.steps,
            memory_store_factory=build_memory_store,
            recovery_enabled=RuntimeDomain.RECOVERY in state.plan.durable_domains,
            conversation_durable=(
                state.plan.route(RuntimeDomain.CONVERSATION).retention
                is RuntimeRetentionMode.DURABLE
            ),
            handoff_contract_digest=state.handoff_contract_digest,
            subagent_dispatcher=dispatcher,
            live_broker=live_broker,
            payload_policy=payload_policy,
            execution_objects_durable=(
                state.plan.route(RuntimeDomain.EXECUTION).retention
                is RuntimeRetentionMode.DURABLE
            ),
            tool_operations=state.recovery.tools,
        )
        state.retention.bind_execution_runtime_release(backend.release_runtime_execution)
        execution.bind_backend(backend)
        execution.bind_local_waiter(backend)
        execution.bind_terminal_committer(backend)
        execution.bind_terminal_verifier(backend.verify_terminal_projection)
        execution.bind_subagent_cancellation(dispatcher)
        session = DefaultSessionService(
            state.conversation,
            state.execution.executions,
            authorization,
            execution,
            HmacCursorSigner("session", grant_key),
            history_reader=session_history_reader,
            transcript_store=state.steps.read_store(RuntimeDomain.CONVERSATION),
            release_terminal=state.retention.release_session,
        )
        task_runner = RuntimeTaskNodeRunner(execution, catalog, compiler)
        task_launcher = LocalTaskGraphLauncher(
            state.task.tasks,
            task_runner,
            owner=f"runtime:{tenant_id}:{uuid.uuid4().hex}",
        )
        task = DefaultTaskService(
            state.task,
            authorization,
            task_launcher,
            release_terminal=state.retention.release_task_graph,
            local_waiter=task_launcher,
        )
        evaluation = DefaultEvaluationService(
            state.evaluation,
            state.execution.executions,
            authorization,
            execution,
            release_terminal=state.retention.release_evaluation,
            acquire_execution_hold=execution.acquire_dependency_hold,
            release_execution_hold=execution.release_dependency_hold,
            request_execution_handoff=execution.request_terminal_handoff,
        )
        approval = DefaultApprovalService(
            state.recovery.approvals,
            state.execution.executions,
            authorization,
            context_reader=backend,
            continuation=backend,
        )
        event = DefaultEventService(
            state.execution.executions,
            state.execution.events,
            authorization,
            backend.worker_failure,
            live_broker,
        )
        artifact = DefaultArtifactService(
            state.artifact,
            authorization,
            grant_key=grant_key,
            cursor_signer=HmacCursorSigner("artifact", grant_key),
        )
        execution.bind_local_stream(
            live_broker.prepare_local_producer,
            live_broker.abandon_prepared_local_producer,
        )
        local_coordinator = _LocalRuntimeCoordinator(execution, event)
        close_actions: list[Callable[[], Awaitable[None]]] = [
            task.drain_owned_finalizers,
            task.preflight_close,
            task_launcher.shutdown,
            execution.preflight_close,
            backend.close,
            state.close,
        ]
        if owned_workspace_close is not None:
            close_actions.append(owned_workspace_close)
        coordinator = _RuntimeCloseCoordinator(tuple(close_actions))
        await _restore_recovery_bindings(catalog, compiler, state, tenant_id=tenant_id)
        if RuntimeDomain.RECOVERY in state.plan.durable_domains:
            await backend.reconcile()
        await task.recover_pending()
    except BaseException:
        if task_launcher is not None:
            await task_launcher.shutdown()
        if backend is not None:
            await backend.close()
        raise
    return _RuntimeComponents(
        catalog=catalog,
        compiler=compiler,
        execution=execution,
        session=session,
        task=task,
        evaluation=evaluation,
        approval=approval,
        event=event,
        artifact=artifact,
        tenant_id=tenant_id,
        close_callback=coordinator.close,
        local_coordinator=local_coordinator,
    )


async def _restore_recovery_bindings(
    catalog: AgentCatalog,
    compiler: AgentCompiler,
    state: RuntimeState,
    *,
    tenant_id: str,
) -> None:
    cursor: str | None = None
    while True:
        page = await state.recovery.checkpoints.list_recoverable_page(
            tenant_id=tenant_id,
            cursor=cursor,
            limit=128,
        )
        for checkpoint in page.items:
            if checkpoint.state is RecoveryCheckpointState.COMPLETED:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            recovery_input = checkpoint.input
            execution = await state.execution.executions.get(
                checkpoint.execution_id,
                tenant_id=tenant_id,
            )
            if execution is not None and (
                execution.binding_digest != recovery_input.binding_digest
                or execution.mode != recovery_input.mode
                or execution.planning is not recovery_input.planning
                or execution.thinking != recovery_input.thinking
                or execution.binding != recovery_input.binding
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                binding = compiler.restore(recovery_input.binding)
            except AIError as error:
                if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                    raise
                if error.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE:
                    _logger.warning(
                        "recovery binding unavailable: execution=%s",
                        checkpoint.execution_id,
                    )
                    continue
                raise
            if binding.digest != recovery_input.binding_digest:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            catalog.register_definition(binding.definition)
            catalog.register_binding(binding)
        if page.next_cursor is None:
            return
        cursor = page.next_cursor


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
