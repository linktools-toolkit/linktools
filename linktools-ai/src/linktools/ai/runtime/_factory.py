#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime composition and local service graph construction."""

import asyncio
import hashlib
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

from linktools.core import environ

from ..agent import AgentCatalog, AgentCompiler
from ..capability import (
    CapabilityContribution,
    CapabilityGroup,
    SkillSourceRegistry,
    TaskExpander,
    WorkspaceAccess,
    workspace_tool_contributions,
)
from ..core import (
    HmacCursorSigner,
    PromptLimits,
    TenantAuthorizationPolicy,
    validate_persistence_namespace,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode
from ..model import ModelRegistry
from ..observe import Metrics
from ..spec import AgentSpec
from ..storage import ObjectStore, PayloadPolicy
from ..task import DefaultTaskGraphService, LocalTaskGraphLauncher, TaskNodeHandler
from ..workspace import (
    LocalRepositoryInstructionResolver,
    LocalRuleCatalog,
    RepositoryInstructionResolver,
    Workspace,
)
from ._agent_executor import AgentExecutor
from ._approval import DefaultApprovalService
from ._artifact import DefaultArtifactService
from ._coordinator import _LocalRuntimeCoordinator
from ._evaluation import DefaultEvaluationService
from ._event import DefaultEventService, LiveExecutionEventBroker
from ._external import DefaultExternalService
from ._execution import DefaultExecutionService, _ExecutionRuntimeBridge
from ._execution_tree import ExecutionTreeBroker, ExecutionTreeStreamer
from ._history import StepExecutionHistoryReader, StepSessionHistoryReader
from ._history_service import DefaultExecutionHistoryService
from ._input import ExecutionInputMaterializer
from ._local import LocalExecutionBackend
from ._memory import MemoryStore, RuntimeMemoryStore
from ._metrics import _RuntimeMetricBuffer
from ._object import RuntimeObjectKeyFactory
from ._planner import RuntimeTaskNodeRunner
from ._session import DefaultSessionService
from ._subagent import SubagentDispatcher
from .service_api import ExecutionHistoryReader, SessionHistoryReader
from .state import RuntimeDomain, RuntimeRetentionMode, RuntimeState
from .state._contracts import RecoveryCheckpointState

AppT = TypeVar("AppT")
_logger = environ.get_logger("ai.runtime.factory")


@dataclass(frozen=True, slots=True)
class _RuntimeComponents:
    catalog: AgentCatalog
    compiler: AgentCompiler
    execution: DefaultExecutionService
    session: DefaultSessionService
    graph: DefaultTaskGraphService
    evaluation: DefaultEvaluationService
    approval: DefaultApprovalService
    external: DefaultExternalService
    event: DefaultEventService
    artifact: DefaultArtifactService
    tenant_id: str
    close_callback: Callable[[], Awaitable[None]]
    task_node_runtime: RuntimeTaskNodeRunner[object]
    tree_streamer: ExecutionTreeStreamer
    metric_control: _RuntimeMetricBuffer | None


async def compose_runtime_components(
    namespace: str,
    *,
    app: "AppT | None" = None,
    tenant_id: "str | None" = None,
    models: ModelRegistry,
    state: RuntimeState,
    capabilities: "Sequence[CapabilityGroup[AppT]]" = (),
    metrics: "Metrics | None" = None,
    limits: "PromptLimits | None" = None,
) -> _RuntimeComponents:
    """Freeze declarations and build Runtime-private services."""
    resolved_namespace = validate_persistence_namespace(namespace)
    if not isinstance(models, ModelRegistry):
        raise TypeError("models must be ModelRegistry")
    if not isinstance(state, RuntimeState):
        raise TypeError("state must be RuntimeState")
    if metrics is not None and not isinstance(metrics, Metrics):
        raise TypeError("metrics must be Metrics")
    selected_limits = PromptLimits() if limits is None else limits
    if not isinstance(selected_limits, PromptLimits):
        raise TypeError("limits must be PromptLimits")
    groups = tuple(capabilities)
    if any(not isinstance(group, CapabilityGroup) for group in groups):
        raise TypeError("capabilities must contain CapabilityGroup values")
    group_ids = tuple(group.id for group in groups)
    if len(group_ids) != len(set(group_ids)):
        raise AIError(ErrorCode.CAPABILITY_CONFLICT)
    workspace_groups = tuple(group for group in groups if group.workspace is not None)
    if len(workspace_groups) > 1:
        raise AIError(ErrorCode.CAPABILITY_CONFLICT)
    workspace = None if not workspace_groups else workspace_groups[0].workspace
    if workspace is not None:
        workspace.policy.validate()

    selected_state: RuntimeState | None = None
    workspace_access: WorkspaceAccess | None = None
    input_materializer: ExecutionInputMaterializer | None = None
    initialized = False
    ownership_transferred = False
    try:
        frozen: list[CapabilityContribution[object]] = []
        if workspace is not None:
            frozen.extend(workspace_tool_contributions(workspace))
        for group in groups:
            frozen.extend(await group.freeze())
        _validate_candidate_uniqueness(frozen)
        skill_sources = SkillSourceRegistry(
            tuple(
                source
                for group in groups
                if (source := group.skill_source) is not None
            )
        )
        task_handlers = tuple(
            cast("TaskNodeHandler[object]", candidate.value)
            for candidate in frozen
            if candidate.kind == "task"
        )
        task_expanders = tuple(
            cast(TaskExpander, candidate.value)
            for candidate in frozen
            if candidate.kind == "task_expander"
        )

        agents = {
            candidate.id: cast(AgentSpec, candidate.value)
            for candidate in frozen
            if candidate.kind == "agent"
        }
        if "default" not in agents:
            agents["default"] = AgentSpec("default")
        resolver = models.snapshot()
        workspace_ref = (
            None
            if workspace is not None
            and workspace.workspace_id == resolved_namespace
            else {
                "id": None if workspace is None else workspace.workspace_id
            }
        )
        compiler = AgentCompiler(
            model_resolver=resolver,
            candidates=tuple(
                candidate
                for candidate in frozen
                if candidate.kind not in {"agent", "task", "task_expander"}
            ),
            agents=agents,
            namespace=resolved_namespace,
            workspace_ref=workspace_ref,
        )
        definitions = {
            agent_id: compiler.compile(agents[agent_id])
            for agent_id in sorted(agents)
        }
        catalog = AgentCatalog(definitions)
        has_mcp = any(definition.mcp_servers for definition in definitions.values())

        effective_tenant_id = (
            "default" if tenant_id is None else validate_tenant_id(tenant_id)
        )
        selected_state = state
        await selected_state.initialize(
            namespace=resolved_namespace,
            tenant_id=effective_tenant_id,
        )
        initialized = True
        if workspace is None:
            instruction_resolver: RepositoryInstructionResolver | None = None
            workspace_access = None
            mcp_cwd = str(Path.cwd().resolve()) if has_mcp else ""
        else:
            rules = await LocalRuleCatalog.load(workspace.root, workspace.policy)
            instruction_resolver = LocalRepositoryInstructionResolver(
                workspace.root,
                workspace.policy,
                rules,
            )
            workspace_access = WorkspaceAccess.for_workspace(workspace)
            mcp_cwd = str(workspace.root)
        object_key_factory = RuntimeObjectKeyFactory(resolved_namespace)
        payload_policy = PayloadPolicy()
        input_materializer = ExecutionInputMaterializer(
            workspace_access,
            selected_limits,
            object_store=selected_state.object_store(RuntimeDomain.EXECUTION),
            object_key_factory=object_key_factory,
            payload_policy=payload_policy,
        )
        grant_key = _grant_key(resolved_namespace)
        history_reader = _execution_history_reader(
            resolved_namespace,
            selected_state,
            grant_key,
        )
        session_history_reader = StepSessionHistoryReader(
            store=selected_state.steps.read_store(RuntimeDomain.CONVERSATION),
            cursor_signer=HmacCursorSigner("session-history", grant_key),
        )
        memory_store_factory = _memory_store_factory(
            resolved_namespace,
            selected_state,
        )
        authorization = TenantAuthorizationPolicy(effective_tenant_id)
        ownership_transferred = True
        return await _build_local_components(
            state=selected_state,
            catalog=catalog,
            compiler=compiler,
            authorization=authorization,
            tenant_id=effective_tenant_id,
            namespace=resolved_namespace,
            workspace=workspace,
            limits=selected_limits,
            mcp_cwd=mcp_cwd,
            app=app,
            task_handlers=task_handlers,
            task_expanders=task_expanders,
            history_reader=history_reader,
            session_history_reader=session_history_reader,
            memory_store_factory=memory_store_factory,
            skill_sources=skill_sources,
            grant_key=grant_key,
            instruction_resolver=instruction_resolver,
            object_key_factory=object_key_factory,
            payload_policy=payload_policy,
            input_materializer=input_materializer,
            session_execution_ready=True,
            metrics=metrics,
        )
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
    metric_buffer: _RuntimeMetricBuffer | None,
    state: RuntimeState,
) -> tuple[tuple[str, Callable[[], Awaitable[None]]], ...]:
    actions: list[tuple[str, Callable[[], Awaitable[None]]]] = []
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
    grant_key: bytes,
) -> StepExecutionHistoryReader:
    return StepExecutionHistoryReader(
        namespace=namespace,
        executions=state.execution.executions,
        store=state.steps.read_store(RuntimeDomain.EXECUTION),
        cursor_signer=HmacCursorSigner("execution-history", grant_key),
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


def _grant_key(namespace: str) -> bytes:
    return hashlib.sha256(f"workspace:{namespace}".encode()).digest()


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
    workspace: "Workspace | None",
    limits: PromptLimits,
    mcp_cwd: str,
    app: AppT,
    task_handlers: Sequence[TaskNodeHandler[AppT]],
    task_expanders: Sequence[TaskExpander],
    history_reader: ExecutionHistoryReader,
    session_history_reader: SessionHistoryReader,
    memory_store_factory: "Callable[[str, str, str, ObjectStore, bool], MemoryStore] | None",
    skill_sources: SkillSourceRegistry,
    grant_key: bytes,
    instruction_resolver: "RepositoryInstructionResolver | None",
    object_key_factory: RuntimeObjectKeyFactory,
    payload_policy: PayloadPolicy,
    input_materializer: ExecutionInputMaterializer,
    session_execution_ready: bool,
    metrics: "Metrics | None",
) -> _RuntimeComponents:
    metric_buffer: _RuntimeMetricBuffer | None = None
    metric_source_namespace: str | None = None
    backend: LocalExecutionBackend | None = None

    async def release_execution_handoff(
        execution_id: str,
        *,
        tenant_id: str,
    ) -> None:
        if backend is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        await backend.release_runtime_execution(
            execution_id,
            tenant_id=tenant_id,
        )
        await state.retention.release_execution_handoff(
            execution_id,
            tenant_id=tenant_id,
        )

    try:
        if not state.ready:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        _require_state_identity(state, namespace=namespace, tenant_id=tenant_id)
        metric_buffer = None if metrics is None else _RuntimeMetricBuffer(metrics)
        metric_source_namespace = None if metric_buffer is None else namespace
        runtime_bridge = _ExecutionRuntimeBridge()
        live_broker = LiveExecutionEventBroker()
        history_service = DefaultExecutionHistoryService(
            state.execution.executions,
            authorization,
            history_reader,
            HmacCursorSigner("execution", grant_key),
        )
        execution = DefaultExecutionService(
            state.execution,
            state.object_store(RuntimeDomain.EXECUTION),
            authorization,
            sessions=state.conversation.sessions,
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
            metrics=metric_buffer,
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

    task_launcher: LocalTaskGraphLauncher | None = None
    graph_service: DefaultTaskGraphService | None = None
    coordinator: _RuntimeCloseCoordinator | None = None
    close_actions: tuple[tuple[str, Callable[[], Awaitable[None]]], ...] | None = None
    try:
        backend = LocalExecutionBackend(
            state.conversation,
            state.execution,
            state.recovery,
            state.object_store(RuntimeDomain.EXECUTION),
            state.object_store(RuntimeDomain.RECOVERY),
            namespace,
            state.steps,
            executor,
            catalog,
            tenant_id=tenant_id,
            workspace=workspace,
            limits=limits,
            mcp_cwd=mcp_cwd,
            instruction_resolver=instruction_resolver,
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
            conversation_durable=(
                state.plan.route(RuntimeDomain.CONVERSATION).retention
                is RuntimeRetentionMode.DURABLE
            ),
            input_materializer=input_materializer,
            subagent_dispatcher=dispatcher,
            live_broker=live_broker,
            payload_policy=payload_policy,
            execution_objects_durable=(
                state.plan.route(RuntimeDomain.EXECUTION).retention
                is RuntimeRetentionMode.DURABLE
            ),
            tool_operations=state.recovery.tools,
            metric_recorder=metric_buffer,
        )
        runtime_bridge.bind(backend)
        session = DefaultSessionService(
            state.conversation,
            state.execution.executions,
            authorization,
            execution,
            HmacCursorSigner("session", grant_key),
            history_reader=session_history_reader,
            transcript_store=state.steps,
            release_terminal=state.retention.release_session,
            workspace_access=input_materializer.access,
        )
        task_runner = RuntimeTaskNodeRunner(
            execution,
            catalog,
            compiler,
            app=app,
            task_state=state.task.tasks,
            task_objects=state.object_store(RuntimeDomain.TASK),
            object_key_factory=object_key_factory,
            payload_policy=payload_policy,
            handlers=task_handlers,
            expanders=task_expanders,
            release_dependency_hold=execution.release_dependency_hold,
            task_durable=(
                state.plan.route(RuntimeDomain.TASK).retention
                is RuntimeRetentionMode.DURABLE
            ),
            execution_durable=(
                state.plan.route(RuntimeDomain.EXECUTION).retention
                is RuntimeRetentionMode.DURABLE
            ),
            recovery_durable=(
                state.plan.route(RuntimeDomain.RECOVERY).retention
                is RuntimeRetentionMode.DURABLE
            ),
        )
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
            metric_recorder=metric_buffer,
            metric_source_namespace=metric_source_namespace,
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
        )
        artifact = DefaultArtifactService(
            state.artifact,
            authorization,
            grant_key=grant_key,
            cursor_signer=HmacCursorSigner("artifact", grant_key),
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
        )
        coordinator = _RuntimeCloseCoordinator(
            tuple(action for _, action in close_actions)
        )
        await _restore_recovery_bindings(catalog, compiler, state, tenant_id=tenant_id)
        if RuntimeDomain.RECOVERY in state.plan.durable_domains:
            await backend.reconcile()
        await graph_service.recover_pending()
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
        task_node_runtime=cast("RuntimeTaskNodeRunner[object]", task_runner),
        tree_streamer=tree_streamer,
        metric_control=metric_buffer,
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
            execution = await state.execution.executions.get(
                checkpoint.execution_id,
                tenant_id=tenant_id,
            )
            if execution is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                binding = compiler.restore(execution.binding)
            except AIError as error:
                if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                    raise
                if error.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE:
                    if error.safe_details.get("reason") == "workspace_mismatch":
                        raise
                    _logger.warning(
                        "recovery binding unavailable: execution=%s",
                        checkpoint.execution_id,
                    )
                    continue
                raise
            if binding.digest != execution.binding_digest:
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
