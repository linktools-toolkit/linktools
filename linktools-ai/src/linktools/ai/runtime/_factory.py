#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime composition and local service graph construction."""

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from linktools.core import environ

from ..asset import AssetStoreReader
from ..agent import AgentCatalog, AgentCompiler
from ..capability import (
    AssetSkillSource,
    CapabilityContribution,
    CapabilityGroup,
    CapabilityGroupCapture,
    SkillSourceRegistry,
    TaskExpander,
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
from ..spec import (
    AgentSpec,
    AssetRuleInstructionResolver,
    MCPServerSpec,
    RepositoryInstructionResolver,
    RepositoryInstructions,
)
from ..storage import ObjectStore, PayloadPolicy
from ..task import DefaultTaskGraphService, LocalTaskGraphLauncher, TaskNodeHandler
from ..workspace import (
    LocalRepositoryInstructionResolver,
    LocalSandbox,
    Sandbox,
    Workspace,
    WorkspaceAccess,
)
from ._agent_executor import AgentExecutor
from ._approval import DefaultApprovalService
from ._binding_resolver import _RuntimeBindingResolver
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
from ._runtime_history import RuntimeHistory
from ._task_capability_capture import TaskCapabilityCaptureStore
from ._runtime_identity import token_seed
from ._session import DefaultSessionService
from ._subagent import SubagentDispatcher
from .service_api import ExecutionHistoryReader, SessionHistoryReader
from .state import RuntimeDomain, RuntimeRetentionMode, RuntimeStorage

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
    binding_resolver: _RuntimeBindingResolver
    history: object


async def compose_runtime_components(
    namespace: str,
    *,
    app: "AppT | None" = None,
    tenant_id: "str | None" = None,
    models: ModelRegistry,
    storage: RuntimeStorage,
    capabilities: "Sequence[CapabilityGroup[AppT] | CapabilityGroupCapture[AppT]]" = (),
    metrics: "Metrics | None" = None,
    limits: "PromptLimits | None" = None,
) -> _RuntimeComponents:
    """Capture declarations and build Runtime-private services."""
    resolved_namespace = validate_persistence_namespace(namespace)
    if not isinstance(storage, RuntimeStorage):
        raise TypeError("storage must be RuntimeStorage")
    if metrics is not None and not isinstance(metrics, Metrics):
        raise TypeError("metrics must be Metrics")
    selected_limits = PromptLimits() if limits is None else limits
    if not isinstance(selected_limits, PromptLimits):
        raise TypeError("limits must be PromptLimits")
    sources = tuple(capabilities)
    if any(
        not isinstance(source, (CapabilityGroup, CapabilityGroupCapture))
        for source in sources
    ):
        raise TypeError(
            "capabilities must contain CapabilityGroup or CapabilityGroupCapture"
        )
    captures: list[CapabilityGroupCapture[AppT]] = []
    for source in sources:
        capture = (
            await source.capture()
            if isinstance(source, CapabilityGroup)
            else source
        )
        await capture.verify_source_revision()
        captures.append(capture)
    groups = tuple(captures)
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
        task_handlers = tuple(
            candidate.value
            for candidate in candidates
            if candidate.kind == "task"
        )
        task_expanders = tuple(
            candidate.value
            for candidate in candidates
            if candidate.kind == "task_expander"
        )

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
            model_resolver=resolver,
            candidates=tuple(
                candidate
                for candidate in candidates
                if candidate.kind not in {"agent", "task", "task_expander"}
            ),
            agents=agents,
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
        await selected_storage.initialize(
            namespace=resolved_namespace,
            tenant_id=effective_tenant_id,
        )
        initialized = True
        for group in groups:
            await group.verify_source_revision()
        if workspace is None:
            instruction_resolver: RepositoryInstructionResolver | None = (
                AssetRuleInstructionResolver(rules)
                if rules.documents
                else None
            )
            workspace_access = None
            execution_cwd = _capture_host_cwd()
        else:
            instruction_resolver = LocalRepositoryInstructionResolver(
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
        authorization = TenantAuthorizationPolicy(effective_tenant_id)
        ownership_transferred = True
        components = await _build_local_components(
            storage=selected_storage,
            catalog=catalog,
            compiler=compiler,
            authorization=authorization,
            tenant_id=effective_tenant_id,
            namespace=resolved_namespace,
            workspace=workspace,
            sandbox=sandbox,
            limits=selected_limits,
            execution_cwd=execution_cwd,
            app=app,
            task_handlers=task_handlers,
            task_expanders=task_expanders,
            history_reader=history_reader,
            session_history_reader=session_history_reader,
            memory_store_factory=memory_store_factory,
            skill_sources=skill_sources,
            asset_sources=asset_sources,
            runtime_token_seed=runtime_token_seed,
            instruction_resolver=instruction_resolver,
            object_key_factory=object_key_factory,
            payload_policy=payload_policy,
            input_materializer=input_materializer,
            session_execution_ready=True,
            metrics=metrics,
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
    metric_buffer: _RuntimeMetricBuffer | None,
    storage: RuntimeStorage,
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
    identities = tuple(
        (
            (candidate.kind, candidate.id, candidate.revision)
            if candidate.kind in {"task", "task_expander"}
            else (candidate.kind, candidate.id)
        )
        for candidate in candidates
    )
    if len(identities) != len(set(identities)):
        raise AIError(ErrorCode.CAPABILITY_CONFLICT)


def _execution_history_reader(
    namespace: str,
    storage: RuntimeStorage,
    runtime_token_seed: bytes,
) -> StepExecutionHistoryReader:
    return StepExecutionHistoryReader(
        namespace=namespace,
        executions=storage.execution.executions,
        store=storage.run_store.read_store(RuntimeDomain.EXECUTION),
        cursor_signer=HmacCursorSigner("execution-history", runtime_token_seed),
        tool_operations=storage.recovery.tools,
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
    authorization: TenantAuthorizationPolicy,
    tenant_id: str,
    namespace: str,
    workspace: "Workspace | None",
    sandbox: "Sandbox | None",
    limits: PromptLimits,
    execution_cwd: "str | None",
    app: AppT,
    task_handlers: Sequence[TaskNodeHandler[AppT]],
    task_expanders: Sequence[TaskExpander],
    history_reader: ExecutionHistoryReader,
    session_history_reader: SessionHistoryReader,
    memory_store_factory: "Callable[[str, str, str, ObjectStore, bool], MemoryStore] | None",
    skill_sources: SkillSourceRegistry,
    asset_sources: Mapping[str, AssetStoreReader],
    runtime_token_seed: bytes,
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
        await storage.retention.release_execution_handoff(
            execution_id,
            tenant_id=tenant_id,
        )

    try:
        if not storage.ready:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        _require_storage_identity(storage, namespace=namespace, tenant_id=tenant_id)
        metric_buffer = None if metrics is None else _RuntimeMetricBuffer(metrics)
        metric_source_namespace = None if metric_buffer is None else namespace
        runtime_bridge = _ExecutionRuntimeBridge()
        live_broker = LiveExecutionEventBroker()
        history_service = DefaultExecutionHistoryService(
            storage.execution.executions,
            authorization,
            history_reader,
            HmacCursorSigner("execution", runtime_token_seed),
        )
        binding_resolver = _RuntimeBindingResolver(
            catalog,
            compiler,
            sandbox=sandbox,
        )
        execution = DefaultExecutionService(
            storage.execution,
            storage.object_store(RuntimeDomain.EXECUTION),
            authorization,
            sessions=storage.conversation.sessions,
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
            asset_sources=asset_sources,
            metrics=metric_buffer,
            sandbox=sandbox,
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

    task_launcher: LocalTaskGraphLauncher | None = None
    graph_service: DefaultTaskGraphService | None = None
    coordinator: _RuntimeCloseCoordinator | None = None
    close_actions: tuple[tuple[str, Callable[[], Awaitable[None]]], ...] | None = None
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
                    RuntimeDomain.EXECUTION,
                    RuntimeDomain.RECOVERY,
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
            metric_recorder=metric_buffer,
        )
        runtime_bridge.bind(backend)
        session = DefaultSessionService(
            storage.conversation,
            storage.execution.executions,
            authorization,
            execution,
            HmacCursorSigner("session", runtime_token_seed),
            history_reader=session_history_reader,
            transcript_store=storage.run_store,
            release_terminal=storage.retention.release_session,
            workspace_access=input_materializer.access,
        )
        task_capability_captures = TaskCapabilityCaptureStore(
            namespace,
            compiler,
            binding_resolver,
            storage.object_store(RuntimeDomain.TASK),
            agent_task_id="linktools.ai.agent",
        )
        task_runner = RuntimeTaskNodeRunner(
            execution,
            catalog,
            compiler,
            session=session,
            namespace=namespace,
            app=app,
            authorization=authorization,
            task_state=storage.task.tasks,
            task_objects=storage.object_store(RuntimeDomain.TASK),
            artifact_state=storage.artifact,
            artifact_objects=storage.object_store(RuntimeDomain.ARTIFACT),
            object_key_factory=object_key_factory,
            capability_captures=task_capability_captures,
            input_materializer=ExecutionInputMaterializer(
                input_materializer.access,
                limits,
                object_store=storage.object_store(RuntimeDomain.TASK),
                object_key_factory=object_key_factory,
                payload_policy=PayloadPolicy(inline_limit_bytes=0),
                object_domain=RuntimeDomain.TASK,
            ),
            handlers=task_handlers,
            expanders=task_expanders,
            release_dependency_hold=execution.release_dependency_hold,
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
            metric_recorder=metric_buffer,
            metric_source_namespace=metric_source_namespace,
        )
        evaluation = DefaultEvaluationService(
            storage.evaluation,
            storage.execution.executions,
            authorization,
            execution,
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
        )
        artifact = DefaultArtifactService(
            storage.artifact,
            authorization,
            token_seed=runtime_token_seed,
            cursor_signer=HmacCursorSigner("artifact", runtime_token_seed),
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
        )
        coordinator = _RuntimeCloseCoordinator(
            tuple(action for _, action in close_actions)
        )
        if RuntimeDomain.RECOVERY in storage.plan.durable_domains:
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
                storage=storage,
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
        history=_borrowed_runtime_history(
            history_service,
            tenant_id=tenant_id,
            storage=storage,
            authorization=authorization,
            artifact=artifact,
        ),
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
        tasks=storage.task.tasks,
        authorization=authorization,
        namespace=storage.namespace,
        execution_objects=storage.object_store(RuntimeDomain.EXECUTION),
        task_objects=storage.object_store(RuntimeDomain.TASK),
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
