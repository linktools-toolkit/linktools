#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-internal service graph construction."""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from linktools.core import environ
from pydantic_ai_harness.memory import SearchableMemoryStore

from ..agent import AgentCompiler, AgentCatalog, AgentExecutor
from ..asset import AssetRepository
from ..core import AuthorizationPolicy, HmacCursorSigner
from ..errors import AIError, ErrorCode
from ..storage import ObjectStore
from ..task import LocalTaskGraphLauncher
from ._approval import DefaultApprovalService
from ._artifact import DefaultArtifactService
from ._coordinator import _LocalRuntimeCoordinator
from ._evaluation import DefaultEvaluationService
from ._event import DefaultEventService, LiveExecutionEventBroker
from ._execution import DefaultExecutionService
from ._local import LocalExecutionBackend
from ._planner import DefaultTaskService, RuntimeTaskNodeRunner
from ._runtime_service import Runtime
from ._session import DefaultSessionService
from ._subagent import SubagentDispatcher
from .service_api import ExecutionHistoryReader, SessionHistoryReader
from .state import RecoveryCheckpointState, RuntimeDomain, RuntimeRetentionMode, RuntimeState

_logger = environ.get_logger("ai.runtime.factory")


async def build_local_runtime(
    *,
    state: RuntimeState,
    catalog: AgentCatalog,
    compiler: AgentCompiler,
    assets: AssetRepository,
    authorization: AuthorizationPolicy,
    tenant_id: str,
    namespace: str,
    execution_root: Path,
    history_reader: ExecutionHistoryReader,
    session_history_reader: SessionHistoryReader,
    memory_store_factory: "Callable[[str, str, str, ObjectStore, bool], SearchableMemoryStore] | None",
    grant_key: bytes,
) -> Runtime:
    if not state.ready or not assets.ready:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    execution = DefaultExecutionService(
        state.execution,
        state.object_store(RuntimeDomain.EXECUTION),
        authorization,
        sessions=state.conversation.sessions,
        catalog=catalog,
        compiler=compiler,
        history_reader=history_reader,
        release_terminal=state.retention.release_execution_handoff,
    )
    dispatcher = SubagentDispatcher(catalog, compiler, execution)
    executor = AgentExecutor(execution_root=execution_root)

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
            state.working_object_store(RuntimeDomain.MEMORY, owner_scope=f"execution:{execution_id}")
            if transient
            else state.object_store(RuntimeDomain.MEMORY)
        )
        return memory_store_factory(memory_tenant, execution_id, memory_scope, store, transient)

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
            execution_root=execution_root,
            step_reads={
                domain: state.steps.read_store(domain)
                for domain in (RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY)
            },
            step_lifecycle=state.steps,
            memory_store_factory=build_memory_store,
            recovery_enabled=RuntimeDomain.RECOVERY in state.plan.durable_domains,
            conversation_durable=state.plan.route(RuntimeDomain.CONVERSATION).retention is RuntimeRetentionMode.DURABLE,
            handoff_contract_digest=state.handoff_contract_digest,
            subagent_dispatcher=dispatcher,
            live_broker=live_broker,
            execution_objects_durable=state.plan.route(RuntimeDomain.EXECUTION).retention is RuntimeRetentionMode.DURABLE,
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
            _cursor_signer("session", grant_key),
            history_reader=session_history_reader,
            transcript_store=state.steps.read_store(RuntimeDomain.CONVERSATION),
            release_terminal=state.retention.release_session,
            gated_execution=execution,
        )
        task_runner = RuntimeTaskNodeRunner(execution, catalog, compiler)
        task_launcher = LocalTaskGraphLauncher(
            state.task.tasks,
            task_runner,
            owner=f"runtime:{tenant_id}",
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
            cursor_signer=_cursor_signer("artifact", grant_key),
        )
        local_coordinator = _LocalRuntimeCoordinator(execution, session, event, backend)
        coordinator = _RuntimeCloseCoordinator((task_launcher.shutdown, backend.close, state.close))
        runtime = Runtime(
            catalog,
            compiler,
            assets,
            execution,
            session,
            task,
            evaluation,
            approval,
            event,
            artifact,
            tenant_id=tenant_id,
            close_callback=coordinator.close,
            local_coordinator=local_coordinator,
        )
        await _restore_recovery_bindings(catalog, compiler, state, tenant_id=tenant_id)
        if RuntimeDomain.RECOVERY in state.plan.durable_domains:
            await backend.reconcile()
    except BaseException:
        if task_launcher is not None:
            await task_launcher.shutdown()
        if backend is not None:
            await backend.close()
        raise
    _logger.info(
        "local Runtime built: namespace=%s tenant=%s durable_domains=%s",
        namespace,
        tenant_id,
        sorted(domain.value for domain in state.plan.durable_domains),
    )
    return runtime


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
                or execution.planning is not recovery_input.planning
                or execution.thinking is not recovery_input.thinking
                or execution.binding != recovery_input.binding
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                binding = compiler.restore(recovery_input.binding)
            except AIError as error:
                if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                    raise
                raise AIError(
                    ErrorCode.AGENT_DEFINITION_UNAVAILABLE,
                    safe_details={"execution_id": checkpoint.execution_id},
                ) from error
            if binding.digest != recovery_input.binding_digest:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            catalog.register_definition(binding.definition)
            binding = catalog.register_binding(binding)
            handoff = checkpoint.terminal_handoff
            if handoff is not None and handoff.outcome.output is not None:
                output = binding.output_binding
                outcome = handoff.outcome
                if (
                    outcome.output_schema_id != output.schema_id
                    or outcome.output_schema_revision != output.schema_revision
                    or outcome.output_schema_fingerprint != output.schema_fingerprint
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if page.next_cursor is None:
            return
        cursor = page.next_cursor


def _cursor_signer(name: str, grant_key: bytes) -> HmacCursorSigner:
    return HmacCursorSigner(name, grant_key)


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


__all__ = ["build_local_runtime"]
