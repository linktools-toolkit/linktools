#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local ExecutionBackend backed by AgentExecutor and durable persistence."""

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Protocol

from linktools.core import environ
from pydantic import ValidationError
from pydantic_ai_harness.memory import SearchableMemoryStore
from pydantic_ai_harness.step_persistence import StepStore, continue_run, fork_run

from ..agent import (
    MEMORY_TOOL_NAMES,
    AgentDefinition,
    AgentDefinitionCatalog,
    AgentExecutor,
    SubagentDelegate,
    select_platform_tool_names,
)
from ..capability import CapabilityMaterializationContext
from ..core import (
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    IdempotencyStatus,
    JsonValue,
    Principal,
    ResourceKind,
    ResourceRef,
    SessionStatus,
    StopReason,
    UsageMetrics,
    canonical_json_bytes,
    step_conversation_id,
    step_run_id,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode
from ..storage import ObjectStore, StorageMetrics
from ._execution import CancelEffectOutcome, ExecutionStartIdentity
from ._object import RuntimeObjectKeyFactory, put_runtime_object, read_runtime_object
from .service_api import ExecutionRequest
from .state import ConversationState, ExecutionState, RecoveryState
from .state._contracts import (
    ConversationCursor,
    ExecutionRecord,
    ExecutionStartClaim,
    ExecutionTerminalCommit,
    IdempotencyRecord,
    IdempotencyTerminalUpdate,
    RecoveryCheckpoint,
    RecoveryCheckpointState,
    RecoveryConversationIntent,
    RecoveryExecutionInput,
    RecoveryHandoffPhase,
    RecoveryIdempotencyInput,
    RecoveryTerminalHandoff,
    RecoveryTerminalOutcome,
    ResultRecord,
)
from .state._plan import RuntimeDomain

if TYPE_CHECKING:
    from ..storage import ObjectRef


class _SubagentDispatcher(Protocol):
    def delegate_for(
        self,
        *,
        parent_execution_id: str,
        root_execution_id: str,
        memory_scope: "str | None",
        principal: Principal,
    ) -> SubagentDelegate: ...

_logger = environ.get_logger("ai.runtime.local")

_HANDOFF_PHASE_RANK = {
    RecoveryHandoffPhase.PREPARED: 0,
    RecoveryHandoffPhase.CONVERSATION_RESOLVED: 1,
    RecoveryHandoffPhase.EXECUTION_COMMITTED: 2,
    RecoveryHandoffPhase.COMPLETED: 3,
}


@dataclass(frozen=True, slots=True)
class _WorkerFailure:
    code: ErrorCode
    safe_details: Mapping[str, JsonValue]


class _StepLifecycle(Protocol):
    async def materialize_conversation(self, *, step_run_id: str) -> None: ...
    async def materialize_from_recovery(self, *, target: RuntimeDomain, step_run_id: str) -> None: ...
    async def materialize_recovery_snapshot(self, *, step_run_id: str, require_complete: bool) -> None: ...
    async def verify_terminal_attempts(self, *, candidate_step_run_ids: tuple[str, ...], required_step_run_id: str | None) -> None: ...
    async def release_staging_many(self, *, candidate_step_run_ids: tuple[str, ...]) -> None: ...


class LocalExecutionBackend:
    """Resolve immutable definitions and persist one execution lifecycle."""

    def __init__(
        self,
        conversation: ConversationState,
        execution_state: ExecutionState,
        recovery: RecoveryState,
        execution_objects: ObjectStore,
        recovery_objects: ObjectStore,
        metrics: StorageMetrics,
        namespace: str,
        steps: StepStore,
        executor: AgentExecutor,
        catalog: AgentDefinitionCatalog,
        *,
        tenant_id: str,
        execution_root: Path,
        step_reads: Mapping[RuntimeDomain, StepStore],
        step_lifecycle: _StepLifecycle,
        memory_store_factory: "Callable[[str, str, str], SearchableMemoryStore] | None" = None,
        recovery_enabled: bool = False,
        conversation_durable: bool = False,
        handoff_contract_digest: "str | None" = None,
        subagent_dispatcher: "_SubagentDispatcher | None" = None,
    ) -> None:
        self._conversation = conversation
        self._execution = execution_state
        self._recovery = recovery
        self._execution_objects = execution_objects
        self._recovery_objects = recovery_objects
        self._metrics = metrics
        self._namespace = namespace
        self._steps = steps
        self._executor = executor
        self._catalog = catalog
        self._tenant_id = validate_tenant_id(tenant_id)
        self._execution_root = execution_root
        self._memory_store_factory = memory_store_factory
        self._recovery_enabled = recovery_enabled
        self._conversation_durable = conversation_durable
        self._handoff_contract_digest = handoff_contract_digest
        self._subagent_dispatcher = subagent_dispatcher
        self._step_reads = dict(step_reads)
        if frozenset(self._step_reads) != frozenset({RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY}):
            raise ValueError("step_reads must contain exactly the three Step owner domains")
        self._step_lifecycle = step_lifecycle
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._worker_failures: dict[str, _WorkerFailure] = {}
        self._captured_usage: dict[str, UsageMetrics] = {}
        self._accepting = True

    async def _validate_start(self, request: ExecutionRequest, execution: ExecutionRecord) -> None:
        if not self._accepting:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if request.principal.tenant_id != self._tenant_id or execution.tenant_id != self._tenant_id:
            _logger.warning(
                "local execution tenant rejected: expected=%s request=%s execution=%s",
                self._tenant_id,
                request.principal.tenant_id,
                execution.tenant_id,
            )
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        self._catalog.definition(execution.binding_digest)

    async def prepare_start(
        self,
        request: ExecutionRequest,
        execution: ExecutionRecord,
        identity: ExecutionStartIdentity,
    ) -> None:
        await self._validate_start(request, execution)
        if execution.status is not ExecutionStatus.PENDING_START:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not self._recovery_enabled:
            await self._ensure_session_admission(execution)
            if await self._cancel_admitted_start_if_closing(execution):
                return
            return
        definition = self._catalog.definition(execution.binding_digest)
        now = datetime.now(timezone.utc)
        recovery_input = RecoveryExecutionInput(
            user_prompt=request.user_prompt,
            principal_id=request.principal.principal_id,
            principal_kind=request.principal.kind,
            session_id=execution.session_id,
            memory_scope=execution.memory_scope,
            agent_id=definition.spec.id,
            binding_digest=execution.binding_digest,
            lineage_kind=execution.lineage_kind.value,
            parent_execution_id=execution.parent_execution_id,
            root_execution_id=execution.root_execution_id,
            source_execution_id=execution.source_execution_id,
            base_execution_id=execution.base_execution_id,
            conversation_step_run_id=execution.conversation_step_run_id,
            idempotency=RecoveryIdempotencyInput(
                scope=identity.scope,
                idempotency_key_digest=identity.idempotency_key_digest,
                request_digest=identity.request_digest,
            ),
        )
        candidate = RecoveryCheckpoint(
            execution_id=execution.execution_id,
            tenant_id=execution.tenant_id,
            input=recovery_input,
            step_run_id=None,
            agent_run_sequence=execution.agent_run_sequence,
            state=RecoveryCheckpointState.ADMITTED,
            handoff_phase=RecoveryHandoffPhase.NONE,
            terminal_handoff=None,
            handoff_contract_digest=None,
            pending_operation_id=None,
            revision=0,
            created_at=now,
            updated_at=now,
        )
        existing = await self._recovery.checkpoints.get(
            execution.execution_id,
            tenant_id=execution.tenant_id,
        )
        if existing is None:
            await self._recovery.checkpoints.create(candidate)
            self._metrics.count("recovery.checkpoint.admitted", domain="recovery", target="runtime")
            _logger.info("recovery admission persisted: execution=%s", execution.execution_id)
        elif not _admission_matches(existing, candidate):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        else:
            _logger.debug("recovery admission replayed: execution=%s", execution.execution_id)
        await self._ensure_session_admission(execution)
        await self._cancel_admitted_start_if_closing(execution)

    async def _cancel_admitted_start_if_closing(self, execution: ExecutionRecord) -> bool:
        if execution.session_id is None:
            return False
        session = await self._conversation.sessions.get(
            execution.session_id,
            tenant_id=execution.tenant_id,
        )
        if session is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if session.status is SessionStatus.OPEN:
            return False
        if session.status not in {
            SessionStatus.CLOSING,
            SessionStatus.CLEANUP_REQUIRED,
        } or session.active_execution_id != execution.execution_id:
            raise AIError(ErrorCode.SESSION_CONFLICT)
        await self._commit_terminal(
            execution,
            ExecutionStatus.CANCELLED,
            None,
            ErrorCode.EXECUTION_CANCELLED.value,
            StopReason.CANCELLED,
        )
        _logger.info(
            "admitted start cancelled by session close: execution=%s",
            execution.execution_id,
        )
        return True

    async def _ensure_session_admission(self, execution: ExecutionRecord) -> None:
        if execution.session_id is None:
            return
        if execution.parent_execution_id is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        expected = await self._expected_session_cursor(execution)
        released_stale_owner = False
        for _ in range(2):
            session = await self._conversation.sessions.get(
                execution.session_id,
                tenant_id=execution.tenant_id,
            )
            if session is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            owner_id = session.active_execution_id
            if owner_id == execution.execution_id:
                if session.status not in {
                    SessionStatus.OPEN,
                    SessionStatus.CLOSING,
                    SessionStatus.CLEANUP_REQUIRED,
                } or session.continuation != expected:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return
            if owner_id is not None:
                owner = await self._execution.executions.get(
                    owner_id,
                    tenant_id=execution.tenant_id,
                )
                if owner is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if (
                    owner.execution_id != owner_id
                    or owner.tenant_id != execution.tenant_id
                    or owner.session_id != execution.session_id
                    or owner.parent_execution_id is not None
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if owner.status not in {
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                }:
                    if session.status is not SessionStatus.OPEN:
                        raise AIError(ErrorCode.SESSION_CONFLICT)
                    raise AIError(ErrorCode.SESSION_BUSY)
                if released_stale_owner:
                    raise AIError(ErrorCode.SESSION_BUSY)
                await self._conversation.sessions.release_execution(
                    execution.session_id,
                    tenant_id=execution.tenant_id,
                    execution_id=owner_id,
                )
                _logger.info(
                    "released terminal session owner: session=%s execution=%s",
                    execution.session_id,
                    owner_id,
                )
                released_stale_owner = True
                continue
            if session.continuation != expected:
                raise AIError(ErrorCode.SESSION_BUSY)
            await self._conversation.sessions.admit_execution(
                execution.session_id,
                tenant_id=execution.tenant_id,
                execution_id=execution.execution_id,
                expected=expected,
            )
            _logger.info(
                "session admission acquired: session=%s execution=%s",
                execution.session_id,
                execution.execution_id,
            )
            return
        raise AIError(ErrorCode.SESSION_BUSY)

    async def launch(self, request: ExecutionRequest, execution: ExecutionRecord) -> None:
        await self._validate_start(request, execution)
        if self._recovery_enabled:
            checkpoint = await self._recovery.checkpoints.get(
                execution.execution_id,
                tenant_id=execution.tenant_id,
            )
            if checkpoint is None or checkpoint.state is RecoveryCheckpointState.COMPLETED:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        current = await self._execution.executions.get(
            execution.execution_id,
            tenant_id=execution.tenant_id,
        )
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            current.tenant_id != execution.tenant_id
            or current.binding_digest != execution.binding_digest
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if current.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.CANCELLING,
            ExecutionStatus.FINALIZING,
        }:
            return
        if current.status is not ExecutionStatus.STARTED:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if current.session_id is not None:
            session = await self._conversation.sessions.get(
                current.session_id,
                tenant_id=current.tenant_id,
            )
            if session is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if session.status is SessionStatus.CLOSED:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if session.status not in {
                SessionStatus.OPEN,
                SessionStatus.CLOSING,
                SessionStatus.CLEANUP_REQUIRED,
            }:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if session.active_execution_id != current.execution_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        existing = self._tasks.get(execution.execution_id)
        if existing is not None:
            if not existing.done():
                return
            self._task_done(execution.execution_id, existing)
            if execution.execution_id in self._worker_failures:
                return
        if execution.execution_id in self._worker_failures:
            return
        task = asyncio.create_task(self._run(request, current), name=f"ai-execution-{execution.execution_id}")
        self._tasks[execution.execution_id] = task
        task.add_done_callback(
            lambda completed, execution_id=execution.execution_id: self._task_done(
                execution_id,
                completed,
            )
        )
        _logger.debug(
            "local execution launched: execution=%s definition=%s",
            execution.execution_id,
            execution.binding_digest,
        )

    async def abort_start(self, execution: ExecutionRecord) -> None:
        current = await self._execution.executions.get(
            execution.execution_id,
            tenant_id=execution.tenant_id,
        )
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if current.status not in {
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if current.session_id is not None:
            await self._conversation.sessions.release_execution(
                current.session_id,
                tenant_id=current.tenant_id,
                execution_id=current.execution_id,
            )
        if not self._recovery_enabled:
            return
        checkpoint = await self._recovery.checkpoints.get(
            current.execution_id,
            tenant_id=current.tenant_id,
        )
        if checkpoint is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if checkpoint.state is RecoveryCheckpointState.COMPLETED:
            return
        if (
            checkpoint.state is not RecoveryCheckpointState.ADMITTED
            or checkpoint.handoff_phase is not RecoveryHandoffPhase.NONE
            or checkpoint.terminal_handoff is not None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._finish_checkpoint(checkpoint)
        _logger.info("start admission aborted: execution=%s", current.execution_id)

    def _task_done(self, execution_id: str, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            error = None
        else:
            try:
                error = task.exception()
            except asyncio.CancelledError:
                error = None
        if self._tasks.get(execution_id) is not task:
            return
        self._tasks.pop(execution_id, None)
        self._captured_usage.pop(execution_id, None)
        if error is None:
            return
        if isinstance(error, AIError):
            failure = _WorkerFailure(error.code, dict(error.safe_details))
        else:
            failure = _WorkerFailure(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                {"phase": "local_execution_worker"},
            )
        details = dict(failure.safe_details)
        details["execution_id"] = execution_id
        self._worker_failures[execution_id] = _WorkerFailure(failure.code, details)
        _logger.error(
            "local execution worker failed: execution=%s code=%s",
            execution_id,
            failure.code.value,
            exc_info=environ.debug,
        )

    def worker_failure(self, execution_id: str, *, tenant_id: str) -> AIError | None:
        if tenant_id != self._tenant_id:
            return AIError(ErrorCode.AUTHORIZATION_DENIED)
        failure = self._worker_failures.get(execution_id)
        if failure is None:
            return None
        return AIError(failure.code, safe_details=dict(failure.safe_details))

    async def cancel(self, execution: ExecutionRecord) -> CancelEffectOutcome:
        current = await self._execution.executions.get(
            execution.execution_id,
            tenant_id=execution.tenant_id,
        )
        if current is not None and current.status is ExecutionStatus.FINALIZING:
            _logger.debug(
                "local cancellation deferred during finalization: execution=%s",
                execution.execution_id,
            )
            return CancelEffectOutcome.CONFIRMED
        task = self._tasks.get(execution.execution_id)
        if task is None:
            if current is not None and current.status in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
                ExecutionStatus.CANCELLING,
            }:
                return CancelEffectOutcome.CONFIRMED
            return CancelEffectOutcome.UNKNOWN
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self._task_done(execution.execution_id, task)
        current = await self._execution.executions.get(
            execution.execution_id,
            tenant_id=execution.tenant_id,
        )
        if current is not None and current.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.CANCELLING,
        }:
            return CancelEffectOutcome.CONFIRMED
        return CancelEffectOutcome.UNKNOWN

    async def reconcile(self) -> None:
        """Rebuild transient execution state from recovery-owned checkpoints."""
        if not self._recovery_enabled:
            return
        checkpoints = await self._recovery.checkpoints.list(tenant_id=self._tenant_id)
        for checkpoint in checkpoints:
            if checkpoint.state is RecoveryCheckpointState.COMPLETED:
                continue
            if checkpoint.handoff_phase is not RecoveryHandoffPhase.NONE:
                if self._handoff_contract_digest is None or checkpoint.handoff_contract_digest != self._handoff_contract_digest:
                    _logger.error("recovery handoff contract mismatch: execution=%s", checkpoint.execution_id)
                    raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
                await self._reconcile_handoff(checkpoint)
                continue
            recovery_input = checkpoint.input
            principal = Principal(recovery_input.principal_id, checkpoint.tenant_id, recovery_input.principal_kind)
            execution = await self._execution.executions.get(checkpoint.execution_id, tenant_id=checkpoint.tenant_id)
            if execution is not None and execution.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                if execution.session_id is not None:
                    await self._conversation.sessions.release_execution(
                        execution.session_id,
                        tenant_id=execution.tenant_id,
                        execution_id=execution.execution_id,
                    )
                await self._finish_checkpoint(checkpoint)
                continue
            if execution is not None:
                if (
                    execution.binding_digest != recovery_input.binding_digest
                    or execution.parent_execution_id != recovery_input.parent_execution_id
                    or execution.root_execution_id != recovery_input.root_execution_id
                    or execution.source_execution_id != recovery_input.source_execution_id
                    or execution.base_execution_id != recovery_input.base_execution_id
                    or execution.conversation_step_run_id != recovery_input.conversation_step_run_id
                    or execution.lineage_kind.value != recovery_input.lineage_kind
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if execution is None:
                execution = await self._create_recovery_execution(checkpoint)
            if checkpoint.state in {
                RecoveryCheckpointState.ADMITTED,
                RecoveryCheckpointState.ACTIVE,
                RecoveryCheckpointState.WAITING,
            }:
                if not await self._reconcile_session_recovery(execution):
                    continue
            if (
                checkpoint.state is RecoveryCheckpointState.ADMITTED
                and execution.status is ExecutionStatus.PENDING_START
            ):
                await self._ensure_recovery_idempotency(
                    checkpoint,
                    expected_status=IdempotencyStatus.RESERVED,
                )
                started = await self._execution.executions.claim_start(
                    ExecutionStartClaim(
                        execution.execution_id,
                        execution.tenant_id,
                        execution.revision,
                        execution.event_sequence,
                        recovery_input.idempotency.scope,
                        recovery_input.idempotency.idempotency_key_digest,
                        recovery_input.idempotency.request_digest,
                        datetime.now(timezone.utc),
                    )
                )
                execution = started
            elif execution.status is ExecutionStatus.CANCELLING:
                await self._commit_terminal(
                    execution,
                    ExecutionStatus.CANCELLED,
                    None,
                    ErrorCode.EXECUTION_CANCELLED.value,
                    StopReason.CANCELLED,
                )
                continue
            elif execution.status is ExecutionStatus.START_UNKNOWN:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            elif checkpoint.state is RecoveryCheckpointState.ADMITTED and execution.status is ExecutionStatus.STARTED:
                await self._ensure_recovery_idempotency(
                    checkpoint,
                    expected_status=IdempotencyStatus.STARTED,
                )
            elif checkpoint.state in {
                RecoveryCheckpointState.ACTIVE,
                RecoveryCheckpointState.WAITING,
            }:
                if execution.status is not ExecutionStatus.STARTED:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self._ensure_recovery_idempotency(
                    checkpoint,
                    expected_status=IdempotencyStatus.STARTED,
                )
            else:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            request = ExecutionRequest(
                user_prompt=recovery_input.user_prompt,
                principal=principal,
                idempotency_key=f"recovery:{checkpoint.execution_id}",
                memory_scope=recovery_input.memory_scope,
            )
            await self.launch(request, execution)
            _logger.info("local recovery execution relaunched: tenant=%s execution=%s", checkpoint.tenant_id, checkpoint.execution_id)

    async def _reconcile_session_recovery(self, execution: ExecutionRecord) -> bool:
        if execution.session_id is None:
            return True
        session = await self._conversation.sessions.get(
            execution.session_id,
            tenant_id=execution.tenant_id,
        )
        if session is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution.status is ExecutionStatus.FINALIZING:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if session.status is SessionStatus.CLOSED:
            if execution.status is not ExecutionStatus.PENDING_START:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._commit_terminal(
                execution,
                ExecutionStatus.FAILED,
                None,
                ErrorCode.SESSION_CONFLICT.value,
                StopReason.ERROR,
            )
            return False
        if session.status in {SessionStatus.CLOSING, SessionStatus.CLEANUP_REQUIRED}:
            if execution.status is ExecutionStatus.FINALIZING:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if execution.status is ExecutionStatus.PENDING_START:
                if session.active_execution_id == execution.execution_id:
                    await self._commit_terminal(
                        execution,
                        ExecutionStatus.CANCELLED,
                        None,
                        ErrorCode.EXECUTION_CANCELLED.value,
                        StopReason.CANCELLED,
                    )
                else:
                    await self._commit_terminal(
                        execution,
                        ExecutionStatus.FAILED,
                        None,
                        ErrorCode.SESSION_CONFLICT.value,
                        StopReason.ERROR,
                    )
            elif session.active_execution_id == execution.execution_id:
                await self._commit_terminal(
                    execution,
                    ExecutionStatus.CANCELLED,
                    None,
                    ErrorCode.EXECUTION_CANCELLED.value,
                    StopReason.CANCELLED,
                )
            else:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return False
        try:
            await self._ensure_session_admission(execution)
        except AIError as error:
            if error.code not in {
                ErrorCode.SESSION_BUSY,
                ErrorCode.SESSION_CONFLICT,
            }:
                raise
            await self._commit_terminal(
                execution,
                ExecutionStatus.FAILED,
                None,
                error.code.value,
                StopReason.ERROR,
            )
            return False
        return True

    async def _reconcile_handoff(self, checkpoint: RecoveryCheckpoint) -> None:
        handoff = checkpoint.terminal_handoff
        if handoff is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        outcome = handoff.outcome
        if outcome.terminal_status in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED} and (
            handoff.conversation is not None
            or outcome.recovery_object_ref is not None
            or any(value is not None for value in (outcome.output_schema_id, outcome.output_schema_revision, outcome.output_schema_fingerprint))
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        execution = await self._execution.executions.get(
            checkpoint.execution_id,
            tenant_id=checkpoint.tenant_id,
        )
        if execution is None:
            execution = await self._create_recovery_execution(checkpoint)
        if execution.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        } and execution.status is not outcome.terminal_status:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if checkpoint.handoff_phase is RecoveryHandoffPhase.PREPARED:
            if outcome.terminal_status is ExecutionStatus.SUCCEEDED:
                if execution.session_id is not None and handoff.conversation is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if handoff.source_step_run_id is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self._step_lifecycle.materialize_from_recovery(
                    target=RuntimeDomain.EXECUTION,
                    step_run_id=handoff.source_step_run_id,
                )
                snapshot = await self._step_reads[RuntimeDomain.EXECUTION].latest_snapshot(
                    run_id=handoff.source_step_run_id,
                )
                if snapshot is None or snapshot.state != "complete":
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if outcome.recovery_object_ref is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await read_runtime_object(
                    self._recovery_objects,
                    outcome.recovery_object_ref,
                )
                if execution.status is ExecutionStatus.CANCELLING:
                    checkpoint = await self._rewrite_prepared_success_handoff(
                        checkpoint,
                        target_status=ExecutionStatus.CANCELLED,
                        error_code=ErrorCode.EXECUTION_CANCELLED.value,
                        stop_reason=StopReason.CANCELLED,
                    )
                    return await self._reconcile_handoff(checkpoint)
                if execution.status is ExecutionStatus.STARTED:
                    execution = await self._claim_session_or_recovery_finalizing(execution)
                if execution.status is ExecutionStatus.CANCELLING:
                    checkpoint = await self._rewrite_prepared_success_handoff(
                        checkpoint,
                        target_status=ExecutionStatus.CANCELLED,
                        error_code=ErrorCode.EXECUTION_CANCELLED.value,
                        stop_reason=StopReason.CANCELLED,
                    )
                    return await self._reconcile_handoff(checkpoint)
                allowed_statuses = {
                    ExecutionStatus.STARTED,
                    ExecutionStatus.FINALIZING,
                    ExecutionStatus.SUCCEEDED,
                }
                if execution.status not in allowed_statuses:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            elif outcome.recovery_object_ref is not None or any(
                value is not None
                for value in (
                    outcome.output_schema_id,
                    outcome.output_schema_revision,
                    outcome.output_schema_fingerprint,
                )
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                await self._resolve_handoff_conversation(checkpoint, handoff)
            except AIError as error:
                if error.code is not ErrorCode.SESSION_CONFLICT:
                    raise
                checkpoint = await self._rewrite_prepared_success_handoff(
                    checkpoint,
                    target_status=ExecutionStatus.FAILED,
                    error_code=error.code.value,
                    stop_reason=StopReason.ERROR,
                )
                return await self._reconcile_handoff(checkpoint)
            checkpoint = await self._advance_handoff(
                checkpoint,
                RecoveryHandoffPhase.CONVERSATION_RESOLVED,
            )
        if checkpoint.handoff_phase is RecoveryHandoffPhase.CONVERSATION_RESOLVED:
            await self._commit_reconciled_terminal(checkpoint)
            checkpoint = await self._advance_handoff(
                checkpoint,
                RecoveryHandoffPhase.EXECUTION_COMMITTED,
            )
        if checkpoint.handoff_phase is RecoveryHandoffPhase.EXECUTION_COMMITTED:
            await self._validate_committed_handoff(checkpoint)
            execution = await self._execution.executions.get(
                checkpoint.execution_id,
                tenant_id=checkpoint.tenant_id,
            )
            if execution is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if execution.session_id is not None:
                await self._conversation.sessions.release_execution(
                    execution.session_id,
                    tenant_id=execution.tenant_id,
                    execution_id=execution.execution_id,
                )
                _logger.info(
                    "session admission released after recovery terminal: execution=%s",
                    execution.execution_id,
                )
            await self._complete_handoff(checkpoint)

    async def _create_recovery_execution(self, checkpoint: RecoveryCheckpoint) -> ExecutionRecord:
        recovery_input = checkpoint.input
        session_id = recovery_input.session_id
        if session_id is not None:
            session = await self._conversation.sessions.get(session_id, tenant_id=checkpoint.tenant_id)
            if session is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        status = (
            ExecutionStatus.PENDING_START
            if checkpoint.state is RecoveryCheckpointState.ADMITTED
            else ExecutionStatus.STARTED
        )
        execution = ExecutionRecord(
            execution_id=checkpoint.execution_id,
            tenant_id=checkpoint.tenant_id,
            session_id=session_id,
            binding_digest=recovery_input.binding_digest,
            parent_execution_id=recovery_input.parent_execution_id,
            root_execution_id=recovery_input.root_execution_id,
            source_execution_id=recovery_input.source_execution_id,
            base_execution_id=recovery_input.base_execution_id,
            lineage_kind=ExecutionLineageKind(recovery_input.lineage_kind),
            status=status,
            revision=0,
            event_sequence=0,
            agent_run_sequence=checkpoint.agent_run_sequence,
            error_code=None,
            safe_error_details={},
            created_at=checkpoint.created_at,
            updated_at=checkpoint.updated_at,
            memory_scope=recovery_input.memory_scope,
            conversation_step_run_id=recovery_input.conversation_step_run_id,
        )
        await self._execution.executions.create(execution)
        return execution

    async def _ensure_recovery_idempotency(
        self,
        checkpoint: RecoveryCheckpoint,
        *,
        expected_status: IdempotencyStatus,
    ) -> IdempotencyRecord:
        recovery_idempotency = checkpoint.input.idempotency
        records = await self._execution.idempotency.list_by_resource(
            ResourceKind.EXECUTION,
            checkpoint.execution_id,
            tenant_id=checkpoint.tenant_id,
        )
        if len(records) > 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        identity = records[0] if records else None
        if identity is not None:
            return await self._restore_recovery_idempotency(
                checkpoint,
                identity,
                expected_status=expected_status,
            )
        now = checkpoint.updated_at
        identity = await self._execution.idempotency.reserve(
            IdempotencyRecord(
                tenant_id=checkpoint.tenant_id,
                runtime_domain=RuntimeDomain.EXECUTION,
                scope=recovery_idempotency.scope,
                idempotency_key_digest=recovery_idempotency.idempotency_key_digest,
                request_digest=recovery_idempotency.request_digest,
                resource_kind=ResourceKind.EXECUTION,
                resource_id=checkpoint.execution_id,
                status=expected_status,
                result_digest=None,
                error_code=None,
                created_at=now,
                updated_at=now,
            )
        )
        return await self._restore_recovery_idempotency(
            checkpoint,
            identity,
            expected_status=expected_status,
        )

    async def _restore_recovery_idempotency(
        self,
        checkpoint: RecoveryCheckpoint,
        identity: IdempotencyRecord,
        *,
        expected_status: IdempotencyStatus,
    ) -> IdempotencyRecord:
        recovery_idempotency = checkpoint.input.idempotency
        if (
            identity.runtime_domain is not RuntimeDomain.EXECUTION
            or identity.scope != recovery_idempotency.scope
            or identity.idempotency_key_digest != recovery_idempotency.idempotency_key_digest
            or identity.request_digest != recovery_idempotency.request_digest
            or identity.resource_kind is not ResourceKind.EXECUTION
            or identity.resource_id != checkpoint.execution_id
            or identity.tenant_id != checkpoint.tenant_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if identity.status is expected_status:
            return identity
        if expected_status is IdempotencyStatus.STARTED and identity.status is IdempotencyStatus.RESERVED:
            return await self._execution.idempotency.compare_and_swap(
                identity.scope,
                identity.idempotency_key_digest,
                tenant_id=identity.tenant_id,
                expected_status=identity.status,
                next_record=replace(
                    identity,
                    status=IdempotencyStatus.STARTED,
                    updated_at=checkpoint.updated_at,
                ),
            )
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _resolve_handoff_conversation(self, checkpoint: RecoveryCheckpoint, handoff: RecoveryTerminalHandoff) -> None:
        intent = handoff.conversation
        if intent is None:
            return
        session = await self._conversation.sessions.get(intent.session_id, tenant_id=checkpoint.tenant_id)
        if session is None:
            if self._conversation_durable:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            _logger.info("recovery conversation resolve skipped: session lost execution=%s", checkpoint.execution_id)
            return
        if self._step_reads.get(RuntimeDomain.CONVERSATION) is not self._steps:
            if handoff.source_step_run_id is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._step_lifecycle.materialize_from_recovery(
                target=RuntimeDomain.CONVERSATION,
                step_run_id=handoff.source_step_run_id,
            )
            target_snapshot = await self._step_reads[RuntimeDomain.CONVERSATION].latest_snapshot(
                run_id=handoff.source_step_run_id,
            )
            if target_snapshot is None or target_snapshot.state != "complete":
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if session.continuation == intent.next_cursor:
            return
        if session.status is SessionStatus.CLOSED:
            raise AIError(ErrorCode.SESSION_CONFLICT)
        if session.active_execution_id != checkpoint.execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if session.continuation != intent.expected_cursor:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            await self._conversation.sessions.advance_continuation(
                intent.session_id,
                tenant_id=checkpoint.tenant_id,
                execution_id=checkpoint.execution_id,
                expected=intent.expected_cursor,
                next_cursor=intent.next_cursor,
            )
        except AIError as error:
            if error.code not in {
                ErrorCode.STORAGE_CONFLICT,
                ErrorCode.STORAGE_INTEGRITY_ERROR,
            }:
                raise
            latest = await self._conversation.sessions.get(
                intent.session_id,
                tenant_id=checkpoint.tenant_id,
            )
            if latest is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if latest.continuation == intent.next_cursor:
                return
            if latest.status is SessionStatus.CLOSED:
                raise AIError(ErrorCode.SESSION_CONFLICT)
            if latest.active_execution_id != checkpoint.execution_id or latest.continuation != intent.expected_cursor:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            raise

    async def _advance_handoff(self, checkpoint: RecoveryCheckpoint, phase: RecoveryHandoffPhase) -> RecoveryCheckpoint:
        if checkpoint.terminal_handoff is None or checkpoint.handoff_contract_digest is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        current_rank = _HANDOFF_PHASE_RANK.get(checkpoint.handoff_phase)
        requested_rank = _HANDOFF_PHASE_RANK.get(phase)
        if current_rank is None or requested_rank is None or requested_rank < current_rank:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if requested_rank == current_rank:
            return checkpoint
        if requested_rank != current_rank + 1:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        updated = replace(
            checkpoint,
            handoff_phase=phase,
            state=RecoveryCheckpointState.HANDOFF,
            revision=checkpoint.revision + 1,
            updated_at=datetime.now(timezone.utc),
        )
        try:
            return await self._recovery.checkpoints.compare_and_swap(
                checkpoint.execution_id,
                tenant_id=checkpoint.tenant_id,
                expected_revision=checkpoint.revision,
                next_record=updated,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._recovery.checkpoints.get(checkpoint.execution_id, tenant_id=checkpoint.tenant_id)
            if current is None:
                raise
            same_handoff = (
                current.handoff_contract_digest == checkpoint.handoff_contract_digest
                and current.terminal_handoff == checkpoint.terminal_handoff
            )
            current_rank = _HANDOFF_PHASE_RANK.get(current.handoff_phase, -1)
            if not same_handoff or current_rank < requested_rank:
                raise
            return current

    async def _complete_handoff(self, checkpoint: RecoveryCheckpoint) -> None:
        if checkpoint.state is RecoveryCheckpointState.COMPLETED:
            return
        if checkpoint.handoff_phase is not RecoveryHandoffPhase.EXECUTION_COMMITTED:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        completed = replace(
            checkpoint,
            handoff_phase=RecoveryHandoffPhase.COMPLETED,
            state=RecoveryCheckpointState.COMPLETED,
            revision=checkpoint.revision + 1,
            updated_at=datetime.now(timezone.utc),
        )
        try:
            await self._recovery.checkpoints.compare_and_swap(
                checkpoint.execution_id,
                tenant_id=checkpoint.tenant_id,
                expected_revision=checkpoint.revision,
                next_record=completed,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._recovery.checkpoints.get(
                checkpoint.execution_id,
                tenant_id=checkpoint.tenant_id,
            )
            if current is None or current.state is not RecoveryCheckpointState.COMPLETED:
                raise

    async def _claim_session_or_recovery_finalizing(
        self,
        execution: ExecutionRecord,
    ) -> ExecutionRecord:
        if execution.session_id is None:
            return execution
        return await self._claim_session_finalizing(execution)

    async def _rewrite_prepared_success_handoff(
        self,
        checkpoint: RecoveryCheckpoint,
        *,
        target_status: ExecutionStatus,
        error_code: str,
        stop_reason: StopReason,
    ) -> RecoveryCheckpoint:
        handoff = checkpoint.terminal_handoff
        if (
            checkpoint.handoff_phase is not RecoveryHandoffPhase.PREPARED
            or handoff is None
            or handoff.outcome.terminal_status is not ExecutionStatus.SUCCEEDED
            or target_status not in {
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        outcome = handoff.outcome
        rewritten = RecoveryTerminalHandoff(
            RecoveryTerminalOutcome(
                terminal_status=target_status,
                error_code=error_code,
                safe_error_details={},
                stop_reason=stop_reason,
                output_schema_id=None,
                output_schema_revision=None,
                output_schema_fingerprint=None,
                recovery_object_ref=None,
                usage=outcome.usage,
                terminal_event_type=(
                    ExecutionEventType.EXECUTION_CANCELLED
                    if target_status is ExecutionStatus.CANCELLED
                    else ExecutionEventType.EXECUTION_FAILED
                ),
                terminal_event_payload=(
                    {"error_code": ErrorCode.EXECUTION_CANCELLED.value}
                    if target_status is ExecutionStatus.CANCELLED
                    else {"error_code": error_code}
                ),
                result_created_at=outcome.result_created_at,
            ),
            None,
            None,
        )
        updated = replace(
            checkpoint,
            terminal_handoff=rewritten,
            revision=checkpoint.revision + 1,
            updated_at=datetime.now(timezone.utc),
        )
        try:
            result = await self._recovery.checkpoints.compare_and_swap(
                checkpoint.execution_id,
                tenant_id=checkpoint.tenant_id,
                expected_revision=checkpoint.revision,
                next_record=updated,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._recovery.checkpoints.get(
                checkpoint.execution_id,
                tenant_id=checkpoint.tenant_id,
            )
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current.handoff_phase is not RecoveryHandoffPhase.PREPARED:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return current
        _logger.warning(
            "recovery success handoff rewritten: execution=%s status=%s",
            checkpoint.execution_id,
            target_status.value,
        )
        return result

    async def _commit_reconciled_terminal(
        self,
        checkpoint: RecoveryCheckpoint,
    ) -> None:
        handoff = checkpoint.terminal_handoff
        if handoff is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        outcome = handoff.outcome
        current = await self._execution.executions.get(
            checkpoint.execution_id,
            tenant_id=checkpoint.tenant_id,
        )
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if current.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            if current.status is not outcome.terminal_status:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        execution_ref = None
        if outcome.terminal_status is ExecutionStatus.SUCCEEDED:
            if outcome.recovery_object_ref is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            payload = await read_runtime_object(
                self._recovery_objects,
                outcome.recovery_object_ref,
            )
            execution_ref = await put_runtime_object(
                self._execution_objects,
                RuntimeObjectKeyFactory(self._namespace),
                RuntimeDomain.EXECUTION,
                current.tenant_id,
                payload,
            )
            if current.session_id is not None and current.status is not ExecutionStatus.FINALIZING:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        elif current.status not in {
            ExecutionStatus.PENDING_START,
            ExecutionStatus.STARTED,
            ExecutionStatus.FINALIZING,
            ExecutionStatus.CANCELLING,
        }:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        expected_idempotency_status = (
            IdempotencyStatus.RESERVED
            if current.status is ExecutionStatus.PENDING_START
            else IdempotencyStatus.STARTED
        )
        await self._ensure_recovery_idempotency(
            checkpoint,
            expected_status=expected_idempotency_status,
        )
        await self.verify_terminal_projection(
            current,
            outcome.terminal_status,
            handoff.source_step_run_id
            if outcome.terminal_status is ExecutionStatus.SUCCEEDED
            else None,
        )
        terminal = _terminal_record(
            current,
            outcome.terminal_status,
            outcome.result_created_at,
            error_code=outcome.error_code,
            safe_error_details=outcome.safe_error_details,
        )
        identity = await _terminal_idempotency(
            self._execution,
            current,
            outcome.terminal_status,
            None if execution_ref is None else execution_ref.digest,
            outcome.error_code,
        )
        result = ResultRecord(
            current.execution_id,
            current.tenant_id,
            outcome.output_schema_id,
            outcome.output_schema_revision,
            outcome.output_schema_fingerprint,
            execution_ref,
            outcome.stop_reason,
            outcome.usage,
            outcome.result_created_at,
        )
        try:
            await self._execution.executions.commit_terminal(
                ExecutionTerminalCommit(
                    current.revision,
                    current.event_sequence,
                    terminal,
                    result,
                    outcome.terminal_event_type,
                    dict(outcome.terminal_event_payload),
                    identity,
                )
            )
        except AIError as error:
            if error.code not in {
                ErrorCode.STORAGE_CONFLICT,
                ErrorCode.EXECUTION_RESULT_CONFLICT,
            }:
                raise
            latest = await self._execution.executions.get(
                current.execution_id,
                tenant_id=current.tenant_id,
            )
            if latest is None or latest.status is not outcome.terminal_status:
                raise
        _logger.info(
            "recovery execution terminal committed: execution=%s status=%s",
            current.execution_id,
            outcome.terminal_status.value,
        )

    async def _validate_committed_handoff(
        self,
        checkpoint: RecoveryCheckpoint,
    ) -> None:
        handoff = checkpoint.terminal_handoff
        if handoff is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        execution = await self._execution.executions.get(
            checkpoint.execution_id,
            tenant_id=checkpoint.tenant_id,
        )
        if (
            execution is None
            or execution.status is not handoff.outcome.terminal_status
            or execution.error_code != handoff.outcome.error_code
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result = await self._execution.executions.get_result(
            checkpoint.execution_id,
            tenant_id=checkpoint.tenant_id,
        )
        if result is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            result.stop_reason is not handoff.outcome.stop_reason
            or result.usage != handoff.outcome.usage
            or result.created_at != handoff.outcome.result_created_at
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if handoff.outcome.terminal_status is ExecutionStatus.SUCCEEDED:
            if (
                result.object_ref is None
                or result.output_schema_id != handoff.outcome.output_schema_id
                or result.output_schema_revision != handoff.outcome.output_schema_revision
                or result.output_schema_fingerprint != handoff.outcome.output_schema_fingerprint
                or handoff.outcome.recovery_object_ref is None
                or result.object_ref.digest != handoff.outcome.recovery_object_ref.digest
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        elif result.object_ref is not None or any(
            value is not None
            for value in (
                result.output_schema_id,
                result.output_schema_revision,
                result.output_schema_fingerprint,
            )
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        records = await self._execution.idempotency.list_by_resource(
            ResourceKind.EXECUTION,
            checkpoint.execution_id,
            tenant_id=checkpoint.tenant_id,
        )
        if len(records) != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        expected_status = (
            IdempotencyStatus.COMPLETED
            if handoff.outcome.terminal_status is ExecutionStatus.SUCCEEDED
            else IdempotencyStatus.CANCELLED
            if handoff.outcome.terminal_status is ExecutionStatus.CANCELLED
            else IdempotencyStatus.FAILED
        )
        if records[0].status is not expected_status or records[0].error_code != handoff.outcome.error_code:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def close(self) -> None:
        self._accepting = False
        tasks = tuple(self._tasks.items())
        cancel_tasks: list[asyncio.Task[None]] = []
        for execution_id, task in tasks:
            current = await self._execution.executions.get(
                execution_id,
                tenant_id=self._tenant_id,
            )
            if current is None or current.status is not ExecutionStatus.FINALIZING:
                task.cancel()
            else:
                _logger.info(
                    "close waiting for execution finalization: execution=%s",
                    execution_id,
                )
            cancel_tasks.append(task)
        if cancel_tasks:
            await asyncio.gather(
                *cancel_tasks,
                return_exceptions=True,
            )
        for execution_id, task in tasks:
            self._task_done(execution_id, task)
        self._tasks.clear()
        self._captured_usage.clear()
        self._worker_failures.clear()

    async def _run(self, request: ExecutionRequest, original: ExecutionRecord) -> None:
        execution_id = original.execution_id
        checkpoint: RecoveryCheckpoint | None = None
        run_id: str | None = None
        recovery_history_run_id: str | None = None
        operation_started_at = monotonic()
        operation_result = "failure"
        try:
            current = await self._execution.executions.get(execution_id, tenant_id=original.tenant_id)
            if current is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = await self._execution.executions.claim_next_agent_run(
                execution_id,
                tenant_id=current.tenant_id,
                expected_revision=current.revision,
                expected_agent_run_sequence=current.agent_run_sequence,
            )
            definition = self._catalog.definition(current.binding_digest)
            run_id = step_run_id(
                namespace=self._namespace,
                tenant_id=current.tenant_id,
                execution_id=execution_id,
                segment_sequence=current.agent_run_sequence,
            )
            conversation_id = step_conversation_id(
                namespace=self._namespace,
                tenant_id=current.tenant_id,
                execution_id=execution_id,
            )
            if self._recovery_enabled:
                checkpoint = await self._recovery.checkpoints.get(
                    execution_id,
                    tenant_id=current.tenant_id,
                )
                if checkpoint is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if checkpoint.state is RecoveryCheckpointState.ADMITTED:
                    pass
                elif checkpoint.state in {
                    RecoveryCheckpointState.ACTIVE,
                    RecoveryCheckpointState.WAITING,
                }:
                    recovery_history_run_id = checkpoint.step_run_id
                    if recovery_history_run_id is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    snapshot = await self._step_store(RuntimeDomain.RECOVERY).latest_snapshot(
                        run_id=recovery_history_run_id,
                        include_interrupted=True,
                    )
                    if snapshot is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                else:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                checkpoint = await self._activate_recovery_attempt(
                    checkpoint,
                    run_id,
                    current.agent_run_sequence,
                )
            if recovery_history_run_id is None:
                history = await self._history(current)
            else:
                history = await self._history(
                    current,
                    recovery_run_id=recovery_history_run_id,
                )

            async def sink(event_type: ExecutionEventType, payload: JsonValue) -> None:
                await self._append_event(current, event_type, payload)

            memory = None
            platform_tool_names = select_platform_tool_names(
                allow_tools=definition.spec.allow_tools,
                memory_scope=current.memory_scope,
                subagent_available=current.parent_execution_id is None and self._subagent_dispatcher is not None,
            )
            selected_memory = tuple(name for name in platform_tool_names if name in MEMORY_TOOL_NAMES)
            if selected_memory:
                if self._memory_store_factory is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                memory = self._memory_store_factory(
                    current.tenant_id,
                    current.execution_id,
                    current.memory_scope or "default",
                )
            try:
                result = await self._executor.execute(
                    definition,
                    request.user_prompt,
                    history,
                    conversation_id,
                    step_store=self._steps,
                    step_run_id=run_id,
                    segment_sequence=current.agent_run_sequence,
                    capability_context=CapabilityMaterializationContext(
                        request.principal,
                        ResourceRef(ResourceKind.EXECUTION, execution_id, current.tenant_id),
                        self._execution_root,
                        definition.spec.allow_tools,
                        definition.spec.allow_skills,
                    ),
                    memory_scope=current.memory_scope,
                    memory_store=memory,
                    platform_tool_names=platform_tool_names,
                    subagent_delegate=(
                        None
                        if current.parent_execution_id is not None
                        or self._subagent_dispatcher is None
                        else self._subagent_dispatcher.delegate_for(
                            parent_execution_id=current.execution_id,
                            root_execution_id=current.root_execution_id,
                            memory_scope=current.memory_scope,
                            principal=request.principal,
                        )
                    ),
                    event_sink=sink,
                    usage_sink=lambda usage: self._capture_usage(execution_id, usage),
                )
            except Exception as error:
                if _is_infrastructure_error(error):
                    raise
                current = await self._execution.executions.get(
                    execution_id,
                    tenant_id=original.tenant_id,
                )
                if current is not None and current.status not in {
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                }:
                    await self._commit_failure(current, error, run_id=run_id)
                persisted = await self._execution.executions.get(
                    execution_id,
                    tenant_id=original.tenant_id,
                )
                if persisted is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                operation_result = _execution_operation_result(persisted.status)
                _logger.error(
                    "local execution failed: execution=%s",
                    execution_id,
                    exc_info=True,
                )
                return
            await self._commit_success(
                current,
                definition,
                result.output,
                result.usage,
                run_id,
            )
            persisted = await self._execution.executions.get(
                execution_id,
                tenant_id=original.tenant_id,
            )
            if persisted is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            operation_result = _execution_operation_result(persisted.status)
            _logger.debug("local execution completed: execution=%s run=%s", execution_id, run_id)
        except asyncio.CancelledError:
            current = await self._execution.executions.get(execution_id, tenant_id=original.tenant_id)
            if current is not None and current.status is ExecutionStatus.FINALIZING:
                raise
            if current is not None and current.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                await self._commit_terminal(current, ExecutionStatus.CANCELLED, None, ErrorCode.EXECUTION_CANCELLED.value, StopReason.CANCELLED, run_id=run_id)
            persisted = await self._execution.executions.get(
                execution_id,
                tenant_id=original.tenant_id,
            )
            if persisted is not None:
                operation_result = _execution_operation_result(persisted.status)
            raise
        except Exception:
            _logger.error(
                "local execution infrastructure failure: execution=%s",
                execution_id,
                exc_info=True,
            )
            raise
        finally:
            self._metrics.operation("execution", "runtime", operation_result, operation_started_at)

    async def _finish_checkpoint(self, checkpoint: RecoveryCheckpoint) -> None:
        if checkpoint.state is RecoveryCheckpointState.COMPLETED:
            return
        if checkpoint.terminal_handoff is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        updated = replace(
            checkpoint,
            state=RecoveryCheckpointState.COMPLETED,
            handoff_phase=RecoveryHandoffPhase.NONE,
            handoff_contract_digest=checkpoint.handoff_contract_digest,
            revision=checkpoint.revision + 1,
            updated_at=datetime.now(timezone.utc),
        )
        try:
            await self._recovery.checkpoints.compare_and_swap(
                checkpoint.execution_id,
                tenant_id=checkpoint.tenant_id,
                expected_revision=checkpoint.revision,
                next_record=updated,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._recovery.checkpoints.get(checkpoint.execution_id, tenant_id=checkpoint.tenant_id)
            if current is None or current.state is not RecoveryCheckpointState.COMPLETED:
                raise

    async def _activate_recovery_attempt(
        self,
        checkpoint: RecoveryCheckpoint,
        run_id: str,
        agent_run_sequence: int,
    ) -> RecoveryCheckpoint:
        updated = replace(
            checkpoint,
            state=RecoveryCheckpointState.ACTIVE,
            step_run_id=run_id,
            agent_run_sequence=agent_run_sequence,
            pending_operation_id=None,
            revision=checkpoint.revision + 1,
            updated_at=datetime.now(timezone.utc),
        )
        try:
            await self._recovery.checkpoints.compare_and_swap(
                checkpoint.execution_id,
                tenant_id=checkpoint.tenant_id,
                expected_revision=checkpoint.revision,
                next_record=updated,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._recovery.checkpoints.get(
                checkpoint.execution_id,
                tenant_id=checkpoint.tenant_id,
            )
            if (
                current is None
                or current.state is not RecoveryCheckpointState.ACTIVE
                or current.step_run_id != run_id
                or current.agent_run_sequence != agent_run_sequence
            ):
                raise
            return current
        return updated

    async def _history(
        self,
        execution: ExecutionRecord,
        recovery_run_id: str | None = None,
    ) -> list[object]:
        execution_steps = self._step_store(RuntimeDomain.EXECUTION)
        conversation_steps = self._step_store(RuntimeDomain.CONVERSATION)
        if recovery_run_id is not None:
            try:
                return list(
                    await continue_run(
                        self._step_store(RuntimeDomain.RECOVERY),
                        run_id=recovery_run_id,
                        include_interrupted=True,
                    )
                )
            except LookupError as error:
                raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE) from error
        if execution.conversation_step_run_id is not None:
            try:
                return list(await continue_run(conversation_steps, run_id=execution.conversation_step_run_id, include_interrupted=True))
            except LookupError as error:
                raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE) from error
        if execution.base_execution_id is None:
            return []
        base = await self._execution.executions.get(execution.base_execution_id, tenant_id=execution.tenant_id)
        if base is None or base.agent_run_sequence < 1:
            raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
        run_id = step_run_id(
            namespace=self._namespace,
            tenant_id=execution.tenant_id,
            execution_id=base.execution_id,
            segment_sequence=base.agent_run_sequence,
        )
        try:
            if execution.lineage_kind is ExecutionLineageKind.FORK:
                return list(await fork_run(execution_steps, run_id=run_id))
            return list(await continue_run(execution_steps, run_id=run_id))
        except LookupError as error:
            raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE) from error

    async def _append_event(self, execution: ExecutionRecord, event_type: ExecutionEventType, payload: JsonValue) -> None:
        current = await self._execution.executions.get(execution.execution_id, tenant_id=execution.tenant_id)
        if current is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        try:
            await self._execution.events.append(
                execution.execution_id,
                tenant_id=execution.tenant_id,
                expected_sequence=current.event_sequence,
                event_type=event_type,
                payload=payload,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            latest = await self._execution.executions.get(execution.execution_id, tenant_id=execution.tenant_id)
            if latest is None:
                raise
            await self._execution.events.append(
                execution.execution_id,
                tenant_id=execution.tenant_id,
                expected_sequence=latest.event_sequence,
                event_type=event_type,
                payload=payload,
            )

    def _step_store(self, runtime_domain: RuntimeDomain) -> StepStore:
        return self._step_reads[runtime_domain]

    async def _commit_success(
        self,
        execution: ExecutionRecord,
        definition: AgentDefinition,
        output: JsonValue,
        usage: UsageMetrics,
        run_id: str,
    ) -> None:
        payload = canonical_json_bytes(output)
        object_ref = await put_runtime_object(
            self._execution_objects,
            RuntimeObjectKeyFactory(self._namespace),
            RuntimeDomain.EXECUTION,
            execution.tenant_id,
            payload,
        )
        current = await self._execution.executions.get(execution.execution_id, tenant_id=execution.tenant_id)
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if current.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            await self._release_session_admission_best_effort(current)
            return
        if self._recovery_enabled:
            await self._commit_terminal(
                current,
                ExecutionStatus.SUCCEEDED,
                object_ref,
                None,
                StopReason.END_TURN,
                definition=definition,
                output_digest=object_ref.digest,
                run_id=run_id,
                usage=usage,
            )
            return
        if current.status is ExecutionStatus.CANCELLING:
            return
        await self.verify_terminal_projection(current, ExecutionStatus.SUCCEEDED, run_id)
        if current.session_id is not None:
            expected_cursor = await self._expected_session_cursor(current)
            current = await self._claim_session_finalizing(current)
            if current.status is ExecutionStatus.CANCELLING:
                _logger.info(
                    "session success lost to cancellation: execution=%s",
                    current.execution_id,
                )
                return
            if current.status in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }:
                return
            await self._commit_session_conversation(
                current,
                source_run_id=run_id,
                expected_cursor=expected_cursor,
            )
        await self._commit_terminal(
            current,
            ExecutionStatus.SUCCEEDED,
            object_ref,
            None,
            StopReason.END_TURN,
            definition=definition,
            output_digest=object_ref.digest,
            run_id=run_id,
            usage=usage,
        )
        return

    async def _expected_session_cursor(
        self,
        execution: ExecutionRecord,
    ) -> ConversationCursor | None:
        if execution.conversation_step_run_id is not None:
            return ConversationCursor(execution.conversation_step_run_id)
        if execution.base_execution_id is None:
            return None
        base = await self._execution.executions.get(
            execution.base_execution_id,
            tenant_id=execution.tenant_id,
        )
        if base is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if base.agent_run_sequence == 0:
            return None
        return ConversationCursor(
            step_run_id(
                namespace=self._namespace,
                tenant_id=execution.tenant_id,
                execution_id=base.execution_id,
                segment_sequence=base.agent_run_sequence,
            )
        )

    async def _claim_session_finalizing(
        self,
        execution: ExecutionRecord,
    ) -> ExecutionRecord:
        if execution.session_id is None or execution.status is not ExecutionStatus.STARTED:
            if execution.status in {
                ExecutionStatus.FINALIZING,
                ExecutionStatus.CANCELLING,
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }:
                return execution
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        finalizing = replace(
            execution,
            status=ExecutionStatus.FINALIZING,
            revision=execution.revision + 1,
            updated_at=datetime.now(timezone.utc),
        )
        try:
            updated = await self._execution.executions.compare_and_swap(
                execution.execution_id,
                tenant_id=execution.tenant_id,
                expected_revision=execution.revision,
                next_record=finalizing,
            )
            _logger.debug(
                "session execution finalization claimed: execution=%s",
                execution.execution_id,
            )
            return updated
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._execution.executions.get(
                execution.execution_id,
                tenant_id=execution.tenant_id,
            )
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current.status in {
                ExecutionStatus.FINALIZING,
                ExecutionStatus.CANCELLING,
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }:
                return current
            raise

    async def _commit_session_conversation(
        self,
        execution: ExecutionRecord,
        *,
        source_run_id: str,
        expected_cursor: ConversationCursor | None,
    ) -> None:
        snapshot = await self._steps.latest_snapshot(run_id=source_run_id)
        if snapshot is None or snapshot.state != "complete":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._step_lifecycle.materialize_conversation(step_run_id=source_run_id)
        session = await self._conversation.sessions.get(
            execution.session_id or "",
            tenant_id=execution.tenant_id,
        )
        if session is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        next_cursor = ConversationCursor(source_run_id)
        if session.continuation == next_cursor:
            return
        if session.status is SessionStatus.CLOSED:
            raise AIError(ErrorCode.SESSION_CONFLICT)
        if session.active_execution_id != execution.execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if session.continuation != expected_cursor:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            await self._conversation.sessions.advance_continuation(
                execution.session_id or "",
                tenant_id=execution.tenant_id,
                execution_id=execution.execution_id,
                expected=expected_cursor,
                next_cursor=next_cursor,
            )
        except AIError as error:
            if error.code not in {
                ErrorCode.STORAGE_CONFLICT,
                ErrorCode.STORAGE_INTEGRITY_ERROR,
            }:
                raise
            latest = await self._conversation.sessions.get(
                execution.session_id or "",
                tenant_id=execution.tenant_id,
            )
            if latest is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if latest.continuation == next_cursor:
                return
            if latest.status is SessionStatus.CLOSED:
                raise AIError(ErrorCode.SESSION_CONFLICT)
            if latest.active_execution_id != execution.execution_id or latest.continuation != expected_cursor:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            raise
        _logger.info(
            "session conversation committed: execution=%s run=%s",
            execution.execution_id,
            source_run_id,
        )

    async def _commit_failure(self, execution: ExecutionRecord, error: Exception, *, run_id: str | None = None) -> None:
        code = ErrorCode.OUTPUT_VALIDATION_FAILED if isinstance(error, ValidationError) else error.code if isinstance(error, AIError) else ErrorCode.EXECUTION_FAILED
        await self._commit_terminal(execution, ExecutionStatus.FAILED, None, code.value, StopReason.ERROR, run_id=run_id)

    async def _commit_terminal(
        self,
        execution: ExecutionRecord,
        status: ExecutionStatus,
        object_ref: "ObjectRef | None",
        error_code: str | None,
        stop_reason: StopReason,
        *,
        definition: AgentDefinition | None = None,
        output_digest: str | None = None,
        run_id: str | None = None,
        usage: UsageMetrics | None = None,
    ) -> None:
        current = await self._execution.executions.get(
            execution.execution_id,
            tenant_id=execution.tenant_id,
        )
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if current.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            await self._release_session_admission_best_effort(current)
            return
        now = datetime.now(timezone.utc)
        captured_usage = usage or self._captured_usage.get(execution.execution_id, UsageMetrics())
        if definition is None:
            schema_id, schema_revision, schema_fingerprint = "none", 1, "none"
        else:
            schema_id = definition.spec.output_schema
            schema_revision = definition.spec.output_schema_revision
            schema_fingerprint = definition.output_schema_fingerprint
        if self._recovery_enabled:
            recovery_checkpoint = await self._recovery.checkpoints.get(
                current.execution_id,
                tenant_id=current.tenant_id,
            )
            if recovery_checkpoint is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if recovery_checkpoint.handoff_phase is RecoveryHandoffPhase.NONE:
                if status is ExecutionStatus.SUCCEEDED and run_id is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if run_id is not None:
                    await self._step_lifecycle.materialize_recovery_snapshot(
                        step_run_id=run_id,
                        require_complete=status is ExecutionStatus.SUCCEEDED,
                    )
                recovery_ref = None
                if object_ref is not None:
                    content = await read_runtime_object(
                        self._execution_objects,
                        object_ref,
                    )
                    recovery_ref = await put_runtime_object(
                        self._recovery_objects,
                        RuntimeObjectKeyFactory(self._namespace),
                        RuntimeDomain.RECOVERY,
                        current.tenant_id,
                        content,
                    )
                conversation = None
                if status is ExecutionStatus.SUCCEEDED and current.session_id is not None and run_id is not None:
                    conversation = RecoveryConversationIntent(
                        current.session_id,
                        await self._expected_session_cursor(current),
                        ConversationCursor(run_id),
                    )
                terminal_event_type = (
                    ExecutionEventType.EXECUTION_SUCCEEDED
                    if status is ExecutionStatus.SUCCEEDED
                    else ExecutionEventType.EXECUTION_CANCELLED
                    if status is ExecutionStatus.CANCELLED
                    else ExecutionEventType.EXECUTION_FAILED
                )
                handoff = RecoveryTerminalHandoff(
                    RecoveryTerminalOutcome(
                        terminal_status=status,
                        error_code=error_code,
                        safe_error_details={},
                        stop_reason=stop_reason,
                        output_schema_id=schema_id if status is ExecutionStatus.SUCCEEDED else None,
                        output_schema_revision=schema_revision if status is ExecutionStatus.SUCCEEDED else None,
                        output_schema_fingerprint=schema_fingerprint if status is ExecutionStatus.SUCCEEDED else None,
                        recovery_object_ref=recovery_ref,
                        usage=captured_usage,
                        terminal_event_type=terminal_event_type,
                        terminal_event_payload=(
                            {"run_id": run_id}
                            if status is ExecutionStatus.SUCCEEDED and run_id is not None
                            else {"error_code": error_code}
                        ),
                        result_created_at=now,
                    ),
                    run_id or recovery_checkpoint.step_run_id,
                    conversation,
                )
                if self._handoff_contract_digest is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                prepared = replace(
                    recovery_checkpoint,
                    state=RecoveryCheckpointState.HANDOFF,
                    handoff_phase=RecoveryHandoffPhase.PREPARED,
                    terminal_handoff=handoff,
                    handoff_contract_digest=self._handoff_contract_digest,
                    revision=recovery_checkpoint.revision + 1,
                    updated_at=now,
                )
                try:
                    recovery_checkpoint = await self._recovery.checkpoints.compare_and_swap(
                        current.execution_id,
                        tenant_id=current.tenant_id,
                        expected_revision=recovery_checkpoint.revision,
                        next_record=prepared,
                    )
                except AIError as error:
                    if error.code is not ErrorCode.STORAGE_CONFLICT:
                        raise
                    recovery_checkpoint = await self._recovery.checkpoints.get(
                        current.execution_id,
                        tenant_id=current.tenant_id,
                    )
                    if recovery_checkpoint is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                _logger.info(
                    "recovery handoff prepared: execution=%s run=%s",
                    current.execution_id,
                    run_id,
                )
            await self._reconcile_handoff(recovery_checkpoint)
            return
        await self.verify_terminal_projection(
            current,
            status,
            run_id if status is ExecutionStatus.SUCCEEDED else None,
        )
        terminal = _terminal_record(current, status, now, error_code=error_code)
        identity = await _terminal_idempotency(
            self._execution,
            current,
            status,
            output_digest,
            error_code,
        )
        await self._execution.executions.commit_terminal(
            ExecutionTerminalCommit(
                current.revision,
                current.event_sequence,
                terminal,
                ResultRecord(
                    current.execution_id,
                    current.tenant_id,
                    schema_id if status is ExecutionStatus.SUCCEEDED else None,
                    schema_revision if status is ExecutionStatus.SUCCEEDED else None,
                    schema_fingerprint if status is ExecutionStatus.SUCCEEDED else None,
                    object_ref if status is ExecutionStatus.SUCCEEDED else None,
                    stop_reason,
                    captured_usage,
                    now,
                ),
                (
                    ExecutionEventType.EXECUTION_SUCCEEDED
                    if status is ExecutionStatus.SUCCEEDED
                    else ExecutionEventType.EXECUTION_CANCELLED
                    if status is ExecutionStatus.CANCELLED
                    else ExecutionEventType.EXECUTION_FAILED
                ),
                (
                    {"run_id": run_id}
                    if status is ExecutionStatus.SUCCEEDED and run_id is not None
                    else {"error_code": error_code}
                ),
                identity,
            )
        )
        _logger.info(
            "execution terminal committed: execution=%s status=%s",
            current.execution_id,
            status.value,
        )
        await self._release_session_admission_best_effort(current)

    async def _release_session_admission_best_effort(
        self,
        execution: ExecutionRecord,
    ) -> None:
        if execution.session_id is None:
            return
        try:
            await self._conversation.sessions.release_execution(
                execution.session_id,
                tenant_id=execution.tenant_id,
                execution_id=execution.execution_id,
            )
        except BaseException:
            _logger.error(
                "session admission release failed after terminal: execution=%s",
                execution.execution_id,
                exc_info=environ.debug,
            )

    async def _capture_usage(self, execution_id: str, usage: UsageMetrics) -> None:
        self._captured_usage[execution_id] = usage
        _logger.debug(
            "execution usage captured: execution=%s requests=%s tool_calls=%s total_tokens=%s",
            execution_id,
            usage.model_requests,
            usage.tool_calls,
            usage.total_tokens,
        )

    async def verify_terminal_projection(self, execution: ExecutionRecord, status: ExecutionStatus, run_id: "str | None") -> None:
        candidates = tuple(
            step_run_id(
                namespace=self._namespace,
                tenant_id=execution.tenant_id,
                execution_id=execution.execution_id,
                segment_sequence=sequence,
            )
            for sequence in range(1, execution.agent_run_sequence + 1)
        )
        await self._step_lifecycle.verify_terminal_attempts(
            candidate_step_run_ids=candidates,
            required_step_run_id=run_id if status is ExecutionStatus.SUCCEEDED else None,
        )

def _terminal_record(
    record: ExecutionRecord,
    status: ExecutionStatus,
    now: datetime,
    *,
    error_code: str | None,
    safe_error_details: Mapping[str, JsonValue] | None = None,
) -> ExecutionRecord:
    return replace(
        record,
        status=status,
        revision=record.revision + 1,
        event_sequence=record.event_sequence + 1,
        error_code=error_code,
        safe_error_details={} if safe_error_details is None else safe_error_details,
        updated_at=now,
    )


async def _terminal_idempotency(state: ExecutionState, execution: ExecutionRecord, status: ExecutionStatus, result_digest: str | None, error_code: str | None) -> IdempotencyTerminalUpdate | None:
    records = await state.idempotency.list_by_resource(ResourceKind.EXECUTION, execution.execution_id, tenant_id=execution.tenant_id)
    if len(records) > 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if not records:
        return None
    identity = records[0]
    next_status = IdempotencyStatus.COMPLETED if status is ExecutionStatus.SUCCEEDED else IdempotencyStatus.CANCELLED if status is ExecutionStatus.CANCELLED else IdempotencyStatus.FAILED
    return IdempotencyTerminalUpdate(
        identity.scope,
        identity.idempotency_key_digest,
        identity.status,
        next_status,
        identity.request_digest,
        result_digest,
        error_code,
    )


def _admission_matches(existing: RecoveryCheckpoint, candidate: RecoveryCheckpoint) -> bool:
    return (
        existing.execution_id == candidate.execution_id
        and existing.tenant_id == candidate.tenant_id
        and existing.input == candidate.input
        and existing.step_run_id is None
        and existing.agent_run_sequence == candidate.agent_run_sequence
        and existing.state is RecoveryCheckpointState.ADMITTED
        and existing.handoff_phase is RecoveryHandoffPhase.NONE
        and existing.terminal_handoff is None
        and existing.handoff_contract_digest is None
        and existing.pending_operation_id is None
    )


def _is_infrastructure_error(error: Exception) -> bool:
    if not isinstance(error, AIError):
        return False
    return error.code.value.startswith("STORAGE_") or error.code in {
        ErrorCode.AGENT_DEFINITION_UNAVAILABLE,
        ErrorCode.EXECUTION_HISTORY_UNAVAILABLE,
        ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
        ErrorCode.SERVICE_NOT_READY,
    }


def _execution_operation_result(status: ExecutionStatus) -> str:
    if status is ExecutionStatus.SUCCEEDED:
        return "success"
    if status is ExecutionStatus.FAILED:
        return "failure"
    if status in {ExecutionStatus.CANCELLED, ExecutionStatus.CANCELLING}:
        return "cancelled"
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


__all__ = ["LocalExecutionBackend"]
