#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local ExecutionBackend backed by AgentExecutor and durable persistence."""

import asyncio
import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Protocol, TypeVar, cast

from linktools.core import environ
from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.tools import DeferredToolResults, ToolApproved, ToolDenied
from pydantic_ai_harness.memory import SearchableMemoryStore
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    RunRecord,
    StepEvent,
    StepStore,
    ToolEffectRecord,
    continue_run,
    fork_run,
)

from ..agent import AgentBinding, AgentCatalog, SubagentRef
from ..capability import RunContext, SubagentDelegate
from ..workspace import RepositoryInstructions, Workspace
from ._agent_executor import (
    AgentExecutionPaused,
    AgentExecutor,
    DurableBoundary,
    LiveDelta,
    PendingToolApproval,
    _RunScope,
)
from ._capabilities import MEMORY_TOOL_NAMES, select_runtime_tool_names
from ._input import UserPromptTransport, user_prompt_transport
from ._plan import RuntimePlanStore
from ..core import (
    ApprovalStatus,
    ExecutionEventType,
    ExecutionMode,
    ExecutionLineageKind,
    ExecutionStatus,
    IdempotencyStatus,
    JsonValue,
    OperationKind,
    OperationLedgerInput,
    OperationStatus,
    Principal,
    ResourceKind,
    ResourceRef,
    SessionStatus,
    StopReason,
    ToolOperationStatus,
    UsageMetrics,
    canonical_json_bytes,
    canonical_sha256,
    normalize_json_value,
    step_conversation_id,
    step_run_id,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode
from ..storage import (
    ObjectStore,
    PayloadPolicy,
    StorageMetrics,
    StoredPayload,
    payload_fits_inline,
)
from ._event import ExecutionDelta, LiveExecutionEventBroker
from ._execution import CancelEffectOutcome, ExecutionStartIdentity
from ._object import RuntimeObjectKeyFactory, put_runtime_object, read_runtime_object
from ._tool import RuntimeToolOperationBridge, _ToolOperationRuntimeRepository
from .service_api import ExecutionRequest, ToolApprovalContext
from .state import (
    ConversationState,
    ConversationStateCommands,
    ExecutionRepositoryImpl,
    ExecutionState,
    ExecutionTerminalSealPlan,
    PendingApprovalContinuation,
    RecoveryState,
    RuntimeStateCommands,
    RuntimeStepStore,
    SessionRepositoryImpl,
    StateStepArchive,
    ToolApprovalAdmission,
)
from .state._contracts import (
    AgentAttemptClaim,
    ApprovalRecord,
    ConversationCursor,
    ExecutionCancelRequestCommit,
    ExecutionEventAppend,
    ExecutionRecord,
    ExecutionStartClaim,
    ExecutionTerminalCommit,
    ExecutionTerminalCommitResult,
    IdempotencyRecord,
    LoadedModelContext,
    RecoveryCheckpoint,
    RecoveryCheckpointState,
    RecoveryConversationIntent,
    RecoveryExecutionInput,
    RecoveryHandoffPhase,
    RecoveryIdempotencyInput,
    RecoveryTerminalHandoff,
    RecoveryTerminalOutcome,
    ResultRecord,
    RuntimePayloadRef,
)
from .state._plan import RuntimeDomain
from .state._repositories import (
    ConversationHistoryRepositoryImpl,
    EventRepositoryImpl,
    OperationLedgerRepository,
    RecoveryCheckpointRepositoryImpl,
    ToolRepositoryImpl,
)


class _SubagentDispatcher(Protocol):
    @property
    def pending_background_tasks(self) -> tuple[asyncio.Task[object], ...]: ...

    @property
    def background_failure(self) -> "AIError | None": ...

    def delegate_for(
        self,
        *,
        parent_execution_id: str,
        root_execution_id: str,
        memory_scope: "str | None",
        principal: Principal,
        refs: "tuple[SubagentRef, ...]",
        mode: ExecutionMode,
    ) -> SubagentDelegate: ...

    def descriptions_for(
        self,
        refs: "tuple[SubagentRef, ...]",
    ) -> "dict[str, str | None]": ...


_logger = environ.get_logger("ai.runtime.local")


_CheckpointT = TypeVar("_CheckpointT")


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


@dataclass(frozen=True, slots=True)
class _RepositoryInstructionAttemptProvenance:
    run_id: str
    messages: tuple[ModelMessage, ...]
    marker_authority: frozenset[tuple[str, str]]


class _RepositoryInstructionProvenanceCache:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.initialized = False
        self.final_attempts: dict[int, _RepositoryInstructionAttemptProvenance] = {}


@dataclass(frozen=True, slots=True)
class _RepositoryInstructionProvenance:
    messages: tuple[ModelMessage, ...]
    marker_authority: frozenset[tuple[str, str]]


@dataclass(frozen=True, slots=True)
class _ApprovalBatchCall:
    approval_id: str
    operation_id: str
    tool_call_id: str
    tool_name: str
    arguments: JsonValue
    args_digest: str


class _StepLifecycle(Protocol):
    async def materialize_conversation(self, *, step_run_id: str) -> None: ...
    async def materialize_from_recovery(
        self,
        *,
        target: RuntimeDomain,
        step_run_id: str,
        execution_id: "str | None" = None,
    ) -> None: ...
    async def materialize_recovery_snapshot(self, *, step_run_id: str, require_complete: bool) -> None: ...
    async def verify_terminal_attempts(self, *, candidate_step_run_ids: tuple[str, ...], required_step_run_id: str | None) -> None: ...
    async def release_staging_many(
        self,
        *,
        candidate_step_run_ids: tuple[str, ...],
        execution_id: "str | None" = None,
    ) -> None: ...
    async def flush_execution_projection(self, step_run_id: str, *, execution_id: str) -> None: ...
    async def wait_projection_flight(self, step_run_id: str) -> None: ...
    async def prepare_execution_terminal_seal(
        self,
        *,
        execution_id: str,
        run_ids: Sequence[str],
        binding_digest: str,
    ) -> ExecutionTerminalSealPlan: ...
    async def finalize_execution_terminal_seal(
        self,
        plan: ExecutionTerminalSealPlan,
    ) -> None: ...
    async def discard_execution_terminal_seal(
        self,
        plan: ExecutionTerminalSealPlan,
    ) -> None: ...


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
        catalog: AgentCatalog,
        *,
        workspace: Workspace,
        app: object,
        tenant_id: str,
        step_reads: Mapping[RuntimeDomain, StepStore],
        step_lifecycle: _StepLifecycle,
        memory_store_factory: "Callable[[str, str, str], SearchableMemoryStore] | None" = None,
        recovery_enabled: bool = False,
        conversation_durable: bool = False,
        handoff_contract_digest: "str | None" = None,
        subagent_dispatcher: "_SubagentDispatcher | None" = None,
        live_broker: "LiveExecutionEventBroker | None" = None,
        payload_policy: "PayloadPolicy | None" = None,
        execution_objects_durable: bool = True,
        tool_operations: "_ToolOperationRuntimeRepository | None" = None,
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
        self._workspace = workspace
        self._app = app
        self._tenant_id = validate_tenant_id(tenant_id)
        self._memory_store_factory = memory_store_factory
        self._recovery_enabled = recovery_enabled
        self._conversation_durable = conversation_durable
        self._handoff_contract_digest = handoff_contract_digest
        self._subagent_dispatcher = subagent_dispatcher
        self._live_broker = live_broker or LiveExecutionEventBroker()
        self._payload_policy = payload_policy or PayloadPolicy()
        self._tool_operations = tool_operations
        self._execution_objects_durable = execution_objects_durable
        self._step_reads = dict(step_reads)
        if frozenset(self._step_reads) != frozenset({RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY}):
            raise ValueError("step_reads must contain exactly the three Step owner domains")
        self._step_lifecycle = step_lifecycle
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._worker_failures: dict[str, _WorkerFailure] = {}
        self._captured_usage: dict[str, UsageMetrics] = {}
        self._terminal_events: dict[str, asyncio.Event] = {}
        self._pending_audit_events: dict[str, list[ExecutionEventAppend]] = {}
        self._pending_audit_locks: dict[str, asyncio.Lock] = {}
        self._recovery_relaunch_ids: set[str] = set()
        self._approval_pause_segments: dict[str, asyncio.Task[None]] = {}
        self._segment_only_worker_exits: set[str] = set()
        self._repository_instruction_provenance: dict[
            str, _RepositoryInstructionProvenanceCache
        ] = {}
        self._checkpoint_tasks: set[asyncio.Task[object]] = set()
        self._execution_durable_tasks: dict[
            str,
            set[asyncio.Task[object]],
        ] = {}
        self._worker_cancel_requests: set[str] = set()
        self._accepting = True
        execution_steps = self._step_reads[RuntimeDomain.EXECUTION]
        conversation_steps = self._step_reads[RuntimeDomain.CONVERSATION]
        execution_repository = cast(ExecutionRepositoryImpl, self._execution.executions)
        session_repository = cast(SessionRepositoryImpl, self._conversation.sessions)
        self._session_state_store = session_repository.state_store
        self._execution_state_store = execution_repository.state_store
        self._conversation_commands = ConversationStateCommands(
            session_repository.state_store,
            session_repository,
            conversation_steps if isinstance(conversation_steps, StateStepArchive) else None,
            cast(ConversationHistoryRepositoryImpl, self._conversation.histories),
        )
        self._runtime_commands = RuntimeStateCommands(
            execution_repository,
            namespace=self._namespace,
            events=cast(EventRepositoryImpl, self._execution.events),
            operations=cast(OperationLedgerRepository, self._execution.operations),
            approvals=self._recovery.approvals,
            conversation=session_repository,
            recovery=cast(RecoveryCheckpointRepositoryImpl, self._recovery.checkpoints),
            conversation_history=cast(
                ConversationHistoryRepositoryImpl,
                self._conversation.histories,
            ),
            tools=cast(ToolRepositoryImpl | None, self._tool_operations),
            conversation_steps=(
                conversation_steps if isinstance(conversation_steps, StateStepArchive) else None
            ),
            execution_steps=execution_steps if isinstance(execution_steps, StateStepArchive) else None,
            recovery_steps=(
                self._step_reads[RuntimeDomain.RECOVERY]
                if isinstance(self._step_reads[RuntimeDomain.RECOVERY], StateStepArchive)
                else None
            ),
            background_tasks=self._checkpoint_tasks,
        )

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
        binding = self._catalog.binding(execution.binding_digest)
        if (
            request.mode != execution.mode
            or request.planning is not execution.planning
            or request.thinking != execution.thinking
            or execution.binding != binding.snapshot
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _execution_task_map(
        self,
    ) -> "dict[str, set[asyncio.Task[object]]]":
        try:
            return self._execution_durable_tasks
        except AttributeError:
            task_map: dict[str, set[asyncio.Task[object]]] = {}
            self._execution_durable_tasks = task_map
            return task_map

    def _execution_task_set(
        self,
        execution_id: str,
    ) -> set[asyncio.Task[object]]:
        return self._execution_task_map().setdefault(execution_id, set())

    def _track_checkpoint_task(
        self,
        task: "asyncio.Task[object]",
        label: str,
        execution_id: str | None = None,
    ) -> None:
        self._checkpoint_tasks.add(task)
        execution_tasks = (
            None
            if execution_id is None
            else self._execution_task_set(execution_id)
        )
        if execution_tasks is not None:
            execution_tasks.add(task)

        def consume(done: "asyncio.Task[object]") -> None:
            try:
                done.result()
            except asyncio.CancelledError:
                pass
            except BaseException as error:  # noqa: BLE001
                _logger.warning(
                    "local durable checkpoint owner failed: label=%s error=%s",
                    label,
                    type(error).__name__,
                )
            finally:
                self._checkpoint_tasks.discard(done)
                if execution_tasks is not None:
                    execution_tasks.discard(done)

        task.add_done_callback(consume)

    async def _await_checkpoint_task(
        self,
        task: "asyncio.Task[_CheckpointT]",
        *,
        label: str,
        execution_id: str | None = None,
    ) -> "tuple[_CheckpointT, asyncio.CancelledError | None]":
        self._track_checkpoint_task(
            cast("asyncio.Task[object]", task),
            label,
            execution_id,
        )
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                value = await asyncio.shield(task)
            except asyncio.CancelledError as error:
                if task.done():
                    if task.cancelled():
                        raise AIError(
                            ErrorCode.STORAGE_COMMIT_UNKNOWN,
                            safe_details={
                                "phase": "local_checkpoint",
                                "operation": label,
                            },
                        ) from error
                    try:
                        value = task.result()
                    except BaseException as task_error:  # noqa: BLE001
                        raise task_error from error
                    return value, cancellation or error
                if cancellation is None:
                    cancellation = error
                continue
            return value, cancellation

    async def _commit_terminal_checkpoint_owned(
        self,
        commit: ExecutionTerminalCommit,
        *,
        session_id: str | None,
    ) -> ExecutionTerminalCommitResult:
        execution_id = commit.execution.execution_id
        async with self._audit_lock(execution_id):
            pending = tuple(self._pending_audit_events.get(execution_id, ()))
            committed = await self._runtime_commands.commit_terminal_checkpoint(
                commit,
                session_id=session_id,
                audit_events=pending,
                background_tasks=self._execution_task_set(execution_id),
            )
            self._pending_audit_events.pop(execution_id, None)
            self._confirm_committed_events(
                execution_id,
                pending_count=len(pending),
                durable_sequence=committed.execution.event_sequence,
            )
        self._publish_terminal_event(
            execution_id,
            event_type=commit.terminal_event_type,
            payload=dict(commit.terminal_event_payload),
            durable_sequence=committed.execution.event_sequence,
        )
        self._live_broker.complete(execution_id)
        return committed

    async def commit_terminal_checkpoint(
        self,
        commit: ExecutionTerminalCommit,
        *,
        session_id: str | None,
    ) -> ExecutionTerminalCommitResult:
        task = asyncio.create_task(
            self._commit_terminal_checkpoint_owned(
                commit,
                session_id=session_id,
            ),
            name=f"local-terminal-checkpoint-{commit.execution.execution_id}",
        )
        committed, cancellation = await self._await_checkpoint_task(
            task,
            label="terminal",
            execution_id=commit.execution.execution_id,
        )
        if cancellation is not None:
            raise cancellation
        return committed

    async def _commit_cancel_checkpoint_owned(
        self,
        commit: ExecutionCancelRequestCommit,
        *,
        expected_status: ExecutionStatus,
    ) -> ExecutionRecord:
        async with self._audit_lock(commit.execution_id):
            current = await self._execution.executions.get(
                commit.execution_id,
                tenant_id=commit.tenant_id,
            )
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            pending = tuple(self._pending_audit_events.get(commit.execution_id, ()))
            effective_commit = commit
            effective_status = expected_status
            if (
                current.status is ExecutionStatus.WAITING_APPROVAL
                and expected_status in {
                    ExecutionStatus.STARTED,
                    ExecutionStatus.WAITING_APPROVAL,
                }
            ):
                checkpoint = await self._recovery.checkpoints.get(
                    commit.execution_id,
                    tenant_id=commit.tenant_id,
                )
                if (
                    checkpoint is None
                    or checkpoint.state is not RecoveryCheckpointState.WAITING
                    or checkpoint.pending_approval is None
                    or checkpoint.agent_run_sequence != current.agent_run_sequence
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                _, batch = await self._approval_batch(current, checkpoint)
                if (
                    current.revision != commit.expected_revision
                    or current.event_sequence != commit.expected_event_sequence
                ):
                    effective_commit = replace(
                        commit,
                        expected_revision=current.revision,
                        expected_event_sequence=current.event_sequence,
                    )
                committed = await self._runtime_commands.commit_waiting_approval_cancel_checkpoint(
                    effective_commit,
                    approval_ids=tuple(item.approval_id for item in batch),
                    expected_recovery_revision=checkpoint.revision,
                    expected_agent_run_sequence=current.agent_run_sequence,
                    expected_pending_approval=checkpoint.pending_approval,
                    audit_events=pending,
                    background_tasks=self._execution_task_set(commit.execution_id),
                )
            elif (
                current.status is expected_status
                and current.revision == commit.expected_revision
                and current.event_sequence == commit.expected_event_sequence
                and current.status is not ExecutionStatus.WAITING_APPROVAL
            ):
                committed = await self._runtime_commands.commit_cancel_checkpoint(
                    effective_commit,
                    expected_status=effective_status,
                    audit_events=pending,
                    background_tasks=self._execution_task_set(commit.execution_id),
                )
            elif (
                current.status is ExecutionStatus.STARTED
                and expected_status
                in {ExecutionStatus.STARTED, ExecutionStatus.WAITING_APPROVAL}
                and (
                    current.revision != commit.expected_revision
                    or current.event_sequence != commit.expected_event_sequence
                )
            ):
                checkpoint = await self._recovery.checkpoints.get(
                    commit.execution_id,
                    tenant_id=commit.tenant_id,
                )
                if (
                    checkpoint is None
                    or checkpoint.state is not RecoveryCheckpointState.ACTIVE
                    or checkpoint.pending_approval is None
                    or checkpoint.step_run_id is None
                    or checkpoint.agent_run_sequence != current.agent_run_sequence
                    or checkpoint.handoff_phase is not RecoveryHandoffPhase.NONE
                    or checkpoint.terminal_handoff is not None
                ):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                effective_commit = replace(
                    commit,
                    expected_revision=current.revision,
                    expected_event_sequence=current.event_sequence,
                )
                effective_status = ExecutionStatus.STARTED
                committed = await self._runtime_commands.commit_cancel_checkpoint(
                    effective_commit,
                    expected_status=effective_status,
                    audit_events=pending,
                    background_tasks=self._execution_task_set(commit.execution_id),
                )
            else:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._pending_audit_events.pop(commit.execution_id, None)
            cancel_sequence = (
                effective_commit.expected_event_sequence + len(pending) + 1
            )
            self._confirm_committed_events(
                commit.execution_id,
                pending_count=len(pending),
                durable_sequence=cancel_sequence,
            )
            self._live_broker.publish_event(
                commit.execution_id,
                event_type=ExecutionEventType.CANCEL_REQUESTED,
                payload={"operation_id": commit.operation_id},
                durable_sequence=cancel_sequence,
            )
            return committed

    async def commit_cancel_checkpoint(
        self,
        commit: ExecutionCancelRequestCommit,
        *,
        expected_status: ExecutionStatus,
    ) -> ExecutionRecord:
        task = asyncio.create_task(
            self._commit_cancel_checkpoint_owned(
                commit,
                expected_status=expected_status,
            ),
            name=f"local-cancel-checkpoint-{commit.execution_id}",
        )
        committed, cancellation = await self._await_checkpoint_task(
            task,
            label="cancel",
            execution_id=commit.execution_id,
        )
        if cancellation is not None:
            raise cancellation
        return committed

    async def prepare_start(
        self,
        request: ExecutionRequest,
        execution: ExecutionRecord,
        identity: ExecutionStartIdentity,
    ) -> ExecutionRecord:
        await self._validate_start(request, execution)
        if execution.status is not ExecutionStatus.PENDING_START:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        binding = self._catalog.binding(execution.binding_digest)
        if execution.binding != binding.snapshot:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        now = datetime.now(timezone.utc)
        recovery_input = RecoveryExecutionInput(
            user_prompt=_recovery_prompt_payload(request.user_prompt),
            user_prompt_codec=request.user_prompt_codec,
            principal_id=request.principal.principal_id,
            principal_kind=request.principal.kind,
            session_id=execution.session_id,
            memory_scope=execution.memory_scope,
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
            mode=execution.mode,
            planning=execution.planning,
            thinking=execution.thinking,
            binding=execution.binding,
            repository_instructions=execution.repository_instructions,
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
        expected = await self._expected_session_cursor(execution) if execution.session_id is not None else None
        started = await self._runtime_commands.commit_start_attempt_checkpoint(
            ExecutionStartClaim(
                execution.execution_id,
                execution.tenant_id,
                execution.revision,
                execution.event_sequence,
                identity.scope,
                identity.idempotency_key_digest,
                identity.request_digest,
                now,
            ),
            recovery_checkpoint=candidate if self._recovery_enabled else None,
            session_id=execution.session_id,
            expected_cursor=expected,
        )
        self._metrics.count("execution.start.checkpoint", domain="execution", target="runtime")
        _logger.info("execution start checkpoint committed: execution=%s", execution.execution_id)
        return started

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
            self._validate_recovery_identity(execution, checkpoint.input)
        current = await self._execution.executions.get(
            execution.execution_id,
            tenant_id=execution.tenant_id,
        )
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            current.tenant_id != execution.tenant_id
            or current.binding_digest != execution.binding_digest
            or current.mode != execution.mode
            or current.planning is not execution.planning
            or current.thinking != execution.thinking
            or current.binding != execution.binding
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
        if not self._live_broker.is_local_producer(current.execution_id):
            self._live_broker.register_local_producer(
                current.execution_id,
                current.event_sequence,
            )
        self._terminal_events[execution.execution_id] = asyncio.Event()
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

    def _validate_recovery_identity(
        self,
        execution: ExecutionRecord,
        recovery_input: RecoveryExecutionInput,
    ) -> None:
        if (
            execution.binding_digest != recovery_input.binding_digest
            or execution.mode != recovery_input.mode
            or execution.planning is not recovery_input.planning
            or execution.thinking != recovery_input.thinking
            or execution.binding != recovery_input.binding
            or execution.repository_instructions != recovery_input.repository_instructions
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

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
        self._worker_cancel_requests.discard(execution_id)
        self._captured_usage.pop(execution_id, None)
        if self._approval_pause_segments.get(execution_id) is task:
            self._approval_pause_segments.pop(execution_id, None)
        try:
            event = self._terminal_events.get(execution_id)
            if event is not None:
                event.set()
        except AttributeError:
            pass
        segment_only = execution_id in self._segment_only_worker_exits
        self._segment_only_worker_exits.discard(execution_id)
        try:
            live_broker = self._live_broker
        except AttributeError:
            live_broker = None
        if error is None:
            if live_broker is not None and not segment_only:
                live_broker.complete(execution_id)
            return
        if isinstance(error, AIError):
            failure = _WorkerFailure(error.code, dict(error.safe_details))
        else:
            failure = _WorkerFailure(
                ErrorCode.INTERNAL_ERROR,
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
        if live_broker is not None:
            live_broker.complete(execution_id)

    def _request_worker_cancel(
        self,
        execution_id: str,
        task: asyncio.Task[None],
    ) -> None:
        if task.done() or execution_id in self._worker_cancel_requests:
            return
        self._worker_cancel_requests.add(execution_id)
        task.cancel()

    async def _drain_worker_task(
        self,
        execution_id: str,
        task: asyncio.Task[None],
    ) -> None:
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.done():
                raise
        except BaseException:  # noqa: BLE001
            pass
        if self._tasks.get(execution_id) is task:
            self._task_done(execution_id, task)

    def worker_failure(self, execution_id: str, *, tenant_id: str) -> AIError | None:
        if tenant_id != self._tenant_id:
            return AIError(ErrorCode.AUTHORIZATION_DENIED)
        failure = self._worker_failures.get(execution_id)
        if failure is None:
            return None
        return AIError(failure.code, safe_details=dict(failure.safe_details))

    def worker_installed(self, execution_id: str) -> bool:
        task = self._tasks.get(execution_id)
        return task is not None and not task.done()

    def owns_execution(self, execution_id: str, *, tenant_id: str) -> bool:
        return tenant_id == self._tenant_id and self.worker_installed(execution_id)

    async def wait_terminal(self, execution_id: str, *, tenant_id: str) -> None:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        event = self._terminal_events.setdefault(execution_id, asyncio.Event())
        await event.wait()

    @property
    def live_broker(self) -> LiveExecutionEventBroker:
        return self._live_broker

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
        if (
            not task.done()
            and current is not None
            and current.status is ExecutionStatus.CANCELLING
            and self._approval_pause_segments.get(execution.execution_id) is task
        ):
            segment_exit = self._terminal_events.get(execution.execution_id)
            if segment_exit is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await segment_exit.wait()
            if task.done() and self._tasks.get(execution.execution_id) is task:
                self._task_done(execution.execution_id, task)
            current = await self._execution.executions.get(
                execution.execution_id,
                tenant_id=execution.tenant_id,
            )
            if current is not None and current.status in {
                ExecutionStatus.CANCELLING,
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }:
                return CancelEffectOutcome.CONFIRMED
            return CancelEffectOutcome.UNKNOWN
        self._request_worker_cancel(execution.execution_id, task)
        await self._drain_worker_task(execution.execution_id, task)
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

    @classmethod
    def _decode_repository_instruction_object(
        cls,
        data: bytes,
    ) -> Mapping[str, object]:
        del cls

        def reject_duplicate_keys(
            pairs: list[tuple[str, object]],
        ) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON object key")
                result[key] = value
            return result

        def reject_constant(value: str) -> object:
            raise ValueError(f"invalid JSON constant: {value}")

        try:
            raw = json.loads(
                data.decode("utf-8"),
                object_pairs_hook=reject_duplicate_keys,
                parse_constant=reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if not isinstance(raw, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return raw

    async def _load_repository_instructions(
        self,
        reference: RuntimePayloadRef | None,
    ) -> RepositoryInstructions | None:
        if reference is None:
            return None
        if reference.source_domain is not RuntimeDomain.EXECUTION:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        payload = reference.payload
        if payload.kind == "inline":
            if payload.encoding != "json":
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                raw = payload.decode()
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if not isinstance(raw, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        elif payload.kind == "object":
            if payload.ref is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            data = await read_runtime_object(self._execution_objects, payload.ref)
            raw = self._decode_repository_instruction_object(data)
        else:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            instructions = RepositoryInstructions.from_payload(raw)
        except AIError as error:
            if error.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED:
                raise
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if (
            instructions.digest != payload.digest
            or canonical_sha256(instructions.to_payload()) != payload.digest
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return instructions

    async def _repository_instruction_marker_authority_for_messages(
        self,
        messages: Sequence[ModelMessage],
        *,
        tenant_id: str,
    ) -> frozenset[tuple[str, str]]:
        candidates_by_run: dict[str, list[str]] = {}
        seen_by_run: dict[str, set[str]] = {}
        for message in messages:
            if not isinstance(message, (ModelRequest, ModelResponse)):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if not isinstance(message, ModelRequest):
                continue
            for part in message.parts:
                if not isinstance(part, ToolReturnPart):
                    continue
                if (
                    part.outcome != "failed"
                    or not isinstance(part.tool_call_id, str)
                    or not part.tool_call_id
                    or not isinstance(part.content, str)
                    or not part.content.startswith("[linktools.repository-instructions.v1]\n")
                ):
                    continue
                if not isinstance(message.run_id, str) or not message.run_id:
                    continue
                seen = seen_by_run.setdefault(message.run_id, set())
                if part.tool_call_id in seen:
                    continue
                seen.add(part.tool_call_id)
                candidates_by_run.setdefault(message.run_id, []).append(part.tool_call_id)
        if not candidates_by_run:
            return frozenset()
        if self._tool_operations is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        authority: set[tuple[str, str]] = set()
        for run_id, tool_call_ids in candidates_by_run.items():
            existing = await self._tool_operations.existing_call_ids(
                run_id,
                tool_call_ids,
                tenant_id=tenant_id,
            )
            for tool_call_id in tool_call_ids:
                if tool_call_id not in existing:
                    authority.add((run_id, tool_call_id))
        return frozenset(authority)

    def _validate_repository_instruction_run(
        self,
        run: RunRecord,
        *,
        execution: ExecutionRecord,
        sequence: int,
        conversation_id: str,
    ) -> None:
        if (
            sequence < 1
            or run.conversation_id != conversation_id
            or run.metadata.get("segment_sequence") != str(sequence)
            or run.run_id
            != step_run_id(
                namespace=self._namespace,
                tenant_id=execution.tenant_id,
                execution_id=execution.execution_id,
                segment_sequence=sequence,
            )
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _load_repository_instruction_attempt_provenance(
        self,
        archive: StateStepArchive,
        run: RunRecord,
        *,
        execution: ExecutionRecord,
        sequence: int,
    ) -> _RepositoryInstructionAttemptProvenance:
        conversation_id = step_conversation_id(
            namespace=self._namespace,
            tenant_id=execution.tenant_id,
            execution_id=execution.execution_id,
        )
        self._validate_repository_instruction_run(
            run,
            execution=execution,
            sequence=sequence,
            conversation_id=conversation_id,
        )
        messages: list[ModelMessage] = []
        async for message in archive.iter_raw_messages(run_id=run.run_id):
            if (
                not isinstance(message, (ModelRequest, ModelResponse))
                or message.run_id != run.run_id
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            messages.append(message)
        authority = await self._repository_instruction_marker_authority_for_messages(
            messages,
            tenant_id=execution.tenant_id,
        )
        if any(run_id != run.run_id for run_id, _ in authority):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return _RepositoryInstructionAttemptProvenance(
            run.run_id,
            tuple(messages),
            authority,
        )

    async def _repository_instruction_provenance_for_scope(
        self,
        execution: ExecutionRecord,
        checkpoint: RecoveryCheckpoint,
        *,
        fallback_history: Sequence[ModelMessage],
    ) -> _RepositoryInstructionProvenance:
        if execution.repository_instructions is None:
            return _RepositoryInstructionProvenance((), frozenset())
        archive = self._step_reads[RuntimeDomain.RECOVERY]
        if not isinstance(archive, StateStepArchive):
            messages = tuple(fallback_history)
            authority = await self._repository_instruction_marker_authority_for_messages(
                messages,
                tenant_id=execution.tenant_id,
            )
            return _RepositoryInstructionProvenance(messages, authority)
        captured_upper_sequence = checkpoint.agent_run_sequence
        if captured_upper_sequence < 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        conversation_id = step_conversation_id(
            namespace=self._namespace,
            tenant_id=execution.tenant_id,
            execution_id=execution.execution_id,
        )
        cache = self._repository_instruction_provenance.setdefault(
            execution.execution_id,
            _RepositoryInstructionProvenanceCache(),
        )
        async with cache.lock:
            current_attempt: _RepositoryInstructionAttemptProvenance | None = None
            if not cache.initialized:
                cache.final_attempts.clear()
                runs = await archive.list_runs(conversation_id=conversation_id)
                by_sequence: dict[int, RunRecord] = {}
                for run in runs:
                    raw_sequence = run.metadata.get("segment_sequence")
                    if (
                        not isinstance(raw_sequence, str)
                        or not raw_sequence.isdigit()
                        or raw_sequence == "0"
                        or str(int(raw_sequence)) != raw_sequence
                    ):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    sequence = int(raw_sequence)
                    self._validate_repository_instruction_run(
                        run,
                        execution=execution,
                        sequence=sequence,
                        conversation_id=conversation_id,
                    )
                    if sequence in by_sequence:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    by_sequence[sequence] = run
                for sequence in range(1, captured_upper_sequence):
                    run = by_sequence.get(sequence)
                    if run is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    cache.final_attempts[sequence] = (
                        await self._load_repository_instruction_attempt_provenance(
                            archive,
                            run,
                            execution=execution,
                            sequence=sequence,
                        )
                    )
                current_run = by_sequence.get(captured_upper_sequence)
                if current_run is not None:
                    current_attempt = await self._load_repository_instruction_attempt_provenance(
                        archive,
                        current_run,
                        execution=execution,
                        sequence=captured_upper_sequence,
                    )
                cache.initialized = True
            else:
                for sequence in range(1, captured_upper_sequence):
                    if sequence in cache.final_attempts:
                        continue
                    run = await archive.get_run(
                        run_id=step_run_id(
                            namespace=self._namespace,
                            tenant_id=execution.tenant_id,
                            execution_id=execution.execution_id,
                            segment_sequence=sequence,
                        )
                    )
                    if run is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    cache.final_attempts[sequence] = (
                        await self._load_repository_instruction_attempt_provenance(
                            archive,
                            run,
                            execution=execution,
                            sequence=sequence,
                        )
                    )
                current_run = await archive.get_run(
                    run_id=step_run_id(
                        namespace=self._namespace,
                        tenant_id=execution.tenant_id,
                        execution_id=execution.execution_id,
                        segment_sequence=captured_upper_sequence,
                    )
                )
                if current_run is not None:
                    current_attempt = await self._load_repository_instruction_attempt_provenance(
                        archive,
                        current_run,
                        execution=execution,
                        sequence=captured_upper_sequence,
                    )
            attempts = [
                cache.final_attempts[sequence]
                for sequence in range(1, captured_upper_sequence)
            ]
            if current_attempt is not None:
                attempts.append(current_attempt)
            messages = tuple(
                message
                for attempt in attempts
                for message in attempt.messages
            )
            authority = frozenset(
                item
                for attempt in attempts
                for item in attempt.marker_authority
            )
            return _RepositoryInstructionProvenance(messages, authority)

    @staticmethod
    def _approval_call_identity(
        *,
        tenant_id: str,
        execution_id: str,
        source_step_run_id: str,
        tool_call: ToolCallPart,
    ) -> _ApprovalBatchCall:
        try:
            arguments = normalize_json_value(tool_call.args_as_dict())
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        tool_call_id = tool_call.tool_call_id
        tool_name = tool_call.tool_name
        if (
            not isinstance(tool_call_id, str)
            or not tool_call_id
            or not isinstance(tool_name, str)
            or not tool_name
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        args_digest = canonical_sha256(arguments)
        approval_id = canonical_sha256(
            {
                "contract": "tool-approval-v1",
                "tenant_id": tenant_id,
                "execution_id": execution_id,
                "source_step_run_id": source_step_run_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "args_digest": args_digest,
            }
        )
        operation_id = canonical_sha256(
            {
                "contract": "tool-approval-operation-v1",
                "approval_id": approval_id,
            }
        )
        return _ApprovalBatchCall(
            approval_id,
            operation_id,
            tool_call_id,
            tool_name,
            arguments,
            args_digest,
        )

    async def _approval_batch(
        self,
        execution: ExecutionRecord,
        checkpoint: RecoveryCheckpoint,
    ) -> tuple[ContinuableSnapshot, tuple[_ApprovalBatchCall, ...]]:
        continuation = checkpoint.pending_approval
        if continuation is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        archive = self._step_reads[RuntimeDomain.RECOVERY]
        if not isinstance(archive, StateStepArchive):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        source_run_id = continuation.source_step_run_id
        run = await archive.get_run(run_id=source_run_id)
        snapshot = await archive.latest_snapshot(
            run_id=source_run_id,
            include_interrupted=True,
        )
        if (
            run is None
            or snapshot is None
            or snapshot.run_id != source_run_id
            or snapshot.state != "interrupted"
            or not snapshot.messages
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        calls = AgentExecutor.pending_tool_calls(snapshot.messages, run_id=source_run_id)
        if not calls:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        batch = tuple(
            self._approval_call_identity(
                tenant_id=execution.tenant_id,
                execution_id=execution.execution_id,
                source_step_run_id=source_run_id,
                tool_call=call,
            )
            for call in calls
        )
        batch_id = canonical_sha256(
            {
                "contract": "tool-approval-batch-v1",
                "execution_id": execution.execution_id,
                "source_step_run_id": source_run_id,
                "approval_ids": sorted(item.approval_id for item in batch),
            }
        )
        if continuation.batch_id != batch_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        tool_call_ids = tuple(item.tool_call_id for item in batch)
        if self._tool_operations is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        existing = await self._tool_operations.existing_call_ids(
            source_run_id,
            tool_call_ids,
            tenant_id=execution.tenant_id,
        )
        if existing:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for tool_call_id in tool_call_ids:
            effect = await archive.get_tool_effect(
                run_id=source_run_id,
                tool_call_id=tool_call_id,
            )
            if effect is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return snapshot, batch

    async def _approval_records(
        self,
        execution: ExecutionRecord,
        batch: Sequence[_ApprovalBatchCall],
    ) -> tuple[ApprovalRecord, ...]:
        records: list[ApprovalRecord] = []
        for item in batch:
            record = await self._recovery.approvals.get(
                item.approval_id,
                tenant_id=execution.tenant_id,
            )
            if (
                record is None
                or record.execution_id != execution.execution_id
                or record.tenant_id != execution.tenant_id
                or record.operation_id != item.operation_id
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            records.append(record)
        return tuple(records)

    def _approval_tool_class(
        self,
        execution: ExecutionRecord,
        binding: AgentBinding,
        tool_name: str,
    ) -> str | None:
        subagent_available = (
            execution.parent_execution_id is None
            and self._subagent_dispatcher is not None
            and bool(binding.snapshot.subagents)
        )
        return self._executor.trusted_tool_class(
            binding,
            tool_name,
            memory_scope=execution.memory_scope,
            planning=execution.planning,
            subagent_available=subagent_available,
        )

    async def tool_approvals(
        self,
        approval_ids: Sequence[str],
        *,
        execution_id: str,
        tenant_id: str,
    ) -> Mapping[str, ToolApprovalContext]:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        ordered = tuple(approval_ids)
        if len(set(ordered)) != len(ordered):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        execution = await self._execution.executions.get(
            execution_id,
            tenant_id=tenant_id,
        )
        if execution is None or execution.status is not ExecutionStatus.WAITING_APPROVAL:
            return {}
        checkpoint = await self._recovery.checkpoints.get(
            execution_id,
            tenant_id=tenant_id,
        )
        if (
            checkpoint is None
            or checkpoint.state is not RecoveryCheckpointState.WAITING
            or checkpoint.pending_approval is None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _, batch = await self._approval_batch(execution, checkpoint)
        by_id = {item.approval_id: item for item in batch}
        result: dict[str, ToolApprovalContext] = {}
        for approval_id in ordered:
            item = by_id.get(approval_id)
            if item is None:
                continue
            result[approval_id] = ToolApprovalContext(
                item.tool_name,
                item.arguments,
                item.args_digest,
                checkpoint.pending_approval.batch_id,
            )
        return result

    async def reconcile_approval(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> None:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        checkpoint = await self._recovery.checkpoints.get(
            execution_id,
            tenant_id=tenant_id,
        )
        if (
            checkpoint is None
            or checkpoint.state is not RecoveryCheckpointState.WAITING
            or checkpoint.pending_approval is None
        ):
            return
        await self._reconcile_checkpoint(checkpoint)

    async def _wait_approval_pause_segment_exit(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> bool:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        task = self._tasks.get(execution_id)
        waited = False
        if task is not None and task.done():
            self._task_done(execution_id, task)
            task = self._tasks.get(execution_id)
        if task is not None and not task.done():
            if self._approval_pause_segments.get(execution_id) is not task:
                waited = True
            else:
                segment_exit = self._terminal_events.get(execution_id)
                if segment_exit is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await segment_exit.wait()
                if self._tasks.get(execution_id) is task:
                    if not task.done():
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    self._task_done(execution_id, task)
                waited = True
        failure = self.worker_failure(execution_id, tenant_id=tenant_id)
        if failure is not None:
            raise failure
        return waited

    def _approval_admission(
        self,
        execution: ExecutionRecord,
        paused: AgentExecutionPaused,
        call: _ApprovalBatchCall,
    ) -> ToolApprovalAdmission:
        occurred_at = paused.paused_at
        record = ApprovalRecord(
            approval_id=call.approval_id,
            execution_id=execution.execution_id,
            tenant_id=execution.tenant_id,
            operation_id=call.operation_id,
            status=ApprovalStatus.PENDING,
            idempotency_key_digest=None,
            decision=None,
            decided_by=None,
            decision_digest=None,
            created_at=occurred_at,
            decided_at=None,
        )
        admission_operation_id = canonical_sha256(
            {
                "contract": "tool-approval-admission-v1",
                "approval_id": call.approval_id,
            }
        )
        request_digest = canonical_sha256(
            {
                "contract": "tool-approval-admission-request-v1",
                "approval_id": call.approval_id,
                "execution_id": execution.execution_id,
                "operation_id": call.operation_id,
                "source_step_run_id": paused.run_id,
                "tool_call_id": call.tool_call_id,
                "tool_name": call.tool_name,
                "args_digest": call.args_digest,
            }
        )
        result_digest = canonical_sha256(
            {
                "contract": "tool-approval-admission-result-v1",
                "approval_id": call.approval_id,
                "execution_id": execution.execution_id,
                "operation_id": call.operation_id,
                "status": "PENDING",
            }
        )
        operation = OperationLedgerInput(
            operation_id=admission_operation_id,
            tenant_id=execution.tenant_id,
            resource_kind=ResourceKind.APPROVAL,
            resource_id=call.approval_id,
            execution_id=execution.execution_id,
            operation_kind=OperationKind.APPROVAL,
            status=OperationStatus.SUCCEEDED,
            request_digest=request_digest,
            result_ref=call.approval_id,
            result_digest=result_digest,
            error_code=None,
            compactable=True,
            created_at=occurred_at,
            updated_at=occurred_at,
        )
        return ToolApprovalAdmission(
            record=record,
            operation=operation,
            tool_name=call.tool_name,
            args_digest=call.args_digest,
        )

    async def _commit_approval_pause(
        self,
        execution: ExecutionRecord,
        paused: AgentExecutionPaused,
    ) -> None:
        pause_worker = asyncio.current_task()
        if (
            pause_worker is None
            or self._tasks.get(execution.execution_id) is not pause_worker
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._approval_pause_segments[execution.execution_id] = pause_worker

        async def commit_owned() -> None:
            async with self._audit_lock(execution.execution_id):
                current = await self._execution.executions.get(
                    execution.execution_id,
                    tenant_id=execution.tenant_id,
                )
                checkpoint = await self._recovery.checkpoints.get(
                    execution.execution_id,
                    tenant_id=execution.tenant_id,
                )
                if current is None or checkpoint is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if current.status in {
                    ExecutionStatus.CANCELLING,
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                }:
                    self._segment_only_worker_exits.add(execution.execution_id)
                    return
                if (
                    current.status is not ExecutionStatus.STARTED
                    or checkpoint.state is not RecoveryCheckpointState.ACTIVE
                    or checkpoint.step_run_id != paused.run_id
                    or checkpoint.agent_run_sequence != current.agent_run_sequence
                ):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                working_run = await self._steps.get_run(run_id=paused.run_id)
                if working_run is None or working_run.run_id != paused.run_id:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                candidate_snapshot = ContinuableSnapshot(
                    run_id=paused.run_id,
                    step_index=paused.step_index,
                    messages=list(paused.messages),
                    conversation_id=working_run.conversation_id,
                    parent_run_id=working_run.parent_run_id,
                    agent_name=working_run.agent_name,
                    timestamp=paused.paused_at,
                    state="interrupted",
                )
                pending_calls = AgentExecutor.pending_tool_calls(
                    candidate_snapshot.messages,
                    run_id=paused.run_id,
                )
                calls = tuple(
                    self._approval_call_identity(
                        tenant_id=current.tenant_id,
                        execution_id=current.execution_id,
                        source_step_run_id=paused.run_id,
                        tool_call=call,
                    )
                    for call in pending_calls
                )
                if len(calls) != len(paused.approvals):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                for call, pending in zip(calls, paused.approvals):
                    if (
                        pending.tool_call_id != call.tool_call_id
                        or pending.tool_name != call.tool_name
                        or pending.arguments != call.arguments
                        or pending.args_digest != call.args_digest
                    ):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                tool_call_ids = tuple(call.tool_call_id for call in calls)
                if self._tool_operations is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                existing = await self._tool_operations.existing_call_ids(
                    paused.run_id,
                    tool_call_ids,
                    tenant_id=current.tenant_id,
                )
                if existing:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                for tool_call_id in tool_call_ids:
                    if await self._steps.get_tool_effect(
                        run_id=paused.run_id,
                        tool_call_id=tool_call_id,
                    ) is not None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                approval_ids = tuple(call.approval_id for call in calls)
                continuation = PendingApprovalContinuation(
                    batch_id=canonical_sha256(
                        {
                            "contract": "tool-approval-batch-v1",
                            "execution_id": current.execution_id,
                            "source_step_run_id": paused.run_id,
                            "approval_ids": sorted(approval_ids),
                        }
                    ),
                    source_step_run_id=paused.run_id,
                )
                admissions = tuple(
                    self._approval_admission(current, paused, call)
                    for call in calls
                )
                pending_audit = tuple(
                    self._pending_audit_events.get(current.execution_id, ())
                )
                expected_event_sequence = current.event_sequence
                try:
                    committed, _ = await self._runtime_commands.commit_approval_wait_checkpoint(
                        execution_id=current.execution_id,
                        tenant_id=current.tenant_id,
                        expected_execution_revision=current.revision,
                        expected_event_sequence=expected_event_sequence,
                        expected_recovery_revision=checkpoint.revision,
                        expected_agent_run_sequence=current.agent_run_sequence,
                        expected_previous_pending_approval=checkpoint.pending_approval,
                        continuation=continuation,
                        admissions=admissions,
                        audit_events=pending_audit,
                        recovery_run=working_run,
                        recovery_snapshot=candidate_snapshot,
                        occurred_at=paused.paused_at,
                        background_tasks=self._execution_task_set(current.execution_id),
                    )
                except AIError as error:
                    if error.code is not ErrorCode.STORAGE_CONFLICT:
                        raise
                    reread = await self._execution.executions.get(
                        current.execution_id,
                        tenant_id=current.tenant_id,
                    )
                    if reread is not None and reread.status in {
                        ExecutionStatus.CANCELLING,
                        ExecutionStatus.SUCCEEDED,
                        ExecutionStatus.FAILED,
                        ExecutionStatus.CANCELLED,
                    }:
                        self._segment_only_worker_exits.add(current.execution_id)
                        return
                    raise
                self._pending_audit_events.pop(current.execution_id, None)
                if pending_audit:
                    self._live_broker.confirm_events(
                        current.execution_id,
                        first_sequence=expected_event_sequence + 1,
                        count=len(pending_audit),
                    )
                for index, admission in enumerate(admissions, 1):
                    self._live_broker.publish_event(
                        current.execution_id,
                        ExecutionEventType.APPROVAL_REQUESTED,
                        {
                            "approval_id": admission.record.approval_id,
                            "tool_name": admission.tool_name,
                            "args_digest": admission.args_digest,
                            "batch_id": continuation.batch_id,
                        },
                        durable_sequence=(
                            expected_event_sequence + len(pending_audit) + index
                        ),
                    )
                if committed.status is not ExecutionStatus.WAITING_APPROVAL:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                self._segment_only_worker_exits.add(current.execution_id)

        task = asyncio.create_task(
            commit_owned(),
            name=f"ai-approval-pause-{execution.execution_id}",
        )
        self._track_checkpoint_task(
            cast("asyncio.Task[object]", task),
            "approval-pause",
            execution.execution_id,
        )
        while True:
            try:
                await asyncio.shield(task)
                return
            except asyncio.CancelledError:
                if task.done():
                    task.result()
                    return
                continue

    def _deferred_tool_results(
        self,
        execution: ExecutionRecord,
        binding: AgentBinding,
        batch: Sequence[_ApprovalBatchCall],
        records: Sequence[ApprovalRecord],
    ) -> DeferredToolResults:
        if len(batch) != len(records):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        results = DeferredToolResults()
        for item, record in zip(batch, records):
            if record.approval_id != item.approval_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            tool_class = self._approval_tool_class(
                execution,
                binding,
                item.tool_name,
            )
            decision = self._workspace.policy.tool_permissions.decide(
                tool_name=item.tool_name,
                tool_class=tool_class,
            )
            if record.status is ApprovalStatus.PENDING:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if (
                decision == "deny"
                or record.status
                in {
                    ApprovalStatus.DENIED,
                    ApprovalStatus.CANCELLED,
                    ApprovalStatus.EXPIRED,
                }
            ):
                results.approvals[item.tool_call_id] = ToolDenied()
            elif record.status is ApprovalStatus.APPROVED:
                results.approvals[item.tool_call_id] = ToolApproved()
            else:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if results.calls or results.metadata:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return results

    async def _reconcile_waiting_approval(
        self,
        checkpoint: RecoveryCheckpoint,
        execution: ExecutionRecord,
    ) -> tuple[ExecutionRecord, RecoveryCheckpoint] | None:
        while True:
            current = await self._execution.executions.get(
                execution.execution_id,
                tenant_id=execution.tenant_id,
            )
            recovery = await self._recovery.checkpoints.get(
                execution.execution_id,
                tenant_id=execution.tenant_id,
            )
            if current is None or recovery is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current.status is ExecutionStatus.CANCELLING:
                return None
            if (
                recovery.state is not RecoveryCheckpointState.WAITING
                or recovery.pending_approval is None
                or current.status is not ExecutionStatus.WAITING_APPROVAL
                or recovery.agent_run_sequence != current.agent_run_sequence
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            _, batch = await self._approval_batch(current, recovery)
            records = await self._approval_records(current, batch)
            binding = self._catalog.binding(current.binding_digest)
            if current.binding != binding.snapshot:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            denied_approval_ids: list[str] = []
            any_pending = False
            for item, record in zip(batch, records):
                tool_class = self._approval_tool_class(
                    current,
                    binding,
                    item.tool_name,
                )
                decision = self._workspace.policy.tool_permissions.decide(
                    tool_name=item.tool_name,
                    tool_class=tool_class,
                )
                if record.status is ApprovalStatus.PENDING:
                    if decision == "deny":
                        denied_approval_ids.append(item.approval_id)
                    else:
                        any_pending = True
            if denied_approval_ids:
                try:
                    await self._runtime_commands.commit_approval_policy_checkpoint(
                        execution_id=current.execution_id,
                        tenant_id=current.tenant_id,
                        expected_recovery_revision=recovery.revision,
                        expected_pending_approval=recovery.pending_approval,
                        batch_approval_ids=tuple(item.approval_id for item in batch),
                        denied_approval_ids=tuple(denied_approval_ids),
                        decided_at=datetime.now(timezone.utc),
                        background_tasks=self._execution_task_set(current.execution_id),
                    )
                except AIError as error:
                    if error.code is ErrorCode.STORAGE_CONFLICT:
                        continue
                    raise
                continue
            if any_pending:
                return None
            self._deferred_tool_results(current, binding, batch, records)
            if await self._wait_approval_pause_segment_exit(
                current.execution_id,
                tenant_id=current.tenant_id,
            ):
                continue
            try:
                return await self._runtime_commands.claim_approval_resume_checkpoint(
                    execution_id=current.execution_id,
                    tenant_id=current.tenant_id,
                    expected_execution_revision=current.revision,
                    expected_event_sequence=current.event_sequence,
                    expected_recovery_revision=recovery.revision,
                    expected_agent_run_sequence=current.agent_run_sequence,
                    expected_pending_approval=recovery.pending_approval,
                    approval_ids=tuple(item.approval_id for item in batch),
                    background_tasks=self._execution_task_set(current.execution_id),
                )
            except AIError as error:
                if error.code is ErrorCode.STORAGE_CONFLICT:
                    continue
                raise

    async def reconcile(self) -> None:
        """Rebuild transient execution state from recovery-owned checkpoints."""
        if not self._recovery_enabled:
            return
        cursor: str | None = None
        while True:
            page = await self._recovery.checkpoints.list_recoverable_page(
                tenant_id=self._tenant_id,
                cursor=cursor,
                limit=128,
            )
            for checkpoint in page.items:
                if checkpoint.state is RecoveryCheckpointState.COMPLETED:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                try:
                    await self._reconcile_checkpoint(checkpoint)
                except AIError as error:
                    if error.code is not ErrorCode.AGENT_DEFINITION_UNAVAILABLE:
                        raise
                    _logger.warning(
                        "recovery reconciliation deferred: execution=%s",
                        checkpoint.execution_id,
                    )
            if page.next_cursor is None:
                return
            cursor = page.next_cursor

    async def _reconcile_checkpoint(self, checkpoint: RecoveryCheckpoint) -> None:
        recovery_input = checkpoint.input
        if (
            recovery_input.session_id is not None
            and self._workspace.policy.tool_permissions.requires_approval
            and not self._conversation_durable
        ):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        principal = Principal(recovery_input.principal_id, checkpoint.tenant_id, recovery_input.principal_kind)
        execution = await self._execution.executions.get(checkpoint.execution_id, tenant_id=checkpoint.tenant_id)
        if execution is not None and (
            execution.execution_id != checkpoint.execution_id
            or execution.tenant_id != checkpoint.tenant_id
            or execution.binding_digest != recovery_input.binding_digest
            or execution.parent_execution_id != recovery_input.parent_execution_id
            or execution.root_execution_id != recovery_input.root_execution_id
            or execution.source_execution_id != recovery_input.source_execution_id
            or execution.base_execution_id != recovery_input.base_execution_id
            or execution.conversation_step_run_id != recovery_input.conversation_step_run_id
            or execution.lineage_kind.value != recovery_input.lineage_kind
            or execution.planning is not recovery_input.planning
            or execution.thinking != recovery_input.thinking
            or execution.binding != recovery_input.binding
            or execution.repository_instructions != recovery_input.repository_instructions
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if checkpoint.handoff_phase is not RecoveryHandoffPhase.NONE:
            self._validate_recovery_handoff_integrity(checkpoint, execution)
        if (
            checkpoint.handoff_phase is RecoveryHandoffPhase.NONE
            and execution is not None
            and execution.status
            in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }
        ):
            if execution.session_id is not None:
                await self._conversation.sessions.release_execution(
                    execution.session_id,
                    tenant_id=execution.tenant_id,
                    execution_id=execution.execution_id,
                )
            await self._finish_checkpoint(checkpoint)
            return
        self._catalog.binding(recovery_input.binding_digest)
        if checkpoint.handoff_phase is not RecoveryHandoffPhase.NONE:
            if (
                self._handoff_contract_digest is None
                or checkpoint.handoff_contract_digest != self._handoff_contract_digest
            ):
                _logger.error("recovery handoff contract mismatch: execution=%s", checkpoint.execution_id)
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            await self._reconcile_handoff(checkpoint)
            return
        if execution is None:
            execution = await self._create_recovery_execution(checkpoint)
        if checkpoint.state in {
            RecoveryCheckpointState.ADMITTED,
            RecoveryCheckpointState.ACTIVE,
            RecoveryCheckpointState.WAITING,
        } and not await self._reconcile_session_recovery(checkpoint, execution):
            return
        if (
            checkpoint.state is RecoveryCheckpointState.ADMITTED
            and execution.status is ExecutionStatus.PENDING_START
        ):
            expected = (
                await self._expected_session_cursor(execution)
                if execution.session_id is not None
                else None
            )
            started = await self._runtime_commands.commit_start_checkpoint(
                ExecutionStartClaim(
                    execution.execution_id,
                    execution.tenant_id,
                    execution.revision,
                    execution.event_sequence,
                    recovery_input.idempotency.scope,
                    recovery_input.idempotency.idempotency_key_digest,
                    recovery_input.idempotency.request_digest,
                    datetime.now(timezone.utc),
                ),
                recovery_checkpoint=checkpoint,
                session_id=execution.session_id,
                expected_cursor=expected,
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
            return
        elif execution.status is ExecutionStatus.START_UNKNOWN:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        elif checkpoint.state is RecoveryCheckpointState.ADMITTED and execution.status is ExecutionStatus.STARTED:
            await self._ensure_recovery_idempotency(
                checkpoint,
                expected_status=IdempotencyStatus.STARTED,
            )
        elif (
            checkpoint.state is RecoveryCheckpointState.WAITING
            and checkpoint.pending_approval is not None
        ):
            if execution.status is not ExecutionStatus.WAITING_APPROVAL:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            resumed = await self._reconcile_waiting_approval(checkpoint, execution)
            if resumed is None:
                return
            execution, checkpoint = resumed
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
            if checkpoint.state is RecoveryCheckpointState.WAITING and checkpoint.pending_approval is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._ensure_recovery_idempotency(
                checkpoint,
                expected_status=IdempotencyStatus.STARTED,
            )
        else:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        request = ExecutionRequest(
            user_prompt=_recovery_prompt_text(recovery_input),
            user_prompt_codec=recovery_input.user_prompt_codec,
            principal=principal,
            idempotency_key=f"recovery:{checkpoint.execution_id}",
            memory_scope=recovery_input.memory_scope,
            mode=recovery_input.mode,
            planning=recovery_input.planning,
            thinking=recovery_input.thinking,
        )
        self._recovery_relaunch_ids.add(checkpoint.execution_id)
        await self.launch(request, execution)
        _logger.info(
            "local recovery execution relaunched: tenant=%s execution=%s",
            checkpoint.tenant_id,
            checkpoint.execution_id,
        )

    async def _reconcile_session_recovery(
        self,
        checkpoint: RecoveryCheckpoint,
        execution: ExecutionRecord,
    ) -> bool:
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
                if execution.status is ExecutionStatus.WAITING_APPROVAL:
                    if (
                        checkpoint.state is not RecoveryCheckpointState.WAITING
                        or checkpoint.pending_approval is None
                    ):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    operation_id = canonical_sha256(
                        {
                            "type": "linktools.ai.session-close-recovery-cancel",
                            "version": 1,
                            "tenant_id": execution.tenant_id,
                            "session_id": execution.session_id,
                            "execution_id": execution.execution_id,
                        }
                    )
                    committed = await self.commit_cancel_checkpoint(
                        ExecutionCancelRequestCommit(
                            execution.execution_id,
                            execution.tenant_id,
                            execution.revision,
                            execution.event_sequence,
                            operation_id,
                            datetime.now(timezone.utc),
                        ),
                        expected_status=ExecutionStatus.WAITING_APPROVAL,
                    )
                    if committed.status is not ExecutionStatus.CANCELLING:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    _, batch = await self._approval_batch(committed, checkpoint)
                    records = await self._approval_records(committed, batch)
                    if any(record.status is ApprovalStatus.PENDING for record in records):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    fresh_checkpoint = await self._recovery.checkpoints.get(
                        execution.execution_id,
                        tenant_id=execution.tenant_id,
                    )
                    if (
                        fresh_checkpoint is None
                        or fresh_checkpoint.state is not RecoveryCheckpointState.WAITING
                        or fresh_checkpoint.pending_approval != checkpoint.pending_approval
                    ):
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                    checkpoint = fresh_checkpoint
                    execution = committed
                elif execution.status is ExecutionStatus.CANCELLING:
                    if checkpoint.pending_approval is not None:
                        _, batch = await self._approval_batch(execution, checkpoint)
                        records = await self._approval_records(execution, batch)
                        if any(record.status is ApprovalStatus.PENDING for record in records):
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                elif (
                    execution.status is ExecutionStatus.STARTED
                    and checkpoint.state is RecoveryCheckpointState.ACTIVE
                    and checkpoint.pending_approval is not None
                ):
                    _, batch = await self._approval_batch(execution, checkpoint)
                    records = await self._approval_records(execution, batch)
                    if any(record.status is ApprovalStatus.PENDING for record in records):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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
        if execution.status is ExecutionStatus.PENDING_START:
            if session.status is SessionStatus.OPEN:
                return True
            if session.status is SessionStatus.CLOSED:
                await self._commit_terminal(
                    execution,
                    ExecutionStatus.FAILED,
                    None,
                    ErrorCode.SESSION_CONFLICT.value,
                    StopReason.ERROR,
                )
                return False
            if session.active_execution_id == execution.execution_id:
                await self._commit_terminal(
                    execution,
                    ExecutionStatus.CANCELLED,
                    None,
                    ErrorCode.EXECUTION_CANCELLED.value,
                    StopReason.CANCELLED,
                )
                return False
            await self._commit_terminal(
                execution,
                ExecutionStatus.FAILED,
                None,
                ErrorCode.SESSION_CONFLICT.value,
                StopReason.ERROR,
            )
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

    def _validate_recovery_handoff_integrity(
        self,
        checkpoint: RecoveryCheckpoint,
        execution: ExecutionRecord | None,
    ) -> None:
        handoff = checkpoint.terminal_handoff
        if handoff is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        outcome = handoff.outcome
        if outcome.terminal_status in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED} and (
            handoff.conversation is not None
            or outcome.output is not None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            execution is not None
            and execution.status
            in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}
            and execution.status is not outcome.terminal_status
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _validate_handoff_output(self, outcome: RecoveryTerminalOutcome) -> None:
        output = outcome.output
        if output is None:
            return
        if output.kind == "inline":
            output.decode()
            return
        source = outcome.object_source_domain
        reference = output.ref
        if source is None or reference is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        store = self._recovery_objects if source is RuntimeDomain.RECOVERY else self._execution_objects
        await read_runtime_object(store, reference)

    async def _reconcile_handoff(self, checkpoint: RecoveryCheckpoint) -> ExecutionRecord:
        handoff = checkpoint.terminal_handoff
        if handoff is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        outcome = handoff.outcome
        if outcome.terminal_status in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED} and (
            handoff.conversation is not None
            or outcome.output is not None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        execution = await self._execution.executions.get(
            checkpoint.execution_id,
            tenant_id=checkpoint.tenant_id,
        )
        if execution is None:
            execution = await self._create_recovery_execution(checkpoint)
        else:
            self._validate_recovery_identity(execution, checkpoint.input)
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
                    execution_id=checkpoint.execution_id,
                )
                snapshot = await self._step_reads[RuntimeDomain.EXECUTION].latest_snapshot(
                    run_id=handoff.source_step_run_id,
                )
                if snapshot is None or snapshot.state != "complete":
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if outcome.output is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self._validate_handoff_output(outcome)
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
            elif outcome.output is not None:
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
            execution = await self._commit_reconciled_terminal(checkpoint)
            await self._validate_committed_handoff(checkpoint)
            if execution.session_id is not None:
                await self._conversation.sessions.release_execution(
                    execution.session_id,
                    tenant_id=execution.tenant_id,
                    execution_id=execution.execution_id,
                )
            await self._complete_handoff(checkpoint)
            self._publish_terminal_event(
                execution.execution_id,
                event_type=outcome.terminal_event_type,
                payload=dict(outcome.terminal_event_payload),
                durable_sequence=execution.event_sequence,
            )
            _logger.info(
                "recovery handoff completed: execution=%s",
                checkpoint.execution_id,
            )
            return execution
        if checkpoint.handoff_phase is RecoveryHandoffPhase.CONVERSATION_RESOLVED:
            execution = await self._commit_reconciled_terminal(checkpoint)
            checkpoint = await self._advance_handoff(
                checkpoint,
                RecoveryHandoffPhase.EXECUTION_COMMITTED,
            )
        if checkpoint.handoff_phase is RecoveryHandoffPhase.EXECUTION_COMMITTED:
            await self._validate_committed_handoff(checkpoint)
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
            self._publish_terminal_event(
                execution.execution_id,
                event_type=outcome.terminal_event_type,
                payload=dict(outcome.terminal_event_payload),
                durable_sequence=execution.event_sequence,
            )
            _logger.info(
                "recovery handoff completed: execution=%s",
                checkpoint.execution_id,
            )
            return execution
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

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
            mode=recovery_input.mode,
            planning=recovery_input.planning,
            thinking=recovery_input.thinking,
            binding=recovery_input.binding,
            repository_instructions=recovery_input.repository_instructions,
        )
        await self._execution.executions.create_with_history_head(execution)
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
            pending_approval=None,
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
        if checkpoint.handoff_phase not in {
            RecoveryHandoffPhase.PREPARED,
            RecoveryHandoffPhase.EXECUTION_COMMITTED,
        }:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        completed = replace(
            checkpoint,
            handoff_phase=RecoveryHandoffPhase.COMPLETED,
            state=RecoveryCheckpointState.COMPLETED,
            pending_approval=None,
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
                output=None,
                object_source_domain=None,
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
    ) -> ExecutionRecord:
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
        self._validate_recovery_identity(current, checkpoint.input)
        if current.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            if current.status is not outcome.terminal_status:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return current
        execution_ref = None
        if outcome.terminal_status is ExecutionStatus.SUCCEEDED:
            if outcome.output is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            execution_ref = outcome.output
            if outcome.output.kind == "object" and outcome.object_source_domain is RuntimeDomain.RECOVERY:
                reference = outcome.output.ref
                if reference is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                payload = await read_runtime_object(self._recovery_objects, reference)
                execution_ref = StoredPayload.object(
                    await put_runtime_object(
                        self._execution_objects,
                        RuntimeObjectKeyFactory(self._namespace),
                        RuntimeDomain.EXECUTION,
                        current.tenant_id,
                        payload,
                    )
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
        result = ResultRecord(
            current.execution_id,
            current.tenant_id,
            execution_ref,
            outcome.stop_reason,
            outcome.usage,
            outcome.result_created_at,
        )
        try:
            terminal_run_id = handoff.source_step_run_id
            if terminal_run_id is None and current.agent_run_sequence > 0:
                terminal_run_id = step_run_id(
                    namespace=self._namespace,
                    tenant_id=current.tenant_id,
                    execution_id=current.execution_id,
                    segment_sequence=current.agent_run_sequence,
                )
            committed = await self._commit_execution_terminal_checkpoint(
                current,
                ExecutionTerminalCommit(
                    current.revision,
                    current.event_sequence,
                    terminal,
                    result,
                    outcome.terminal_event_type,
                    dict(outcome.terminal_event_payload),
                ),
                run_id=terminal_run_id,
            )
            current = committed.execution
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
            current = latest
        _logger.info(
            "recovery execution terminal committed: execution=%s status=%s",
            current.execution_id,
            outcome.terminal_status.value,
        )
        return current

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
        self._validate_recovery_identity(execution, checkpoint.input)
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
                result.output is None
                or handoff.outcome.output is None
                or result.output.digest != handoff.outcome.output.digest
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        elif result.output is not None:
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
        for execution_id, task in tasks:
            current = await self._execution.executions.get(
                execution_id,
                tenant_id=self._tenant_id,
            )
            if current is None or current.status not in {
                ExecutionStatus.FINALIZING,
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }:
                self._request_worker_cancel(execution_id, task)
            else:
                _logger.info(
                    "close draining terminal execution worker: execution=%s status=%s",
                    execution_id,
                    current.status.value,
                )

        for execution_id, task in tasks:
            await self._drain_worker_task(execution_id, task)

        while True:
            background = self._executor.pending_background_tasks
            dispatcher_background = (
                ()
                if self._subagent_dispatcher is None
                else self._subagent_dispatcher.pending_background_tasks
            )
            checkpoint_background = tuple(
                task for task in self._checkpoint_tasks if not task.done()
            )
            execution_task_map = self._execution_task_map()
            execution_background = tuple(
                task
                for tasks_for_execution in execution_task_map.values()
                for task in tasks_for_execution
                if isinstance(task, asyncio.Task) and not task.done()
            )
            pending_background_by_identity = {
                id(task): task
                for task in (
                    *background,
                    *dispatcher_background,
                    *checkpoint_background,
                    *execution_background,
                )
                if isinstance(task, asyncio.Task) and not task.done()
            }
            pending_background = tuple(pending_background_by_identity.values())
            if not pending_background:
                break
            await asyncio.gather(
                *(asyncio.shield(task) for task in pending_background),
                return_exceptions=True,
            )

        dispatcher_failure = (
            None
            if self._subagent_dispatcher is None
            else self._subagent_dispatcher.background_failure
        )
        if dispatcher_failure is not None:
            raise AIError(
                dispatcher_failure.code,
                safe_details=dict(dispatcher_failure.safe_details),
            )
        self._tasks.clear()
        self._captured_usage.clear()
        self._terminal_events.clear()
        self._worker_failures.clear()
        self._pending_audit_events.clear()
        self._pending_audit_locks.clear()
        self._approval_pause_segments.clear()
        self._segment_only_worker_exits.clear()
        self._repository_instruction_provenance.clear()
        self._checkpoint_tasks.clear()
        self._worker_cancel_requests.clear()
        self._execution_task_map().clear()

    async def release_runtime_execution(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> None:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        task = self._tasks.get(execution_id)
        if task is not None and not task.done():
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        execution_tasks = self._execution_task_map().get(execution_id)
        pending_execution_tasks = tuple(
            task
            for task in (() if execution_tasks is None else execution_tasks)
            if not task.done()
        )
        if pending_execution_tasks:
            _logger.debug(
                "local execution release blocked by durable tasks: execution=%s count=%s",
                execution_id,
                len(pending_execution_tasks),
            )
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if task is not None:
            self._tasks.pop(execution_id, None)
        self._worker_cancel_requests.discard(execution_id)
        self._terminal_events.pop(execution_id, None)
        self._worker_failures.pop(execution_id, None)
        self._captured_usage.pop(execution_id, None)
        self._pending_audit_events.pop(execution_id, None)
        self._pending_audit_locks.pop(execution_id, None)
        self._approval_pause_segments.pop(execution_id, None)
        self._segment_only_worker_exits.discard(execution_id)
        self._repository_instruction_provenance.pop(execution_id, None)
        self._execution_task_map().pop(execution_id, None)
        _logger.debug(
            "local execution runtime cache released: tenant=%s execution=%s",
            tenant_id,
            execution_id,
        )

    async def _reconcile_unresolved_tool_effects(
        self,
        step_run_id: str,
        effects: list[ToolEffectRecord],
        *,
        tenant_id: str,
    ) -> None:
        for effect in effects:
            while True:
                if self._tool_operations is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                operation = await self._tool_operations.get_by_call(
                    step_run_id,
                    effect.tool_call_id,
                    tenant_id=tenant_id,
                )
                if operation is None:
                    raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
                if operation.status in {
                    ToolOperationStatus.COMPLETED,
                    ToolOperationStatus.FAILED,
                }:
                    break
                if operation.status in {
                    ToolOperationStatus.EFFECT_UNKNOWN,
                    ToolOperationStatus.CANCELLED,
                }:
                    raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
                if operation.status is ToolOperationStatus.CLAIMED:
                    expires = operation.lease_expires_at
                    if expires is None:
                        raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
                    remaining = (expires - datetime.now(timezone.utc)).total_seconds()
                    if remaining > 0:
                        await asyncio.sleep(min(1.0, remaining))
                        continue
                    if not operation.replay_safe:
                        raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
                break
        _logger.info(
            "recovery tool effects reconciled: run=%s count=%s",
            step_run_id,
            len(effects),
        )

    async def _run(self, request: ExecutionRequest, original: ExecutionRecord) -> None:
        execution_id = original.execution_id
        execution_tasks = self._execution_task_set(execution_id)
        checkpoint: RecoveryCheckpoint | None = None
        run_id: str | None = None
        recovery_history_run_id: str | None = None
        try:
            recovery_relaunch_ids = self._recovery_relaunch_ids
        except AttributeError:
            recovery_relaunch_ids = set()
        exact_recovery_context = execution_id in recovery_relaunch_ids
        recovery_relaunch_ids.discard(execution_id)
        operation_started_at = monotonic()
        operation_result = "failure"
        claimed_from_admitted = False
        try:
            current = await self._execution.executions.get(execution_id, tenant_id=original.tenant_id)
            if current is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            if self._recovery_enabled:
                checkpoint = await self._recovery.checkpoints.get(
                    execution_id,
                    tenant_id=current.tenant_id,
                )
                if checkpoint is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                self._validate_recovery_identity(current, checkpoint.input)
                if checkpoint.state is RecoveryCheckpointState.ADMITTED:
                    current, checkpoint = (
                        await self._runtime_commands.commit_agent_attempt_checkpoint(
                            AgentAttemptClaim(
                                execution_id=execution_id,
                                tenant_id=current.tenant_id,
                                expected_execution_revision=current.revision,
                                expected_agent_run_sequence=current.agent_run_sequence,
                                expected_recovery_revision=checkpoint.revision,
                                expected_recovery_state=checkpoint.state,
                            )
                        )
                    )
                    claimed_from_admitted = True
                    _logger.info(
                        "agent attempt admitted and activated: execution=%s sequence=%s",
                        execution_id,
                        current.agent_run_sequence,
                    )
                elif checkpoint.state in {
                    RecoveryCheckpointState.ACTIVE,
                    RecoveryCheckpointState.WAITING,
                }:
                    exact_recovery_context = True
                    if (
                        checkpoint.step_run_id is None
                        or checkpoint.agent_run_sequence != current.agent_run_sequence
                    ):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    recovery_history_run_id = checkpoint.step_run_id
                else:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            else:
                current = await self._execution.executions.claim_next_agent_run(
                    execution_id,
                    tenant_id=current.tenant_id,
                    expected_revision=current.revision,
                    expected_agent_run_sequence=current.agent_run_sequence,
                )
            binding = self._catalog.binding(current.binding_digest)
            if current.binding != binding.snapshot:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            definition = binding.definition
            repository_instructions = await self._load_repository_instructions(
                current.repository_instructions
            )
            run_id = (
                checkpoint.step_run_id
                if checkpoint is not None and checkpoint.step_run_id is not None
                else step_run_id(
                    namespace=self._namespace,
                    tenant_id=current.tenant_id,
                    execution_id=execution_id,
                    segment_sequence=current.agent_run_sequence,
                )
            )
            conversation_id = step_conversation_id(
                namespace=self._namespace,
                tenant_id=current.tenant_id,
                execution_id=execution_id,
            )
            session = (
                None
                if current.session_id is None
                else await self._conversation.sessions.get(
                    current.session_id,
                    tenant_id=current.tenant_id,
                )
            )
            history_id = (
                None
                if session is None
                else session.history_id
                or (None if session.continuation is None else session.continuation.history_id)
            )
            tool_owner = f"tool:{execution_id}:{uuid.uuid4().hex}"
            source_replay_history: list[ModelMessage] | None = None
            deferred_tool_results: DeferredToolResults | None = None
            resumed_approval_attempt = False
            if self._recovery_enabled:
                if checkpoint is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if checkpoint.state in {
                    RecoveryCheckpointState.ACTIVE,
                    RecoveryCheckpointState.WAITING,
                }:
                    current_run_id = checkpoint.step_run_id
                    if current_run_id is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if checkpoint.pending_approval is not None:
                        if (
                            checkpoint.state is not RecoveryCheckpointState.ACTIVE
                            or current.status is not ExecutionStatus.STARTED
                            or current_run_id == checkpoint.pending_approval.source_step_run_id
                        ):
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        resumed_approval_attempt = True
                        await self._step_lifecycle.wait_projection_flight(current_run_id)
                        recovery_archive = self._step_reads[RuntimeDomain.RECOVERY]
                        if not isinstance(recovery_archive, StateStepArchive):
                            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                        durable_current_run = await recovery_archive.get_run(
                            run_id=current_run_id
                        )
                        current_snapshot = await recovery_archive.latest_snapshot(
                            run_id=current_run_id,
                            include_interrupted=True,
                        )
                        if self._tool_operations is None:
                            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                        current_has_tool_operation = await self._tool_operations.has_by_step_run(
                            current_run_id,
                            tenant_id=current.tenant_id,
                        )
                        if current_snapshot is not None:
                            if (
                                durable_current_run is None
                                or current_snapshot.run_id != current_run_id
                            ):
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                            unresolved = await recovery_archive.list_unresolved_tool_effects(
                                run_id=current_run_id
                            )
                            if unresolved:
                                await self._reconcile_unresolved_tool_effects(
                                    current_run_id,
                                    unresolved,
                                    tenant_id=current.tenant_id,
                                )
                            recovery_history_run_id = current_run_id
                        elif durable_current_run is not None or current_has_tool_operation:
                            raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
                        else:
                            source_snapshot, batch = await self._approval_batch(
                                current,
                                checkpoint,
                            )
                            records = await self._approval_records(current, batch)
                            if any(record.status is ApprovalStatus.PENDING for record in records):
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                            deferred_tool_results = self._deferred_tool_results(
                                current,
                                binding,
                                batch,
                                records,
                            )
                            source_replay_history = list(source_snapshot.messages)
                            recovery_history_run_id = None
                    else:
                        recovery_history_run_id = current_run_id
                        recovery_archive = self._step_store(RuntimeDomain.RECOVERY)
                        recovery_run = await recovery_archive.get_run(
                            run_id=recovery_history_run_id
                        )
                        snapshot = await recovery_archive.latest_snapshot(
                            run_id=recovery_history_run_id,
                            include_interrupted=True,
                        )
                        unresolved = await recovery_archive.list_unresolved_tool_effects(
                            run_id=recovery_history_run_id
                        )
                        if snapshot is None:
                            if recovery_run is None:
                                _logger.info(
                                    "recovery attempt has no durable step progress: execution=%s run=%s",
                                    execution_id,
                                    recovery_history_run_id,
                                )
                                recovery_history_run_id = None
                            else:
                                raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
                        elif unresolved:
                            await self._reconcile_unresolved_tool_effects(
                                recovery_history_run_id,
                                unresolved,
                                tenant_id=current.tenant_id,
                            )
            try:
                tool_repository = self._tool_operations
            except AttributeError:
                tool_repository = None
            tool_operations = RuntimeToolOperationBridge(
                tool_repository,
                self._recovery_objects,
                namespace=self._namespace,
                tenant_id=current.tenant_id,
                execution_id=execution_id,
                step_run_id=run_id,
                binding_digest=current.binding_digest,
                owner=tool_owner,
                background_tasks=execution_tasks,
                payload_policy=self._payload_policy,
                recovery_step_run_id=recovery_history_run_id,
                terminal_commands=(
                    self._runtime_commands
                    if isinstance(
                        self._step_reads[RuntimeDomain.RECOVERY],
                        StateStepArchive,
                    )
                    else None
                ),
            ) if tool_repository is not None else None
            loaded_context = LoadedModelContext(())
            session_history_source = (
                current.lineage_kind is ExecutionLineageKind.SESSION_RESUME
                and history_id is not None
            )
            if source_replay_history is not None:
                history = source_replay_history
            elif recovery_history_run_id is not None:
                loaded_context = await self._steps.load_loaded_model_context(
                    RuntimeDomain.RECOVERY,
                    recovery_history_run_id,
                )
                history = list(loaded_context.model_messages())
            else:
                if session_history_source:
                    loaded_context = await self._steps.load_loaded_model_context(
                        RuntimeDomain.CONVERSATION,
                        history_id,
                    )
                if session_history_source and loaded_context.messages:
                    history = list(loaded_context.model_messages())
                else:
                    history = cast("list[ModelMessage]", await self._history(current))
            session_history_start = (
                current.lineage_kind is ExecutionLineageKind.SESSION_RESUME
                and bool(history)
            )
            if isinstance(self._steps, RuntimeStepStore):
                self._steps.register_context_baseline(run_id, loaded_context)
            if repository_instructions is None:
                repository_provenance = _RepositoryInstructionProvenance((), frozenset())
            elif checkpoint is None or claimed_from_admitted:
                repository_provenance = _RepositoryInstructionProvenance(
                    tuple(history),
                    frozenset(),
                )
            else:
                repository_provenance = await self._repository_instruction_provenance_for_scope(
                    current,
                    checkpoint,
                    fallback_history=history,
                )
            run_user_prompt = (
                None
                if resumed_approval_attempt
                else user_prompt_transport(request.user_prompt, request.user_prompt_codec)
            )

            async def sink(emission: "LiveDelta | DurableBoundary") -> None:
                if isinstance(emission, LiveDelta):
                    self._live_broker.publish(
                        ExecutionDelta(
                            current.execution_id,
                            emission.kind,
                            emission.content,
                        )
                    )
                    return
                await self._append_event(current, emission.kind, emission.payload)

            subagent_refs = binding.snapshot.subagents
            subagent_available = (
                current.parent_execution_id is None
                and self._subagent_dispatcher is not None
                and bool(subagent_refs)
            )
            runtime_tool_names = select_runtime_tool_names(
                ordinary_tool_policy=definition.ordinary_tool_policy,
                memory_scope=current.memory_scope,
                planning=current.planning,
                subagent_available=subagent_available,
            )
            memory = None
            selected_memory = tuple(name for name in runtime_tool_names if name in MEMORY_TOOL_NAMES)
            if selected_memory:
                if self._memory_store_factory is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                memory = self._memory_store_factory(
                    current.tenant_id,
                    current.execution_id,
                    current.memory_scope or "default",
                )
            session_metadata: Mapping[str, JsonValue] = {}
            if current.session_id is not None:
                session = await self._conversation.sessions.get(
                    current.session_id,
                    tenant_id=current.tenant_id,
                )
                if session is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                session_metadata = session.metadata
            public_context = RunContext(
                app=self._app,
                principal=request.principal,
                workspace=self._workspace,
                session_id=current.session_id,
                execution_id=current.execution_id,
                session_metadata=session_metadata,
            )
            plan_store = RuntimePlanStore(
                self._session_state_store if current.session_id is not None else self._execution_state_store,
                namespace=self._namespace,
                tenant_id=current.tenant_id,
                owner_kind="session" if current.session_id is not None else "execution",
                owner_id=current.session_id or current.execution_id,
            )
            try:
                result = await self._executor.execute(
                    _RunScope(
                        binding=binding,
                        context=public_context,
                        user_prompt=run_user_prompt,
                        history=history,
                        conversation_id=conversation_id,
                        step_store=self._steps,
                        step_run_id=run_id,
                        segment_sequence=current.agent_run_sequence,
                        history_id=history_id,
                        memory_scope=current.memory_scope,
                        memory_store=memory,
                        plan_store_resolver=lambda _ctx: plan_store,
                        mode=current.mode,
                        planning=current.planning,
                        thinking=current.thinking,
                        parent_step_run_id=None,
                        subagent_available=subagent_available,
                        subagent_descriptions=(
                            {}
                            if self._subagent_dispatcher is None
                            else self._subagent_dispatcher.descriptions_for(subagent_refs)
                        ),
                        subagent_delegate=(
                            None
                            if not subagent_available
                            else self._subagent_dispatcher.delegate_for(
                                parent_execution_id=current.execution_id,
                                root_execution_id=current.root_execution_id,
                                memory_scope=current.memory_scope,
                                principal=request.principal,
                                refs=subagent_refs,
                                mode=current.mode,
                            )
                        ),
                        event_sink=sink,
                        usage_sink=lambda usage: self._capture_usage(execution_id, usage),
                        tool_operations=tool_operations,
                        background_tasks=execution_tasks,
                        replace_history_system_prompt=(
                            session_history_start and not exact_recovery_context
                        ),
                        repository_instructions=repository_instructions,
                        repository_instruction_history=repository_provenance.messages,
                        repository_instruction_marker_authority=(
                            repository_provenance.marker_authority
                        ),
                        deferred_tool_results=deferred_tool_results,
                    )
                )
            except Exception as error:
                if _is_infrastructure_error(error):
                    raise
                current = await self._execution.executions.get(
                    execution_id,
                    tenant_id=original.tenant_id,
                )
                if current is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if current.status not in {
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                }:
                    try:
                        committed_failure = await self._commit_failure(
                            current,
                            error,
                            run_id=run_id,
                        )
                        current = committed_failure
                    except asyncio.CancelledError:
                        raise
                    except Exception as commit_error:
                        try:
                            persisted = await self._execution.executions.get(
                                execution_id,
                                tenant_id=original.tenant_id,
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as readback_error:  # noqa: BLE001
                            raise _secondary_execution_error(readback_error, error) from error
                        if persisted is not None and persisted.status in {
                            ExecutionStatus.SUCCEEDED,
                            ExecutionStatus.FAILED,
                            ExecutionStatus.CANCELLED,
                        }:
                            _logger.exception(
                                "terminal finalization failed after durable execution terminal: execution=%s",
                                execution_id,
                            )
                        raise _secondary_execution_error(commit_error, error) from error
                operation_result = _execution_operation_result(current.status)
                _logger.exception(
                    "local execution failed: execution=%s",
                    execution_id,
                )
                return
            if isinstance(result, AgentExecutionPaused):
                if not self._recovery_enabled or checkpoint is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                await self._commit_approval_pause(current, result)
                operation_result = "success"
                return
            committed = await self._commit_success(
                current,
                binding,
                result.output,
                result.usage,
                run_id,
            )
            operation_result = _execution_operation_result(committed.status)
            _logger.debug("local execution completed: execution=%s run=%s", execution_id, run_id)
        except asyncio.CancelledError:
            current = await self._execution.executions.get(execution_id, tenant_id=original.tenant_id)
            if current is not None and current.status is ExecutionStatus.FINALIZING:
                raise
            if current is not None and current.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                current = await self._commit_terminal(current, ExecutionStatus.CANCELLED, None, ErrorCode.EXECUTION_CANCELLED.value, StopReason.CANCELLED, run_id=run_id)
            if current is not None:
                operation_result = _execution_operation_result(current.status)
            raise
        except Exception:
            _logger.exception(
                "local execution infrastructure failure: execution=%s",
                execution_id,
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
            pending_approval=None,
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

    async def _history(
        self,
        execution: ExecutionRecord,
        recovery_run_id: str | None = None,
    ) -> list[object]:
        execution_steps = self._step_store(RuntimeDomain.EXECUTION)
        conversation_steps = self._step_store(RuntimeDomain.CONVERSATION)
        if recovery_run_id is not None:
            try:
                recovery_archive = self._step_store(RuntimeDomain.RECOVERY)
                if isinstance(recovery_archive, StateStepArchive):
                    return list(
                        (
                            await recovery_archive.load_loaded_model_context(
                                owner_id=recovery_run_id,
                            )
                        ).model_messages()
                    )
                return list(
                    await continue_run(
                        self._step_store(RuntimeDomain.RECOVERY),
                        run_id=recovery_run_id,
                        include_interrupted=True,
                    )
                )
            except LookupError as error:
                raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE) from error
        session = (
            None
            if execution.session_id is None
            else await self._conversation.sessions.get(
                execution.session_id,
                tenant_id=execution.tenant_id,
            )
        )
        if (
            execution.lineage_kind is ExecutionLineageKind.SESSION_RESUME
            and session is not None
            and (
                session.history_id is not None
                or (
                    session.continuation is not None
                    and session.continuation.history_id is not None
                )
            )
        ):
            history_id = session.history_id or session.continuation.history_id
            try:
                return [
                    message
                    async for message in self._steps.iter_session_messages(
                        history_id,
                        tenant_id=execution.tenant_id,
                    )
                ]
            except AIError as error:
                if error.code is ErrorCode.SESSION_HISTORY_UNAVAILABLE:
                    raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE) from error
                raise
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

    async def _append_event(
        self,
        execution: ExecutionRecord,
        event_type: ExecutionEventType,
        payload: JsonValue,
    ) -> None:
        event_payload = payload if isinstance(payload, Mapping) else {"value": payload}
        async with self._audit_lock(execution.execution_id):
            self._pending_audit_events.setdefault(execution.execution_id, []).append(
                ExecutionEventAppend(event_type, event_payload)
            )
            self._live_broker.publish_event(
                execution.execution_id,
                event_type,
                event_payload,
                durable_sequence=None,
            )
        _logger.debug(
            "execution audit event buffered: execution=%s type=%s pending=%s",
            execution.execution_id,
            event_type.value,
            len(self._pending_audit_events.get(execution.execution_id, ())),
        )

    def _audit_lock(self, execution_id: str) -> asyncio.Lock:
        try:
            pending_locks = self._pending_audit_locks
        except AttributeError:
            pending_locks = {}
            self._pending_audit_locks = pending_locks
        return pending_locks.setdefault(execution_id, asyncio.Lock())

    def _confirm_committed_events(
        self,
        execution_id: str,
        *,
        pending_count: int,
        durable_sequence: int,
    ) -> None:
        if pending_count:
            self._live_broker.confirm_events(
                execution_id,
                first_sequence=durable_sequence - pending_count,
                count=pending_count,
            )

    def _publish_terminal_event(
        self,
        execution_id: str,
        *,
        event_type: ExecutionEventType,
        payload: JsonValue,
        durable_sequence: int,
    ) -> None:
        self._live_broker.publish_event(
            execution_id,
            event_type,
            payload,
            durable_sequence=durable_sequence,
        )

    def _step_store(self, runtime_domain: RuntimeDomain) -> StepStore:
        return self._step_reads[runtime_domain]

    async def _commit_success(
        self,
        execution: ExecutionRecord,
        binding: AgentBinding,
        output: JsonValue,
        usage: UsageMetrics,
        run_id: str,
    ) -> ExecutionRecord:
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
            return current
        payload = canonical_json_bytes(output)
        inline = StoredPayload.inline_json(output)
        output_payload = inline
        if not payload_fits_inline(inline, self._payload_policy):
            object_ref = await put_runtime_object(
                self._execution_objects,
                RuntimeObjectKeyFactory(self._namespace),
                RuntimeDomain.EXECUTION,
                execution.tenant_id,
                payload,
            )
            output_payload = StoredPayload.object(object_ref)
        if self._recovery_enabled:
            return await self._commit_terminal(
                current,
                ExecutionStatus.SUCCEEDED,
                output_payload,
                None,
                StopReason.END_TURN,
                binding=binding,
                run_id=run_id,
                usage=usage,
            )
        if current.status is ExecutionStatus.CANCELLING:
            return current
        await self.verify_terminal_projection(current, ExecutionStatus.SUCCEEDED, run_id)
        if current.session_id is not None:
            expected_cursor = await self._expected_session_cursor(current)
            current = await self._claim_session_finalizing(current)
            if current.status is ExecutionStatus.CANCELLING:
                _logger.info(
                    "session success lost to cancellation: execution=%s",
                    current.execution_id,
                )
                return current
            if current.status in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }:
                return current
            snapshot = await self._steps.latest_snapshot(run_id=run_id)
            step_run = await self._steps.get_run(run_id=run_id)
            if snapshot is None or snapshot.state != "complete" or step_run is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        else:
            expected_cursor = None
            snapshot = None
            step_run = None
        return await self._commit_terminal(
            current,
            ExecutionStatus.SUCCEEDED,
            output_payload,
            None,
            StopReason.END_TURN,
            binding=binding,
            run_id=run_id,
            usage=usage,
            expected_cursor=expected_cursor,
            conversation_run=step_run,
            conversation_snapshot=snapshot,
        )

    async def _expected_session_cursor(
        self,
        execution: ExecutionRecord,
    ) -> ConversationCursor | None:
        history_id = None
        if execution.session_id is not None:
            session = await self._conversation.sessions.get(
                execution.session_id,
                tenant_id=execution.tenant_id,
            )
            if session is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            history_id = session.history_id or (
                None
                if session.continuation is None
                else session.continuation.history_id
            )
        if execution.conversation_step_run_id is not None:
            return ConversationCursor(
                execution.conversation_step_run_id,
                history_id=history_id,
            )
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
            ),
            history_id=history_id,
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
        session = await self._conversation.sessions.get(
            execution.session_id or "",
            tenant_id=execution.tenant_id,
        )
        if session is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        next_cursor = ConversationCursor(
            source_run_id,
            history_id=session.history_id
            or (None if session.continuation is None else session.continuation.history_id),
        )
        conversation_archive = self._step_reads[RuntimeDomain.CONVERSATION]
        if isinstance(conversation_archive, StateStepArchive):
            run = await self._steps.get_run(run_id=source_run_id)
            if run is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                await self._conversation_commands.commit_snapshot_and_advance(
                    execution.session_id or "",
                    tenant_id=execution.tenant_id,
                    execution_id=execution.execution_id,
                    expected=expected_cursor,
                    next_cursor=next_cursor,
                    step_run=run,
                    snapshot=snapshot,
                )
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN:
                    raise
                current = await self._conversation.sessions.get(
                    execution.session_id or "",
                    tenant_id=execution.tenant_id,
                )
                if current is None or current.continuation != next_cursor:
                    raise
                _logger.warning(
                    "conversation checkpoint commit unknown but cursor advanced: execution=%s run=%s",
                    execution.execution_id,
                    source_run_id,
                )
            _logger.info(
                "conversation snapshot checkpoint committed: execution=%s run=%s",
                execution.execution_id,
                source_run_id,
            )
            return
        await self._step_lifecycle.materialize_conversation(step_run_id=source_run_id)
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

    async def _commit_failure(
        self,
        execution: ExecutionRecord,
        error: Exception,
        *,
        run_id: str | None = None,
    ) -> ExecutionRecord:
        code = _execution_error_code(error)
        details = _execution_error_details(error)
        cancelled = code is ErrorCode.EXECUTION_CANCELLED
        return await self._commit_terminal(
            execution,
            ExecutionStatus.CANCELLED if cancelled else ExecutionStatus.FAILED,
            None,
            code.value,
            StopReason.CANCELLED if cancelled else StopReason.ERROR,
            run_id=run_id,
            safe_error_details=details,
        )

    async def _commit_terminal(
        self,
        execution: ExecutionRecord,
        status: ExecutionStatus,
        output: "StoredPayload | None",
        error_code: str | None,
        stop_reason: StopReason,
        *,
        binding: AgentBinding | None = None,
        run_id: str | None = None,
        usage: UsageMetrics | None = None,
        safe_error_details: Mapping[str, JsonValue] | None = None,
        expected_cursor: ConversationCursor | None = None,
        conversation_run: RunRecord | None = None,
        conversation_snapshot: ContinuableSnapshot | None = None,
        recovery_checkpoint: RecoveryCheckpoint | None = None,
    ) -> ExecutionRecord:
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
            return current
        now = datetime.now(timezone.utc)
        captured_usage = usage or self._captured_usage.get(execution.execution_id, UsageMetrics())
        if status is ExecutionStatus.SUCCEEDED:
            if binding is None or output is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        elif output is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if self._recovery_enabled:
            recovery_checkpoint = await self._recovery.checkpoints.get(
                current.execution_id,
                tenant_id=current.tenant_id,
            )
            if recovery_checkpoint is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._validate_recovery_identity(current, recovery_checkpoint.input)
            if (
                recovery_checkpoint.handoff_phase is RecoveryHandoffPhase.NONE
                and self._same_terminal_storage_group(
                    current,
                    status=status,
                    run_id=run_id,
                )
            ):
                return await self._commit_same_group_recovery_terminal(
                    current,
                    recovery_checkpoint,
                    status=status,
                    output=output,
                    error_code=error_code,
                    stop_reason=stop_reason,
                    binding=binding,
                    run_id=run_id,
                    usage=captured_usage,
                    safe_error_details=safe_error_details,
                )
            if recovery_checkpoint.handoff_phase is RecoveryHandoffPhase.NONE:
                if status is ExecutionStatus.SUCCEEDED and run_id is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if run_id is not None:
                    await self._step_lifecycle.materialize_recovery_snapshot(
                        step_run_id=run_id,
                        require_complete=status is ExecutionStatus.SUCCEEDED,
                    )
                handoff_output = output
                source_domain = None
                if output is not None and output.kind == "object":
                    if self._execution_objects_durable:
                        source_domain = RuntimeDomain.EXECUTION
                    else:
                        reference = output.ref
                        if reference is None:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        content = await read_runtime_object(self._execution_objects, reference)
                        handoff_output = StoredPayload.object(
                            await put_runtime_object(
                                self._recovery_objects,
                                RuntimeObjectKeyFactory(self._namespace),
                                RuntimeDomain.RECOVERY,
                                current.tenant_id,
                                content,
                            )
                        )
                        source_domain = RuntimeDomain.RECOVERY
                conversation = None
                if status is ExecutionStatus.SUCCEEDED and current.session_id is not None and run_id is not None:
                    session = await self._conversation.sessions.get(
                        current.session_id,
                        tenant_id=current.tenant_id,
                    )
                    if session is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    conversation = RecoveryConversationIntent(
                        current.session_id,
                        await self._expected_session_cursor(current),
                        ConversationCursor(
                            run_id,
                            history_id=session.history_id
                            or (
                                None
                                if session.continuation is None
                                else session.continuation.history_id
                            ),
                        ),
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
                        safe_error_details=safe_error_details or {},
                        stop_reason=stop_reason,
                        output=handoff_output,
                        object_source_domain=source_domain,
                        usage=captured_usage,
                        terminal_event_type=terminal_event_type,
                        terminal_event_payload=(
                            {"run_id": run_id}
                            if status is ExecutionStatus.SUCCEEDED and run_id is not None
                            else {
                                "error_code": error_code,
                                "safe_error_details": safe_error_details or {},
                            }
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
                    pending_approval=None,
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
            return await self._reconcile_handoff(recovery_checkpoint)
        await self.verify_terminal_projection(
            current,
            status,
            run_id if status is ExecutionStatus.SUCCEEDED else None,
        )
        terminal = _terminal_record(
            current,
            status,
            now,
            error_code=error_code,
            safe_error_details=safe_error_details,
        )
        terminal_commit = ExecutionTerminalCommit(
            current.revision,
            current.event_sequence,
            terminal,
            ResultRecord(
                current.execution_id,
                current.tenant_id,
                output if status is ExecutionStatus.SUCCEEDED else None,
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
                else {
                    "error_code": error_code,
                    "safe_error_details": safe_error_details or {},
                }
            ),
        )
        committed = await self._commit_execution_terminal_checkpoint(
            current,
            terminal_commit,
            run_id=run_id,
            expected_cursor=expected_cursor,
            conversation_run=conversation_run,
            conversation_snapshot=conversation_snapshot,
            recovery_checkpoint=recovery_checkpoint,
        )
        self._publish_terminal_event(
            current.execution_id,
            event_type=terminal_commit.terminal_event_type,
            payload=dict(terminal_commit.terminal_event_payload),
            durable_sequence=committed.execution.event_sequence,
        )
        _logger.info(
            "execution terminal committed: execution=%s status=%s",
            current.execution_id,
            status.value,
        )
        return committed.execution

    def _same_terminal_storage_group(
        self,
        execution: ExecutionRecord,
        *,
        status: ExecutionStatus,
        run_id: str | None,
    ) -> bool:
        stores = [
            self._execution.executions.state_store,
            self._recovery.checkpoints.state_store,
        ]
        if execution.session_id is not None:
            stores.append(self._conversation.sessions.state_store)
        if run_id is not None:
            for domain in (RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY):
                archive = self._step_reads[domain]
                if isinstance(archive, StateStepArchive):
                    stores.append(archive.state_store)
            if execution.session_id is not None and status is ExecutionStatus.SUCCEEDED:
                archive = self._step_reads[RuntimeDomain.CONVERSATION]
                if isinstance(archive, StateStepArchive):
                    stores.append(archive.state_store)
        return all(store.storage_group is stores[0].storage_group for store in stores[1:])

    async def _commit_same_group_recovery_terminal(
        self,
        current: ExecutionRecord,
        checkpoint: RecoveryCheckpoint,
        *,
        status: ExecutionStatus,
        output: StoredPayload | None,
        error_code: str | None,
        stop_reason: StopReason,
        binding: AgentBinding | None,
        run_id: str | None,
        usage: UsageMetrics,
        safe_error_details: Mapping[str, JsonValue] | None,
    ) -> ExecutionRecord:
        if current.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            return current
        self._validate_recovery_identity(current, checkpoint.input)
        recovery_run = None
        recovery_snapshot = None
        if run_id is not None and status is not ExecutionStatus.SUCCEEDED:
            candidate_run = await self._steps.get_run(run_id=run_id)
            candidate_snapshot = await self._steps.latest_snapshot(
                run_id=run_id,
                include_interrupted=True,
            )
            if candidate_snapshot is not None:
                if candidate_run is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                recovery_run = candidate_run
                recovery_snapshot = candidate_snapshot
        await self.verify_terminal_projection(
            current,
            status,
            run_id if status is ExecutionStatus.SUCCEEDED else None,
        )
        now = datetime.now(timezone.utc)
        if status is ExecutionStatus.SUCCEEDED:
            if binding is None or output is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        elif output is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        terminal = _terminal_record(
            current,
            status,
            now,
            error_code=error_code,
            safe_error_details=safe_error_details,
        )
        commit = ExecutionTerminalCommit(
            current.revision,
            current.event_sequence,
            terminal,
            ResultRecord(
                current.execution_id,
                current.tenant_id,
                output if status is ExecutionStatus.SUCCEEDED else None,
                stop_reason,
                usage,
                now,
            ),
            ExecutionEventType.EXECUTION_SUCCEEDED
            if status is ExecutionStatus.SUCCEEDED
            else ExecutionEventType.EXECUTION_CANCELLED
            if status is ExecutionStatus.CANCELLED
            else ExecutionEventType.EXECUTION_FAILED,
            {"run_id": run_id}
            if status is ExecutionStatus.SUCCEEDED and run_id is not None
            else {"error_code": error_code, "safe_error_details": safe_error_details or {}},
        )
        target = replace(
            checkpoint,
            state=RecoveryCheckpointState.COMPLETED,
            handoff_phase=RecoveryHandoffPhase.COMPLETED,
            terminal_handoff=None,
            handoff_contract_digest=None,
            pending_operation_id=None,
            pending_approval=None,
            revision=checkpoint.revision + 1,
            updated_at=now,
        )
        conversation_run = None
        conversation_snapshot = None
        expected_cursor = None
        if current.session_id is not None and status is ExecutionStatus.SUCCEEDED:
            conversation_run = await self._steps.get_run(run_id=run_id or "")
            conversation_snapshot = await self._steps.latest_snapshot(run_id=run_id or "")
            expected_cursor = await self._expected_session_cursor(current)
            if conversation_run is None or conversation_snapshot is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        committed = await self._commit_execution_terminal_checkpoint(
            current,
            commit,
            run_id=run_id,
            expected_cursor=expected_cursor,
            conversation_run=conversation_run,
            conversation_snapshot=conversation_snapshot,
            recovery_checkpoint=target,
            recovery_run=recovery_run,
            recovery_snapshot=recovery_snapshot,
        )
        self._publish_terminal_event(
            current.execution_id,
            event_type=commit.terminal_event_type,
            payload=dict(commit.terminal_event_payload),
            durable_sequence=committed.execution.event_sequence,
        )
        _logger.info(
            "same-group terminal checkpoint committed: execution=%s status=%s",
            current.execution_id,
            status.value,
        )
        return committed.execution

    async def _commit_execution_terminal_checkpoint(
        self,
        current: ExecutionRecord,
        commit: ExecutionTerminalCommit,
        *,
        run_id: str | None,
        expected_cursor: ConversationCursor | None = None,
        conversation_run: RunRecord | None = None,
        conversation_snapshot: ContinuableSnapshot | None = None,
        recovery_checkpoint: RecoveryCheckpoint | None = None,
        recovery_run: RunRecord | None = None,
        recovery_snapshot: ContinuableSnapshot | None = None,
    ) -> ExecutionTerminalCommitResult:
        async def commit_owned() -> ExecutionTerminalCommitResult:
            plan: ExecutionTerminalSealPlan | None = None
            durable_commit = False
            try:
                state_archive = isinstance(
                    self._step_reads[RuntimeDomain.EXECUTION],
                    StateStepArchive,
                )
                if state_archive:
                    if plan is None:
                        candidate_run_ids = tuple(
                            step_run_id(
                                namespace=self._namespace,
                                tenant_id=current.tenant_id,
                                execution_id=current.execution_id,
                                segment_sequence=sequence,
                            )
                            for sequence in range(1, current.agent_run_sequence + 1)
                        )
                        candidate_run_ids = await self._existing_execution_run_ids(
                            candidate_run_ids
                        )
                        plan = await self._step_lifecycle.prepare_execution_terminal_seal(
                            execution_id=current.execution_id,
                            run_ids=candidate_run_ids,
                            binding_digest=current.binding_digest,
                        )
                if run_id is not None and not state_archive:
                    await self._step_lifecycle.flush_execution_projection(
                        run_id,
                        execution_id=current.execution_id,
                    )
                async with self._audit_lock(current.execution_id):
                    pending_count = len(
                        self._pending_audit_events.get(current.execution_id, ())
                    )
                    committed = await self._commit_execution_terminal_checkpoint_locked_body(
                        current,
                        commit,
                        run_id=run_id,
                        expected_cursor=expected_cursor,
                        conversation_run=conversation_run,
                        conversation_snapshot=conversation_snapshot,
                        recovery_checkpoint=recovery_checkpoint,
                        recovery_run=recovery_run,
                        recovery_snapshot=recovery_snapshot,
                        terminal_plan=plan,
                    )
                    durable_commit = True
                    self._confirm_committed_events(
                        current.execution_id,
                        pending_count=pending_count,
                        durable_sequence=committed.execution.event_sequence,
                    )
                if plan is not None:
                    try:
                        await self._step_lifecycle.finalize_execution_terminal_seal(plan)
                    except BaseException:
                        _logger.error(
                            "terminal seal finalization failed after durable commit: execution=%s",
                            current.execution_id,
                            exc_info=environ.debug,
                        )
                        raise
                return committed
            except BaseException as error:
                if (
                    plan is not None
                    and not durable_commit
                    and not (
                        isinstance(error, AIError)
                        and error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
                    )
                ):
                    await self._step_lifecycle.discard_execution_terminal_seal(plan)
                raise

        task = asyncio.create_task(
            commit_owned(),
            name=f"ai-terminal-boundary-{current.execution_id}",
        )
        committed, cancellation = await self._await_checkpoint_task(
            task,
            label="terminal",
            execution_id=current.execution_id,
        )
        if cancellation is not None:
            raise cancellation
        return committed

    async def _existing_execution_run_ids(
        self,
        candidate_run_ids: Sequence[str],
    ) -> tuple[str, ...]:
        existing: list[str] = []
        execution_archive = self._step_reads[RuntimeDomain.EXECUTION]
        for run_id in dict.fromkeys(candidate_run_ids):
            staged = await self._steps.get_run(run_id=run_id)
            if staged is not None:
                existing.append(run_id)
                continue
            archived = await execution_archive.get_run(run_id=run_id)
            if archived is not None:
                existing.append(run_id)
                continue
            if isinstance(execution_archive, StateStepArchive):
                head = await execution_archive.execution_history_head(run_id)
                if head != (0, 0, 0, "empty"):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return tuple(existing)

    async def _commit_execution_terminal_checkpoint_locked_body(
        self,
        current: ExecutionRecord,
        commit: ExecutionTerminalCommit,
        *,
        run_id: str | None,
        expected_cursor: ConversationCursor | None = None,
        conversation_run: RunRecord | None = None,
        conversation_snapshot: ContinuableSnapshot | None = None,
        recovery_checkpoint: RecoveryCheckpoint | None = None,
        recovery_run: RunRecord | None = None,
        recovery_snapshot: ContinuableSnapshot | None = None,
        terminal_plan: ExecutionTerminalSealPlan | None,
    ) -> ExecutionTerminalCommitResult:
        pending_audit = tuple(self._pending_audit_events.get(current.execution_id, ()))
        step_run = None
        step_events: Sequence[StepEvent] = ()
        snapshots: Sequence[ContinuableSnapshot] = ()
        if terminal_plan is not None:
            execution_projections = terminal_plan.projections
            if execution_projections:
                step_run = execution_projections[0].run
        else:
            execution_projections = ()
        next_cursor = None
        if run_id is not None:
            history_id = None
            if current.session_id is not None:
                session = await self._conversation.sessions.get(
                    current.session_id,
                    tenant_id=current.tenant_id,
                )
                if session is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                history_id = session.history_id or (
                    None
                    if session.continuation is None
                    else session.continuation.history_id
                )
            next_cursor = ConversationCursor(run_id, history_id=history_id)
        committed = await self._runtime_commands.commit_terminal_checkpoint(
            commit,
            session_id=current.session_id,
            expected_cursor=expected_cursor,
            next_cursor=next_cursor,
            conversation_run=conversation_run,
            conversation_snapshot=conversation_snapshot,
            recovery_checkpoint=recovery_checkpoint,
            recovery_run=recovery_run,
            recovery_snapshot=recovery_snapshot,
            execution_run=step_run,
            execution_events=step_events,
            execution_snapshots=snapshots,
            execution_projections=execution_projections,
            audit_events=pending_audit,
            background_tasks=self._execution_task_set(current.execution_id),
        )
        self._pending_audit_events.pop(current.execution_id, None)
        return committed

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


def _recovery_prompt_payload(value: str) -> StoredPayload:
    if isinstance(value, UserPromptTransport) and value.codec != "text":
        return StoredPayload.inline_json(
            {"codec": value.codec, "value": str(value)}
        )
    return StoredPayload.inline_text(str(value))


def _recovery_prompt_text(recovery_input: RecoveryExecutionInput) -> str:
    payload = recovery_input.user_prompt
    if not isinstance(payload, StoredPayload) or payload.kind != "inline":
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        decoded = payload.decode()
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if payload.encoding == "utf-8":
        if not isinstance(decoded, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            return user_prompt_transport(decoded, "text")
        except AIError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if payload.encoding != "json" or not isinstance(decoded, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if set(decoded) != {"codec", "value"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    codec = decoded.get("codec")
    value = decoded.get("value")
    if not isinstance(codec, str) or not isinstance(value, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return user_prompt_transport(value, codec)
    except AIError as error:
        if error.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED:
            raise
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


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


def _execution_error_code(error: Exception) -> ErrorCode:
    if isinstance(error, ValidationError):
        return ErrorCode.OUTPUT_VALIDATION_FAILED
    if isinstance(error, AIError):
        return error.code
    return ErrorCode.INTERNAL_ERROR


def _execution_error_details(error: Exception) -> dict[str, JsonValue]:
    return dict(error.safe_details) if isinstance(error, AIError) else {}


def _secondary_execution_error(error: Exception, primary: Exception) -> AIError:
    primary_details: dict[str, JsonValue] = {
        "primary_error_code": _execution_error_code(primary).value,
        "primary_safe_error_details": _execution_error_details(primary),
    }
    if isinstance(error, AIError):
        details = dict(error.safe_details)
        details.update(primary_details)
        return AIError(
            error.code,
            category=error.category,
            retryable=error.retryable,
            operation_id=error.operation_id,
            safe_details=details,
        )
    return AIError(
        ErrorCode.INTERNAL_ERROR,
        safe_details={
            "phase": "execution_terminal_commit",
            **primary_details,
        },
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
