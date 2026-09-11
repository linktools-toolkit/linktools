#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local ExecutionBackend backed by AgentExecutor and durable persistence."""

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from time import monotonic_ns
from typing import Protocol, TypeVar, cast

from linktools.core import environ
from pydantic import ValidationError
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.tools import (
    DeferredToolRequests,
    DeferredToolResults,
    ToolApproved,
    ToolDenied,
)

from ..agent import AgentBinding, AgentCatalog, SubagentRef
from ..capability import AgentContext, SubagentDelegate
from ..workspace import (
    RepositoryInstructionResolver,
    RepositoryInstructions,
    Workspace,
)
from ._agent_executor import (
    AgentExecutor,
    AgentExecutionResult,
    DurableBoundary,
    LiveDelta,
    _RunScope,
)
from ._capabilities import MEMORY_TOOL_NAMES, select_runtime_tool_names
from ._input import CanonicalUserInput, ExecutionInputMaterializer
from ._plan import RuntimePlanStore
from ..core import (
    ApprovalStatus,
    ExecutionEventType,
    ExecutionMode,
    ExecutionLineageKind,
    ExecutionStatus,
    ExternalCallStatus,
    IdempotencyStatus,
    JsonValue,
    OperationKind,
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationStatus,
    Page,
    Principal,
    ResourceKind,
    SessionStatus,
    StopReason,
    ThinkingValue,
    ToolOperationStatus,
    UsageMetrics,
    canonical_json_bytes,
    canonical_sha256,
    idempotency_key_digest,
    normalize_json_value,
    step_conversation_id,
    step_run_id,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode, ErrorDiagnostics
from ..observe import MetricRecorder
from ..storage import (
    ObjectStore,
    PayloadPolicy,
    StoredPayload,
    payload_fits_inline,
)
from ._event import ExecutionDelta, LiveExecutionEventBroker
from ._execution import CancelEffectOutcome, ExecutionStartIdentity
from ._metrics import (
    _record_execution_terminal,
    _record_storage_operation,
    _release_metric_execution_context,
)
from ._message import encode_model_messages
from ._memory import MemoryStore
from ._object import RuntimeObjectKeyFactory, put_runtime_object, read_runtime_object
from ._tool import (
    RuntimeToolOperationBridge,
    ToolOperationBridge,
    _ToolOperationRuntimeRepository,
)
from ._tool_boundary import RepositoryInstructionBoundary
from .recovery import (
    ExecutionRecoveryEffect,
    ResolveToolEffectRequest,
    ToolEffectApplied,
    ToolEffectFailed,
    ToolEffectNotApplied,
    ToolEffectResolutionResult,
)
from .service_api import ExecutionRequest
from .state import RuntimeDomain
from .state._commands import ConversationStateCommands, RuntimeStateCommands
from .state._contracts import (
    ApprovalRecord,
    AgentAttemptClaim,
    ConversationState,
    ConversationCursor,
    ExecutionCancelRequestCommit,
    ExecutionEventAppend,
    ExecutionRepository,
    ExecutionRecord,
    ExecutionStartClaim,
    ExecutionState,
    ExecutionTerminalCommit,
    ExecutionTerminalCommitResult,
    ExternalCallRecord,
    IdempotencyRecord,
    LoadedModelContext,
    PendingDeferredCall,
    PendingToolContinuation,
    RecoveryCheckpointState,
    RecoveryHandoffPhase,
    RecoveryState,
    RuntimePayloadRef,
    RuntimeStorageContract,
    RecoveryCheckpoint,
    RecoveryConversationIntent,
    RecoveryTerminalHandoff,
    RecoveryTerminalOutcome,
    RepositoryInstructionBarrier,
    ResultRecord,
    SessionRepository,
    ToolOperationRecord,
)
from .state._step_contracts import (
    ContinuableSnapshot,
    RunRecord,
    StepEvent,
    StepStore,
)
from .state._recovery_commands import RuntimeRecoveryCommands
from .state._steps import ExecutionTerminalSealPlan, RuntimeStepStore, StateStepArchive
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

    async def cancel_children(
        self,
        parent_execution_id: str,
        principal: Principal,
    ) -> None: ...


def _merge_repository_instructions(
    initial: RepositoryInstructions | None,
    overlay: RepositoryInstructions | None,
) -> RepositoryInstructions | None:
    if initial is None:
        return overlay
    if overlay is None:
        return initial
    initial_sources = {document.source for document in initial.documents}
    if any(document.source in initial_sources for document in overlay.documents):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return RepositoryInstructions((*initial.documents, *overlay.documents))


class _RepositoryInstructionBoundary:
    def __init__(
        self,
        runtime: "_RecoveryCoordinator",
        execution: ExecutionRecord,
        initial: RepositoryInstructions | None,
        active: RepositoryInstructions | None,
    ) -> None:
        self._runtime = runtime
        self._execution = execution
        self._initial = initial
        self._active = active

    def render(self) -> str:
        if self._active is None:
            return ""
        return self._active.render()

    async def check(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, object],
        path_fields: tuple[str, ...],
    ) -> None:
        self._active, reconsider = (
            await self._runtime.check_repository_instructions(
                execution=self._execution,
                initial=self._initial,
                active=self._active,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                arguments=arguments,
                path_fields=path_fields,
            )
        )
        if reconsider:
            raise ToolFailed(_REPOSITORY_INSTRUCTION_RECONSIDER)


_REPOSITORY_INSTRUCTION_RECONSIDER = (
    "Repository instructions changed; reconsider the call"
)


def _instruction_paths(
    arguments: Mapping[str, object],
    path_fields: tuple[str, ...],
) -> tuple[str, ...]:
    paths: list[str] = []
    for field in path_fields:
        value = arguments.get(field)
        if value is None:
            continue
        if isinstance(value, str):
            paths.append(value)
            continue
        if isinstance(value, (list, tuple)) and all(
            isinstance(item, str) for item in value
        ):
            paths.extend(value)
            continue
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return tuple(paths)


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
    diagnostics: ErrorDiagnostics | None = None
    category: str | None = None
    retryable: bool | None = None
    operation_id: str | None = None


class _StepLifecycle(Protocol):
    async def materialize_conversation(self, *, step_run_id: str) -> None: ...
    async def materialize_from_recovery(
        self,
        *,
        target: RuntimeDomain,
        step_run_id: str,
        execution_id: "str | None" = None,
    ) -> None: ...
    async def materialize_recovery_snapshot(
        self, *, step_run_id: str, require_complete: bool
    ) -> None: ...
    async def verify_terminal_attempts(
        self,
        *,
        candidate_step_run_ids: tuple[str, ...],
        required_step_run_id: str | None,
    ) -> None: ...
    async def release_staging_many(
        self,
        *,
        candidate_step_run_ids: tuple[str, ...],
        execution_id: "str | None" = None,
    ) -> None: ...
    async def flush_execution_projection(
        self, step_run_id: str, *, execution_id: str
    ) -> None: ...
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


@dataclass(frozen=True, slots=True)
class _AgentSegmentInput:
    """The immutable inputs needed to materialize one agent segment."""

    binding: AgentBinding
    context: AgentContext[object]
    user_prompt: CanonicalUserInput | None
    history: list[ModelMessage]
    conversation_id: str
    step_store: StepStore
    step_run_id: str
    segment_sequence: int
    history_id: str | None
    memory_store: MemoryStore | None
    plan_store_resolver: Callable[..., RuntimePlanStore] | None
    mode: ExecutionMode
    planning: bool
    thinking: ThinkingValue
    parent_step_run_id: str | None
    subagent_available: bool
    subagent_descriptions: Mapping[str, str | None]
    subagent_delegate: SubagentDelegate | None
    event_sink: Callable[[LiveDelta | DurableBoundary], Awaitable[None]]
    usage_sink: Callable[[UsageMetrics], Awaitable[None]]
    tool_operations: ToolOperationBridge | None
    background_tasks: set[asyncio.Task[object]]
    replace_history_system_prompt: bool
    repository_instructions: RepositoryInstructions | None
    repository_instruction_boundary: RepositoryInstructionBoundary | None
    deferred_tool_results: DeferredToolResults | None


@dataclass(frozen=True, slots=True)
class _SegmentCompleted:
    result: AgentExecutionResult


@dataclass(frozen=True, slots=True)
class _SegmentDeferred:
    requests: DeferredToolRequests


@dataclass(frozen=True, slots=True)
class _SegmentFailed:
    error: Exception


@dataclass(frozen=True, slots=True)
class _SegmentCancelled:
    pass


@dataclass(frozen=True, slots=True)
class _DeferredResume:
    execution: ExecutionRecord
    checkpoint: RecoveryCheckpoint
    history: tuple[ModelMessage, ...]
    results: DeferredToolResults


class _AgentSegmentRunner:
    """Run exactly one AgentExecutor segment without changing runtime state."""

    def __init__(self, executor: AgentExecutor) -> None:
        self._executor = executor

    async def run(
        self,
        segment: _AgentSegmentInput,
    ) -> "_SegmentCompleted | _SegmentDeferred | _SegmentFailed | _SegmentCancelled":
        scope = _RunScope(
            binding=segment.binding,
            context=segment.context,
            user_prompt=segment.user_prompt,
            history=segment.history,
            conversation_id=segment.conversation_id,
            step_store=segment.step_store,
            step_run_id=segment.step_run_id,
            segment_sequence=segment.segment_sequence,
            history_id=segment.history_id,
            memory_store=segment.memory_store,
            plan_store_resolver=segment.plan_store_resolver,
            mode=segment.mode,
            planning=segment.planning,
            thinking=segment.thinking,
            parent_step_run_id=segment.parent_step_run_id,
            subagent_available=segment.subagent_available,
            subagent_descriptions=segment.subagent_descriptions,
            subagent_delegate=segment.subagent_delegate,
            event_sink=segment.event_sink,
            usage_sink=segment.usage_sink,
            tool_operations=segment.tool_operations,
            background_tasks=segment.background_tasks,
            replace_history_system_prompt=segment.replace_history_system_prompt,
            repository_instructions=segment.repository_instructions,
            repository_instruction_boundary=segment.repository_instruction_boundary,
            deferred_tool_results=segment.deferred_tool_results,
        )
        try:
            result = await self._executor.execute(scope)
        except asyncio.CancelledError:
            return _SegmentCancelled()
        except Exception as error:
            return _SegmentFailed(error)
        if isinstance(result, DeferredToolRequests):
            return _SegmentDeferred(result)
        return _SegmentCompleted(result)


class _RecoveryCoordinatorPort(Protocol):
    """Durable and orchestration operations consumed by recovery coordination."""

    @property
    def tenant_id(self) -> str: ...

    async def _list_recoverable_checkpoints(
        self,
        *,
        cursor: str | None,
    ) -> Page[RecoveryCheckpoint]: ...

    async def _recovery_failure_effects(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> tuple[ExecutionRecoveryEffect, ...]: ...

    async def _get_tool_operation(
        self,
        operation_id: str,
        *,
        tenant_id: str,
    ) -> ToolOperationRecord | None: ...

    async def _tool_result_payload(
        self,
        execution: ExecutionRecord,
        operation_id: str,
        result: object,
    ) -> StoredPayload: ...

    async def _tool_resolution_error_payload(
        self,
        execution: ExecutionRecord,
    ) -> StoredPayload: ...

    async def _resolve_tool_effect_command(
        self,
        execution_id: str,
        ledger: OperationLedgerInput,
        *,
        expected_fence: int,
        target_status: ToolOperationStatus,
        result_payload: StoredPayload | None,
        error_code: str | None,
        error_payload: StoredPayload | None,
    ) -> ToolOperationRecord: ...

    async def _persist_cancel_intent(
        self,
        execution: ExecutionRecord,
        operation: OperationLedgerInput,
    ) -> OperationLedgerRecord: ...

    async def load_execution(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ExecutionRecord | None: ...

    async def load_recovery_checkpoint(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> RecoveryCheckpoint | None: ...

    async def materialize_deferred_call(
        self,
        execution: ExecutionRecord,
        source_step_run_id: str,
        call: ToolCallPart,
        metadata: Mapping[str, object],
    ) -> PendingDeferredCall: ...

    async def commit_deferred_frontier(
        self,
        execution: ExecutionRecord,
        checkpoint: RecoveryCheckpoint,
        continuation: PendingToolContinuation,
        approval_records: tuple[ApprovalRecord, ...],
        external_records: tuple[ExternalCallRecord, ...],
        occurred_at: datetime,
    ) -> tuple[ExecutionRecord, RecoveryCheckpoint]: ...

    async def load_approval(
        self,
        approval_id: str,
        *,
        tenant_id: str,
    ) -> ApprovalRecord | None: ...

    async def load_external_call(
        self,
        call_id: str,
        *,
        tenant_id: str,
    ) -> ExternalCallRecord | None: ...

    async def read_deferred_payload(self, payload: StoredPayload) -> JsonValue: ...

    async def load_interrupted_messages(
        self,
        run_id: str,
    ) -> tuple[ModelMessage, ...]: ...

    async def claim_deferred_resume(
        self,
        checkpoint: RecoveryCheckpoint,
        execution: ExecutionRecord,
    ) -> tuple[ExecutionRecord, RecoveryCheckpoint]: ...

    def validate_binding(self, execution: ExecutionRecord) -> None: ...

    def _validate_recovery_identity(self, execution: ExecutionRecord) -> None: ...

    async def _commit_recovery_required(
        self,
        execution: ExecutionRecord,
        error: AIError,
        effects: tuple[ExecutionRecoveryEffect, ...],
    ) -> ExecutionRecord: ...

    async def _reconcile_session_recovery(
        self,
        checkpoint: RecoveryCheckpoint,
        execution: ExecutionRecord,
    ) -> bool: ...

    async def _reconcile_handoff(
        self,
        checkpoint: RecoveryCheckpoint,
    ) -> ExecutionRecord: ...

    async def _release_session_execution(
        self,
        execution: ExecutionRecord,
    ) -> None: ...

    async def _finish_checkpoint(self, checkpoint: RecoveryCheckpoint) -> None: ...

    async def _recovery_idempotency(
        self,
        execution: ExecutionRecord,
    ) -> IdempotencyRecord: ...

    async def _ensure_recovery_idempotency(
        self,
        execution: ExecutionRecord,
        *,
        expected_status: IdempotencyStatus,
    ) -> IdempotencyRecord: ...

    async def _expected_session_cursor(
        self,
        execution: ExecutionRecord,
    ) -> ConversationCursor | None: ...

    async def _commit_terminal(
        self,
        execution: ExecutionRecord,
        status: ExecutionStatus,
        output: StoredPayload | None,
        error_code: str | None,
        stop_reason: StopReason,
        *,
        binding: AgentBinding | None = None,
        run_id: str | None = None,
        usage: UsageMetrics | None = None,
        safe_error_details: Mapping[str, JsonValue] | None = None,
        error_diagnostics: ErrorDiagnostics | None = None,
        expected_cursor: ConversationCursor | None = None,
        conversation_run: RunRecord | None = None,
        conversation_snapshot: ContinuableSnapshot | None = None,
        recovery_checkpoint: RecoveryCheckpoint | None = None,
    ) -> ExecutionRecord: ...

    async def _commit_start_recovery_checkpoint(
        self,
        execution: ExecutionRecord,
        checkpoint: RecoveryCheckpoint,
        identity: IdempotencyRecord,
    ) -> ExecutionRecord: ...

    async def _pending_cancel_operations(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> tuple[OperationLedgerRecord, ...]: ...

    async def _complete_recovered_cancel(
        self,
        resumed: ExecutionRecord,
        checkpoint: RecoveryCheckpoint,
        operations: tuple[OperationLedgerRecord, ...],
    ) -> ExecutionRecord: ...

    async def _commit_recovery_resume(
        self,
        execution: ExecutionRecord,
    ) -> tuple[ExecutionRecord, RecoveryCheckpoint]: ...

    def _reset_local_producer(self, execution_id: str) -> None: ...

    def _publish_recovery_resumed(
        self,
        execution_id: str,
        event_sequence: int,
    ) -> None: ...

    async def restore_user_input(
        self,
        execution: ExecutionRecord,
    ) -> CanonicalUserInput: ...

    async def load_repository_instructions(
        self,
        reference: RuntimePayloadRef | None,
    ) -> RepositoryInstructions | None: ...

    async def commit_repository_instruction_barrier(
        self,
        execution: ExecutionRecord,
        checkpoint: RecoveryCheckpoint,
        overlay: RepositoryInstructions,
        barrier: RepositoryInstructionBarrier,
    ) -> RecoveryCheckpoint: ...

    def _mark_recovery_relaunch(self, execution_id: str) -> None: ...

    def execution_task_set(
        self,
        execution_id: str,
    ) -> set[asyncio.Task[object]]: ...

    async def launch(
        self,
        request: ExecutionRequest,
        execution: ExecutionRecord,
        *,
        resume: _DeferredResume | None = None,
    ) -> None: ...


class _RecoveryCoordinator:
    """Own durable recovery decisions while the backend owns worker lifecycle."""

    def __init__(
        self,
        port: _RecoveryCoordinatorPort,
        instruction_resolver: RepositoryInstructionResolver | None,
    ) -> None:
        self._port = port
        self._instruction_resolver = instruction_resolver

    async def check_repository_instructions(
        self,
        *,
        execution: ExecutionRecord,
        initial: RepositoryInstructions | None,
        active: RepositoryInstructions | None,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, object],
        path_fields: tuple[str, ...],
    ) -> tuple[RepositoryInstructions | None, bool]:
        del tool_name
        paths = _instruction_paths(arguments, path_fields)
        resolver = self._instruction_resolver
        if not paths or resolver is None:
            return active, False
        checkpoint = await self._port.load_recovery_checkpoint(
            execution.execution_id,
            tenant_id=execution.tenant_id,
        )
        if (
            checkpoint is None
            or checkpoint.state is not RecoveryCheckpointState.ACTIVE
            or checkpoint.step_run_id is None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        arguments_digest = canonical_sha256(normalize_json_value(arguments))
        matching = tuple(
            barrier
            for barrier in checkpoint.repository_instruction_barriers
            if barrier.step_run_id == checkpoint.step_run_id
            and barrier.tool_call_id == tool_call_id
        )
        if matching:
            barrier = matching[0]
            if barrier.arguments_digest != arguments_digest:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            overlay = await self._port.load_repository_instructions(
                checkpoint.repository_instruction_overlay
            )
            if (
                checkpoint.repository_instruction_overlay is None
                or overlay is None
                or overlay.digest != barrier.resulting_overlay_digest
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return _merge_repository_instructions(initial, overlay), True

        overlay = await self._port.load_repository_instructions(
            checkpoint.repository_instruction_overlay
        )
        active = _merge_repository_instructions(initial, overlay)
        excluded = frozenset(
            () if active is None else document.source for document in active.documents
        )
        discovered: list[object] = []
        discovered_sources = set(excluded)
        for path in paths:
            resolved = await resolver.resolve(
                path,
                exclude_sources=frozenset(discovered_sources),
            )
            for document in resolved.documents:
                if document.source in discovered_sources:
                    continue
                discovered_sources.add(document.source)
                discovered.append(document)
        if not discovered:
            return active, False
        new_documents = RepositoryInstructions(tuple(discovered))
        next_overlay = _merge_repository_instructions(overlay, new_documents)
        if next_overlay is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        barrier = RepositoryInstructionBarrier(
            checkpoint.step_run_id,
            tool_call_id,
            arguments_digest,
            next_overlay.digest,
        )
        committed = await self._port.commit_repository_instruction_barrier(
            execution,
            checkpoint,
            next_overlay,
            barrier,
        )
        committed_overlay = await self._port.load_repository_instructions(
            committed.repository_instruction_overlay
        )
        if committed_overlay is None or committed_overlay.digest != next_overlay.digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _logger.info(
            "repository instructions extended: execution=%s step=%s tool_call=%s",
            execution.execution_id,
            checkpoint.step_run_id,
            tool_call_id,
        )
        return _merge_repository_instructions(initial, committed_overlay), True

    async def commit_deferred_pause(
        self,
        execution: ExecutionRecord,
        requests: DeferredToolRequests,
        *,
        step_run_id: str,
        paused_at: datetime,
    ) -> None:
        current = await self._port.load_execution(
            execution.execution_id,
            tenant_id=execution.tenant_id,
        )
        checkpoint = await self._port.load_recovery_checkpoint(
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
            return
        if (
            current.status is not ExecutionStatus.STARTED
            or checkpoint.state is not RecoveryCheckpointState.ACTIVE
            or checkpoint.step_run_id != step_run_id
            or checkpoint.agent_run_sequence != current.agent_run_sequence
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        metadata = requests.metadata
        approvals_list: list[PendingDeferredCall] = []
        for call in requests.approvals:
            approvals_list.append(
                await self._port.materialize_deferred_call(
                    current,
                    step_run_id,
                    call,
                    metadata.get(call.tool_call_id, {}),
                )
            )
        approvals = tuple(approvals_list)
        calls_list: list[PendingDeferredCall] = []
        for call in requests.calls:
            calls_list.append(
                await self._port.materialize_deferred_call(
                    current,
                    step_run_id,
                    call,
                    metadata.get(call.tool_call_id, {}),
                )
            )
        calls = tuple(calls_list)
        continuation = PendingToolContinuation(
            source_step_run_id=step_run_id,
            requests_digest=canonical_sha256(
                {
                    "version": 1,
                    "approvals": [
                        {
                            "tool_call_id": item.tool_call_id,
                            "tool_name": item.tool_name,
                            "arguments_digest": item.arguments_digest,
                            "metadata": item.metadata,
                        }
                        for item in approvals
                    ],
                    "calls": [
                        {
                            "tool_call_id": item.tool_call_id,
                            "tool_name": item.tool_name,
                            "arguments_digest": item.arguments_digest,
                            "metadata": item.metadata,
                        }
                        for item in calls
                    ],
                }
            ),
            approvals=approvals,
            calls=calls,
        )
        approval_records = tuple(
            ApprovalRecord(
                _deferred_id(
                    "approval-v1",
                    current.tenant_id,
                    current.execution_id,
                    step_run_id,
                    item.tool_call_id,
                ),
                current.execution_id,
                current.tenant_id,
                ApprovalStatus.PENDING,
                None,
                None,
                None,
                None,
                paused_at,
                None,
            )
            for item in approvals
        )
        external_records = tuple(
            ExternalCallRecord(
                _deferred_id(
                    "external-call-v1",
                    current.tenant_id,
                    current.execution_id,
                    step_run_id,
                    item.tool_call_id,
                ),
                current.execution_id,
                current.tenant_id,
                ExternalCallStatus.PENDING,
                None,
                paused_at,
                None,
            )
            for item in calls
        )
        committed, _ = await self._port.commit_deferred_frontier(
            current,
            checkpoint,
            continuation,
            approval_records,
            external_records,
            paused_at,
        )
        if committed.status is not ExecutionStatus.WAITING_DEFERRED:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _logger.info(
            "deferred execution checkpoint committed: execution=%s approvals=%s calls=%s",
            current.execution_id,
            len(approvals),
            len(calls),
        )

    async def recovery_effects(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> tuple[ExecutionRecoveryEffect, ...]:
        return await self._port._recovery_failure_effects(
            execution_id,
            tenant_id=tenant_id,
        )

    async def resolve_tool_effect(
        self,
        execution_id: str,
        request: ResolveToolEffectRequest,
    ) -> ToolEffectResolutionResult:
        current = await self._port.load_execution(
            execution_id,
            tenant_id=request.principal.tenant_id,
        )
        if current is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if current.status is not ExecutionStatus.RECOVERY_REQUIRED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        tool = await self._port._get_tool_operation(
            request.operation_id,
            tenant_id=current.tenant_id,
        )
        if tool is None or tool.execution_id != execution_id:
            raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)

        result_payload: StoredPayload | None = None
        error_code: str | None = None
        error_payload: StoredPayload | None = None
        if isinstance(request.resolution, ToolEffectApplied):
            target_status = ToolOperationStatus.COMPLETED
            result_payload = await self._port._tool_result_payload(
                current,
                request.operation_id,
                request.resolution.result,
            )
        elif isinstance(request.resolution, ToolEffectNotApplied):
            target_status = ToolOperationStatus.PENDING
        elif isinstance(request.resolution, ToolEffectFailed):
            target_status = ToolOperationStatus.FAILED
            error_code = ErrorCode.TOOL_EXECUTION_FAILED.value
            error_payload = await self._port._tool_resolution_error_payload(current)
        else:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        payload_digest = (
            result_payload.digest
            if result_payload is not None
            else error_payload.digest
            if error_payload is not None
            else None
        )
        request_digest = canonical_sha256(
            {
                "kind": "tool_effect_resolution",
                "execution_id": execution_id,
                "operation_id": request.operation_id,
                "expected_fence": request.expected_fence,
                "resolution": type(request.resolution).__name__,
                "payload_digest": payload_digest,
            }
        )
        resolution_operation_id = canonical_sha256(
            {
                "scope": "execution.tool_effect.resolve",
                "tenant_id": current.tenant_id,
                "execution_id": execution_id,
                "idempotency_key_digest": idempotency_key_digest(
                    request.idempotency_key
                ),
            }
        )
        result_digest = canonical_sha256(
            {
                "operation_id": request.operation_id,
                "fence": request.expected_fence,
                "status": target_status.value,
                "payload_digest": payload_digest,
            }
        )
        now = datetime.now(timezone.utc)
        ledger = OperationLedgerInput(
            resolution_operation_id,
            current.tenant_id,
            ResourceKind.TOOL_OPERATION,
            request.operation_id,
            execution_id,
            OperationKind.TOOL_EFFECT_RESOLVE,
            OperationStatus.SUCCEEDED,
            request_digest,
            request.operation_id,
            result_digest,
            None,
            True,
            now,
            now,
        )
        resolved = await self._port._resolve_tool_effect_command(
            execution_id,
            ledger,
            expected_fence=request.expected_fence,
            target_status=target_status,
            result_payload=result_payload,
            error_code=error_code,
            error_payload=error_payload,
        )
        return ToolEffectResolutionResult(
            operation_id=resolved.tool_operation_id,
            execution_id=resolved.execution_id,
            status=resolved.status,
            fence=resolved.fence,
        )

    async def persist_cancel_intent(
        self,
        execution: ExecutionRecord,
        operation: OperationLedgerInput,
    ) -> OperationLedgerRecord:
        return await self._port._persist_cancel_intent(execution, operation)

    async def reconcile_checkpoint(self, checkpoint: RecoveryCheckpoint) -> None:
        execution = await self._port.load_execution(
            checkpoint.execution_id,
            tenant_id=checkpoint.tenant_id,
        )
        if execution is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        resume: _DeferredResume | None = None
        if execution.status is ExecutionStatus.RECOVERY_REQUIRED:
            self._port._validate_recovery_identity(execution)
            if (
                checkpoint.handoff_phase is not RecoveryHandoffPhase.NONE
                or checkpoint.state
                not in {
                    RecoveryCheckpointState.ACTIVE,
                    RecoveryCheckpointState.WAITING,
                }
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        if (
            execution.status
            in {ExecutionStatus.STARTED, ExecutionStatus.CANCELLING}
            and checkpoint.handoff_phase is RecoveryHandoffPhase.NONE
            and checkpoint.state
            in {
                RecoveryCheckpointState.ACTIVE,
                RecoveryCheckpointState.WAITING,
            }
        ):
            effects = await self._port._recovery_failure_effects(
                execution.execution_id,
                tenant_id=execution.tenant_id,
            )
            if effects:
                first = effects[0]
                await self._port._commit_recovery_required(
                    execution,
                    AIError(
                        ErrorCode.TOOL_EFFECT_UNKNOWN,
                        safe_details={
                            "execution_id": execution.execution_id,
                            "operation_id": first.operation_id,
                            "fence": first.fence,
                            "phase": "startup_reconcile",
                        },
                    ),
                    effects,
                )
                return
        self._port._validate_recovery_identity(execution)
        principal = Principal(
            execution.principal_id,
            execution.tenant_id,
            execution.principal_kind,
        )
        if checkpoint.handoff_phase is not RecoveryHandoffPhase.NONE:
            self._port.validate_binding(execution)
            await self._port._reconcile_handoff(checkpoint)
            return
        if execution.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            if execution.session_id is not None:
                await self._port._release_session_execution(execution)
            await self._port._finish_checkpoint(checkpoint)
            return
        self._port.validate_binding(execution)
        if checkpoint.state in {
            RecoveryCheckpointState.ADMITTED,
            RecoveryCheckpointState.ACTIVE,
            RecoveryCheckpointState.WAITING,
        } and not await self._port._reconcile_session_recovery(
            checkpoint,
            execution,
        ):
            return
        identity = await self._port._recovery_idempotency(execution)
        if (
            checkpoint.state is RecoveryCheckpointState.ADMITTED
            and execution.status is ExecutionStatus.PENDING_START
        ):
            execution = await self._port._commit_start_recovery_checkpoint(
                execution,
                checkpoint,
                identity,
            )
        elif execution.status is ExecutionStatus.CANCELLING:
            await self._port._commit_terminal(
                execution,
                ExecutionStatus.CANCELLED,
                None,
                ErrorCode.EXECUTION_CANCELLED.value,
                StopReason.CANCELLED,
            )
            return
        elif execution.status is ExecutionStatus.START_UNKNOWN:
            raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
        elif checkpoint.state is RecoveryCheckpointState.ADMITTED:
            await self._port._ensure_recovery_idempotency(
                execution,
                expected_status=IdempotencyStatus.STARTED,
            )
        elif checkpoint.state is RecoveryCheckpointState.WAITING:
            if execution.status is not ExecutionStatus.WAITING_DEFERRED:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            resume = await self.reconcile_waiting_deferred(
                checkpoint,
                execution,
            )
            if resume is None:
                return
            execution = resume.execution
            checkpoint = resume.checkpoint
            await self._port._ensure_recovery_idempotency(
                execution,
                expected_status=IdempotencyStatus.STARTED,
            )
        elif checkpoint.state is RecoveryCheckpointState.ACTIVE:
            if execution.status is not ExecutionStatus.STARTED:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._port._ensure_recovery_idempotency(
                execution,
                expected_status=IdempotencyStatus.STARTED,
            )
        else:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        request = ExecutionRequest(
            user_prompt=await self._port.restore_user_input(execution),
            principal=principal,
            idempotency_key=f"recovery:{execution.execution_id}",
            memory_scope=execution.memory_scope,
            mode=execution.mode,
            planning=execution.planning,
            thinking=execution.thinking,
            correlation=execution.correlation,
        )
        self._port._mark_recovery_relaunch(execution.execution_id)
        await self._port.launch(request, execution, resume=resume)
        _logger.info(
            "local recovery execution relaunched: tenant=%s execution=%s",
            checkpoint.tenant_id,
            checkpoint.execution_id,
        )

    async def recover_execution(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ExecutionRecord:
        current = await self._port.load_execution(
            execution_id,
            tenant_id=tenant_id,
        )
        if current is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if current.status is not ExecutionStatus.RECOVERY_REQUIRED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        unresolved = await self.recovery_effects(
            execution_id,
            tenant_id=tenant_id,
        )
        if unresolved:
            first = unresolved[0]
            raise AIError(
                ErrorCode.TOOL_EFFECT_UNKNOWN,
                safe_details={
                    "execution_id": execution_id,
                    "operation_id": first.operation_id,
                    "fence": first.fence,
                    "phase": "execution_recover",
                },
            )
        checkpoint = await self._port.load_recovery_checkpoint(
            execution_id,
            tenant_id=tenant_id,
        )
        if (
            checkpoint is None
            or checkpoint.state
            not in {
                RecoveryCheckpointState.ACTIVE,
                RecoveryCheckpointState.WAITING,
            }
            or checkpoint.handoff_phase is not RecoveryHandoffPhase.NONE
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        cancel_operations = await self._port._pending_cancel_operations(
            execution_id,
            tenant_id=tenant_id,
        )
        self._port._reset_local_producer(execution_id)
        resumed, _ = await self._port._commit_recovery_resume(current)
        self._port._publish_recovery_resumed(
            execution_id,
            resumed.event_sequence,
        )
        if cancel_operations:
            return await self._port._complete_recovered_cancel(
                resumed,
                checkpoint,
                cancel_operations,
            )
        await self.reconcile_checkpoint(checkpoint)
        latest = await self._port.load_execution(
            execution_id,
            tenant_id=tenant_id,
        )
        if latest is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return latest

    async def reconcile_waiting_deferred(
        self,
        checkpoint: RecoveryCheckpoint,
        execution: ExecutionRecord,
    ) -> _DeferredResume | None:
        current = await self._port.load_execution(
            execution.execution_id,
            tenant_id=execution.tenant_id,
        )
        recovery = await self._port.load_recovery_checkpoint(
            execution.execution_id,
            tenant_id=execution.tenant_id,
        )
        if current is None or recovery is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if current.status is ExecutionStatus.CANCELLING:
            return None
        if (
            current.status is not ExecutionStatus.WAITING_DEFERRED
            or recovery != checkpoint
            or recovery.state is not RecoveryCheckpointState.WAITING
            or recovery.pending_tools is None
            or recovery.agent_run_sequence != current.agent_run_sequence
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        deferred_results = DeferredToolResults()
        for pending in recovery.pending_tools.approvals:
            approval_id = _deferred_id(
                "approval-v1",
                current.tenant_id,
                current.execution_id,
                recovery.pending_tools.source_step_run_id,
                pending.tool_call_id,
            )
            record = await self._port.load_approval(
                approval_id,
                tenant_id=current.tenant_id,
            )
            if record is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if record.status is ApprovalStatus.PENDING:
                return None
            arguments = await self._port.read_deferred_payload(
                pending.arguments_payload
            )
            if record.status is ApprovalStatus.APPROVED:
                deferred_results.approvals[pending.tool_call_id] = ToolApproved(
                    override_args=arguments
                )
            elif record.status in {
                ApprovalStatus.DENIED,
                ApprovalStatus.CANCELLED,
            }:
                deferred_results.approvals[pending.tool_call_id] = ToolDenied(
                    message=record.decision_message or "The tool call was denied."
                )
            else:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            deferred_results.metadata[pending.tool_call_id] = dict(
                record.resolution_metadata
            )
        for pending in recovery.pending_tools.calls:
            call_id = _deferred_id(
                "external-call-v1",
                current.tenant_id,
                current.execution_id,
                recovery.pending_tools.source_step_run_id,
                pending.tool_call_id,
            )
            record = await self._port.load_external_call(
                call_id,
                tenant_id=current.tenant_id,
            )
            if record is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if record.status is ExternalCallStatus.PENDING:
                return None
            if record.status is not ExternalCallStatus.SUPPLIED:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if record.result_payload is None or record.resolution_kind is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result = await self._port.read_deferred_payload(record.result_payload)
            if record.resolution_kind == "succeeded":
                deferred_results.calls[pending.tool_call_id] = result
            elif record.resolution_kind == "retry":
                if not isinstance(result, str):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                deferred_results.calls[pending.tool_call_id] = ModelRetry(result)
            elif record.resolution_kind == "failed":
                if not isinstance(result, str):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                deferred_results.calls[pending.tool_call_id] = ToolFailed(result)
            else:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            deferred_results.metadata[pending.tool_call_id] = dict(
                record.resolution_metadata
            )
        history = await self._port.load_interrupted_messages(
            recovery.pending_tools.source_step_run_id
        )
        resumed_execution, resumed_checkpoint = (
            await self._port.claim_deferred_resume(checkpoint, current)
        )
        return _DeferredResume(
            resumed_execution,
            resumed_checkpoint,
            history,
            deferred_results,
        )

    async def reconcile_deferred(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> _DeferredResume | None:
        if tenant_id != self._port.tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        execution = await self._port.load_execution(
            execution_id,
            tenant_id=tenant_id,
        )
        checkpoint = await self._port.load_recovery_checkpoint(
            execution_id,
            tenant_id=tenant_id,
        )
        if (
            execution is None
            or checkpoint is None
            or execution.status is not ExecutionStatus.WAITING_DEFERRED
            or checkpoint.state is not RecoveryCheckpointState.WAITING
            or checkpoint.pending_tools is None
        ):
            return None
        return await self.reconcile_waiting_deferred(
            checkpoint,
            execution,
        )

    async def reconcile(self) -> None:
        """Reconcile each durable checkpoint exactly once per startup page."""
        cursor: str | None = None
        while True:
            page = await self._port._list_recoverable_checkpoints(
                cursor=cursor,
            )
            for checkpoint in page.items:
                if checkpoint.state is RecoveryCheckpointState.COMPLETED:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                try:
                    await self.reconcile_checkpoint(checkpoint)
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


async def _step_messages(
    store: StepStore,
    run_id: str,
    *,
    include_interrupted: bool = False,
) -> list[ModelMessage]:
    snapshot = await store.latest_snapshot(
        run_id=run_id,
        include_interrupted=include_interrupted,
    )
    if snapshot is None:
        raise LookupError(run_id)
    return list(snapshot.messages)


class LocalExecutionBackend:
    """Resolve immutable definitions and persist one execution lifecycle."""

    def __init__(
        self,
        conversation: ConversationState,
        execution_state: ExecutionState,
        recovery: RecoveryState,
        execution_objects: ObjectStore,
        recovery_objects: ObjectStore,
        namespace: str,
        steps: StepStore,
        executor: AgentExecutor,
        catalog: AgentCatalog,
        *,
        workspace: Workspace,
        instruction_resolver: RepositoryInstructionResolver | None = None,
        app: object,
        tenant_id: str,
        step_reads: Mapping[RuntimeDomain, StepStore],
        step_lifecycle: _StepLifecycle,
        memory_store_factory: "Callable[[str, str, str], MemoryStore] | None" = None,
        conversation_durable: bool = False,
        input_materializer: ExecutionInputMaterializer | None = None,
        storage_contract: RuntimeStorageContract | None = None,
        storage_contract_factory: (
            "Callable[[Collection[RuntimeDomain]], RuntimeStorageContract] | None"
        ) = None,
        subagent_dispatcher: "_SubagentDispatcher | None" = None,
        live_broker: "LiveExecutionEventBroker | None" = None,
        payload_policy: "PayloadPolicy | None" = None,
        execution_objects_durable: bool = True,
        tool_operations: "_ToolOperationRuntimeRepository | None" = None,
        metric_recorder: "MetricRecorder | None" = None,
    ) -> None:
        self._conversation = conversation
        self._execution = execution_state
        self._recovery = recovery
        self._execution_objects = execution_objects
        self._recovery_objects = recovery_objects
        self._metric_recorder = metric_recorder
        self._namespace = namespace
        self._steps = steps
        self._executor = executor
        self._segment_runner = _AgentSegmentRunner(executor)
        self._catalog = catalog
        self._workspace = workspace
        self._app = app
        self._tenant_id = validate_tenant_id(tenant_id)
        self._memory_store_factory = memory_store_factory
        self._conversation_durable = conversation_durable
        self._input_materializer = input_materializer
        self._storage_contract = storage_contract
        self._storage_contract_factory = storage_contract_factory
        self._subagent_dispatcher = subagent_dispatcher
        self._live_broker = live_broker or LiveExecutionEventBroker()
        self._payload_policy = payload_policy or PayloadPolicy()
        self._tool_operations = tool_operations
        self._execution_objects_durable = execution_objects_durable
        self._step_reads = dict(step_reads)
        if frozenset(self._step_reads) != frozenset(
            {
                RuntimeDomain.CONVERSATION,
                RuntimeDomain.EXECUTION,
                RuntimeDomain.RECOVERY,
            }
        ):
            raise ValueError(
                "step_reads must contain exactly the three Step owner domains"
            )
        self._step_lifecycle = step_lifecycle
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._worker_failures: dict[str, _WorkerFailure] = {}
        self._captured_usage: dict[str, UsageMetrics] = {}
        self._terminal_events: dict[str, asyncio.Event] = {}
        self._pending_audit_events: dict[str, list[ExecutionEventAppend]] = {}
        self._pending_audit_locks: dict[str, asyncio.Lock] = {}
        self._recovery_relaunch_ids: set[str] = set()
        self._segment_only_worker_exits: set[str] = set()
        self._checkpoint_tasks: set[asyncio.Task[object]] = set()
        self._execution_durable_tasks: dict[
            str,
            set[asyncio.Task[object]],
        ] = {}
        self._worker_cancel_requests: set[str] = set()
        self._worker_shutdown_requests: set[str] = set()
        self._accepting = True
        execution_steps = self._step_reads[RuntimeDomain.EXECUTION]
        conversation_steps = self._step_reads[RuntimeDomain.CONVERSATION]
        execution_repository: ExecutionRepository = self._execution.executions
        session_repository: SessionRepository = self._conversation.sessions
        self._session_state_store = session_repository.state_store
        self._execution_state_store = execution_repository.state_store
        self._conversation_commands = ConversationStateCommands(
            session_repository.state_store,
            session_repository,
            conversation_steps
            if isinstance(conversation_steps, StateStepArchive)
            else None,
            self._conversation.histories,
        )
        self._runtime_commands = RuntimeStateCommands(
            execution_repository,
            namespace=self._namespace,
            events=self._execution.events,
            operations=self._execution.operations,
            approvals=self._recovery.approvals,
            external_calls=self._recovery.external_calls,
            conversation=session_repository,
            recovery=self._recovery.checkpoints,
            conversation_history=self._conversation.histories,
            tools=self._tool_operations,
            conversation_steps=(
                conversation_steps
                if isinstance(conversation_steps, StateStepArchive)
                else None
            ),
            execution_steps=execution_steps
            if isinstance(execution_steps, StateStepArchive)
            else None,
            recovery_steps=(
                self._step_reads[RuntimeDomain.RECOVERY]
                if isinstance(
                    self._step_reads[RuntimeDomain.RECOVERY], StateStepArchive
                )
                else None
            ),
            background_tasks=self._checkpoint_tasks,
        )
        self._recovery_coordinator = _RecoveryCoordinator(
            self,
            instruction_resolver,
        )

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    def validate_binding(self, execution: ExecutionRecord) -> None:
        binding = self._catalog.binding(execution.binding_digest)
        if execution.binding != binding.snapshot:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def load_execution(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ExecutionRecord | None:
        return await self._execution.executions.get(
            execution_id,
            tenant_id=tenant_id,
        )

    async def _list_recoverable_checkpoints(
        self,
        *,
        cursor: str | None,
    ) -> Page[RecoveryCheckpoint]:
        return await self._recovery.checkpoints.list_recoverable_page(
            tenant_id=self._tenant_id,
            cursor=cursor,
            limit=128,
        )

    async def load_recovery_checkpoint(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> RecoveryCheckpoint | None:
        return await self._recovery.checkpoints.get(
            execution_id,
            tenant_id=tenant_id,
        )

    async def _release_session_execution(
        self,
        execution: ExecutionRecord,
    ) -> None:
        if execution.session_id is None:
            return
        await self._conversation.sessions.release_execution(
            execution.session_id,
            tenant_id=execution.tenant_id,
            execution_id=execution.execution_id,
        )

    async def _commit_start_recovery_checkpoint(
        self,
        execution: ExecutionRecord,
        checkpoint: RecoveryCheckpoint,
        identity: IdempotencyRecord,
    ) -> ExecutionRecord:
        expected = (
            await self._expected_session_cursor(execution)
            if execution.session_id is not None
            else None
        )
        return await self._runtime_commands.commit_start_checkpoint(
            ExecutionStartClaim(
                execution.execution_id,
                execution.tenant_id,
                execution.revision,
                execution.event_sequence,
                identity.scope,
                identity.idempotency_key_digest,
                identity.request_digest,
                datetime.now(timezone.utc),
            ),
            recovery_checkpoint=checkpoint,
            session_id=execution.session_id,
            expected_cursor=expected,
        )

    async def _commit_recovery_resume(
        self,
        execution: ExecutionRecord,
    ) -> tuple[ExecutionRecord, RecoveryCheckpoint]:
        resumed = await self._recovery_commands_for(
            execution.execution_id
        ).commit_resumed(execution)
        checkpoint = await self.load_recovery_checkpoint(
            execution.execution_id,
            tenant_id=execution.tenant_id,
        )
        if checkpoint is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return resumed, checkpoint

    def _reset_local_producer(self, execution_id: str) -> None:
        self._live_broker.reset_completed_local_producer(execution_id)

    def _publish_recovery_resumed(
        self,
        execution_id: str,
        event_sequence: int,
    ) -> None:
        self._live_broker.publish_event(
            execution_id,
            ExecutionEventType.EXECUTION_RESUMED,
            {},
            durable_sequence=event_sequence,
        )

    def _mark_recovery_relaunch(self, execution_id: str) -> None:
        self._recovery_relaunch_ids.add(execution_id)

    async def _validate_start(
        self, request: ExecutionRequest, execution: ExecutionRecord
    ) -> None:
        if not self._accepting:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if (
            request.principal.tenant_id != self._tenant_id
            or execution.tenant_id != self._tenant_id
        ):
            _logger.warning(
                "local execution tenant rejected: expected=%s request=%s execution=%s",
                self._tenant_id,
                request.principal.tenant_id,
                execution.tenant_id,
            )
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if request.correlation != execution.correlation:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
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
        return self._execution_durable_tasks

    def _execution_task_set(
        self,
        execution_id: str,
    ) -> set[asyncio.Task[object]]:
        return self._execution_task_map().setdefault(execution_id, set())

    def execution_task_set(
        self,
        execution_id: str,
    ) -> set[asyncio.Task[object]]:
        return self._execution_task_set(execution_id)

    def _track_checkpoint_task(
        self,
        task: "asyncio.Task[object]",
        label: str,
        execution_id: str | None = None,
    ) -> None:
        self._checkpoint_tasks.add(task)
        execution_tasks = (
            None if execution_id is None else self._execution_task_set(execution_id)
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

    def _record_committed_terminal(
        self,
        committed: ExecutionTerminalCommitResult,
        *,
        session_id: str | None,
    ) -> None:
        recorder = self._metric_recorder
        if recorder is None:
            return
        _record_execution_terminal(
            recorder,
            source_namespace=self._namespace,
            result=committed,
            session_id=session_id,
        )
        _release_metric_execution_context(
            recorder,
            committed.execution.execution_id,
        )

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
        self._record_committed_terminal(committed, session_id=session_id)
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
                current.status is ExecutionStatus.WAITING_DEFERRED
                and expected_status
                in {
                    ExecutionStatus.STARTED,
                    ExecutionStatus.WAITING_DEFERRED,
                }
            ):
                checkpoint = await self._recovery.checkpoints.get(
                    commit.execution_id,
                    tenant_id=commit.tenant_id,
                )
                if (
                    checkpoint is None
                    or checkpoint.state is not RecoveryCheckpointState.WAITING
                    or checkpoint.pending_tools is None
                    or checkpoint.agent_run_sequence != current.agent_run_sequence
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if (
                    current.revision != commit.expected_revision
                    or current.event_sequence != commit.expected_event_sequence
                ):
                    effective_commit = replace(
                        commit,
                        expected_revision=current.revision,
                        expected_event_sequence=current.event_sequence,
                    )
                committed = await self._runtime_commands.commit_deferred_cancel_checkpoint(
                    effective_commit,
                    expected_recovery_revision=checkpoint.revision,
                    expected_pending_tools=checkpoint.pending_tools,
                    background_tasks=self._execution_task_set(commit.execution_id),
                )
            elif (
                current.status is expected_status
                and current.revision == commit.expected_revision
                and current.event_sequence == commit.expected_event_sequence
                and current.status is not ExecutionStatus.WAITING_DEFERRED
            ):
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

    async def restore_user_input(
        self,
        execution: ExecutionRecord,
    ) -> CanonicalUserInput:
        if self._input_materializer is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if execution.stored_user_input is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await self._input_materializer.restore(execution.stored_user_input)

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
        storage_contract = execution.storage_contract
        if storage_contract is None:
            storage_contract = self._require_storage_contract(execution.session_id)
        if storage_contract != self._require_storage_contract(execution.session_id):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        now = datetime.now(timezone.utc)
        candidate = RecoveryCheckpoint(
            execution_id=execution.execution_id,
            tenant_id=execution.tenant_id,
            step_run_id=None,
            agent_run_sequence=execution.agent_run_sequence,
            state=RecoveryCheckpointState.ADMITTED,
            revision=0,
            created_at=now,
            updated_at=now,
            handoff_phase=RecoveryHandoffPhase.NONE,
            terminal_handoff=None,
            pending_operation_id=None,
        )
        expected = (
            await self._expected_session_cursor(execution)
            if execution.session_id is not None
            else None
        )
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
            recovery_checkpoint=candidate,
            session_id=execution.session_id,
            expected_cursor=expected,
        )
        _logger.info(
            "execution start checkpoint committed: execution=%s", execution.execution_id
        )
        return started

    async def _cancel_admitted_start_if_closing(
        self, execution: ExecutionRecord
    ) -> bool:
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
        if (
            session.status
            not in {
                SessionStatus.CLOSING,
                SessionStatus.CLEANUP_REQUIRED,
            }
            or session.active_execution_id != execution.execution_id
        ):
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
                if (
                    session.status
                    not in {
                        SessionStatus.OPEN,
                        SessionStatus.CLOSING,
                        SessionStatus.CLEANUP_REQUIRED,
                    }
                    or session.continuation != expected
                ):
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

    async def launch(
        self,
        request: ExecutionRequest,
        execution: ExecutionRecord,
        *,
        resume: _DeferredResume | None = None,
    ) -> None:
        await self._validate_start(request, execution)
        checkpoint = await self._recovery.checkpoints.get(
            execution.execution_id,
            tenant_id=execution.tenant_id,
        )
        if checkpoint is None or checkpoint.state is RecoveryCheckpointState.COMPLETED:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution.stored_user_input is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        request = replace(
            request,
            user_prompt=await self.restore_user_input(execution),
            files=(),
        )
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
            or current.correlation != execution.correlation
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
        task = asyncio.create_task(
            self._run(request, current, resume),
            name=f"ai-execution-{execution.execution_id}",
        )
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

    def _validate_recovery_identity(self, execution: ExecutionRecord) -> None:
        if execution.storage_contract != self._require_storage_contract(
            execution.session_id
        ):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

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
        self._worker_shutdown_set().discard(execution_id)
        self._captured_usage.pop(execution_id, None)
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
            failure = _WorkerFailure(
                error.code,
                dict(error.safe_details),
                error.diagnostics,
                error.category,
                error.retryable,
                error.operation_id,
            )
        else:
            failure = _WorkerFailure(
                ErrorCode.INTERNAL_ERROR,
                {"phase": "local_execution_worker"},
                ErrorDiagnostics.from_exception(error),
            )
        details = dict(failure.safe_details)
        details["execution_id"] = execution_id
        self._worker_failures[execution_id] = _WorkerFailure(
            failure.code,
            details,
            failure.diagnostics,
            failure.category,
            failure.retryable,
            failure.operation_id,
        )
        _logger.error(
            "local execution worker failed: execution=%s code=%s",
            execution_id,
            failure.code.value,
            exc_info=environ.debug,
        )
        if live_broker is not None:
            live_broker.complete(execution_id)

    def _worker_shutdown_set(self) -> set[str]:
        try:
            return self._worker_shutdown_requests
        except AttributeError:
            requests: set[str] = set()
            self._worker_shutdown_requests = requests
            return requests

    def _request_worker_shutdown(
        self,
        execution_id: str,
        task: asyncio.Task[None],
    ) -> None:
        if task.done() or execution_id in self._worker_cancel_requests:
            return
        self._worker_shutdown_set().add(execution_id)
        self._request_worker_cancel(execution_id, task)

    def _request_worker_cancel(
        self,
        execution_id: str,
        task: asyncio.Task[None],
    ) -> None:
        requests = self._worker_cancel_requests
        if task.done() or execution_id in requests:
            return
        requests.add(execution_id)
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
        return AIError(
            failure.code,
            category=failure.category,
            retryable=failure.retryable,
            operation_id=failure.operation_id,
            safe_details=dict(failure.safe_details),
            diagnostics=failure.diagnostics,
        )

    def worker_installed(self, execution_id: str) -> bool:
        task = self._tasks.get(execution_id)
        return task is not None and not task.done()

    def owns_execution(self, execution_id: str, *, tenant_id: str) -> bool:
        return tenant_id == self._tenant_id and self.worker_installed(execution_id)

    async def cancel_children(
        self,
        parent_execution_id: str,
        principal: Principal,
    ) -> None:
        if self._subagent_dispatcher is not None:
            await self._subagent_dispatcher.cancel_children(
                parent_execution_id,
                principal,
            )

    async def wait_terminal(self, execution_id: str, *, tenant_id: str) -> None:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        event = self._terminal_events.setdefault(execution_id, asyncio.Event())
        await event.wait()

    @property
    def live_broker(self) -> LiveExecutionEventBroker:
        return self._live_broker

    async def cancel(self, execution: ExecutionRecord) -> CancelEffectOutcome:
        self._worker_shutdown_set().discard(execution.execution_id)
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
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if not isinstance(raw, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return raw

    async def load_repository_instructions(
        self,
        reference: RuntimePayloadRef | None,
    ) -> RepositoryInstructions | None:
        if reference is None:
            return None
        if reference.source_domain not in {
            RuntimeDomain.EXECUTION,
            RuntimeDomain.RECOVERY,
        }:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        objects = (
            self._execution_objects
            if reference.source_domain is RuntimeDomain.EXECUTION
            else self._recovery_objects
        )
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
            data = await read_runtime_object(objects, payload.ref)
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

    async def reconcile_deferred(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> None:
        decision = await self._recovery_coordinator.reconcile_deferred(
            execution_id,
            tenant_id=tenant_id,
        )
        if decision is None:
            return
        current = decision.execution
        principal = Principal(
            current.principal_id,
            current.tenant_id,
            current.principal_kind,
        )
        request = ExecutionRequest(
            user_prompt=await self.restore_user_input(current),
            principal=principal,
            idempotency_key=f"recovery:{current.execution_id}",
            memory_scope=current.memory_scope,
            mode=current.mode,
            planning=current.planning,
            thinking=current.thinking,
            correlation=current.correlation,
        )
        self._mark_recovery_relaunch(current.execution_id)
        await self.launch(request, current, resume=decision)
        _logger.info(
            "deferred execution relaunched: execution=%s",
            current.execution_id,
        )

    async def materialize_deferred_call(
        self,
        execution: ExecutionRecord,
        source_step_run_id: str,
        call: ToolCallPart,
        metadata: Mapping[str, object],
    ) -> PendingDeferredCall:
        try:
            arguments = normalize_json_value(call.args_as_dict())
            raw_metadata = normalize_json_value(metadata)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT) from error
        if not isinstance(arguments, Mapping) or not isinstance(raw_metadata, Mapping):
            raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
        arguments = dict(arguments)
        metadata_value = raw_metadata.get("linktools")
        if metadata_value is not None:
            if not isinstance(metadata_value, Mapping):
                raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
            if (
                metadata_value.get("kind") != "workspace_approval"
                or not isinstance(metadata_value.get("version"), int)
                or isinstance(metadata_value.get("version"), bool)
                or metadata_value.get("version") != 1
            ):
                raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
            canonical_args = metadata_value.get("canonical_args")
            arguments_digest = metadata_value.get("arguments_digest")
            if (
                not isinstance(canonical_args, Mapping)
                or not isinstance(arguments_digest, str)
                or canonical_sha256(canonical_args) != arguments_digest
            ):
                raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
            arguments = dict(canonical_args)
            raw_metadata = {
                key: value
                for key, value in raw_metadata.items()
                if key != "linktools"
            }
        arguments_payload = StoredPayload.inline_json(arguments)
        if not payload_fits_inline(arguments_payload, self._payload_policy):
            reference = await put_runtime_object(
                self._recovery_objects,
                RuntimeObjectKeyFactory(self._namespace),
                RuntimeDomain.RECOVERY,
                execution.tenant_id,
                canonical_json_bytes(arguments),
            )
            arguments_payload = StoredPayload.object(reference)
        return PendingDeferredCall(
            call.tool_call_id,
            call.tool_name,
            arguments_payload,
            arguments_payload.digest,
            raw_metadata,
        )

    async def commit_deferred_frontier(
        self,
        execution: ExecutionRecord,
        checkpoint: RecoveryCheckpoint,
        continuation: PendingToolContinuation,
        approval_records: tuple[ApprovalRecord, ...],
        external_records: tuple[ExternalCallRecord, ...],
        occurred_at: datetime,
    ) -> tuple[ExecutionRecord, RecoveryCheckpoint]:
        pending_audit = tuple(
            self._pending_audit_events.get(execution.execution_id, ())
        )
        committed, committed_checkpoint = (
            await self._runtime_commands.commit_deferred_checkpoint(
                execution_id=execution.execution_id,
                tenant_id=execution.tenant_id,
                expected_execution_revision=execution.revision,
                expected_event_sequence=execution.event_sequence,
                expected_recovery_revision=checkpoint.revision,
                expected_agent_run_sequence=execution.agent_run_sequence,
                continuation=continuation,
                audit_events=pending_audit,
                approval_records=approval_records,
                external_records=external_records,
                occurred_at=occurred_at,
                background_tasks=self._execution_task_set(execution.execution_id),
            )
        )
        self._pending_audit_events.pop(execution.execution_id, None)
        if pending_audit:
            self._live_broker.confirm_events(
                execution.execution_id,
                first_sequence=execution.event_sequence + 1,
                count=len(pending_audit),
            )
        offset = execution.event_sequence + len(pending_audit)
        for index, (kind, item) in enumerate(
            (
                *(
                    (ExecutionEventType.APPROVAL_REQUESTED, value)
                    for value in continuation.approvals
                ),
                *(
                    (ExecutionEventType.EXTERNAL_REQUESTED, value)
                    for value in continuation.calls
                ),
            ),
            1,
        ):
            self._live_broker.publish_event(
                execution.execution_id,
                kind,
                {
                    "tool_call_id": item.tool_call_id,
                    "tool_name": item.tool_name,
                    "arguments_digest": item.arguments_digest,
                },
                durable_sequence=offset + index,
            )
        self._segment_only_worker_exits.add(execution.execution_id)
        return committed, committed_checkpoint

    async def load_approval(
        self,
        approval_id: str,
        *,
        tenant_id: str,
    ) -> ApprovalRecord | None:
        return await self._recovery.approvals.get(
            approval_id,
            tenant_id=tenant_id,
        )

    async def load_external_call(
        self,
        call_id: str,
        *,
        tenant_id: str,
    ) -> ExternalCallRecord | None:
        return await self._recovery.external_calls.get(
            call_id,
            tenant_id=tenant_id,
        )

    async def load_interrupted_messages(
        self,
        run_id: str,
    ) -> tuple[ModelMessage, ...]:
        archive = self._step_reads[RuntimeDomain.RECOVERY]
        snapshot = await archive.latest_snapshot(
            run_id=run_id,
            include_interrupted=True,
        )
        if snapshot is None or snapshot.state != "interrupted":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return tuple(snapshot.messages)

    async def claim_deferred_resume(
        self,
        checkpoint: RecoveryCheckpoint,
        execution: ExecutionRecord,
    ) -> tuple[ExecutionRecord, RecoveryCheckpoint]:
        if checkpoint.pending_tools is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await self._runtime_commands.claim_deferred_resume_checkpoint(
            execution_id=execution.execution_id,
            tenant_id=execution.tenant_id,
            expected_execution_revision=execution.revision,
            expected_event_sequence=execution.event_sequence,
            expected_recovery_revision=checkpoint.revision,
            expected_agent_run_sequence=execution.agent_run_sequence,
            expected_pending_tools=checkpoint.pending_tools,
            background_tasks=self._execution_task_set(execution.execution_id),
        )

    async def read_deferred_payload(self, payload: StoredPayload) -> JsonValue:
        if payload.kind == "inline":
            value = payload.decode()
        else:
            if payload.ref is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            raw = await read_runtime_object(self._recovery_objects, payload.ref)
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeError, TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        try:
            return normalize_json_value(value)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    async def reconcile(self) -> None:
        await self._recovery_coordinator.reconcile()

    def _require_storage_contract(
        self,
        session_id: str | None = None,
    ) -> RuntimeStorageContract:
        factory = self._storage_contract_factory
        if factory is not None:
            domains = {
                RuntimeDomain.EXECUTION,
                RuntimeDomain.RECOVERY,
            }
            if session_id is not None:
                domains.add(RuntimeDomain.CONVERSATION)
            return factory(tuple(domains))
        storage_contract = self._storage_contract
        if storage_contract is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return storage_contract

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
                if execution.status is ExecutionStatus.WAITING_DEFERRED:
                    if (
                        checkpoint.state is not RecoveryCheckpointState.WAITING
                        or checkpoint.pending_tools is None
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
                        expected_status=ExecutionStatus.WAITING_DEFERRED,
                    )
                    if committed.status is not ExecutionStatus.CANCELLING:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    fresh_checkpoint = await self._recovery.checkpoints.get(
                        execution.execution_id,
                        tenant_id=execution.tenant_id,
                    )
                    if (
                        fresh_checkpoint is None
                        or fresh_checkpoint.state is not RecoveryCheckpointState.ACTIVE
                        or fresh_checkpoint.pending_tools is not None
                    ):
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                    execution = committed
                elif execution.status is ExecutionStatus.CANCELLING:
                    if checkpoint.pending_tools is not None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                elif (
                    execution.status is ExecutionStatus.STARTED
                    and checkpoint.state is RecoveryCheckpointState.ACTIVE
                    and checkpoint.pending_tools is not None
                ):
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
                error_diagnostics=_execution_error_diagnostics(error),
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
        if outcome.terminal_status in {
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        } and (handoff.conversation is not None or outcome.output is not None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            execution is not None
            and execution.status
            in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }
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
        store = (
            self._recovery_objects
            if source is RuntimeDomain.RECOVERY
            else self._execution_objects
        )
        await read_runtime_object(store, reference)

    async def _reconcile_handoff(
        self, checkpoint: RecoveryCheckpoint
    ) -> ExecutionRecord:
        handoff = checkpoint.terminal_handoff
        if handoff is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        outcome = handoff.outcome
        if outcome.terminal_status in {
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        } and (handoff.conversation is not None or outcome.output is not None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        execution = await self._execution.executions.get(
            checkpoint.execution_id,
            tenant_id=checkpoint.tenant_id,
        )
        if execution is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._validate_recovery_identity(execution)
        if (
            execution.status
            in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }
            and execution.status is not outcome.terminal_status
        ):
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
                snapshot = await self._step_reads[
                    RuntimeDomain.EXECUTION
                ].latest_snapshot(
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
                    execution = await self._claim_session_or_recovery_finalizing(
                        execution
                    )
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
                    error_diagnostics=_execution_error_diagnostics(error),
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

    async def _recovery_idempotency(
        self,
        execution: ExecutionRecord,
    ) -> IdempotencyRecord:
        records = await self._execution.idempotency.list_by_resource(
            ResourceKind.EXECUTION,
            execution.execution_id,
            tenant_id=execution.tenant_id,
        )
        if len(records) != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        identity = records[0]
        if (
            identity.runtime_domain is not RuntimeDomain.EXECUTION
            or identity.resource_kind is not ResourceKind.EXECUTION
            or identity.resource_id != execution.execution_id
            or identity.tenant_id != execution.tenant_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return identity

    async def _ensure_recovery_idempotency(
        self,
        execution: ExecutionRecord,
        *,
        expected_status: IdempotencyStatus,
    ) -> IdempotencyRecord:
        identity = await self._recovery_idempotency(execution)
        return await self._restore_recovery_idempotency(
            execution,
            identity,
            expected_status=expected_status,
        )

    async def _restore_recovery_idempotency(
        self,
        execution: ExecutionRecord,
        identity: IdempotencyRecord,
        *,
        expected_status: IdempotencyStatus,
    ) -> IdempotencyRecord:
        if (
            identity.runtime_domain is not RuntimeDomain.EXECUTION
            or identity.resource_kind is not ResourceKind.EXECUTION
            or identity.resource_id != execution.execution_id
            or identity.tenant_id != execution.tenant_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if identity.status is expected_status:
            return identity
        if (
            expected_status is IdempotencyStatus.STARTED
            and identity.status is IdempotencyStatus.RESERVED
        ):
            return await self._execution.idempotency.compare_and_swap(
                identity.scope,
                identity.idempotency_key_digest,
                tenant_id=identity.tenant_id,
                expected_status=identity.status,
                next_record=replace(
                    identity,
                    status=IdempotencyStatus.STARTED,
                    updated_at=execution.updated_at,
                ),
            )
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    async def _resolve_handoff_conversation(
        self, checkpoint: RecoveryCheckpoint, handoff: RecoveryTerminalHandoff
    ) -> None:
        intent = handoff.conversation
        if intent is None:
            return
        session = await self._conversation.sessions.get(
            intent.session_id, tenant_id=checkpoint.tenant_id
        )
        if session is None:
            if self._conversation_durable:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            _logger.info(
                "recovery conversation resolve skipped: session lost execution=%s",
                checkpoint.execution_id,
            )
            return
        if self._step_reads.get(RuntimeDomain.CONVERSATION) is not self._steps:
            if handoff.source_step_run_id is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._step_lifecycle.materialize_from_recovery(
                target=RuntimeDomain.CONVERSATION,
                step_run_id=handoff.source_step_run_id,
            )
            target_snapshot = await self._step_reads[
                RuntimeDomain.CONVERSATION
            ].latest_snapshot(
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
            if (
                latest.active_execution_id != checkpoint.execution_id
                or latest.continuation != intent.expected_cursor
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            raise

    async def _advance_handoff(
        self, checkpoint: RecoveryCheckpoint, phase: RecoveryHandoffPhase
    ) -> RecoveryCheckpoint:
        if checkpoint.terminal_handoff is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        current_rank = _HANDOFF_PHASE_RANK.get(checkpoint.handoff_phase)
        requested_rank = _HANDOFF_PHASE_RANK.get(phase)
        if (
            current_rank is None
            or requested_rank is None
            or requested_rank < current_rank
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if requested_rank == current_rank:
            return checkpoint
        if requested_rank != current_rank + 1:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        updated = replace(
            checkpoint,
            handoff_phase=phase,
            state=RecoveryCheckpointState.HANDOFF,
            pending_tools=None,
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
            current = await self._recovery.checkpoints.get(
                checkpoint.execution_id, tenant_id=checkpoint.tenant_id
            )
            if current is None:
                raise
            same_handoff = current.terminal_handoff == checkpoint.terminal_handoff
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
            pending_tools=None,
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
            if (
                current is None
                or current.state is not RecoveryCheckpointState.COMPLETED
            ):
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
        error_diagnostics: ErrorDiagnostics | None = None,
    ) -> RecoveryCheckpoint:
        handoff = checkpoint.terminal_handoff
        if (
            checkpoint.handoff_phase is not RecoveryHandoffPhase.PREPARED
            or handoff is None
            or handoff.outcome.terminal_status is not ExecutionStatus.SUCCEEDED
            or target_status
            not in {
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            target_status is not ExecutionStatus.FAILED
            and error_diagnostics is not None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        outcome = handoff.outcome
        terminal_payload = _terminal_error_payload(
            error_code,
            {},
            error_diagnostics,
        )
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
                terminal_event_payload=terminal_payload,
                result_created_at=outcome.result_created_at,
                error_diagnostics=error_diagnostics,
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
        self._validate_recovery_identity(current)
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
            if (
                outcome.output.kind == "object"
                and outcome.object_source_domain is RuntimeDomain.RECOVERY
            ):
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
            if (
                current.session_id is not None
                and current.status is not ExecutionStatus.FINALIZING
            ):
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
            current,
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
            error_diagnostics=outcome.error_diagnostics,
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
            or dict(execution.safe_error_details)
            != dict(handoff.outcome.safe_error_details)
            or execution.error_diagnostics != handoff.outcome.error_diagnostics
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._validate_recovery_identity(execution)
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
        if (
            records[0].status is not expected_status
            or records[0].error_code != handoff.outcome.error_code
        ):
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
                self._request_worker_shutdown(execution_id, task)
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
            owned_background_by_identity = {
                id(task): task
                for task in (*background, *dispatcher_background)
                if isinstance(task, asyncio.Task) and not task.done()
            }
            owned_background = tuple(owned_background_by_identity.values())
            if not owned_background:
                break
            await asyncio.gather(
                *(asyncio.shield(task) for task in owned_background),
                return_exceptions=True,
            )

        pending_workers = tuple(
            task for task in self._tasks.values() if not task.done()
        )
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
        dispatcher_failure = (
            None
            if self._subagent_dispatcher is None
            else self._subagent_dispatcher.background_failure
        )
        if pending_workers or pending_background:
            raise AIError(
                ErrorCode.STORAGE_RECOVERY_REQUIRED,
                safe_details={
                    "phase": "local_execution_close",
                    "pending_workers": len(pending_workers),
                    "pending_background_tasks": len(pending_background),
                    "pending_checkpoint_tasks": len(checkpoint_background),
                    "pending_execution_tasks": len(execution_background),
                    "background_failures": int(dispatcher_failure is not None),
                },
            )
        if self._worker_failures:
            worker_failure = self._worker_failures[sorted(self._worker_failures)[0]]
            raise AIError(
                worker_failure.code,
                category=worker_failure.category,
                retryable=worker_failure.retryable,
                operation_id=worker_failure.operation_id,
                safe_details=dict(worker_failure.safe_details),
                diagnostics=worker_failure.diagnostics,
            )
        if dispatcher_failure is not None:
            details = dict(dispatcher_failure.safe_details)
            details.setdefault("phase", "local_execution_close")
            details.setdefault("pending_workers", 0)
            details.setdefault("pending_background_tasks", 0)
            details.setdefault("pending_checkpoint_tasks", 0)
            details.setdefault("pending_execution_tasks", 0)
            details.setdefault("background_failures", 1)
            raise AIError(
                dispatcher_failure.code,
                category=dispatcher_failure.category,
                retryable=dispatcher_failure.retryable,
                operation_id=dispatcher_failure.operation_id,
                safe_details=details,
                diagnostics=dispatcher_failure.diagnostics,
            )
        self._tasks.clear()
        self._captured_usage.clear()
        self._terminal_events.clear()
        self._worker_failures.clear()
        self._pending_audit_events.clear()
        self._pending_audit_locks.clear()
        self._segment_only_worker_exits.clear()
        self._checkpoint_tasks.clear()
        self._worker_cancel_requests.clear()
        self._worker_shutdown_set().clear()
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
        self._worker_shutdown_set().discard(execution_id)
        self._terminal_events.pop(execution_id, None)
        self._worker_failures.pop(execution_id, None)
        self._captured_usage.pop(execution_id, None)
        self._pending_audit_events.pop(execution_id, None)
        self._pending_audit_locks.pop(execution_id, None)
        self._segment_only_worker_exits.discard(execution_id)
        self._execution_task_map().pop(execution_id, None)
        _logger.debug(
            "local execution runtime cache released: tenant=%s execution=%s",
            tenant_id,
            execution_id,
        )

    async def _reconcile_unresolved_tool_operations(
        self,
        step_run_id: str,
        operations: Sequence[ToolOperationRecord],
        *,
        tenant_id: str,
    ) -> None:
        for operation in operations:
            while True:
                if self._tool_operations is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                current = await self._tool_operations.get_by_call(
                    step_run_id,
                    operation.tool_call_id,
                    tenant_id=tenant_id,
                )
                if current is None:
                    raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
                if current.status in {
                    ToolOperationStatus.COMPLETED,
                    ToolOperationStatus.FAILED,
                }:
                    break
                if current.status in {
                    ToolOperationStatus.EFFECT_UNKNOWN,
                    ToolOperationStatus.CANCELLED,
                }:
                    raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
                if current.status is ToolOperationStatus.CLAIMED:
                    expires = current.lease_expires_at
                    if expires is None:
                        raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
                    remaining = (expires - datetime.now(timezone.utc)).total_seconds()
                    if remaining > 0:
                        await asyncio.sleep(min(1.0, remaining))
                        continue
                    if not current.replay_safe:
                        raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
                break
        _logger.info(
            "recovery tool operations reconciled: run=%s count=%s",
            step_run_id,
            len(operations),
        )

    async def _run(
        self,
        request: ExecutionRequest,
        original: ExecutionRecord,
        resume: _DeferredResume | None = None,
    ) -> None:
        execution_id = original.execution_id
        execution_tasks = self._execution_task_set(execution_id)
        checkpoint: RecoveryCheckpoint | None = None
        run_id: str | None = None
        recovery_history_run_id: str | None = None
        recovery_relaunch_ids = self._recovery_relaunch_ids
        exact_recovery_context = execution_id in recovery_relaunch_ids
        recovery_relaunch_ids.discard(execution_id)
        metric_recorder = self._metric_recorder
        metric_id = uuid.uuid4().hex if metric_recorder is not None else None
        metric_started = monotonic_ns() if metric_id is not None else None
        metric_status = "FAILED"
        metric_error_code: str | None = None
        claimed_from_admitted = False
        try:
            current = await self._execution.executions.get(
                execution_id, tenant_id=original.tenant_id
            )
            if current is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            checkpoint = await self._recovery.checkpoints.get(
                execution_id,
                tenant_id=current.tenant_id,
            )
            if checkpoint is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._validate_recovery_identity(current)
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
                    "agent attempt activated: execution=%s sequence=%s",
                    execution_id,
                    current.agent_run_sequence,
                )
            elif checkpoint.state is RecoveryCheckpointState.WAITING:
                if current.status is not ExecutionStatus.WAITING_DEFERRED:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                resume = await self._recovery_coordinator.reconcile_waiting_deferred(
                    checkpoint,
                    current,
                )
                if resume is None:
                    return
                current = resume.execution
                checkpoint = resume.checkpoint
            elif checkpoint.state is not RecoveryCheckpointState.ACTIVE:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if (
                current.status is not ExecutionStatus.STARTED
                or checkpoint.step_run_id is None
                or checkpoint.agent_run_sequence != current.agent_run_sequence
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            binding = self._catalog.binding(current.binding_digest)
            if current.binding != binding.snapshot:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            definition = binding.definition
            initial_repository_instructions = await self.load_repository_instructions(
                current.repository_instructions
            )
            repository_overlay = await self.load_repository_instructions(
                checkpoint.repository_instruction_overlay
            )
            repository_instructions = _merge_repository_instructions(
                initial_repository_instructions,
                repository_overlay,
            )
            run_id = checkpoint.step_run_id
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
                or (
                    None
                    if session.continuation is None
                    else session.continuation.history_id
                )
            )
            tool_owner = f"tool:{execution_id}:{uuid.uuid4().hex}"
            source_replay_history = (
                None if resume is None else list(resume.history)
            )
            deferred_tool_results = None if resume is None else resume.results
            resumed_deferred_attempt = deferred_tool_results is not None
            exact_recovery_context = not claimed_from_admitted
            recovery_history_run_id = None if claimed_from_admitted else run_id
            if recovery_history_run_id is not None:
                recovery_archive = self._step_store(RuntimeDomain.RECOVERY)
                recovery_run = await recovery_archive.get_run(
                    run_id=recovery_history_run_id
                )
                snapshot = await recovery_archive.latest_snapshot(
                    run_id=recovery_history_run_id,
                    include_interrupted=True,
                )
                if snapshot is None:
                    if recovery_run is not None:
                        raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
                elif self._tool_operations is not None:
                    unresolved = tuple(
                        operation
                        for operation in await self._tool_operations.list_by_step_run(
                            recovery_history_run_id,
                            tenant_id=current.tenant_id,
                        )
                        if operation.status
                        not in {
                            ToolOperationStatus.COMPLETED,
                            ToolOperationStatus.FAILED,
                        }
                    )
                    if unresolved:
                        await self._reconcile_unresolved_tool_operations(
                            recovery_history_run_id,
                            unresolved,
                            tenant_id=current.tenant_id,
                        )
            try:
                tool_repository = self._tool_operations
            except AttributeError:
                tool_repository = None
            tool_operations = (
                RuntimeToolOperationBridge(
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
                )
                if tool_repository is not None
                else None
            )
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
            repository_boundary = _RepositoryInstructionBoundary(
                self._recovery_coordinator,
                current,
                initial_repository_instructions,
                repository_instructions,
            )
            run_user_prompt = None if resumed_deferred_attempt else request.user_prompt

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
            selected_memory = tuple(
                name for name in runtime_tool_names if name in MEMORY_TOOL_NAMES
            )
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
            public_context = AgentContext(
                app=self._app,
                principal=request.principal,
                workspace=self._workspace,
                session_id=current.session_id,
                execution_id=current.execution_id,
                session_metadata=session_metadata,
                memory_scope=current.memory_scope,
                correlation=current.correlation,
            )
            plan_store = RuntimePlanStore(
                self._session_state_store
                if current.session_id is not None
                else self._execution_state_store,
                namespace=self._namespace,
                tenant_id=current.tenant_id,
                owner_kind="session" if current.session_id is not None else "execution",
                owner_id=current.session_id or current.execution_id,
            )
            try:
                segment = await self._segment_runner.run(
                    _AgentSegmentInput(
                        binding=binding,
                        context=public_context,
                        user_prompt=run_user_prompt,
                        history=history,
                        conversation_id=conversation_id,
                        step_store=self._steps,
                        step_run_id=run_id,
                        segment_sequence=current.agent_run_sequence,
                        history_id=history_id,
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
                            else self._subagent_dispatcher.descriptions_for(
                                subagent_refs
                            )
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
                        usage_sink=lambda usage: self._capture_usage(
                            execution_id, usage
                        ),
                        tool_operations=tool_operations,
                        background_tasks=execution_tasks,
                        replace_history_system_prompt=(
                            session_history_start and not exact_recovery_context
                        ),
                        repository_instructions=repository_instructions,
                        repository_instruction_boundary=repository_boundary,
                        deferred_tool_results=deferred_tool_results,
                    )
                )
                if isinstance(segment, _SegmentCancelled):
                    raise asyncio.CancelledError
                if isinstance(segment, _SegmentFailed):
                    raise segment.error
                result = (
                    segment.requests
                    if isinstance(segment, _SegmentDeferred)
                    else segment.result
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
                            raise _secondary_execution_error(
                                readback_error, error
                            ) from error
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
                metric_status = current.status.value
                metric_error_code = current.error_code
                _logger.exception(
                    "local execution failed: execution=%s",
                    execution_id,
                )
                return
            if isinstance(result, DeferredToolRequests):
                if checkpoint is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self._recovery_coordinator.commit_deferred_pause(
                    current,
                    result,
                    step_run_id=run_id,
                    paused_at=datetime.now(timezone.utc),
                )
                metric_status = ExecutionStatus.WAITING_DEFERRED.value
                return
            committed = await self._commit_success(
                current,
                binding,
                result.output,
                result.usage,
                run_id,
            )
            metric_status = (
                "CANCELLED"
                if committed.status is ExecutionStatus.CANCELLING
                else committed.status.value
            )
            metric_error_code = committed.error_code
            _logger.debug(
                "local execution completed: execution=%s run=%s", execution_id, run_id
            )
        except asyncio.CancelledError:
            current = await self._execution.executions.get(
                execution_id,
                tenant_id=original.tenant_id,
            )
            if current is not None and current.status is ExecutionStatus.FINALIZING:
                metric_status = "CANCELLED"
                raise
            if current is not None and current.status is ExecutionStatus.CANCELLING:
                self._worker_shutdown_set().discard(execution_id)
                current = await self._commit_terminal(
                    current,
                    ExecutionStatus.CANCELLED,
                    None,
                    ErrorCode.EXECUTION_CANCELLED.value,
                    StopReason.CANCELLED,
                    run_id=run_id,
                )
            elif (
                current is not None
                and current.status
                not in {
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                }
            ):
                current = await self._commit_terminal(
                    current,
                    ExecutionStatus.CANCELLED,
                    None,
                    ErrorCode.EXECUTION_CANCELLED.value,
                    StopReason.CANCELLED,
                    run_id=run_id,
                )
            if current is not None and current.status in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }:
                metric_status = current.status.value
                metric_error_code = current.error_code
            else:
                metric_status = "CANCELLED"
            raise
        except Exception as error:
            metric_status = "FAILED"
            metric_error_code = _execution_error_code(error).value
            _logger.exception(
                "local execution infrastructure failure: execution=%s",
                execution_id,
            )
            raise
        finally:
            _record_storage_operation(
                metric_recorder,
                observation_id=metric_id,
                started_at_ns=metric_started,
                source_namespace=self._namespace,
                tenant_id=original.tenant_id,
                execution_id=execution_id,
                session_id=original.session_id,
                correlation=original.correlation,
                status=metric_status,
                error_code=metric_error_code,
                domain="execution",
                target="runtime",
            )

    async def _finish_checkpoint(self, checkpoint: RecoveryCheckpoint) -> None:
        if checkpoint.state is RecoveryCheckpointState.COMPLETED:
            return
        if checkpoint.terminal_handoff is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        updated = replace(
            checkpoint,
            state=RecoveryCheckpointState.COMPLETED,
            handoff_phase=RecoveryHandoffPhase.NONE,
            pending_tools=None,
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
                checkpoint.execution_id, tenant_id=checkpoint.tenant_id
            )
            if (
                current is None
                or current.state is not RecoveryCheckpointState.COMPLETED
            ):
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
                    await _step_messages(
                        self._step_store(RuntimeDomain.RECOVERY),
                        recovery_run_id,
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
                return list(
                    await _step_messages(
                        conversation_steps,
                        execution.conversation_step_run_id,
                        include_interrupted=True,
                    )
                )
            except LookupError as error:
                raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE) from error
        if execution.base_execution_id is None:
            return []
        base = await self._execution.executions.get(
            execution.base_execution_id, tenant_id=execution.tenant_id
        )
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
                return list(await _step_messages(execution_steps, run_id))
            return list(await _step_messages(execution_steps, run_id))
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
            str(event_type),
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
        if (
            execution.session_id is None
            or execution.status is not ExecutionStatus.STARTED
        ):
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
            or (
                None
                if session.continuation is None
                else session.continuation.history_id
            ),
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
                    "conversation checkpoint commit unknown but cursor advanced: "
                    "execution=%s run=%s",
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
            if (
                latest.active_execution_id != execution.execution_id
                or latest.continuation != expected_cursor
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            raise
        _logger.info(
            "session conversation committed: execution=%s run=%s",
            execution.execution_id,
            source_run_id,
        )

    def _recovery_commands_for(self, execution_id: str) -> RuntimeRecoveryCommands:
        if self._tool_operations is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return RuntimeRecoveryCommands(
            self._execution.executions,
            self._execution.events,
            self._recovery.operations,
            self._tool_operations,
            execution_operations=self._execution.operations,
            background_tasks=self._execution_task_set(execution_id),
        )

    async def _commit_recovery_required(
        self,
        execution: ExecutionRecord,
        error: AIError,
        effects: tuple[ExecutionRecoveryEffect, ...],
    ) -> ExecutionRecord:
        current = await self._execution.executions.get(
            execution.execution_id,
            tenant_id=execution.tenant_id,
        )
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if current.status is ExecutionStatus.RECOVERY_REQUIRED:
            return current
        if current.status not in {
            ExecutionStatus.STARTED,
            ExecutionStatus.CANCELLING,
        }:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        details: dict[str, JsonValue] = {
            "execution_id": current.execution_id,
            "phase": "tool_effect",
        }
        requested_operation_id = error.safe_details.get("operation_id")
        selected = None
        if isinstance(requested_operation_id, str):
            selected = next(
                (
                    value
                    for value in effects
                    if value.operation_id == requested_operation_id
                ),
                None,
            )
            if selected is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        elif len(effects) == 1:
            selected = effects[0]
        if selected is not None:
            details.update(
                {
                    "operation_id": selected.operation_id,
                    "step_run_id": selected.step_run_id,
                    "tool_call_id": selected.tool_call_id,
                    "tool_name": selected.tool_name,
                    "fence": selected.fence,
                }
            )
        else:
            details["unknown_effect_count"] = len(effects)

        async with self._audit_lock(current.execution_id):
            pending = tuple(
                self._pending_audit_events.get(current.execution_id, ())
            )
            committed = await self._recovery_commands_for(
                current.execution_id
            ).commit_recovery_required(
                current,
                error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value,
                safe_error_details=details,
                audit_events=pending,
            )
            if pending:
                self._pending_audit_events.pop(current.execution_id, None)
            self._confirm_committed_events(
                current.execution_id,
                pending_count=len(pending),
                durable_sequence=committed.event_sequence,
            )
            self._live_broker.publish_event(
                current.execution_id,
                ExecutionEventType.EXECUTION_RECOVERY_REQUIRED,
                {
                    "error_code": ErrorCode.TOOL_EFFECT_UNKNOWN.value,
                    "safe_error_details": details,
                },
                durable_sequence=committed.event_sequence,
            )
            self._live_broker.complete(current.execution_id)
        _logger.error(
            "execution entered recovery-required state: execution=%s",
            current.execution_id,
        )
        return committed

    async def _recovery_failure_effects(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> tuple[ExecutionRecoveryEffect, ...]:
        if self._tool_operations is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        records = await self._tool_operations.list_by_execution(
            execution_id,
            tenant_id=tenant_id,
        )
        return tuple(
            ExecutionRecoveryEffect(
                operation_id=record.tool_operation_id,
                execution_id=record.execution_id,
                step_run_id=record.step_run_id,
                tool_call_id=record.tool_call_id,
                tool_name=record.tool_name,
                fence=record.fence,
                idempotency_key_digest=record.idempotency_key_digest,
                replay_safe=record.replay_safe,
                error_code=record.error_code,
            )
            for record in records
            if record.status is ToolOperationStatus.EFFECT_UNKNOWN
        )

    async def recovery_effects(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> tuple[ExecutionRecoveryEffect, ...]:
        """Return unresolved tool effects through the recovery coordinator."""
        return await self._recovery_coordinator.recovery_effects(
            execution_id,
            tenant_id=tenant_id,
        )

    async def _get_tool_operation(
        self,
        operation_id: str,
        *,
        tenant_id: str,
    ) -> ToolOperationRecord | None:
        if self._tool_operations is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await self._tool_operations.get_operation(
            operation_id,
            tenant_id=tenant_id,
        )

    async def _resolve_tool_effect_command(
        self,
        execution_id: str,
        ledger: OperationLedgerInput,
        *,
        expected_fence: int,
        target_status: ToolOperationStatus,
        result_payload: StoredPayload | None,
        error_code: str | None,
        error_payload: StoredPayload | None,
    ) -> ToolOperationRecord:
        return await self._recovery_commands_for(
            execution_id
        ).resolve_tool_effect(
            ledger,
            expected_fence=expected_fence,
            target_status=target_status,
            result_payload=result_payload,
            error_code=error_code,
            error_payload=error_payload,
        )

    async def resolve_tool_effect(
        self,
        execution_id: str,
        request: ResolveToolEffectRequest,
    ) -> ToolEffectResolutionResult:
        return await self._recovery_coordinator.resolve_tool_effect(
            execution_id,
            request,
        )

    async def _persist_cancel_intent(
        self,
        execution: ExecutionRecord,
        operation: OperationLedgerInput,
    ) -> OperationLedgerRecord:
        return await self._recovery_commands_for(
            execution.execution_id
        ).commit_cancel_intent(execution, operation)

    async def persist_cancel_intent(
        self,
        execution: ExecutionRecord,
        operation: OperationLedgerInput,
    ) -> OperationLedgerRecord:
        return await self._recovery_coordinator.persist_cancel_intent(
            execution,
            operation,
        )

    async def _pending_cancel_operations(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> tuple[OperationLedgerRecord, ...]:
        values = await self._execution.operations.list_pending(
            ResourceKind.EXECUTION,
            execution_id,
            tenant_id=tenant_id,
            limit=257,
        )
        cancel = tuple(
            value
            for value in values
            if value.operation_kind is OperationKind.EXECUTION_CANCEL
        )
        if len(cancel) > 256:
            raise AIError(ErrorCode.TOO_MANY_PENDING_OPERATIONS)
        return cancel

    async def _complete_recovered_cancel(
        self,
        resumed: ExecutionRecord,
        checkpoint: RecoveryCheckpoint,
        operations: tuple[OperationLedgerRecord, ...],
    ) -> ExecutionRecord:
        cancelling = await self._recovery_commands_for(
            resumed.execution_id
        ).commit_cancel_claim(resumed)
        terminal = await self._commit_terminal(
            cancelling,
            ExecutionStatus.CANCELLED,
            None,
            ErrorCode.EXECUTION_CANCELLED.value,
            StopReason.CANCELLED,
            run_id=checkpoint.step_run_id,
        )
        for candidate in operations:
            await self._settle_cancel_operation(candidate, terminal)
        return terminal

    async def _settle_cancel_operation(
        self,
        operation: OperationLedgerRecord,
        execution: ExecutionRecord,
    ) -> None:
        current = await self._execution.operations.get(
            operation.operation_id,
            tenant_id=execution.tenant_id,
        )
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if current.status in {
            OperationStatus.SUCCEEDED,
            OperationStatus.CANCELLED,
        }:
            return
        if current.status not in {
            OperationStatus.PENDING,
            OperationStatus.RUNNING,
        }:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        updated = OperationLedgerRecord(
            current.operation_id,
            current.tenant_id,
            current.resource_kind,
            current.resource_id,
            current.execution_id,
            current.operation_kind,
            OperationStatus.SUCCEEDED,
            current.request_digest,
            execution.execution_id,
            None,
            None,
            current.compactable,
            current.sequence,
            current.created_at,
            datetime.now(timezone.utc),
        )
        try:
            await self._execution.operations.compare_and_swap(
                current.operation_id,
                tenant_id=execution.tenant_id,
                expected_status=current.status,
                next_record=updated,
            )
        except AIError as error:
            if error.code not in {
                ErrorCode.STORAGE_COMMIT_UNKNOWN,
                ErrorCode.STORAGE_CONFLICT,
            }:
                raise
            latest = await self._execution.operations.get(
                current.operation_id,
                tenant_id=execution.tenant_id,
            )
            if (
                latest is not None
                and latest.status is OperationStatus.SUCCEEDED
                and latest.result_ref == execution.execution_id
                and latest.request_digest == current.request_digest
            ):
                return
            raise

    async def _tool_result_payload(
        self,
        execution: ExecutionRecord,
        operation_id: str,
        result: object,
    ) -> StoredPayload:
        encoded = encode_model_messages(
            (
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            "runtime",
                            result,
                            tool_call_id=operation_id,
                        )
                    ]
                ),
            )
        )
        return await self._recovery_payload(execution, encoded)

    async def _tool_resolution_error_payload(
        self,
        execution: ExecutionRecord,
    ) -> StoredPayload:
        encoded = json.dumps(
            {
                "kind": "error",
                "code": ErrorCode.TOOL_EXECUTION_FAILED.value,
                "safe_details": {"phase": "tool_effect_resolution"},
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return await self._recovery_payload(execution, encoded)

    async def _recovery_payload(
        self,
        execution: ExecutionRecord,
        data: bytes,
    ) -> StoredPayload:
        inline = StoredPayload.inline_bytes(data)
        if payload_fits_inline(inline, self._payload_policy):
            return inline
        return StoredPayload.object(
            await put_runtime_object(
                self._recovery_objects,
                RuntimeObjectKeyFactory(self._namespace),
                RuntimeDomain.RECOVERY,
                execution.tenant_id,
                data,
            )
        )

    async def _store_repository_instructions(
        self,
        execution: ExecutionRecord,
        instructions: RepositoryInstructions,
    ) -> RuntimePayloadRef:
        payload = instructions.to_payload()
        if canonical_sha256(payload) != instructions.digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        inline = StoredPayload.inline_json(payload)
        if payload_fits_inline(inline, self._payload_policy):
            stored = inline
        else:
            reference = await put_runtime_object(
                self._recovery_objects,
                RuntimeObjectKeyFactory(self._namespace),
                RuntimeDomain.RECOVERY,
                execution.tenant_id,
                canonical_json_bytes(payload),
            )
            stored = StoredPayload.object(reference)
        if stored.digest != instructions.digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return RuntimePayloadRef(stored, RuntimeDomain.RECOVERY)

    async def commit_repository_instruction_barrier(
        self,
        execution: ExecutionRecord,
        checkpoint: RecoveryCheckpoint,
        overlay: RepositoryInstructions,
        barrier: RepositoryInstructionBarrier,
    ) -> RecoveryCheckpoint:
        existing = tuple(
            item
            for item in checkpoint.repository_instruction_barriers
            if item.step_run_id == barrier.step_run_id
            and item.tool_call_id == barrier.tool_call_id
        )
        if existing:
            if existing[0] != barrier:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            current = await self._recovery.checkpoints.get(
                execution.execution_id,
                tenant_id=execution.tenant_id,
            )
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return current
        overlay_reference = await self._store_repository_instructions(
            execution,
            overlay,
        )
        next_checkpoint = replace(
            checkpoint,
            repository_instruction_overlay=overlay_reference,
            repository_instruction_barriers=(
                *checkpoint.repository_instruction_barriers,
                barrier,
            ),
            revision=checkpoint.revision + 1,
            updated_at=datetime.now(timezone.utc),
        )
        try:
            return await self._recovery.checkpoints.compare_and_swap(
                execution.execution_id,
                tenant_id=execution.tenant_id,
                expected_revision=checkpoint.revision,
                next_record=next_checkpoint,
            )
        except AIError as error:
            current = await self._recovery.checkpoints.get(
                execution.execution_id,
                tenant_id=execution.tenant_id,
            )
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            matching = tuple(
                item
                for item in current.repository_instruction_barriers
                if item.step_run_id == barrier.step_run_id
                and item.tool_call_id == barrier.tool_call_id
            )
            if matching:
                if matching[0] != barrier:
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT) from error
                if (
                    current.repository_instruction_overlay is None
                    or current.repository_instruction_overlay.payload.digest
                    != barrier.resulting_overlay_digest
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                return current
            raise

    async def _commit_failure(
        self,
        execution: ExecutionRecord,
        error: Exception,
        *,
        run_id: str | None = None,
    ) -> ExecutionRecord:
        unknown = _tool_effect_unknown_cause(error)
        if unknown is not None:
            effects = await self._recovery_failure_effects(
                execution.execution_id,
                tenant_id=execution.tenant_id,
            )
            if effects:
                return await self._commit_recovery_required(
                    execution,
                    unknown,
                    effects,
                )
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
            error_diagnostics=(
                None if cancelled else _execution_error_diagnostics(error)
            ),
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
        error_diagnostics: ErrorDiagnostics | None = None,
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
        if status is not ExecutionStatus.FAILED and error_diagnostics is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        now = datetime.now(timezone.utc)
        captured_usage = usage or self._captured_usage.get(
            execution.execution_id, UsageMetrics()
        )
        if status is ExecutionStatus.SUCCEEDED:
            if binding is None or output is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        elif output is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if recovery_checkpoint is None:
            recovery_checkpoint = await self._recovery.checkpoints.get(
                current.execution_id,
                tenant_id=current.tenant_id,
            )
            if recovery_checkpoint is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._validate_recovery_identity(current)
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
                    error_diagnostics=error_diagnostics,
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
                        content = await read_runtime_object(
                            self._execution_objects, reference
                        )
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
                if (
                    status is ExecutionStatus.SUCCEEDED
                    and current.session_id is not None
                    and run_id is not None
                ):
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
                            if status is ExecutionStatus.SUCCEEDED
                            and run_id is not None
                            else _terminal_error_payload(
                                error_code,
                                safe_error_details or {},
                                error_diagnostics,
                            )
                        ),
                        result_created_at=now,
                        error_diagnostics=error_diagnostics,
                    ),
                    run_id or recovery_checkpoint.step_run_id,
                    conversation,
                )
                prepared = replace(
                    recovery_checkpoint,
                    state=RecoveryCheckpointState.HANDOFF,
                    handoff_phase=RecoveryHandoffPhase.PREPARED,
                    terminal_handoff=handoff,
                    pending_tools=None,
                    revision=recovery_checkpoint.revision + 1,
                    updated_at=now,
                )
                try:
                    recovery_checkpoint = (
                        await self._recovery.checkpoints.compare_and_swap(
                            current.execution_id,
                            tenant_id=current.tenant_id,
                            expected_revision=recovery_checkpoint.revision,
                            next_record=prepared,
                        )
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
            error_diagnostics=error_diagnostics,
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
                else _terminal_error_payload(
                    error_code,
                    safe_error_details or {},
                    error_diagnostics,
                )
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
        return all(
            store.storage_group is stores[0].storage_group for store in stores[1:]
        )

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
        error_diagnostics: ErrorDiagnostics | None,
    ) -> ExecutionRecord:
        if current.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            return current
        self._validate_recovery_identity(current)
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
            error_diagnostics=error_diagnostics,
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
            else _terminal_error_payload(
                error_code,
                safe_error_details or {},
                error_diagnostics,
            ),
        )
        target = replace(
            checkpoint,
            state=RecoveryCheckpointState.COMPLETED,
            handoff_phase=RecoveryHandoffPhase.COMPLETED,
            terminal_handoff=None,
            pending_operation_id=None,
            pending_tools=None,
            revision=checkpoint.revision + 1,
            updated_at=now,
        )
        conversation_run = None
        conversation_snapshot = None
        expected_cursor = None
        if current.session_id is not None and status is ExecutionStatus.SUCCEEDED:
            conversation_run = await self._steps.get_run(run_id=run_id or "")
            conversation_snapshot = await self._steps.latest_snapshot(
                run_id=run_id or ""
            )
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
                        plan = (
                            await self._step_lifecycle.prepare_execution_terminal_seal(
                                execution_id=current.execution_id,
                                run_ids=candidate_run_ids,
                                binding_digest=current.binding_digest,
                            )
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
                    committed = (
                        await self._commit_execution_terminal_checkpoint_locked_body(
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
                    )
                    durable_commit = True
                    self._confirm_committed_events(
                        current.execution_id,
                        pending_count=pending_count,
                        durable_sequence=committed.execution.event_sequence,
                    )
                if plan is not None:
                    try:
                        await self._step_lifecycle.finalize_execution_terminal_seal(
                            plan
                        )
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
        self._record_committed_terminal(committed, session_id=current.session_id)
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

    async def verify_terminal_projection(
        self, execution: ExecutionRecord, status: ExecutionStatus, run_id: "str | None"
    ) -> None:
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
            required_step_run_id=run_id
            if status is ExecutionStatus.SUCCEEDED
            else None,
        )


def _terminal_record(
    record: ExecutionRecord,
    status: ExecutionStatus,
    now: datetime,
    *,
    error_code: str | None,
    safe_error_details: Mapping[str, JsonValue] | None = None,
    error_diagnostics: ErrorDiagnostics | None = None,
) -> ExecutionRecord:
    if status is not ExecutionStatus.FAILED and error_diagnostics is not None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return replace(
        record,
        status=status,
        revision=record.revision + 1,
        event_sequence=record.event_sequence + 1,
        error_code=error_code,
        safe_error_details={} if safe_error_details is None else safe_error_details,
        error_diagnostics=error_diagnostics,
        updated_at=now,
    )


def _terminal_error_payload(
    error_code: str | None,
    safe_error_details: Mapping[str, JsonValue],
    error_diagnostics: ErrorDiagnostics | None,
) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "error_code": error_code,
        "safe_error_details": dict(safe_error_details),
    }
    if error_diagnostics is not None:
        payload["error_diagnostics"] = {
            "exception_type": error_diagnostics.exception_type,
            "exception_message": error_diagnostics.exception_message,
            "cause_digest": error_diagnostics.cause_digest,
        }
    return payload


def _admission_matches(
    existing: RecoveryCheckpoint, candidate: RecoveryCheckpoint
) -> bool:
    return (
        existing.execution_id == candidate.execution_id
        and existing.tenant_id == candidate.tenant_id
        and existing.step_run_id is None
        and existing.agent_run_sequence == candidate.agent_run_sequence
        and existing.state is RecoveryCheckpointState.ADMITTED
        and existing.handoff_phase is RecoveryHandoffPhase.NONE
        and existing.terminal_handoff is None
        and existing.pending_operation_id is None
    )


def _deferred_id(
    contract: str,
    tenant_id: str,
    execution_id: str,
    source_step_run_id: str,
    tool_call_id: str,
) -> str:
    return canonical_sha256(
        {
            "contract": contract,
            "tenant_id": tenant_id,
            "execution_id": execution_id,
            "source_step_run_id": source_step_run_id,
            "tool_call_id": tool_call_id,
        }
    )


def _execution_error_code(error: Exception) -> ErrorCode:
    if isinstance(error, ValidationError):
        return ErrorCode.OUTPUT_VALIDATION_FAILED
    if isinstance(error, AIError):
        return error.code
    return ErrorCode.INTERNAL_ERROR


def _tool_effect_unknown_cause(error: BaseException) -> AIError | None:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, AIError) and current.code is ErrorCode.TOOL_EFFECT_UNKNOWN:
            return current
        current = current.__cause__ or current.__context__
    return None


def _execution_error_details(error: Exception) -> dict[str, JsonValue]:
    return dict(error.safe_details) if isinstance(error, AIError) else {}


def _execution_error_diagnostics(error: Exception) -> ErrorDiagnostics:
    if isinstance(error, AIError) and error.diagnostics is not None:
        return error.diagnostics
    return ErrorDiagnostics.from_exception(error)


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
            diagnostics=error.diagnostics,
        )
    return AIError(
        ErrorCode.INTERNAL_ERROR,
        safe_details={
            "phase": "execution_terminal_commit",
            **primary_details,
        },
        diagnostics=ErrorDiagnostics.from_exception(error),
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
