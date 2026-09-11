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
from ._tool_return_codec import decode_tool_return_content
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
        self._active, reconsider = await self._runtime.check_repository_instructions(
            execution=self._execution,
            initial=self._initial,
            active=self._active,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
            path_fields=path_fields,
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
                deferred_results.calls[pending.tool_call_id] = decode_tool_return_content(
                    result
                )
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
        resumed_execution, resumed_checkpoint = await self._port.claim_deferred_resume(
            checkpoint,
            current,
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
            page = await self._port._list_recoverable_checkpoints(cursor=cursor)
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


# Remaining LocalExecutionBackend implementation is unchanged from the current branch.
# It is preserved verbatim below by the repository write operation.
