#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local ExecutionBackend backed by AgentExecutor and durable persistence."""

import asyncio

from ._handoff import ExecutionHandoffScope, execution_handoff_scope
import json
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from time import monotonic_ns
from typing import Protocol, TypeVar, cast

from linktools.core import environ
from ..steps import NativeStepSource
from pydantic import ValidationError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.tools import (
    DeferredToolRequests,
)

from ..agent import AgentBinding, AgentBindingContract, AgentCatalog, SubagentRef
from ..capability import AgentContext, SubagentDelegate
from ..spec import CapabilityRef, RepositoryInstructionResolver, RepositoryInstructions
from ..workspace import (
    Workspace,
)
from ..inputs import ExecutionContextArchive
from ._feature_policy import _runtime_feature_policy
from ._capabilities import runtime_feature_leaves
from ..capability import TraceCapability
from ._agent_executor import (
    AgentExecutor,
    _AgentRunScope,
    DurableBoundary,
    LiveDelta,
)
from ..capability import MemoryCapability, tool_class_from_metadata
from ..inputs import (
    CanonicalUserInput,
    ExecutionInputMaterializer,
    stored_input_attachment_views,
    stored_user_input_view,
)
from ._plan import RuntimePlanStore
from ._execution_payload import _materialize_execution_json, _materialize_execution_payloads
from ._execution_buffer import _ExecutionPayloadBuffer, _ExecutionPayloadCapture
from ._pending_turn import _PendingExecutionTurn, _PendingExecutionTurns
from ..core import (
    CommitPolicy,
    ExecutionEventType,
    ExecutionMode,
    ExecutionLineageKind,
    ExecutionStatus,
    IdempotencyStatus,
    JsonValue,
    OperationKind,
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationStatus,
    Page,
    Principal,
    PromptLimits,
    ResourceKind,
    SessionStatus,
    StopReason,
    ToolOperationStatus,
    UsageMetrics,
    canonical_json_bytes,
    canonical_sha256,
    normalize_json_value,
    agent_conversation_id as make_agent_conversation_id,
    agent_run_id as make_agent_run_id,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode, ErrorDiagnostics
from ..observe import MetricRecorder
from ..storage import (
    ObjectRef,
    ObjectStore,
    PayloadPolicy,
    StoredPayload,
    TurnCoordination,
    TurnScope,
    payload_fits_inline,
)
from ..execution.events import ExecutionDelta, LiveExecutionEventBroker
from ._execution import CancelEffectOutcome, ExecutionStartIdentity, _TerminalMetricDelivery
from ..observe.service import (
    capture_execution_terminal,
    record_metric_observation,
    record_storage_operation,
    release_metric_execution_context,
)
from ..model.messages import encode_model_messages
from ..execution import ExecutionInputContext
from ._memory import MemoryStore, RuntimeMemoryStore
from ..steps import NativeLifecycleEvent
from ..persistence.objects import ObjectKeyFactory, put_runtime_object, read_runtime_object
from ._tool import (
    RuntimeToolOperationBridge,
    _ToolOperationRuntimeRepository,
)
from ..execution import ExecutionRecoveryEffect, ResolveToolEffectRequest, ToolEffectResolutionResult
from ._budget import RunBudgetContext
from ..execution import ExecutionRequest
from ._conversation_service import ConversationService
from ..persistence import PersistenceDomain
from ..persistence.commands import StateCommands
from ..persistence.repositories import ExecutionRepositoryImpl
from ..persistence.repositories import SessionRepositoryImpl
from ..persistence.objects import runtime_object_source_reads
from ..persistence.contracts import (
    ApprovalRecord,
    ExecutionStartReceipt,
    ExecutionTerminalPublication,
    AgentAttemptClaim,
    ConversationRepositories,
    ExecutionEventAppend,
    ExecutionRepository,
    ExecutionRepositories,
    ExternalCallRecord,
    LoadedModelContext,
    LoadedContextMessage,
    PendingDeferredCall,
    PendingToolContinuation,
    ExecutionRecoveryState,
    RecoveryHandoffPhase,
    CheckpointRepositories,
    ExecutionRecoveryDetail,
    RecoveryConversationIntent,
    RecoveryTerminalHandoff,
    RecoveryTerminalOutcome,
    RepositoryInstructionBarrier,
    SessionRepository,
    ToolOperationRecord,
)
from ..session import ConversationCursor, SessionRecord, SessionTurnRef
from ..execution import (
    ExecutionCancelRequestCommit,
    ExecutionRecord,
    ExecutionHandle,
    ExecutionStartClaim,
    ExecutionStartReservation,
    ExecutionStartReservationResult,
    ExecutionTerminalCommit,
    ExecutionTerminalCommitResult,
    IdempotencyRecord,
    RuntimePayloadRef,
    ResultRecord,
)
from ..persistence.runs import AgentRunCheckpoint, AgentRunRecord, StepEvent, AgentRunStore
from ..persistence.commands import RuntimeRecoveryCommands
from ..persistence.repositories import operation_identity_matches
from ..persistence.runs import ExecutionTerminalSealPlan, RuntimeAgentRunStore, StateStepArchive
from ..persistence.state import Observed, SqlStateTransaction, StateGroupTransaction
from ..persistence import SqlStepStore, StepWriteReceipt
from ..storage import SqlStorageContext


class _SubagentDispatcher(Protocol):
    @property
    def pending_background_tasks(self) -> tuple[asyncio.Task[object], ...]: ...

    @property
    def background_failure(self) -> "AIError | None": ...

    def delegate_for(
        self,
        *,
        parent_execution_id: str,
        parent_producer_generation: int,
        root_execution_id: str,
        memory_scope: "str | None",
        principal: Principal,
        refs: "tuple[SubagentRef, ...]",
        binding: AgentBindingContract,
        mode: ExecutionMode,
        require_frozen_bindings: bool = False,
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
















from ._agent_run import (
    _AgentRunLifecycle,
    _WorkerFailure,
)
from ._recovery_coordinator import _DeferredResume, _RecoveryCoordinator
from ._repository_instructions import (
    _RepositoryInstructionBoundary,
    _repository_instructions_contain,
    _validate_repository_instruction_frontier,
)

_logger = environ.get_logger("ai.runtime.local")

_CheckpointT = TypeVar("_CheckpointT")


























@dataclass(frozen=True, slots=True)
class _InitialAttemptCapture:
    execution: ExecutionRecord
    session: SessionRecord | None = None


@dataclass(slots=True)
class _OwnedExecutionTurn:
    reservation: ExecutionStartReservation
    coordination: TurnCoordination | None
    principal: Principal
    scope: TurnScope | None = None
    admitting: bool = True
    producer_generation: int | None = None
    attempt: _InitialAttemptCapture | None = None
    boundary: asyncio.Lock = field(default_factory=asyncio.Lock)
    observed: Observed[ExecutionRecord] | None = None
    session_turn: SessionTurnRef | None = None
    tool_admission: Callable[..., Awaitable[object]] | None = None
    tool_terminal: Callable[..., Awaitable[object]] | None = None
    tool_unknown: Callable[..., Awaitable[object]] | None = None
    stopping: bool = False
    cancel_requested: bool = False
    lifetime: asyncio.Task[object] | None = None
    closed: asyncio.Event = field(default_factory=asyncio.Event)
    children: dict[str, "_OwnedExecutionTurn"] = field(default_factory=dict)
    delegated_children: set[str] = field(default_factory=set)
    admissions: set[asyncio.Future[None]] = field(default_factory=set)
    admission: asyncio.Future[ExecutionHandle] | None = None


_TurnActionT = TypeVar("_TurnActionT")


class LocalExecutionBackend:
    """Resolve compiled Agent semantics and persist one execution lifecycle."""

    def __init__(
        self,
        conversation: ConversationRepositories,
        execution_state: ExecutionRepositories,
        checkpoint: CheckpointRepositories,
        execution_objects: ObjectStore,
        checkpoint_objects: ObjectStore,
        namespace: str,
        run_store: AgentRunStore,
        executor: AgentExecutor,
        catalog: AgentCatalog,
        *,
        archive_objects: ObjectStore | None = None,
        archive_object_reader: Callable[[ObjectRef], Awaitable[bytes]] | None = None,
        restore_binding: "Callable[[AgentBindingContract], Awaitable[AgentBinding]] | None" = None,
        native_step_source: Callable[[AgentBinding], Awaitable[NativeStepSource | None]] | None = None,
        workspace: "Workspace | None",
        limits: PromptLimits,
        execution_cwd: "str | None",
        instruction_resolver: RepositoryInstructionResolver | None = None,
        app: object,
        tenant_id: str,
        run_stores: Mapping[PersistenceDomain, AgentRunStore],
        agent_run_lifecycle: _AgentRunLifecycle,
        memory_store_factory: "Callable[[str, str, str], MemoryStore] | None" = None,
        conversation_durable: bool = False,
        input_materializer: ExecutionInputMaterializer | None = None,
        subagent_dispatcher: "_SubagentDispatcher | None" = None,
        live_broker: "LiveExecutionEventBroker | None" = None,
        payload_policy: "PayloadPolicy | None" = None,
        execution_objects_durable: bool = True,
        tool_operations: "_ToolOperationRuntimeRepository | None" = None,
        metric_recorder: "MetricRecorder | None" = None,
        worker_completed: "Callable[[str], Awaitable[None]] | None" = None,
        terminal_publication: Callable[[ExecutionRecord | ExecutionTerminalPublication], None] | None = None,
        retained_publication: Callable[..., ExecutionTerminalPublication | None] | None = None,
        exclusive_execution_writer: bool = False,
        commit: CommitPolicy = CommitPolicy.sync(),
        trace_commit: CommitPolicy | None = None,
        pending_turns: _PendingExecutionTurns | None = None,
        pending_cancel: Callable[[str, Principal], Awaitable[object]] | None = None,
    ) -> None:
        if not isinstance(commit, CommitPolicy) or commit.mode not in {"sync", "terminal"}:
            raise ValueError("Execution requires sync or terminal commit")
        self._commit = commit
        self._trace_commit = trace_commit
        self._execution_buffer = None if commit.buffer is None else _ExecutionPayloadBuffer(commit.buffer)
        self._pending_turns = pending_turns or _PendingExecutionTurns()
        self._pending_cancel = pending_cancel
        self._turn_monitors: dict[str, asyncio.Task[None]] = {}
        self._turn_owners: dict[str, _OwnedExecutionTurn] = {}
        self._conversation = conversation
        self._execution = execution_state
        self._checkpoint = checkpoint
        self._execution_objects = execution_objects
        self._context_archive = ExecutionContextArchive(execution_state.executions.state_store, archive_objects or execution_objects,
            namespace=namespace, tenant_id=tenant_id, object_reader=archive_object_reader)
        self._checkpoint_objects = checkpoint_objects
        self._metric_recorder = metric_recorder
        self._worker_completed = worker_completed
        self._terminal_publication = terminal_publication
        self._retained_publication = retained_publication
        self._namespace = namespace
        self._run_store = run_store
        self._executor = executor
        self._catalog = catalog
        self._restore_binding = restore_binding
        self._native_step_source = native_step_source
        self._workspace = workspace
        self._limits = limits
        self._execution_cwd = execution_cwd
        self._app = app
        self._tenant_id = validate_tenant_id(tenant_id)
        self._memory_store_factory = memory_store_factory
        self._conversation_durable = conversation_durable
        self._input_materializer = input_materializer
        self._subagent_dispatcher = subagent_dispatcher
        self._live_broker = live_broker or LiveExecutionEventBroker()
        self._payload_policy = payload_policy or PayloadPolicy()
        self._tool_operations = tool_operations
        self._execution_objects_durable = execution_objects_durable
        self._run_stores = dict(run_stores)
        if frozenset(self._run_stores) != frozenset(
            {
                PersistenceDomain.CONVERSATION,
                PersistenceDomain.TRACE,
                PersistenceDomain.CHECKPOINT,
            }
        ):
            raise ValueError(
                "run_stores must contain exactly the three AgentRun owner domains"
            )
        self._agent_run_lifecycle = agent_run_lifecycle
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._worker_failures: dict[str, _WorkerFailure] = {}
        self._captured_usage: dict[str, UsageMetrics] = {}
        self._terminal_events: dict[str, asyncio.Event] = {}
        self._pending_audit_events: dict[str, list[ExecutionEventAppend]] = {}
        self._pending_audit_locks: dict[str, asyncio.Lock] = {}
        self._agent_run_only_worker_exits: set[str] = set()
        self._checkpoint_tasks: set[asyncio.Task[object]] = set()
        self._execution_durable_tasks: dict[
            str,
            set[asyncio.Task[object]],
        ] = {}
        self._worker_cancel_requests: set[str] = set()
        self._worker_shutdown_requests: set[str] = set()
        self._recovery_reconcile_tasks: dict[str, asyncio.Task[None]] = {}
        self._accepting = True
        self._handoff_requested = False
        self._handoff_scopes: dict[str, ExecutionHandoffScope] = {}
        execution_run_store = self._run_stores[PersistenceDomain.TRACE]
        conversation_run_store = self._run_stores[PersistenceDomain.CONVERSATION]
        execution_repository: ExecutionRepository = self._execution.executions
        session_repository: SessionRepository = self._conversation.sessions
        self._session_state_store = session_repository.state_store
        self._execution_state_store = execution_repository.state_store
        self._runtime_commands = StateCommands(
            execution_repository,
            namespace=self._namespace,
            events=self._execution.events,
            operations=self._execution.operations,
            conversation=session_repository,
            checkpoint=self._checkpoint,
            conversation_history=self._conversation.histories,
            tools=self._tool_operations,
            budgets=self._execution.budgets,
            conversation_run_store=(
                conversation_run_store
                if isinstance(conversation_run_store, StateStepArchive)
                else None
            ),
            execution_run_store=execution_run_store
            if isinstance(execution_run_store, StateStepArchive)
            else None,
            background_tasks=self._checkpoint_tasks,
            observe_cancellation=not exclusive_execution_writer,
            capability_writes=(run_store.capability_writes if isinstance(run_store, RuntimeAgentRunStore) else None),
        )
        self._recovery_coordinator = _RecoveryCoordinator(
            self,
            instruction_resolver,
        )

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    async def _execution_binding(self, execution: ExecutionRecord, binding: AgentBinding | None = None) -> AgentBinding:
        if binding is None:
            binding = (
                self._catalog.binding(execution.binding_digest)
                if self._restore_binding is None
                else await self._restore_binding(execution.binding)
            )
        if binding.binding_digest != execution.binding_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution.native_step_source is not None:
            source = (None if self._native_step_source is None else await self._native_step_source(binding))
            if source != execution.native_step_source:
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        return binding

    async def validate_binding(self, execution: ExecutionRecord, *, binding: AgentBinding | None = None) -> AgentBinding:
        return await self._execution_binding(execution, binding)

    async def validate_recovery_inputs(self, execution: ExecutionRecord, *, binding: AgentBinding | None = None) -> AgentBinding:
        binding = await self._execution_binding(execution, binding)
        await self._executor.validate_recovery_inputs(binding, execution.tool_response_ref)
        return binding

    async def load_execution(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ExecutionRecord | None:
        pending = await self._pending_turns.read_execution(execution_id, tenant_id=tenant_id)
        if pending is not None:
            return pending
        repository = self._execution.executions
        return await repository.state_store.read(
            lambda _transaction: repository.get(execution_id, tenant_id=tenant_id)
        )

    async def _load_session(self, session_id: str, *, tenant_id: str) -> SessionRecord | None:
        repository = self._conversation.sessions
        return await repository.state_store.read(
            lambda _transaction: repository.get(session_id, tenant_id=tenant_id)
        )

    async def _list_recoverable_executions(
        self,
        *,
        cursor: str | None,
    ) -> Page[ExecutionRecord]:
        return await self._execution.executions.list_recoverable_page(
            tenant_id=self._tenant_id,
            cursor=cursor,
            limit=128,
        )

    async def load_recovery_execution(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ExecutionRecord | None:
        pending = self._pending_turns.get(execution_id, tenant_id=tenant_id)
        if pending is not None:
            return pending.execution
        return await self._execution.executions.get_recovery(
            execution_id,
            tenant_id=tenant_id,
        )

    async def _release_session_execution(
        self,
        execution: ExecutionRecord,
    ) -> None:
        if execution.session_id is None:
            return
        if execution.conversation and execution.status in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            result = await self._execution.executions.get_result(execution.execution_id, tenant_id=self._tenant_id)
            await self._runtime_commands.finish_conversation_turn(execution, result)
            return
        await self._conversation.sessions.release_execution(
            execution.session_id,
            tenant_id=self._tenant_id,
            execution_id=execution.execution_id,
            history_unavailable=execution.invalidates_session_history,
            require_turn_commit=execution.conversation,
        )

    async def _commit_start_recovery_checkpoint(
        self,
        execution: ExecutionRecord,
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
                execution.revision,
                execution.event_seq,
                identity.scope,
                identity.idempotency_key_digest,
                identity.request_digest,
                datetime.now(timezone.utc),
                identity.request_identity_version,
            ),
            session_id=execution.session_id,
            expected_cursor=expected,
        )

    async def _commit_recovery_resume(
        self,
        execution: ExecutionRecord,
        *,
        recovery_operation: OperationLedgerInput | None = None,
    ) -> ExecutionRecord:
        if execution.recovery is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        source_agent_run_id = execution.recovery.source_agent_run_id
        completed_tool_call_ids = frozenset()
        if execution.recovery.agent_run_id is not None:
            archive = self._get_run_store(PersistenceDomain.CHECKPOINT)
            snapshot = await archive.latest_checkpoint(
                agent_run_id=execution.recovery.agent_run_id, include_interrupted=True, include_transcript=False,
            )
            if snapshot is not None:
                source_agent_run_id = execution.recovery.agent_run_id
                if execution.recovery.pending_tools is not None:
                    unresolved: set[str] = set()
                    context = snapshot.context_messages
                    if context is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    for message in context:
                        for part in message.parts:
                            if isinstance(part, ToolCallPart):
                                unresolved.add(part.tool_call_id)
                            elif isinstance(part, ToolReturnPart):
                                unresolved.discard(part.tool_call_id)
                    completed_tool_call_ids = frozenset(
                        call.tool_call_id for call in (
                            *execution.recovery.pending_tools.approvals, *execution.recovery.pending_tools.calls,
                        ) if call.tool_call_id not in unresolved
                    )
        result = await self._runtime_commands.commit_agent_attempt_checkpoint(
            AgentAttemptClaim(
                execution_id=execution.execution_id,
                expected_execution_revision=execution.revision,
                expected_agent_run_seq=execution.agent_run_seq,
                expected_recovery_state=execution.recovery_state,
            ), recovery_operation=recovery_operation,
            source_agent_run_id=source_agent_run_id,
            completed_tool_call_ids=completed_tool_call_ids,
        )
        if recovery_operation is not None:
            receipt = await self._execution.operations.get(
                recovery_operation.operation_id, tenant_id=self._tenant_id,
            )
            if receipt is None or not operation_identity_matches(receipt, recovery_operation):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if receipt.result_ref != recovery_operation.result_ref:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        owner = self._turn_owners.get(execution.execution_id)
        if owner is not None:
            owner.observed = result
        return result.value

    async def _load_admitted_recovery_execution(
        self,
        admitted: ExecutionRecord,
        *,
        producer_generation: int,
        recovery_operation: OperationLedgerInput | None,
    ) -> ExecutionRecord:
        repository = self._execution.executions
        current = await repository.get(
            admitted.execution_id, tenant_id=self._tenant_id,
        )
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            current.status is not ExecutionStatus.STARTED
            or current.producer_generation != producer_generation
            or (recovery_operation is not None and current.producer_claim_id != recovery_operation.result_ref)
            or current.revision < admitted.revision
            or replace(
                current, revision=admitted.revision, updated_at=admitted.updated_at,
                dependency_hold_ids=admitted.dependency_hold_ids,
            ) != admitted
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT, safe_details={
                "phase": "recovery_admission_changed",
                "expected_revision": admitted.revision,
                "actual_revision": current.revision,
                "expected_status": admitted.status.value,
                "actual_status": current.status.value,
            })
        return current

    async def _admit_recovery_without_producer(
        self, execution: ExecutionRecord, operation: OperationLedgerInput,
    ) -> None:
        receipt = await self._recovery_commands_for(execution.execution_id).admit_recovery_without_producer(
            execution, replace(operation, result_ref=None),
        )
        if receipt.result_ref is not None and receipt.result_ref != operation.result_ref:
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    def _reset_local_producer(self, execution_id: str) -> None:
        self._live_broker.reset_completed_local_producer(execution_id)

    def _publish_recovery_resumed(
        self,
        execution_id: str,
        event_seq: int,
    ) -> None:
        self._live_broker.publish_event(
            execution_id,
            ExecutionEventType.EXECUTION_RESUMED,
            {},
            durable_seq=event_seq,
        )

    def _prepare_recovery_relaunch(self, execution_id: str) -> bool:
        if self.worker_installed(execution_id):
            return False
        self._worker_failures.pop(execution_id, None)
        return True

    async def _validate_start(
        self, request: ExecutionRequest, execution: ExecutionRecord
    ) -> None:
        if not self._accepting:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if request.principal.tenant_id != self._tenant_id:
            _logger.warning(
                "local execution tenant rejected: expected=%s request=%s execution=%s",
                self._tenant_id,
                request.principal.tenant_id,
                self._tenant_id,
            )
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._execution_binding(execution, request.resolved_binding)
        if (
            request.mode != execution.mode
            or request.planning is not execution.planning
            or request.thinking != execution.thinking
            or request.tool_response_ref != execution.tool_response_ref
            or request.tool_effects != execution.tool_effects
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

    async def _record_committed_terminal(
        self,
        committed: ExecutionTerminalCommitResult,
        *,
        session_id: str | None,
        capture_metric: Callable[[_TerminalMetricDelivery], None] | None = None,
    ) -> None:
        if self._terminal_publication is not None:
            self._terminal_publication(committed if isinstance(committed, ExecutionTerminalPublication) else committed.execution)
        recorder = self._metric_recorder
        if recorder is None:
            return
        observation = capture_execution_terminal(
            recorder,
            source_namespace=self._namespace,
            tenant_id=self._tenant_id,
            result=committed,
            session_id=session_id,
        )
        release_metric_execution_context(
            recorder,
            committed.execution.execution_id,
        )

        async def deliver() -> None:
            await record_metric_observation(recorder, observation)

        if capture_metric is not None:
            capture_metric(deliver)
        else:
            await deliver()

    async def _commit_terminal_checkpoint_owned(
        self,
        commit: ExecutionTerminalCommit,
        *,
        session_id: str | None,
        expected_execution: ExecutionRecord | None = None,
    ) -> ExecutionTerminalCommitResult:
        execution_id = commit.execution.execution_id
        async with self._audit_lock(execution_id):
            pending = await self._source_audit_prefix(execution_id)
            committed = await self._runtime_commands.commit_terminal_checkpoint(
                commit,
                session_id=session_id,
                expected_execution=expected_execution,
                audit_events=pending,
                background_tasks=self._execution_task_set(execution_id),
            )
            if (committed.execution.recovery is not None
                    and committed.execution.recovery.handoff_phase is RecoveryHandoffPhase.PREPARED):
                completed = await self._complete_handoff(committed.execution)
                if completed != committed.execution:
                    committed = ExecutionTerminalCommitResult(completed, committed.result)
            self._confirm_source_audits(execution_id, len(pending))
            if self._execution_buffer is not None:
                # This necessary control terminal abandons any candidate payload;
                # only its selected audit prefix was published by this command.
                self._execution_buffer.discard(execution_id)
            self._confirm_committed_events(
                execution_id,
                pending_count=len(pending),
                durable_seq=committed.execution.event_seq,
            )
        self._publish_terminal_event(
            execution_id,
            event_type=commit.terminal_event_type,
            payload=dict(commit.terminal_event_payload),
            durable_seq=committed.execution.event_seq,
        )
        self._live_broker.complete(execution_id)
        return committed

    async def commit_terminal_checkpoint(
        self,
        commit: ExecutionTerminalCommit,
        *,
        session_id: str | None,
        expected_execution: ExecutionRecord | None = None,
        capture_metric: Callable[[_TerminalMetricDelivery], None] | None = None,
    ) -> ExecutionTerminalCommitResult:
        task = asyncio.create_task(
            self._commit_terminal_checkpoint_owned(
                commit,
                session_id=session_id,
                expected_execution=expected_execution,
            ),
            name=f"local-terminal-checkpoint-{commit.execution.execution_id}",
        )
        committed, cancellation = await self._await_checkpoint_task(
            task,
            label="terminal",
            execution_id=commit.execution.execution_id,
        )
        await self._record_committed_terminal(
            committed, session_id=session_id, capture_metric=capture_metric,
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
            current = await self.load_execution(
                commit.execution_id,
                tenant_id=self._tenant_id,
            )
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            pending = await self._source_audit_prefix(commit.execution_id)
            # Deferred and recovery boundaries settle earlier local audit tails
            # before admitting another producer. Retained runs may span attempts.
            producer_generation = self._agent_run_lifecycle.latest_local_producer_generation(
                execution_id=current.execution_id,
            ) if current.agent_run_seq > 0 else None
            if producer_generation is None:
                if pending:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                self._live_broker.handoff(current.execution_id)
            else:
                control = await self._execution.executions.get(
                    current.execution_id, tenant_id=self._tenant_id,
                )
                if control is None or control.producer_generation < producer_generation:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if control.producer_generation > producer_generation:
                    pending = ()
                    producer_generation = None
                    self._pending_audit_events.pop(current.execution_id, None)
                    self._live_broker.handoff(current.execution_id)
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
                checkpoint = await self._execution.executions.get_recovery(
                    commit.execution_id,
                    tenant_id=self._tenant_id,
                )
                if (
                    checkpoint is None
                    or checkpoint.recovery_state is not ExecutionRecoveryState.WAITING
                    or checkpoint.recovery.pending_tools is None
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if (
                    current.revision != commit.expected_revision
                    or current.event_seq != commit.expected_event_seq
                ):
                    effective_commit = replace(
                        commit,
                        expected_revision=current.revision,
                        expected_event_seq=current.event_seq,
                    )
                committed = await self._runtime_commands.commit_deferred_cancel_checkpoint(
                    effective_commit,

                    expected_pending_tools=checkpoint.recovery.pending_tools,
                    background_tasks=self._execution_task_set(commit.execution_id),
                )
            elif (
                current.status is expected_status
                and current.revision == commit.expected_revision
                and current.event_seq == commit.expected_event_seq
                and current.status is not ExecutionStatus.WAITING_DEFERRED
            ):
                committed = await self._runtime_commands.commit_cancel_checkpoint(
                    effective_commit,
                    expected_status=effective_status,
                    audit_events=pending,
                    producer_generation=producer_generation,
                    background_tasks=self._execution_task_set(commit.execution_id),
                )
            else:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._confirm_source_audits(commit.execution_id, len(pending))
            cancel_sequence = (
                effective_commit.expected_event_seq + len(pending) + 1
            )
            self._confirm_committed_events(
                commit.execution_id,
                pending_count=len(pending),
                durable_seq=cancel_sequence,
            )
            self._live_broker.publish_event(
                commit.execution_id,
                event_type=ExecutionEventType.CANCEL_REQUESTED,
                payload={"operation_id": commit.operation_id},
                durable_seq=cancel_sequence,
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

    async def _can_prepare_initial_attempt(self, execution: ExecutionRecord, binding: AgentBinding | None = None) -> bool:
        binding = await self._execution_binding(execution, binding)
        policy = _runtime_feature_policy(binding.compiled_agent.runtime_contributions)
        trace_commit = next((leaf.commit for feature in policy.capabilities for leaf in runtime_feature_leaves(feature)
                             if isinstance(leaf, TraceCapability)), None)
        if (type(self) is not LocalExecutionBackend
                or execution.durable_trace and (
                    trace_commit is None or trace_commit.mode != "terminal"
                    or self._execution_buffer is None and trace_commit.buffer is None)
                or execution.binding_kind != "agent"
                or execution.parent_execution_id is not None
                or execution.requires_task_invocation_capture
                or execution.input_context is not None or execution.agent_run_seq != 0
                or type(self._execution.executions) is not ExecutionRepositoryImpl
                or type(self._conversation.sessions) is not SessionRepositoryImpl):
            return False
        stores = (self._execution.executions.state_store,
                  self._conversation.sessions.state_store, self._execution.executions.state_store)
        if any(store.storage_group is not stores[0].storage_group for store in stores):
            return False
        binding = await self._execution_binding(execution, binding)
        return not (execution.memory_scope is not None
                    and any(tool_class_from_metadata(tool.contract.get("metadata")) == "memory"
                            for tool in binding.compiled_agent.selected_tools))

    async def pending_execution(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        local = await self._pending_turns.read_execution(execution_id, tenant_id=tenant_id)
        if local is not None:
            return local
        return await self._pending_turns.read_remote_execution(execution_id, tenant_id=tenant_id)

    def require_pending_producer(
        self, execution_id: str, generation: int, allow_cancelling: bool,
    ) -> bool:
        owner = self._turn_owners.get(execution_id)
        if owner is not None and not allow_cancelling:
            self._require_turn_dispatch(owner)
        turn = self._pending_turns.get(execution_id, tenant_id=self._tenant_id)
        if turn is None or turn.committed is not None:
            return False
        turn.require_active(generation, allow_cancelling=allow_cancelling)
        owner = self._turn_owners.get(execution_id)
        if owner is None or owner.scope is None:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        owner.scope.require_active()
        if owner.cancel_requested and not allow_cancelling:
            raise AIError(ErrorCode.EXECUTION_CANCELLED)
        return True

    async def start_pending(
        self, request: ExecutionRequest, reservation: ExecutionStartReservation,
        session: SessionRecord | None, coordination: TurnCoordination | None,
    ) -> str | None:
        execution = reservation.execution
        binding = await self._execution_binding(execution, request.resolved_binding)
        if (_runtime_feature_policy(binding.compiled_agent.runtime_contributions).execution.commit.mode != "terminal" or execution.restart_checkpoints
                or coordination is None or self._pending_turns.coordination is None
                or not self._pending_turns.coordination.supports_pending_views
                or not execution.conversation or not await self._can_prepare_initial_attempt(execution, request.resolved_binding)
                or execution.budget_scope_id is not None or execution.memory_scope is not None
                or not self._conversation_durable or not self._execution_objects_durable
                or not isinstance(self._run_store, RuntimeAgentRunStore)):
            return None
        if (execution.planning or execution.fork_base_execution_id is not None
                or execution.previous_execution_id is not None):
            return None
        owner = self._turn_owners.get(execution.execution_id)
        if owner is None:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        self._require_turn_dispatch(owner)
        await self._validate_start(request, execution)
        if not self._accepting:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if coordination.lease is None or coordination.execution_id != execution.execution_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        coordinator = self._pending_turns.coordination
        identity = reservation.idempotency
        turn = None
        try:
            # The previous owner may have committed and released after the
            # service's idempotency read. Confirm the source after winning a new
            # ephemeral claim, before starting any model work.
            repository = self._execution.idempotency
            existing = await repository.state_store.read(lambda _transaction: repository.get(
                identity.scope, identity.idempotency_key_digest, tenant_id=self._tenant_id,
            ))
            if existing is not None:
                if (existing.request_digest != identity.request_digest
                        or existing.request_identity_version != identity.request_identity_version):
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                if existing.status is IdempotencyStatus.START_UNKNOWN:
                    raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
                # Let the canonical reservation owner apply terminal replay semantics.
                return None
            if session is not None:
                sessions = self._conversation.sessions
                current_session = await sessions.state_store.read(
                    lambda _transaction: sessions.get(session.session_id, tenant_id=self._tenant_id)
                )
                if current_session is None:
                    raise AIError(ErrorCode.STORAGE_NOT_FOUND)
                if current_session.owner_principal_id != session.owner_principal_id:
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                if current_session.agent_id != session.agent_id:
                    raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)
                if current_session.cwd != session.cwd:
                    raise AIError(ErrorCode.SESSION_CONFLICT)
                if current_session.status is not SessionStatus.OPEN:
                    raise AIError(ErrorCode.SESSION_CONFLICT)
                if (current_session.active_execution_id is not None
                        or current_session.continuation != session.continuation):
                    raise AIError(ErrorCode.SESSION_BUSY)
                session = current_session
            turn, created = self._pending_turns.admit(
                self._tenant_id, reservation, session, now=datetime.now(timezone.utc),
            )
            if not created:
                return turn.execution.execution_id
            turn.coordination = coordination
            turn.promote_tools = lambda: self.promote_pending_execution(execution.execution_id, allow_cancelled=True)
            turn.promotion_claim_id = coordination.owner_id
            if self._execution_buffer is not None:
                await self._execution_buffer.append(_ExecutionPayloadCapture(turn.execution, turn.reservation))
            await coordinator.publish(
                turn.execution, coordination,
                request_identity_version=reservation.idempotency.request_identity_version,
            )
            if not self._accepting:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            self._live_broker.prepare_local_producer(execution.execution_id)
            self._live_broker.register_local_producer(execution.execution_id, 0)
            self._live_broker.publish_event(
                execution.execution_id, ExecutionEventType.EXECUTION_STARTED, {}, durable_seq=None,
            )
            self._require_turn_dispatch(owner)
            self._terminal_events[execution.execution_id] = asyncio.Event()
            task = asyncio.create_task(
                self._run(request, turn.execution, initial_attempt=_InitialAttemptCapture(
                    turn.execution, turn.session,
                )), name=f"ai-execution-{execution.execution_id}",
            )
            self._tasks[execution.execution_id] = task
            task.add_done_callback(
                lambda completed: self._task_done(execution.execution_id, completed)
            )
            return turn.execution.execution_id
        except BaseException:
            if turn is not None:
                if owner.stopping:
                    failure = self.worker_failure(execution.execution_id, tenant_id=self._tenant_id)
                    if failure is None:
                        failure = AIError(ErrorCode.STORAGE_CONFLICT,
                                          safe_details={"phase": "turn_owner_lost"})
                        self._record_worker_failure(execution.execution_id, failure)
                    self._pending_turns.fail(turn, failure)
                    # Reuse the unsubmitted/unknown checks after this exact
                    # owner has settled and released. Waiting here would cycle
                    # with owner.closed; the global close drain owns cleanup.
                    cleanup = asyncio.create_task(self._cleanup_failed_pending_turn(turn, forget=True))
                    self._track_checkpoint_task(cleanup, "pending_admission_release")
                else:
                    self._pending_turns.release(turn)
                    if self._execution_buffer is not None:
                        self._execution_buffer.discard(execution.execution_id)
                        await self._execution_buffer.release(execution.execution_id)
                        self._execution_buffer.release_owner(execution.execution_id)
            # Until this method returns an ID, the admission owner retains the
            # lease and releases it on failure.
            raise

    async def _check_pending_lease(self, turn: _PendingExecutionTurn, *, allow_cancelled: bool) -> None:
        owner = self._turn_owners.get(turn.execution.execution_id)
        if owner is None or owner.scope is None:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        owner.scope.require_active()
        if not allow_cancelled and (owner.cancel_requested
                                    or turn.execution.status is ExecutionStatus.CANCELLING):
            raise asyncio.CancelledError

    def _parent_turn(self, reservation: ExecutionStartReservation) -> _OwnedExecutionTurn | None:
        return self._parent_execution_turn(reservation.execution, reservation.parent_producer_generation)

    def _parent_execution_turn(
        self, execution: ExecutionRecord, parent_generation: int | None,
    ) -> _OwnedExecutionTurn | None:
        parent_id = execution.parent_execution_id
        if parent_id is None:
            return None
        parent = self._turn_owners.get(parent_id)
        if (parent is None or parent.stopping or parent.producer_generation is None
                or parent.producer_generation != parent_generation):
            raise AIError(ErrorCode.STORAGE_CONFLICT, safe_details={"phase": "subagent_parent_unavailable"})
        self._require_turn_dispatch(parent)
        return parent

    def _require_turn_dispatch(self, owner: _OwnedExecutionTurn) -> None:
        if (owner.stopping
                or self._turn_owners.get(owner.reservation.execution.execution_id) is not owner):
            raise asyncio.CancelledError
        if owner.scope is not None:
            try:
                owner.scope.require_active()
            except AIError as error:
                self._stop_owned_turn(owner)
                raise asyncio.CancelledError from error

    async def run_turn(
        self, reservation: ExecutionStartReservation, principal: Principal,
        admit: Callable[[TurnCoordination | None], Awaitable[ExecutionHandle]],
    ) -> ExecutionHandle:
        parent = self._parent_turn(reservation)
        admission = asyncio.get_running_loop().create_future()
        entered = asyncio.get_running_loop().create_future() if parent is not None else None
        if parent is not None:
            parent.admissions.add(entered)
        task = asyncio.create_task(
            self._turn_lifetime(reservation, principal, admit, admission, parent, entered),
            name=f"ai-turn-owner-{reservation.execution.execution_id}",
        )
        self._track_checkpoint_task(task, "turn_lifetime")
        return await asyncio.shield(admission)

    async def _run_owned_action(
        self, execution: ExecutionRecord, parent_generation: int | None,
        action: Callable[[], Awaitable[_TurnActionT]],
    ) -> _TurnActionT:
        identity = await self._recovery_idempotency(execution)
        result: list[_TurnActionT] = []

        async def admit(coordination: TurnCoordination | None) -> ExecutionHandle:
            if coordination is not None and coordination.lease is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT, safe_details={"phase": "turn_recovery_busy"})
            result.append(await action())
            return ExecutionHandle(execution.execution_id)

        await self.run_turn(ExecutionStartReservation(
            execution, identity, parent_producer_generation=parent_generation,
        ), Principal(execution.principal_id, self._tenant_id, execution.principal_kind), admit)
        return result[0]

    async def run_existing_turn(
        self, execution: ExecutionRecord, parent_producer_generation: int | None,
        action: Callable[[], Awaitable[None]],
    ) -> None:
        owner = self._turn_owners.get(execution.execution_id)
        if owner is not None:
            if owner.lifetime is asyncio.current_task():
                self._parent_turn(owner.reservation)
                self._require_turn_dispatch(owner)
                await action()
                return
            if (owner.stopping
                    or owner.reservation.parent_producer_generation != parent_producer_generation):
                raise AIError(ErrorCode.STORAGE_CONFLICT, safe_details={"phase": "turn_recovery_busy"})
            self._parent_turn(owner.reservation)
            self._require_turn_dispatch(owner)
            entered = asyncio.get_running_loop().create_future()
            owner.admissions.add(entered)
            try:
                if owner.admitting:
                    assert owner.admission is not None
                    await asyncio.shield(owner.admission)
                self._require_turn_dispatch(owner)
                await action()
            finally:
                entered.set_result(None)
                owner.admissions.discard(entered)
            return
        await self._run_owned_action(execution, parent_producer_generation, action)

    async def run_recovery_turn(
        self, execution_id: str, action: Callable[[], Awaitable[_TurnActionT]],
    ) -> _TurnActionT:
        owner = self._turn_owners.get(execution_id)
        if owner is not None:
            if (owner.lifetime is asyncio.current_task()
                    or self._tasks.get(execution_id) is asyncio.current_task()):
                self._require_turn_dispatch(owner)
                return await action()
            if owner.admitting or self.worker_installed(execution_id):
                raise AIError(ErrorCode.STORAGE_CONFLICT, safe_details={"phase": "turn_recovery_busy"})
            await owner.closed.wait()
        execution = await self.load_execution(execution_id, tenant_id=self._tenant_id)
        if execution is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        parent_generation = None
        if execution.parent_execution_id is not None:
            parent = self._turn_owners.get(execution.parent_execution_id)
            # Approval may continue a delegation held by this exact live parent.
            # A cold scan or replacement parent must first explicitly replay it.
            if (parent is None or execution_id not in parent.delegated_children
                    or parent.producer_generation is None):
                raise AIError(ErrorCode.STORAGE_CONFLICT, safe_details={"phase": "subagent_parent_unavailable"})
            self._require_turn_dispatch(parent)
            parent_generation = parent.producer_generation
        return await self._run_owned_action(execution, parent_generation, action)

    async def resume_subagent(self, execution: ExecutionRecord, parent_generation: int) -> None:
        # Validate the caller's exact parent before touching any child owner.
        parent = self._parent_execution_turn(execution, parent_generation)
        if parent is None:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        entered = asyncio.get_running_loop().create_future()
        parent.admissions.add(entered)
        try:
            current = await self.load_execution(execution.execution_id, tenant_id=self._tenant_id)
            self._require_turn_dispatch(parent)
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED,
                                  ExecutionStatus.CANCELLED, ExecutionStatus.CANCELLING,
                                  ExecutionStatus.FINALIZING}:
                return
            owner = self._turn_owners.get(current.execution_id)
            if owner is not None:
                if not owner.stopping and owner.reservation.parent_producer_generation == parent_generation:
                    self._require_turn_dispatch(owner)
                    return
                self._stop_owned_turn(owner)
                await owner.closed.wait()
                self._require_turn_dispatch(parent)

            async def resume() -> None:
                latest = await self.load_execution(current.execution_id, tenant_id=self._tenant_id)
                self._require_turn_dispatch(parent)
                if latest is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if latest.status is ExecutionStatus.WAITING_DEFERRED:
                    await self._reconcile_deferred_owned(latest.execution_id, tenant_id=self._tenant_id)
                elif latest.status in {ExecutionStatus.STARTED, ExecutionStatus.RECOVERY_REQUIRED}:
                    await self._recovery_coordinator.recover_execution(
                        latest.execution_id, tenant_id=self._tenant_id, expected_revision=latest.revision,
                    )

            await self._run_owned_action(current, parent_generation, resume)
        finally:
            entered.set_result(None)
            parent.admissions.discard(entered)

    async def _turn_lifetime(
        self, reservation: ExecutionStartReservation, principal: Principal,
        admit: Callable[[TurnCoordination | None], Awaitable[ExecutionHandle]],
        admission: asyncio.Future[ExecutionHandle], parent: _OwnedExecutionTurn | None,
        entered: asyncio.Future[None] | None,
    ) -> None:
        execution = reservation.execution
        identity = reservation.idempotency
        coordinator = self._pending_turns.coordination
        owner: _OwnedExecutionTurn | None = None

        def finish_admission() -> None:
            if entered is not None and not entered.done():
                entered.set_result(None)
                assert parent is not None
                parent.admissions.discard(entered)

        async def execute(coordination: TurnCoordination | None, scope: TurnScope | None) -> None:
            nonlocal owner
            if coordinator is not None and coordination is None:
                raise AIError(ErrorCode.SESSION_BUSY)
            if parent is not None:
                self._require_turn_dispatch(parent)
            if coordination is not None and coordination.lease is None:
                result = await admit(coordination)
                if not admission.done():
                    admission.set_result(result)
                return
            if scope is not None:
                scope.require_active()
            if execution.execution_id in self._turn_owners:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            owner = _OwnedExecutionTurn(reservation, coordination, principal, scope=scope,
                                        lifetime=asyncio.current_task(), admission=admission)
            self._turn_owners[execution.execution_id] = owner
            if parent is not None:
                parent.children[execution.execution_id] = owner
                parent.delegated_children.add(execution.execution_id)
            if scope is not None:
                monitor = asyncio.create_task(self._watch_turn(owner), name=f"ai-turn-control-{execution.execution_id}")
                self._turn_monitors[execution.execution_id] = monitor
            try:
                result = await admit(coordination)
                if not admission.done():
                    admission.set_result(result)
            except BaseException as error:
                # Joined starts must observe admission failure before drain
                # waits for their registered actions to leave this owner.
                if not admission.done():
                    admission.set_exception(error)
                raise
            finally:
                owner.admitting = False
                finish_admission()
                await self._drain_owned_turn(owner)

        try:
            if coordinator is None:
                await execute(None, None)
            else:
                scope = coordinator.enter(
                    session_key=(f"session:{execution.session_id}" if execution.session_id is not None else
                                 f"request:{identity.scope}:{identity.idempotency_key_digest}"),
                    request_key=f"{identity.scope}:{identity.idempotency_key_digest}",
                    request_digest=identity.request_digest, execution_id=execution.execution_id,
                    owner_id=uuid.uuid4().hex,
                )
                # A context may own ContextVars or a TaskGroup. Its task never changes.
                async with scope as coordination:
                    await execute(coordination, scope)
        except BaseException as error:
            if not admission.done():
                admission.set_exception(error)
            elif (not admission.cancelled() and admission.exception() is None
                  and not isinstance(error, asyncio.CancelledError)):
                self._record_worker_failure(execution.execution_id, error)
        finally:
            finish_admission()
            if owner is not None:
                if self._turn_owners.get(execution.execution_id) is owner:
                    self._turn_owners.pop(execution.execution_id)
                owner.closed.set()
                if parent is not None and parent.children.get(execution.execution_id) is owner:
                    parent.children.pop(execution.execution_id)

    def _stop_owned_turn(self, owner: _OwnedExecutionTurn) -> None:
        if owner.stopping:
            return
        owner.stopping = True
        execution_id = owner.reservation.execution.execution_id
        if self._turn_owners.get(execution_id) is not owner:
            return
        scope = self._handoff_scopes.get(execution_id)
        if scope is not None:
            scope.stop()
        for child in tuple(owner.children.values()):
            self._stop_owned_turn(child)
        worker = self._tasks.get(execution_id)
        if worker is not None and worker is not asyncio.current_task():
            self._request_worker_shutdown(execution_id, worker)

    async def _drain_owned_turn(self, owner: _OwnedExecutionTurn) -> None:
        execution_id = owner.reservation.execution.execution_id
        worker = self._tasks.get(execution_id)
        if worker is not None:
            await self._drain_worker_task(execution_id, worker)
        self._stop_owned_turn(owner)
        while owner.admissions:
            await asyncio.wait(tuple(owner.admissions))
        children = tuple(owner.children.values())
        if children:
            await asyncio.gather(*(child.closed.wait() for child in children))
        while True:
            recovery = self._recovery_reconcile_tasks.get(execution_id)
            pending = tuple(task for task in (*self._execution_task_map().get(execution_id, ()), recovery)
                            if task is not None and task is not asyncio.current_task() and not task.done())
            if not pending:
                break
            await asyncio.wait(pending)
        monitor = self._turn_monitors.pop(execution_id, None)
        if monitor is not None:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
        # Monitor cancellation can leave an admitted durable operation settling.
        while True:
            pending = tuple(task for task in self._execution_task_map().get(execution_id, ()) if not task.done())
            if not pending:
                break
            await asyncio.wait(pending)

    async def _watch_turn(self, owner: _OwnedExecutionTurn) -> None:
        assert owner.scope is not None
        execution_id = owner.reservation.execution.execution_id
        promotion: asyncio.Task[None] | None = None
        try:
            while self._turn_owners.get(execution_id) is owner:
                # Promotion is an upstream cache control hint, not lease policy.
                if (promotion is None and owner.reservation.execution.conversation
                        and self._pending_turns.coordination.supports_pending_views):
                    promotion = asyncio.create_task(self._watch_pending_promotion(owner))
                signal = await owner.scope.wait_signal()
                if signal == "owner_lost":
                    error = AIError(ErrorCode.STORAGE_CONFLICT, safe_details={"phase": "turn_owner_lost"})
                    turn = self._pending_turns.get(execution_id, tenant_id=self._tenant_id)
                    if turn is not None:
                        self._pending_turns.fail(turn, error)
                    self._record_worker_failure(execution_id, error)
                    self._stop_owned_turn(owner)
                    return
                # Pending publication must see cancellation before its control
                # operation can wait for promotion on the same source boundary.
                owner.cancel_requested = True
                if self._pending_turns.get(execution_id, tenant_id=self._tenant_id) is not None:
                    # Promotion may release the pending view before the source
                    # cancellation is written. Stop its worker while this owner
                    # still holds the accepted cancellation intent.
                    handoff = self._handoff_scopes.get(execution_id)
                    if handoff is not None:
                        handoff.cancel_tools()
                    worker = self._tasks.get(execution_id)
                    if worker is not None:
                        self._request_worker_cancel(execution_id, worker)
                if self._pending_cancel is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                cancellation = asyncio.create_task(self._cancel_admitted_turn(owner))
                # Cancellation waits for the worker; it is not one of the
                # worker's durability tasks. Runtime shutdown still joins it.
                self._track_checkpoint_task(cancellation, "turn_cancel")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._record_worker_failure(execution_id, error, phase="turn_owner_lost")
            self._stop_owned_turn(owner)
        finally:
            if promotion is not None:
                promotion.cancel()
                await asyncio.gather(promotion, return_exceptions=True)

    async def _cancel_admitted_turn(self, owner: _OwnedExecutionTurn) -> None:
        assert owner.admission is not None and self._pending_cancel is not None
        try:
            await asyncio.shield(owner.admission)
        except asyncio.CancelledError:
            if owner.admission.done():
                return
            raise
        except Exception:
            return
        execution_id = owner.reservation.execution.execution_id
        if not owner.stopping and self._turn_owners.get(execution_id) is owner:
            await self._pending_cancel(execution_id, owner.principal)
            # A remote force cancel can settle the source before its signal
            # arrives. The idempotent API then returns without draining us.
            if self._turn_owners.get(execution_id) is owner and self.worker_installed(execution_id):
                await self.cancel(owner.reservation.execution)

    async def _watch_pending_promotion(self, owner: _OwnedExecutionTurn) -> None:
        coordinator = self._pending_turns.coordination
        assert coordinator is not None
        execution_id = owner.reservation.execution.execution_id
        try:
            while not owner.stopping:
                await asyncio.sleep(3)
                if owner.admitting:
                    continue
                turn = self._pending_turns.get(execution_id, tenant_id=self._tenant_id)
                if turn is None:
                    return
                if not turn.committing and not turn.uncertain and owner.coordination is not None:
                    if await coordinator.promotion_requested(owner.coordination):
                        await self.promote_pending_execution(execution_id)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._record_worker_failure(execution_id, error)
            self._stop_owned_turn(owner)

    async def require_tool_authority(self, execution_id: str, agent_run_id: str, *, authoritative: bool = False) -> None:
        """Check the current producer without promoting a replay-safe pending turn."""
        try:
            turn = self._pending_turns.get(execution_id, tenant_id=self._tenant_id)
            if turn is not None:
                async with turn.boundary:
                    if self._pending_turns.get(execution_id, tenant_id=self._tenant_id) is turn:
                        if type(self._run_store) is not RuntimeAgentRunStore:
                            await self._check_pending_lease(turn, allow_cancelled=False)
                        await self._run_store.require_active_producer(agent_run_id, authoritative=authoritative)
                        return
            await self._run_store.require_active_producer(agent_run_id, authoritative=authoritative)
        except AIError as error:
            if error.code is ErrorCode.EXECUTION_CANCELLED:
                raise asyncio.CancelledError from error
            raise

    async def promote_pending_execution(self, execution_id: str, *, allow_cancelled: bool = False) -> None:
        """Establish source authority before effects or dependent work can begin."""
        turn = self._pending_turns.get(execution_id, tenant_id=self._tenant_id)
        if turn is None:
            await self._promote_remote_pending(execution_id)
            return
        await self._pending_turns.settle(turn)
        async with turn.boundary:
            if self._pending_turns.get(execution_id, tenant_id=self._tenant_id) is not turn:
                return
            if turn.committing or turn.uncertain:
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            if turn.failure is not None:
                raise turn.failure

            await self._check_pending_lease(turn, allow_cancelled=allow_cancelled)
            reservation = turn.reservation
            context = turn.execution.input_context
            if self._execution_buffer is not None:
                captured, _ids = await self._execution_buffer.read(execution_id)
                if captured.reservation is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                reservation = captured.reservation
                context = captured.execution.input_context
            reservation = replace(reservation, execution=await _materialize_execution_payloads(
                reservation.execution, input_materializer=self._input_materializer,
                objects=self._execution_objects, keys=ObjectKeyFactory(self._namespace),
                policy=self._payload_policy, tenant_id=self._tenant_id,
            ))
            context_write = None
            if context is not None:
                prepared = await self._context_archive.prepare(await self.load_input_context(context),
                    history_root=turn.history_root, history_digest=turn.history_digest)
                context = prepared.reference
                context_write = (prepared.archive, prepared.records)
            if turn.coordination is None or turn.coordination.execution_id != execution_id:
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            claim = turn.promotion_claim_id or turn.coordination.owner_id
            if (claim != turn.coordination.owner_id
                    or (turn.execution.agent_run_seq > 0 and turn.execution.producer_claim_id != claim)):
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            turn.promotion_claim_id = claim
            tool_operations = tuple(turn.tool_operations.values())
            confirmed: ExecutionRecord | None = None

            async def accept(execution: ExecutionRecord) -> None:
                nonlocal confirmed
                # Keep the original promotion receipt even if a later outcome
                # write is unknown. Its submitted tuple must not be replaced by
                # the working set, nor checked again after those writes settle.
                confirmed = execution
                await self._settle_promoted_tools(turn, tool_operations)
                turn.uncertain = False
                turn.failure = None
                self._worker_failures.pop(execution_id, None)
                turn.promoted = True
                self._pending_turns.release(turn)
                self._live_broker.confirm_events(execution_id, first_event_seq=1, count=1)

            async def confirm() -> None:
                value = confirmed
                if value is None:
                    value = await self._runtime_commands.read_pending_turn_promotion(
                        reservation, expected_session=turn.session,
                        captured_input_context=context, producer_claim_id=claim, tool_operations=tool_operations,
                    )
                if value is None:
                    # Absence does not prove rollback of the original unknown attempt.
                    raise turn.failure or AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
                execution = value
                checkpoint = execution
                failure = turn.failure
                await accept(execution)
                if failure is not None and not self.worker_installed(execution_id):
                    # Its known effects are settled before the stopped worker's
                    # failure. Confirmation never permits a model replay.
                    terminal = await self._commit_failure(
                        execution, failure, agent_run_id=checkpoint.recovery.agent_run_id,
                        producer_generation=None if context is None else 2,
                    )
                    if context is not None:
                        await self._discard_terminal_producer_tail(
                            terminal, agent_run_id=checkpoint.recovery.agent_run_id, producer_generation=2,
                        )

            async def promote_owned() -> None:
                execution = await self._runtime_commands.promote_pending_turn(
                    reservation, expected_session=turn.session,
                    captured_input_context=context, producer_claim_id=claim, context_write=context_write,
                    tool_operations=tool_operations,
                )
                await accept(execution)

            await self._check_pending_lease(turn, allow_cancelled=allow_cancelled)
            turn.reconcile = lambda: self._reconcile_pending_boundary(turn, confirm)
            task = asyncio.create_task(promote_owned(), name=f"ai-promote-pending-{execution_id}")
            try:
                _result, cancellation = await self._await_checkpoint_task(
                    task, label="pending promotion", execution_id=execution_id,
                )
            except AIError as error:
                if error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
                    turn.uncertain = True
                raise
            if cancellation is not None:
                raise cancellation

    async def _settle_promoted_tools(
        self, turn: _PendingExecutionTurn, submitted: Sequence[ToolOperationRecord],
    ) -> None:
        """Transfer late outcomes only after the original admission is confirmed."""
        if len(submitted) != len(turn.tool_operations):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for original in submitted:
            current = turn.tool_operations.get(original.tool_operation_id)
            if current == original:
                continue
            if (current is None or original.owner is None
                    or current.fence != original.fence
                    or current.owner != (None if current.status is ToolOperationStatus.PENDING else original.owner)):
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
            if current.status in {ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED}:
                await self._runtime_commands.commit_tool_terminal(
                    current.tool_operation_id, tenant_id=self._tenant_id,
                    owner=original.owner, fence=original.fence,
                    result_payload=current.result_payload, error_code=current.error_code,
                    error_payload=current.error_payload,
                )
            elif current.status is ToolOperationStatus.PENDING:
                await self._runtime_commands.commit_tool_deferred(
                    current.tool_operation_id, tenant_id=self._tenant_id,
                    owner=original.owner, fence=original.fence,
                )
            elif current.status is ToolOperationStatus.EFFECT_UNKNOWN:
                if self._tool_operations is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                await self._tool_operations.mark_effect_unknown(
                    current.tool_operation_id, tenant_id=self._tenant_id,
                    owner=original.owner, fence=original.fence, error_code=current.error_code,
                )
            else:
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)

    async def _promote_remote_pending(self, execution_id: str) -> None:
        coordinator = self._pending_turns.coordination
        if coordinator is None:
            return
        if await self.load_execution(execution_id, tenant_id=self._tenant_id) is not None:
            return
        owner_id = await coordinator.request_promotion(execution_id, tenant_id=self._tenant_id)
        if owner_id is None:
            if await self.load_execution(execution_id, tenant_id=self._tenant_id) is not None:
                return
            raise AIError(ErrorCode.EXECUTION_NOT_READY)
        pause = 0.05
        deadline_ns = monotonic_ns() + 30_000_000_000
        while True:
            # Only an authoritative source row permits the original dependency
            # operation. The cache hint is neither an ACK nor an admission.
            if await self.load_execution(execution_id, tenant_id=self._tenant_id) is not None:
                return
            if monotonic_ns() >= deadline_ns:
                raise AIError(ErrorCode.EXECUTION_NOT_READY, safe_details={"phase": "pending_promotion_timeout"})
            if not await coordinator.remote_promotion_pending(
                execution_id, owner_id, tenant_id=self._tenant_id,
            ):
                if await self.load_execution(execution_id, tenant_id=self._tenant_id) is not None:
                    return
                raise AIError(ErrorCode.EXECUTION_NOT_READY)
            await asyncio.sleep(pause)
            pause = min(1.0, pause * 2)

    async def prepare_reserved_start(
        self, request: ExecutionRequest, reservation: ExecutionStartReservation,
        session: SessionRecord | None,
    ) -> ExecutionStartReservationResult | None:
        await self._validate_start(request, reservation.execution)
        expected = (
            await self._expected_session_cursor(reservation.execution, session=session)
            if reservation.execution.session_id is not None else None
        )
        result = await self._runtime_commands.reserve_start_checkpoint(
            reservation, expected_cursor=expected, expected_session=session,
        )
        owner = self._turn_owners.get(reservation.execution.execution_id)
        if (owner is not None and result is not None and result.created
                and result.execution.recovery is not None):
            owner.attempt = _InitialAttemptCapture(result.execution, result.session)
            if isinstance(result, ExecutionStartReceipt):
                owner.observed = result.observed
                owner.reservation = replace(owner.reservation, idempotency=result.idempotency)
                owner.session_turn = result.turn
        return result

    async def prepare_start(
        self,
        request: ExecutionRequest,
        execution: ExecutionRecord,
        identity: ExecutionStartIdentity,
    ) -> ExecutionRecord:
        await self._validate_start(request, execution)
        if execution.status is not ExecutionStatus.PENDING_START:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._execution_binding(execution, request.resolved_binding)
        now = datetime.now(timezone.utc)
        expected = (
            await self._expected_session_cursor(execution)
            if execution.session_id is not None
            else None
        )
        started = await self._runtime_commands.commit_start_attempt_checkpoint(
            ExecutionStartClaim(
                execution.execution_id,
                execution.revision,
                execution.event_seq,
                identity.scope,
                identity.idempotency_key_digest,
                identity.request_digest,
                now,
                identity.request_identity_version,
            ),
            session_id=execution.session_id,
            expected_cursor=expected, expected_session=identity.expected_session,
        )
        _logger.info(
            "execution start checkpoint committed: execution=%s", execution.execution_id
        )
        return started

    async def _ensure_session_admission(self, execution: ExecutionRecord) -> None:
        if execution.session_id is None:
            return
        if execution.parent_execution_id is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        expected = await self._expected_session_cursor(execution)
        released_stale_owner = False
        for _ in range(2):
            session = await self._load_session(
                execution.session_id,
                tenant_id=self._tenant_id,
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
                owner = await self.load_execution(
                    owner_id,
                    tenant_id=self._tenant_id,
                )
                if owner is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if (
                    owner.execution_id != owner_id
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
                    tenant_id=self._tenant_id,
                    execution_id=owner_id,
                    history_unavailable=owner.invalidates_session_history,
                    require_turn_commit=owner.conversation,
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
                tenant_id=self._tenant_id,
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
        producer_generation: int | None = None,
        recovery_operation: OperationLedgerInput | None = None,
    ) -> None:
        owner = self._turn_owners.get(execution.execution_id)
        if owner is None and self._pending_turns.coordination is not None:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if owner is not None:
            self._require_turn_dispatch(owner)
        if recovery_operation is not None and not self._accepting:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                safe_details={"phase": "recovery_launch_rejected"})
        if request.resolved_binding is None:
            request = replace(request, resolved_binding=await self._execution_binding(execution))
        await self._validate_start(request, execution)
        captured = (None if owner is None or resume is not None
                    or producer_generation is not None or recovery_operation is not None
                    else owner.attempt)
        if captured is not None and captured.execution != execution:
            captured = None
        current = (captured.execution if captured is not None else
            await self.load_execution(execution.execution_id, tenant_id=self._tenant_id))
        if current is None or current.recovery is None or current.recovery_state is ExecutionRecoveryState.COMPLETED:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        checkpoint = current
        if execution.stored_user_input is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        request = replace(
            request,
            user_prompt=await self.restore_user_input(execution),
            files=(),
        )
        # The captured execution revision is conditionally claimed before dispatch.
        if (
            current.binding_digest != execution.binding_digest
            or current.mode != execution.mode
            or current.planning is not execution.planning
            or current.thinking != execution.thinking
            or current.binding != execution.binding
            or current.correlation != execution.correlation
            or current.tool_response_ref != execution.tool_response_ref
            or current.tool_effects != execution.tool_effects
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if current.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.CANCELLING,
            ExecutionStatus.FINALIZING,
        }:
            if recovery_operation is not None:
                raise AIError(ErrorCode.STORAGE_CONFLICT,
                    safe_details={"phase": "recovery_launch_rejected"})
            return
        if current.status is not ExecutionStatus.STARTED:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        session = None
        if current.session_id is not None:
            session = (captured.session if captured is not None else
                await self._load_session(current.session_id, tenant_id=self._tenant_id))
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
                if recovery_operation is not None:
                    raise AIError(ErrorCode.STORAGE_CONFLICT,
                        safe_details={"phase": "recovery_launch_rejected"})
                return
            self._task_done(execution.execution_id, existing)
        if execution.execution_id in self._worker_failures:
            if recovery_operation is not None:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED,
                    safe_details={"phase": "recovery_launch_rejected"})
            return
        if recovery_operation is not None:
            control = await self._execution.executions.get(
                execution.execution_id, tenant_id=self._tenant_id,
            )
            if not self._accepting:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                    safe_details={"phase": "recovery_launch_rejected"})
            if (control is None or control.status is not ExecutionStatus.STARTED
                    or control.producer_generation != producer_generation
                    or control.producer_claim_id != recovery_operation.result_ref):
                raise AIError(ErrorCode.STORAGE_CONFLICT,
                    safe_details={"phase": "recovery_launch_rejected"})
        if current.budget_scope_id is not None and request.budget is None:
            # Cold recovery and deferred resumption reconstruct the launch request.
            usage = await self._execution.budgets.read(current.budget_scope_id)
            request = replace(request, budget=usage.limits)
        if not self._accepting:
            if recovery_operation is not None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                    safe_details={"phase": "recovery_launch_rejected"})
            return
        if not self._live_broker.is_local_producer(current.execution_id):
            self._live_broker.register_local_producer(
                current.execution_id,
                current.event_seq,
            )
        initial_attempt = None
        if (resume is None and producer_generation is None and recovery_operation is None
                and checkpoint.recovery_state is ExecutionRecoveryState.ADMITTED
                and type(self) is LocalExecutionBackend):
            # Reusing reads does not opt into batched admission. The worker still
            # claims these revisions through the normal attempt CAS.
            initial_attempt = _InitialAttemptCapture(current, session)
        if owner is not None:
            self._require_turn_dispatch(owner)
        self._terminal_events[execution.execution_id] = asyncio.Event()
        run = (self._run(request, execution, resume, producer_generation=producer_generation)
               if initial_attempt is None else
               self._run(request, execution, resume, producer_generation=producer_generation,
                         initial_attempt=initial_attempt))
        task = asyncio.create_task(run, name=f"ai-execution-{execution.execution_id}")
        self._tasks[execution.execution_id] = task
        task.add_done_callback(
            lambda completed, execution_id=execution.execution_id: self._task_done(
                execution_id,
                completed,
            )
        )
        _logger.debug(
            "local execution launched: execution=%s compiled_agent=%s",
            execution.execution_id,
            execution.binding_digest,
        )

    async def abort_start(self, execution: ExecutionRecord) -> None:
        current = await self.load_execution(
            execution.execution_id,
            tenant_id=self._tenant_id,
        )
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if current.status not in {
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        } or current.started_at is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if current.recovery is not None and current.recovery.terminal_handoff is not None:
            if current.recovery.handoff_phase is RecoveryHandoffPhase.COMPLETED:
                return
            await self._reconcile_handoff(current)
            return
        if current.session_id is not None:
            released = await self._conversation.sessions.release_execution(
                current.session_id,
                tenant_id=self._tenant_id,
                execution_id=current.execution_id,
                history_unavailable=current.invalidates_session_history,
                require_turn_commit=current.conversation,
            )
            if released.active_execution_id == current.execution_id:
                raise AIError(
                    ErrorCode.STORAGE_RECOVERY_REQUIRED,
                    safe_details={
                        "execution_id": current.execution_id,
                        "phase": "start_cleanup_session_release",
                    },
                )
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
        self._worker_shutdown_requests.discard(execution_id)
        self._captured_usage.pop(execution_id, None)
        event = self._terminal_events.get(execution_id)
        if event is not None:
            event.set()
        agent_run_only = execution_id in self._agent_run_only_worker_exits
        self._agent_run_only_worker_exits.discard(execution_id)
        if error is None:
            if not agent_run_only:
                self._live_broker.complete(execution_id)
            if self._worker_completed is not None:
                cleanup = asyncio.create_task(self._complete_worker_handoff(
                    execution_id, resume_handoff=not agent_run_only,
                ))
                self._track_checkpoint_task(cleanup, "execution_handoff")
            return
        self._record_worker_failure(execution_id, error)
        turn = self._pending_turns.get(execution_id, tenant_id=self._tenant_id)
        if turn is not None:
            failure = self.worker_failure(execution_id, tenant_id=self._tenant_id)
            assert failure is not None
            self._pending_turns.fail(turn, failure)
            if not turn.uncertain:
                if self._execution_buffer is not None:
                    self._execution_buffer.discard(execution_id)
                cleanup = asyncio.create_task(self._cleanup_failed_pending_turn(turn))
                self._track_checkpoint_task(cleanup, "pending_release")
        self._live_broker.complete(execution_id)

    async def _complete_worker_handoff(
        self, execution_id: str, *, resume_handoff: bool = True,
    ) -> None:
        try:
            while True:
                if self.worker_installed(execution_id) or execution_id in self._worker_failures:
                    return
                owned = tuple(self._execution_task_map().get(execution_id, ()))
                recovery = self._recovery_reconcile_tasks.get(execution_id)
                pending = tuple(task for task in (*owned, recovery) if task is not None
                                and task is not asyncio.current_task() and not task.done())
                if not pending:
                    break
                # Operation callers own their outcomes; cleanup only waits for quiescence.
                await asyncio.wait(pending)
            await self.release_runtime_execution(execution_id, tenant_id=self._tenant_id)
            if resume_handoff and self._worker_completed is not None:
                await self._worker_completed(execution_id)
        except BaseException as error:
            self._record_worker_failure(execution_id, error)
            raise

    def _record_worker_failure(
        self, execution_id: str, error: BaseException, *, phase: str | None = None,
    ) -> None:
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
        if phase is not None:
            details["phase"] = phase
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
            exc_info=True,
        )

    def _request_worker_shutdown(
        self,
        execution_id: str,
        task: asyncio.Task[None],
    ) -> None:
        if task.done() or execution_id in self._worker_cancel_requests:
            return
        self._worker_shutdown_requests.add(execution_id)
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
        if not self.worker_installed(execution_id):
            return
        event = self._terminal_events.setdefault(execution_id, asyncio.Event())
        await event.wait()

    @property
    def live_broker(self) -> LiveExecutionEventBroker:
        return self._live_broker

    async def cancel(self, execution: ExecutionRecord, *, force: bool = False) -> CancelEffectOutcome:
        self._worker_shutdown_requests.discard(execution.execution_id)
        current = await self.load_execution(
            execution.execution_id,
            tenant_id=self._tenant_id,
        )
        if current is not None and current.status is ExecutionStatus.FINALIZING:
            _logger.debug(
                "local cancellation deferred during finalization: execution=%s",
                execution.execution_id,
            )
            return CancelEffectOutcome.CONFIRMED
        task = self._tasks.get(execution.execution_id)
        if task is None:
            return await self._confirm_cancel_effects(current, force=force)
        scope = self._handoff_scopes.get(execution.execution_id)
        if scope is not None:
            scope.cancel_tools()
        self._request_worker_cancel(execution.execution_id, task)
        await self._drain_worker_task(execution.execution_id, task)
        current = await self.load_execution(
            execution.execution_id,
            tenant_id=self._tenant_id,
        )
        return await self._confirm_cancel_effects(current, force=force, producer_drained=True)

    async def _confirm_cancel_effects(
        self,
        execution: ExecutionRecord | None,
        *, force: bool = False, producer_drained: bool = False,
    ) -> CancelEffectOutcome:
        if execution is None:
            return CancelEffectOutcome.UNKNOWN
        if execution.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.FINALIZING,
        }:
            return CancelEffectOutcome.CONFIRMED
        revoke = (force and not execution.restart_checkpoints
                  and execution.binding_kind == "agent"
                  and not execution.requires_task_invocation_capture)
        if revoke and execution.root_execution_id != execution.execution_id:
            root = await self.load_execution(
                execution.root_execution_id, tenant_id=self._tenant_id,
            )
            if root is None or root.root_execution_id != root.execution_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            # Delegated calls inherit their root's task ownership, not the
            # child's optional invocation-capture marker.
            revoke = root.binding_kind == "agent" and not root.requires_task_invocation_capture
        if execution.status is ExecutionStatus.RECOVERY_REQUIRED and not revoke:
            return CancelEffectOutcome.UNKNOWN
        if execution.status not in {
            ExecutionStatus.STARTED,
            ExecutionStatus.CANCELLING,
            ExecutionStatus.RECOVERY_REQUIRED,
        }:
            return CancelEffectOutcome.UNKNOWN
        if execution.status is ExecutionStatus.CANCELLING and producer_drained:
            # A remote cancel intent is not evidence that the live owner's
            # leaves stopped. Let that owner settle and drain before recovery.
            await self._reconcile_cancelled_tool_claims(execution)
        effects, active_claims = await self._reconcile_tool_effects(
            execution.execution_id,
            tenant_id=self._tenant_id,
        )
        if effects:
            if execution.status is not ExecutionStatus.RECOVERY_REQUIRED:
                await self._require_tool_effect_recovery(execution, effects)
            return CancelEffectOutcome.UNKNOWN
        if active_claims:
            return CancelEffectOutcome.UNKNOWN
        if revoke and execution.status is ExecutionStatus.RECOVERY_REQUIRED:
            execution = await self._recovery_commands_for(execution.execution_id).commit_cancel_claim(execution)
        # CANCELLING blocks new tool admission. Terminal execution status also
        # prevents a remote model response from publishing afterward.
        return (
            CancelEffectOutcome.CONFIRMED
            if execution.status is ExecutionStatus.CANCELLING and (execution.agent_run_seq == 0 or revoke)
            else CancelEffectOutcome.UNKNOWN
        )

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
            PersistenceDomain.EXECUTION,
            PersistenceDomain.CHECKPOINT,
        }:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        objects = (
            self._execution_objects
            if reference.source_domain is PersistenceDomain.EXECUTION
            else self._checkpoint_objects
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
        if canonical_sha256(instructions.to_payload()) != payload.digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return instructions

    async def reconcile_deferred(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> None:
        execution = await self.load_execution(execution_id, tenant_id=tenant_id)
        if execution is None or execution.status is not ExecutionStatus.WAITING_DEFERRED:
            return
        await self.run_recovery_turn(
            execution_id, lambda: self._reconcile_deferred_owned(execution_id, tenant_id=tenant_id),
        )

    async def _reconcile_deferred_owned(self, execution_id: str, *, tenant_id: str) -> None:
        decision = await self._recovery_coordinator.reconcile_deferred(
            execution_id,
            tenant_id=tenant_id,
        )
        if decision is None:
            return
        current = decision.execution
        principal = Principal(
            current.principal_id,
            self._tenant_id,
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
            tool_response_ref=current.tool_response_ref,
            tool_effects=current.tool_effects,
            session_context=current.session_context,
            resolved_binding=decision.binding,
        )
        if not self._prepare_recovery_relaunch(current.execution_id):
            return
        await self.launch(request, current, resume=decision)
        _logger.info(
            "deferred execution relaunched: execution=%s",
            current.execution_id,
        )

    async def materialize_deferred_call(
        self,
        execution: ExecutionRecord,
        source_agent_run_id: str,
        call: ToolCallPart,
        metadata: Mapping[str, object],
        *,
        repository_instruction_frontier: str | None = None,
        capability_ref: CapabilityRef | None = None,
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
        if repository_instruction_frontier is not None:
            if (
                len(repository_instruction_frontier) != 64
                or any(character not in "0123456789abcdef" for character in repository_instruction_frontier)
            ):
                raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
            # This reserved evidence comes from the runtime's observation point,
            # never from tool-authored metadata or the later approval decision.
            raw_metadata["linktools"] = {
                "repository_instruction_frontier": repository_instruction_frontier,
            }
        arguments_payload = StoredPayload.inline_json(arguments)
        if not payload_fits_inline(arguments_payload, self._payload_policy):
            reference = await put_runtime_object(
                self._checkpoint_objects,
                ObjectKeyFactory(self._namespace),
                PersistenceDomain.CHECKPOINT,
                self._tenant_id,
                canonical_json_bytes(arguments),
            )
            arguments_payload = StoredPayload.object(reference)
        return PendingDeferredCall(
            call.tool_call_id,
            call.tool_name,
            arguments_payload,
            raw_metadata,
            capability_ref=capability_ref,
        )

    async def commit_deferred_frontier(
        self,
        execution: ExecutionRecord,
        continuation: PendingToolContinuation,
        approval_records: tuple[ApprovalRecord, ...],
        external_records: tuple[ExternalCallRecord, ...],
        occurred_at: datetime,
        *,
        producer_generation: int | None = None,
    ) -> ExecutionRecord:
        if not execution.restart_checkpoints and (execution.conversation or not execution.durable_trace):
            # Deferred input retains its interrupted context even when public
            # history is disabled. It does not enable native restart checkpoints.
            await self._run_store.materialize_conversation(
                agent_run_id=continuation.source_agent_run_id, include_interrupted=True,
            )
        await self._agent_run_lifecycle.flush_execution_projection(
            continuation.source_agent_run_id, execution_id=execution.execution_id,
        )
        async with self._audit_lock(execution.execution_id):
            payload_ids: tuple[str, ...] = ()
            if self._execution_buffer is not None:
                captured, payload_ids = await self._execution_buffer.source_prefix(execution.execution_id)
                if captured is not None and captured.terminal is not None:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
            pending_audit = await self._source_audit_prefix(execution.execution_id)
            committed = (
                await self._runtime_commands.commit_deferred_checkpoint(
                    execution_id=execution.execution_id,
                    tenant_id=self._tenant_id,
                    expected_execution_revision=execution.revision,
                    expected_event_seq=execution.event_seq,
                    expected_agent_run_seq=execution.agent_run_seq,
                    continuation=continuation,
                    audit_events=pending_audit,
                    approval_records=approval_records,
                    external_records=external_records,
                    occurred_at=occurred_at,
                    background_tasks=self._execution_task_set(execution.execution_id),
                    producer_generation=producer_generation,
                )
            )
            if self._execution_buffer is not None:
                self._execution_buffer.confirm(execution.execution_id, payload_ids)
            self._confirm_source_audits(execution.execution_id, len(pending_audit))
            if self._execution_buffer is not None:
                if producer_generation is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                self._execution_buffer.retire(execution.execution_id, producer_generation)
        if pending_audit:
            self._live_broker.confirm_events(
                execution.execution_id,
                first_event_seq=execution.event_seq + 1,
                count=len(pending_audit),
            )
        offset = execution.event_seq + len(pending_audit)
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
                    "arguments_digest": item.arguments_payload.digest,
                },
                durable_seq=offset + index,
            )
        self._agent_run_only_worker_exits.add(execution.execution_id)
        return committed

    async def load_approval(
        self,
        approval_id: str,
        *,
        tenant_id: str,
    ) -> ApprovalRecord | None:
        return await self._checkpoint.approvals.get(
            approval_id,
            tenant_id=tenant_id,
        )

    async def load_external_call(
        self,
        call_id: str,
        *,
        tenant_id: str,
    ) -> ExternalCallRecord | None:
        return await self._checkpoint.external_calls.get(
            call_id,
            tenant_id=tenant_id,
        )

    async def load_interrupted_checkpoint(
        self,
        agent_run_id: str,
        *, execution: ExecutionRecord,
    ) -> AgentRunCheckpoint:
        domain = (PersistenceDomain.CHECKPOINT if execution.restart_checkpoints else
                  PersistenceDomain.TRACE if execution.durable_trace else PersistenceDomain.CONVERSATION)
        archive = self._run_stores[domain]
        with runtime_object_source_reads():
            checkpoint = await archive.latest_checkpoint(
                agent_run_id=agent_run_id, include_interrupted=True,
                **({"include_transcript": False} if execution.restart_checkpoints else {}),
            )
        if checkpoint is None or checkpoint.state != "interrupted":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return checkpoint

    async def claim_deferred_resume(
        self,
        execution: ExecutionRecord,
        *, recovery_operation: OperationLedgerInput | None = None,
    ) -> ExecutionRecord:
        if execution.recovery is None or execution.recovery.pending_tools is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        resumed = await self._runtime_commands.claim_deferred_resume_checkpoint(
            execution_id=execution.execution_id,
            tenant_id=self._tenant_id,
            expected_execution_revision=execution.revision,
            expected_event_seq=execution.event_seq,
            expected_agent_run_seq=execution.agent_run_seq,
            expected_pending_tools=execution.recovery.pending_tools,
            background_tasks=self._execution_task_set(execution.execution_id),
            recovery_operation=recovery_operation,
        )
        if recovery_operation is not None:
            receipt = await self._recovery_receipt(recovery_operation)
            if receipt is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if receipt.result_ref != recovery_operation.result_ref:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        return resumed

    async def read_deferred_payload(self, payload: StoredPayload) -> JsonValue:
        if payload.kind == "inline":
            value = payload.decode()
        else:
            if payload.ref is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            raw = await read_runtime_object(self._checkpoint_objects, payload.ref)
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeError, TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        try:
            return normalize_json_value(value)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def bind_recovery_relaunch_guard(
        self, guard: Callable[[ExecutionRecord], Awaitable[bool]],
    ) -> None:
        self._recovery_coordinator.bind_relaunch_guard(guard)

    async def reconcile(self, *, allow_active_recovery: bool = True) -> None:
        await self._recovery_coordinator.reconcile(
            allow_active_recovery=allow_active_recovery,
        )

    def _defer_recovery_reconcile(self, execution_id: str) -> None:
        task = self._recovery_reconcile_tasks.get(execution_id)
        if task is not None and not task.done():
            return
        task = asyncio.create_task(
            self._wait_and_reconcile_tool_claims(execution_id),
            name=f"ai-recovery-reconcile-{execution_id}",
        )
        self._recovery_reconcile_tasks[execution_id] = task

        def consume(done: asyncio.Task[None]) -> None:
            if self._recovery_reconcile_tasks.get(execution_id) is done:
                self._recovery_reconcile_tasks.pop(execution_id, None)
            try:
                done.result()
            except asyncio.CancelledError:
                return
            except BaseException as error:  # noqa: BLE001
                if (
                    isinstance(error, AIError)
                    and error.code is ErrorCode.AGENT_BINDING_UNAVAILABLE
                    and error.safe_details.get("reason") != "workspace_mismatch"
                ):
                    _logger.warning(
                        "recovery reconciliation deferred: execution=%s",
                        execution_id,
                    )
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
                        {"phase": "recovery_reconcile"},
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
                self._live_broker.complete(execution_id)
                _logger.error(
                    "deferred recovery reconciliation failed: execution=%s code=%s",
                    execution_id,
                    failure.code.value,
                    exc_info=True,
                )

        task.add_done_callback(consume)

    async def _wait_and_reconcile_tool_claims(self, execution_id: str) -> None:
        while self._accepting:
            if self.worker_installed(execution_id):
                return
            effects, active_claims = await self._reconcile_tool_effects(
                execution_id,
                tenant_id=self._tenant_id,
            )
            if self.worker_installed(execution_id):
                return
            if effects or active_claims == 0:
                checkpoint = await self._execution.executions.get_recovery(
                    execution_id,
                    tenant_id=self._tenant_id,
                )
                if checkpoint is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if checkpoint.recovery_state is ExecutionRecoveryState.COMPLETED:
                    return
                if not self._accepting:
                    return
                try:
                    await self._recovery_coordinator.reconcile_execution(checkpoint)
                except AIError as error:
                    if error.code is not ErrorCode.STORAGE_CONFLICT:
                        raise
                    current = await self.load_execution(
                        execution_id,
                        tenant_id=self._tenant_id,
                    )
                    if current is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                    if (
                        current.status
                        not in {
                            ExecutionStatus.STARTED,
                            ExecutionStatus.CANCELLING,
                        }
                        or self.worker_installed(execution_id)
                    ):
                        return
                    await asyncio.sleep(1)
                    continue
                current = await self.load_execution(
                    execution_id,
                    tenant_id=self._tenant_id,
                )
                if (
                    current is None
                    or current.status
                    not in {
                        ExecutionStatus.STARTED,
                        ExecutionStatus.CANCELLING,
                    }
                    or self.worker_installed(execution_id)
                ):
                    return
            await asyncio.sleep(1)

    async def _reconcile_session_recovery(
        self,
        execution: ExecutionRecord,
        *,
        recovery_operation: OperationLedgerInput | None = None,
    ) -> bool:
        if execution.session_id is None:
            return True
        if execution.recovery is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        async def finish(
            status: ExecutionStatus, error_code: str,
            *, error_diagnostics: ErrorDiagnostics | None = None,
        ) -> None:
            if recovery_operation is not None:
                await self._admit_recovery_without_producer(execution, recovery_operation)
            await self._commit_terminal(
                execution, status, None, error_code,
                StopReason.CANCELLED if status is ExecutionStatus.CANCELLED else StopReason.ERROR,
                error_diagnostics=error_diagnostics,
                expected_revision=execution.revision if recovery_operation is not None else None,
            )

        session = await self._load_session(
            execution.session_id,
            tenant_id=self._tenant_id,
        )
        if session is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution.status is ExecutionStatus.FINALIZING:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if session.status is SessionStatus.CLOSED:
            if execution.status is not ExecutionStatus.PENDING_START:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await finish(ExecutionStatus.FAILED, ErrorCode.SESSION_CONFLICT.value)
            return False
        if session.status in {SessionStatus.CLOSING, SessionStatus.CLEANUP_REQUIRED}:
            if execution.status is ExecutionStatus.FINALIZING:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if execution.status is ExecutionStatus.PENDING_START:
                if session.active_execution_id == execution.execution_id:
                    await finish(ExecutionStatus.CANCELLED, ErrorCode.EXECUTION_CANCELLED.value)
                else:
                    await finish(ExecutionStatus.FAILED, ErrorCode.SESSION_CONFLICT.value)
            elif session.active_execution_id == execution.execution_id:
                if execution.status is ExecutionStatus.WAITING_DEFERRED:
                    if (
                        execution.recovery_state is not ExecutionRecoveryState.WAITING
                        or execution.recovery.pending_tools is None
                    ):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    operation_id = canonical_sha256(
                        {
                            "type": "linktools.ai.session-close-recovery-cancel",
                            "version": 1,
                            "tenant_id": self._tenant_id,
                            "session_id": execution.session_id,
                            "execution_id": execution.execution_id,
                        }
                    )
                    committed = await self.commit_cancel_checkpoint(
                        ExecutionCancelRequestCommit(
                            execution.execution_id,
                            execution.revision,
                            execution.event_seq,
                            operation_id,
                            datetime.now(timezone.utc),
                        ),
                        expected_status=ExecutionStatus.WAITING_DEFERRED,
                    )
                    if committed.status is not ExecutionStatus.CANCELLING:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if (committed.recovery is None
                            or committed.recovery_state is not ExecutionRecoveryState.ACTIVE
                            or committed.recovery.pending_tools is not None):
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                    execution = committed
                elif execution.status is ExecutionStatus.CANCELLING:
                    if execution.recovery.pending_tools is not None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                elif (
                    execution.status is ExecutionStatus.STARTED
                    and execution.recovery_state is ExecutionRecoveryState.ACTIVE
                    and execution.recovery.pending_tools is not None
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await finish(ExecutionStatus.CANCELLED, ErrorCode.EXECUTION_CANCELLED.value)
            else:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return False
        if execution.status is ExecutionStatus.PENDING_START:
            if session.status is SessionStatus.OPEN:
                return True
            if session.status is SessionStatus.CLOSED:
                await finish(ExecutionStatus.FAILED, ErrorCode.SESSION_CONFLICT.value)
                return False
            if session.active_execution_id == execution.execution_id:
                await finish(ExecutionStatus.CANCELLED, ErrorCode.EXECUTION_CANCELLED.value)
                return False
            await finish(ExecutionStatus.FAILED, ErrorCode.SESSION_CONFLICT.value)
            return False
        try:
            await self._ensure_session_admission(execution)
        except AIError as error:
            if error.code not in {
                ErrorCode.SESSION_BUSY,
                ErrorCode.SESSION_CONFLICT,
            }:
                raise
            await finish(
                ExecutionStatus.FAILED, error.code.value,
                error_diagnostics=_execution_error_diagnostics(error),
            )
            return False
        return True

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
            self._checkpoint_objects
            if source is PersistenceDomain.CHECKPOINT
            else self._execution_objects
        )
        await read_runtime_object(store, reference)

    async def _reconcile_handoff(
        self, checkpoint: ExecutionRecord
    ) -> ExecutionRecord:
        handoff = checkpoint.recovery.terminal_handoff
        if handoff is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        outcome = handoff.outcome
        if outcome.terminal_status in {
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        } and (handoff.conversation is not None or outcome.output is not None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        execution = await self.load_execution(
            checkpoint.execution_id,
            tenant_id=self._tenant_id,
        )
        if execution is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if handoff.source_agent_run_id is not None:
            self._run_store.bind_execution_persistence(
                handoff.source_agent_run_id, conversation=execution.conversation,
                restart_checkpoints=execution.restart_checkpoints,
                durable_trace=execution.durable_trace,
            )
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
        if checkpoint.recovery.handoff_phase is RecoveryHandoffPhase.COMPLETED:
            await self._validate_committed_handoff(checkpoint)
            if execution.session_id is not None and execution.conversation:
                result = await self._execution.executions.get_result(
                    execution.execution_id, tenant_id=self._tenant_id,
                )
                await self._runtime_commands.verify_conversation_turn(execution, result)
            return execution
        if checkpoint.recovery.handoff_phase is RecoveryHandoffPhase.PREPARED:
            if outcome.terminal_status is ExecutionStatus.SUCCEEDED:
                if execution.conversation and execution.session_id is not None and handoff.conversation is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if handoff.source_agent_run_id is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if execution.durable_trace and execution.status is not ExecutionStatus.SUCCEEDED:
                    await self._agent_run_lifecycle.materialize_from_recovery(
                        target=PersistenceDomain.TRACE,
                        agent_run_id=handoff.source_agent_run_id,
                        execution_id=checkpoint.execution_id,
                    )
                if execution.conversation or execution.restart_checkpoints or execution.durable_trace:
                    checkpoint_domain = (
                        PersistenceDomain.TRACE
                        if execution.durable_trace
                        else (PersistenceDomain.CHECKPOINT if execution.restart_checkpoints
                              else PersistenceDomain.CONVERSATION)
                    )
                    agent_run_checkpoint = await self._run_stores[checkpoint_domain].latest_checkpoint(
                        agent_run_id=handoff.source_agent_run_id,
                    )
                    if (
                        agent_run_checkpoint is None
                        or agent_run_checkpoint.state != "complete"
                    ):
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

            async def finish_handoff(execution: ExecutionRecord) -> ExecutionRecord:
                await self._validate_committed_handoff(checkpoint)
                if execution.session_id is not None:
                    if execution.conversation and execution.status is not ExecutionStatus.SUCCEEDED:
                        await self._runtime_commands.finish_conversation_turn(execution, execution.result)
                    else:
                        await self._conversation.sessions.release_execution(
                            execution.session_id,
                            tenant_id=self._tenant_id,
                            execution_id=execution.execution_id,
                            history_unavailable=execution.invalidates_session_history,
                            require_turn_commit=execution.conversation,
                        )
                completed = await self._complete_handoff(execution)
                self._publish_terminal_event(
                    execution.execution_id,
                    event_type=outcome.terminal_event_type,
                    payload=dict(outcome.terminal_event_payload),
                    durable_seq=completed.event_seq,
                )
                return completed

            execution = await self._commit_reconciled_terminal(
                checkpoint, finish=finish_handoff,
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
            tenant_id=self._tenant_id,
        )
        if len(records) != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        identity = records[0]
        if (
            identity.resource_kind is not ResourceKind.EXECUTION
            or identity.resource_id != execution.execution_id
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
            identity.resource_kind is not ResourceKind.EXECUTION
            or identity.resource_id != execution.execution_id
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
                tenant_id=self._tenant_id,
                expected_status=identity.status,
                next_record=replace(
                    identity,
                    status=IdempotencyStatus.STARTED,
                    updated_at=execution.updated_at,
                ),
            )
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _resolve_handoff_conversation(
        self, checkpoint: ExecutionRecord, handoff: RecoveryTerminalHandoff
    ) -> None:
        intent = handoff.conversation
        if intent is None:
            return
        terminal = await self.load_execution(checkpoint.execution_id, tenant_id=self._tenant_id)
        if terminal is not None and terminal.status in {
            ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
        }:
            result = await self._execution.executions.get_result(
                checkpoint.execution_id, tenant_id=self._tenant_id,
            )
            try:
                await self._runtime_commands.verify_conversation_turn(terminal, result)
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_RECOVERY_REQUIRED:
                    raise
            else:
                # An acknowledged turn must not replay its cursor over a later turn.
                return
        session = await self._load_session(
            intent.session_id, tenant_id=self._tenant_id
        )
        if session is None:
            if self._conversation_durable:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            _logger.info(
                "recovery conversation resolve skipped: session lost execution=%s",
                checkpoint.execution_id,
            )
            return
        conversation_archive = self._run_stores[PersistenceDomain.CONVERSATION]
        if conversation_archive is not self._run_store:
            if handoff.source_agent_run_id is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            # PREPARED already references the published Conversation projection,
            # including any compaction summary. Native CP owns a different context.
            target_checkpoint = await conversation_archive.latest_checkpoint(
                agent_run_id=handoff.source_agent_run_id,
            )
            if target_checkpoint is None or target_checkpoint.state != "complete":
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        history_id = session.history_id or intent.next_cursor.history_id
        if history_id is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        end_message_index = await self._run_store.conversation_persistence.conversation_message_count(
            history_id=history_id,
            agent_run_id=intent.next_cursor.agent_run_id,
            tenant_id=self._tenant_id,
        )
        effective_next_cursor = replace(
            intent.next_cursor,
            history_id=history_id,
            message_count=end_message_index,
        )
        if intent.expected_cursor is None:
            start_message_index = 0
        else:
            start_message_index = intent.expected_cursor.message_count
            if start_message_index is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if end_message_index <= start_message_index:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        async def commit_conversation(transaction) -> None:
            current = await self._conversation.sessions.get_in_transaction(
                transaction,
                intent.session_id,
                tenant_id=self._tenant_id,
            )
            if current.continuation == effective_next_cursor:
                return
            if current.status is SessionStatus.CLOSED:
                raise AIError(ErrorCode.SESSION_CONFLICT)
            if (
                current.active_execution_id != checkpoint.execution_id
                or current.continuation != intent.expected_cursor
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._conversation.sessions.advance_continuation_in_transaction(
                transaction,
                intent.session_id,
                tenant_id=self._tenant_id,
                execution_id=checkpoint.execution_id,
                expected=intent.expected_cursor,
                next_cursor=effective_next_cursor,
                release_execution=False,
                history_quality="complete",
            )


        try:
            if session.continuation != effective_next_cursor:
                await self._conversation.sessions.state_store.mutate(commit_conversation)
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise
            latest = await self._load_session(
                intent.session_id,
                tenant_id=self._tenant_id,
            )
            if latest is None or latest.continuation != effective_next_cursor:
                raise

        terminal = await self.load_execution(checkpoint.execution_id, tenant_id=self._tenant_id)
        if terminal is not None and terminal.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            result = await self._execution.executions.get_result(checkpoint.execution_id, tenant_id=self._tenant_id)
            await self._runtime_commands.finish_conversation_turn(
                terminal, result, message_range=(start_message_index, end_message_index),
            )

    async def _complete_handoff(self, checkpoint: ExecutionRecord) -> ExecutionRecord:
        if checkpoint.recovery is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if checkpoint.recovery.handoff_phase is RecoveryHandoffPhase.COMPLETED:
            return checkpoint
        if checkpoint.recovery.handoff_phase is not RecoveryHandoffPhase.PREPARED:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        completed = replace(
            checkpoint,
            recovery=replace(checkpoint.recovery, handoff_phase=RecoveryHandoffPhase.COMPLETED, pending_tools=None),
            revision=checkpoint.revision + 1,
            updated_at=datetime.now(timezone.utc),
        )
        try:
            return await self._execution.executions.compare_and_swap(
                checkpoint.execution_id,
                tenant_id=self._tenant_id,
                expected_revision=checkpoint.revision,
                next_record=completed,
            )
        except AIError as error:
            if error.code not in {ErrorCode.STORAGE_CONFLICT, ErrorCode.STORAGE_COMMIT_UNKNOWN}:
                raise
            current = await self._execution.executions.get_recovery(
                checkpoint.execution_id,
                tenant_id=self._tenant_id,
            )
            if (
                current is None
                or current.revision < completed.revision
                or replace(
                    current,
                    revision=completed.revision,
                    updated_at=completed.updated_at,
                    dependency_hold_ids=completed.dependency_hold_ids,
                ) != completed
            ):
                raise
            return current

    async def _claim_session_or_recovery_finalizing(
        self,
        execution: ExecutionRecord,
    ) -> ExecutionRecord:
        if execution.session_id is None:
            return execution
        if execution.status is not ExecutionStatus.STARTED:
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
                tenant_id=self._tenant_id,
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
            current = await self.load_execution(
                execution.execution_id,
                tenant_id=self._tenant_id,
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

    async def _rewrite_prepared_success_handoff(
        self,
        checkpoint: ExecutionRecord,
        *,
        target_status: ExecutionStatus,
        error_code: str,
        stop_reason: StopReason,
        error_diagnostics: ErrorDiagnostics | None = None,
    ) -> ExecutionRecord:
        handoff = checkpoint.recovery.terminal_handoff
        if (
            checkpoint.recovery.handoff_phase is not RecoveryHandoffPhase.PREPARED
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
        updated = replace(checkpoint, recovery=replace(checkpoint.recovery, terminal_handoff=rewritten) , revision=checkpoint.revision + 1, updated_at=datetime.now(timezone.utc))
        try:
            result = await self._execution.executions.compare_and_swap(
                checkpoint.execution_id,
                tenant_id=self._tenant_id,
                expected_revision=checkpoint.revision,
                next_record=updated,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._execution.executions.get_recovery(
                checkpoint.execution_id,
                tenant_id=self._tenant_id,
            )
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current.recovery.handoff_phase is not RecoveryHandoffPhase.PREPARED:
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
        checkpoint: ExecutionRecord,
        *,
        finish: Callable[[ExecutionRecord], Awaitable[ExecutionRecord]],
    ) -> ExecutionRecord:
        handoff = checkpoint.recovery.terminal_handoff
        if handoff is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        outcome = handoff.outcome
        current = await self.load_execution(
            checkpoint.execution_id,
            tenant_id=self._tenant_id,
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
            return await finish(current)
        execution_ref = None
        if outcome.terminal_status is ExecutionStatus.SUCCEEDED:
            if outcome.output is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            execution_ref = outcome.output
            if (
                outcome.output.kind == "object"
                and outcome.object_source_domain is PersistenceDomain.CHECKPOINT
            ):
                reference = outcome.output.ref
                if reference is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                payload = await read_runtime_object(self._checkpoint_objects, reference)
                execution_ref = StoredPayload.object(
                    await put_runtime_object(
                        self._execution_objects,
                        ObjectKeyFactory(self._namespace),
                        PersistenceDomain.EXECUTION,
                        self._tenant_id,
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
            handoff.source_agent_run_id
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
            execution_ref,
            outcome.stop_reason,
            outcome.usage,
            outcome.result_created_at,
        )
        conversation_message_range = None
        if handoff.conversation is not None:
            intent = handoff.conversation
            retained_session = await self._load_session(intent.session_id, tenant_id=self._tenant_id)
            start = 0 if intent.expected_cursor is None else intent.expected_cursor.message_count
            history_id = intent.next_cursor.history_id
            if history_id is None and retained_session is not None:
                history_id = retained_session.history_id
            if history_id is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            end = await self._run_store.conversation_persistence.conversation_message_count(
                history_id=history_id, agent_run_id=intent.next_cursor.agent_run_id,
                tenant_id=self._tenant_id,
            )
            expected_continuation = replace(intent.next_cursor, history_id=history_id, message_count=end)
            if (start is None or end <= start or retained_session is None
                    or retained_session.continuation != expected_continuation):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            conversation_message_range = (start, end)
        handoff_started = False

        async def finish_handoff(committed: ExecutionTerminalCommitResult) -> ExecutionRecord:
            nonlocal handoff_started
            handoff_started = True
            return await finish(committed.execution)

        try:
            terminal_agent_run_id = handoff.source_agent_run_id
            if terminal_agent_run_id is None and current.agent_run_seq > 0:
                terminal_agent_run_id = make_agent_run_id(
                    namespace=self._namespace,
                    tenant_id=self._tenant_id,
                    execution_id=current.execution_id,
                    agent_run_seq=current.agent_run_seq,
                )
            committed = await self._commit_execution_terminal_checkpoint(
                current,
                ExecutionTerminalCommit(
                    current.revision,
                    current.event_seq,
                    terminal,
                    result,
                    outcome.terminal_event_type,
                    dict(outcome.terminal_event_payload),
                ),
                agent_run_id=terminal_agent_run_id,
                conversation_message_range=conversation_message_range,
                handoff=finish_handoff,
            )
            current = committed.execution
        except AIError as error:
            if handoff_started:
                raise
            if error.code not in {
                ErrorCode.STORAGE_CONFLICT,
                ErrorCode.EXECUTION_RESULT_CONFLICT,
            }:
                raise
            latest = await self.load_execution(
                current.execution_id,
                tenant_id=self._tenant_id,
            )
            if latest is None or latest.status is not outcome.terminal_status:
                raise
            current = await finish(latest)
        _logger.info(
            "recovery execution terminal committed: execution=%s status=%s",
            current.execution_id,
            outcome.terminal_status.value,
        )
        return current

    async def _validate_committed_handoff(
        self,
        checkpoint: ExecutionRecord,
    ) -> None:
        handoff = checkpoint.recovery.terminal_handoff
        if handoff is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        execution = await self.load_execution(
            checkpoint.execution_id,
            tenant_id=self._tenant_id,
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
        repository = self._execution.executions
        result = await repository.state_store.read(
            lambda _transaction: repository.get_result(checkpoint.execution_id, tenant_id=self._tenant_id)
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
            tenant_id=self._tenant_id,
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

    def preflight_handoff(self) -> None:
        if any(owner.admitting or any(not task.done() for task in owner.admissions)
               for owner in self._turn_owners.values()):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, retryable=True,
                safe_details={"phase": "runtime_handoff", "reason": "admission_in_progress"})
        if self._pending_turns.has_pending:
            raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING,
                safe_details={"phase": "runtime_handoff", "reason": "nonrecoverable_active_execution"})
        for execution_id, task in self._tasks.items():
            if task.done():
                continue
            owner = self._turn_owners.get(execution_id)
            if owner is None or not owner.reservation.execution.restart_checkpoints:
                raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING,
                    safe_details={"phase": "runtime_handoff", "execution_id": execution_id,
                                  "reason": ("unknown_active_execution" if owner is None
                                             else "nonrecoverable_active_execution")})
        # Validation and admission stop share one event-loop turn. A rejected
        # handoff must leave every existing producer and admission flag intact.
        self._accepting = False

    async def handoff(self) -> None:
        self._accepting = False
        self._handoff_requested = True
        for owner in tuple(self._turn_owners.values()):
            self._stop_owned_turn(owner)
        for scope in self._handoff_scopes.values():
            scope.stop()
        await self.close()

    async def close(self) -> None:
        self._accepting = False
        await self._pending_turns.settle_all()
        recovery_tasks = tuple(self._recovery_reconcile_tasks.values())
        for task in recovery_tasks:
            task.cancel()
        if recovery_tasks:
            await asyncio.gather(*recovery_tasks, return_exceptions=True)
        self._recovery_reconcile_tasks.clear()

        tasks = tuple(self._tasks.items())
        for execution_id, task in tasks:
            current = await self.load_execution(
                execution_id,
                tenant_id=self._tenant_id,
            )
            if current is None or current.status not in {
                ExecutionStatus.FINALIZING,
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
                ExecutionStatus.RECOVERY_REQUIRED,
            }:
                self._request_worker_shutdown(execution_id, task)
            else:
                _logger.info(
                    "close draining execution worker: execution=%s status=%s",
                    execution_id,
                    current.status.value,
                )

        for execution_id, task in tasks:
            await self._drain_worker_task(execution_id, task)

        while True:
            dispatcher_background = (
                ()
                if self._subagent_dispatcher is None
                else self._subagent_dispatcher.pending_background_tasks
            )
            execution_background = {
                task for tasks_for_execution in self._execution_task_map().values()
                for task in tasks_for_execution
            }
            owned_background = tuple(
                task
                for task in (
                    *dispatcher_background,
                    *(self._checkpoint_tasks - execution_background),
                )
                if isinstance(task, asyncio.Task) and not task.done()
            )
            if not owned_background:
                break
            await asyncio.gather(
                *(asyncio.shield(task) for task in owned_background),
                return_exceptions=True,
            )

        await self._live_broker.close_shared()
        if self._execution_buffer is not None:
            try:
                await self._execution_buffer.close()
            except AIError as error:
                if (error.code is not ErrorCode.STORAGE_RECOVERY_REQUIRED
                        or error.safe_details.get("phase") != "execution_buffer_close"):
                    raise
        for execution_id in tuple(self._worker_failures):
            turn = self._pending_turns.get(execution_id, tenant_id=self._tenant_id)
            if turn is not None:
                await self._cleanup_failed_pending_turn(turn, forget=True)
        await self._release_revoked_failed_producers()
        if self._execution_buffer is not None:
            await self._execution_buffer.close()

        pending_workers = tuple(
            task for task in self._tasks.values() if not task.done()
        )
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
        self._agent_run_only_worker_exits.clear()
        self._checkpoint_tasks.clear()
        self._worker_cancel_requests.clear()
        self._worker_shutdown_requests.clear()
        self._execution_task_map().clear()

    async def _cleanup_failed_pending_turn(
        self, turn: _PendingExecutionTurn, *, forget: bool = False,
    ) -> None:
        """Drop only a stopped turn which never submitted a source boundary."""
        execution_id = turn.execution.execution_id

        def failed_turn() -> bool:
            return (self._pending_turns.get(execution_id, tenant_id=self._tenant_id) is turn
                    and turn.failure is not None and not turn.uncertain
                    and execution_id in self._worker_failures and not self.worker_installed(execution_id))

        def eligible() -> bool:
            return (failed_turn() and not turn.committing
                    and turn.committed is None and turn.reconcile is None)

        def quiescent() -> bool:
            recovery = self._recovery_reconcile_tasks.get(execution_id)
            monitor = self._turn_monitors.get(execution_id)
            return (eligible() and (recovery is None or recovery.done())
                    and (monitor is None or monitor.done())
                    and not any(task is not asyncio.current_task() and not task.done()
                                for task in self._execution_task_map().get(execution_id, ())))

        if not failed_turn():
            return
        owner = self._turn_owners.get(execution_id)
        if owner is not None and owner.lifetime is not asyncio.current_task():
            await owner.closed.wait()
        else:
            monitor = self._turn_monitors.get(execution_id)
            if monitor is not None and monitor is not asyncio.current_task():
                monitor.cancel()
                await asyncio.gather(asyncio.shield(monitor), return_exceptions=True)
                if self._turn_monitors.get(execution_id) is monitor:
                    self._turn_monitors.pop(execution_id)
        if not failed_turn():
            return
        if not eligible() or not isinstance(self._run_store, RuntimeAgentRunStore):
            return
        while True:
            pending = tuple(task for task in self._execution_task_map().get(execution_id, ())
                            if task is not asyncio.current_task() and not task.done())
            if not pending:
                break
            await asyncio.gather(*(asyncio.shield(task) for task in pending), return_exceptions=True)
        async with turn.boundary:
            if not quiescent():
                return
            coordinator = self._pending_turns.coordination
            if coordinator is None or turn.coordination is None:
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            if await coordinator.read_owner(execution_id, tenant_id=self._tenant_id) is not None:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED,
                              safe_details={"phase": "pending_cleanup_owner"})
            if not quiescent():
                return
            if self._execution_buffer is not None:
                self._execution_buffer.require_unsubmitted(execution_id)
            await self._run_store.discard_unsubmitted_execution_staging(execution_id)
            if self._execution_buffer is not None:
                self._execution_buffer.discard(execution_id)
                await self._execution_buffer.flush_cleanup(execution_id)
            async with self._audit_lock(execution_id):
                self._pending_audit_events.pop(execution_id, None)
            # Live observers still receive the original failure, not a missing
            # execution or a fabricated terminal result. Close alone forgets it.
            if forget:
                self._pending_turns.release(turn)
                self._worker_failures.pop(execution_id, None)

    async def _release_revoked_failed_producers(self, execution_ids: Sequence[str] | None = None) -> None:
        if not isinstance(self._run_store, RuntimeAgentRunStore):
            return
        archive = self._run_stores[PersistenceDomain.TRACE]
        if not isinstance(archive, StateStepArchive):
            return

        def quiescent(execution_id: str) -> bool:
            owner = self._turn_owners.get(execution_id)
            recovery = self._recovery_reconcile_tasks.get(execution_id)
            monitor = self._turn_monitors.get(execution_id)
            return (not self.worker_installed(execution_id)
                    and (owner is None or owner.closed.is_set())
                    and (recovery is None or recovery.done())
                    and (monitor is None or monitor.done())
                    and not any(task is not asyncio.current_task() and not task.done()
                                for task in self._execution_task_map().get(execution_id, ())))

        for execution_id in tuple(self._worker_failures) if execution_ids is None else execution_ids:
            if execution_id not in self._worker_failures:
                continue
            if self._pending_turns.get(execution_id, tenant_id=self._tenant_id) is not None:
                continue
            if not quiescent(execution_id):
                continue
            current = await self.load_execution(execution_id, tenant_id=self._tenant_id)
            if current is None or current.status not in {
                ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
                ExecutionStatus.RECOVERY_REQUIRED,
            }:
                continue
            if current.status is ExecutionStatus.RECOVERY_REQUIRED:
                if self._tool_operations is None:
                    continue
                operations = await self._tool_operations.list_by_execution(execution_id, tenant_id=self._tenant_id)
                if any(operation.status not in {
                    ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED, ToolOperationStatus.CANCELLED,
                } for operation in operations):
                    continue
                # Recovery still owns the business outcome; only this stopped
                # producer's rejected capture may be abandoned.
            if not quiescent(execution_id):
                continue
            buffered = self._execution_buffer is not None and self._execution_buffer.has_scope(execution_id)
            if not buffered and not self._run_store.has_active_execution_producer(execution_id):
                continue
            if buffered:
                assert self._execution_buffer is not None
                generation = self._execution_buffer.producer_generation(execution_id)
                # A later owner may have revoked this failed worker after it
                # stopped. Recheck authority before releasing its retained bytes.
                await archive.require_revoked_execution_producer(
                    execution_id=execution_id, producer_generation=generation,
                    terminal_execution=current,
                )
                if not quiescent(execution_id):
                    continue
                try:
                    self._execution_buffer.retire(execution_id, generation, abandon=True)
                except AIError as error:
                    if error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
                        continue
                    raise
            for sequence in range(1, current.agent_run_seq + 1):
                agent_run_id = make_agent_run_id(
                    namespace=self._namespace, tenant_id=self._tenant_id,
                    execution_id=execution_id, agent_run_seq=sequence,
                )
                generation = self._run_store.execution_producer_generation(
                    agent_run_id, execution_id=execution_id,
                )
                if generation is not None:
                    await self._agent_run_lifecycle.discard_revoked_producer_staging(
                        agent_run_id, execution_id=execution_id, producer_generation=generation,
                        terminal_execution=current,
                        settle_cleanup=True,
                    )
            if buffered:
                await self._execution_buffer.flush_cleanup(execution_id)
                await self._execution_buffer.release(execution_id)
                self._execution_buffer.release_owner(execution_id)
            async with self._audit_lock(execution_id):
                self._pending_audit_events.pop(execution_id, None)
            self._worker_failures.pop(execution_id, None)

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
        recovery_task = self._recovery_reconcile_tasks.get(execution_id)
        if recovery_task is not None and not recovery_task.done():
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
        if execution_id in self._worker_failures and (
            self._execution_buffer is not None and self._execution_buffer.has_scope(execution_id)
            or isinstance(self._run_store, RuntimeAgentRunStore)
            and self._run_store.has_active_execution_producer(execution_id)
        ):
            await self._release_revoked_failed_producers((execution_id,))
            if execution_id in self._worker_failures:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        if self._execution_buffer is not None:
            self._execution_buffer.release_owner(execution_id)
        if task is not None:
            self._tasks.pop(execution_id, None)
        self._worker_cancel_requests.discard(execution_id)
        self._worker_shutdown_requests.discard(execution_id)
        self._terminal_events.pop(execution_id, None)
        self._worker_failures.pop(execution_id, None)
        self._captured_usage.pop(execution_id, None)
        self._pending_audit_events.pop(execution_id, None)
        self._pending_audit_locks.pop(execution_id, None)
        self._agent_run_only_worker_exits.discard(execution_id)
        self._recovery_reconcile_tasks.pop(execution_id, None)
        self._execution_task_map().pop(execution_id, None)
        _logger.debug(
            "local execution runtime cache released: tenant=%s execution=%s",
            tenant_id,
            execution_id,
        )

    async def _reconcile_unresolved_tool_operations(
        self, agent_run_id: str, operations: Sequence[ToolOperationRecord], *,
        tenant_id: str, execution: ExecutionRecord, producer_generation: int,
    ) -> None:
        if (execution.producer_generation != producer_generation
                or execution.producer_claim_id is None):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        for operation in operations:
            current = await self._runtime_commands.reconcile_tool_turn_takeover(
                operation.tool_operation_id, execution_id=execution.execution_id,
                producer_generation=producer_generation,
                producer_claim_id=execution.producer_claim_id,
            )
            if current.status in {ToolOperationStatus.EFFECT_UNKNOWN, ToolOperationStatus.CANCELLED}:
                raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
            if current.status is ToolOperationStatus.CLAIMED:
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)

    async def validate_preload_skill_inputs(
        self, binding: AgentBinding, inputs: tuple[Mapping[str, JsonValue], ...],
    ) -> None:
        await self._executor.validate_preload_skill_inputs(binding, inputs)

    async def load_input_context(self, reference: RuntimePayloadRef | None) -> ExecutionInputContext | None:
        if reference is None:
            return None
        payload = reference.payload
        value = payload.decode() if payload.kind == "inline" else json.loads(await read_runtime_object(self._execution_objects, payload.ref))
        if isinstance(value, Mapping) and value.get("kind") == "execution-context":
            return await self._context_archive.load(reference)
        return ExecutionInputContext.from_payload(value)

    async def load_input_repository_instructions(
        self, reference: RuntimePayloadRef,
    ) -> RepositoryInstructions | None:
        if reference.source_domain is not PersistenceDomain.EXECUTION:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        payload = reference.payload
        value = payload.decode() if payload.kind == "inline" else json.loads(await read_runtime_object(self._execution_objects, payload.ref))
        if isinstance(value, Mapping) and value.get("kind") == "execution-context":
            instructions = await self._context_archive.load_repository_instructions(reference)
        else:
            instructions = ExecutionInputContext.from_payload(value).repository_instructions
        return None if instructions is None else RepositoryInstructions.from_payload(instructions)

    async def _prepare_input_context(self, context: ExecutionInputContext) -> RuntimePayloadRef:
        # A pending owner retains this view only in-process. Coordination publishes
        # control facts; promotion/terminal publication retains canonical content.
        return RuntimePayloadRef(StoredPayload.inline_json(context.to_payload()), PersistenceDomain.EXECUTION)

    async def _persist_input_context(
        self, execution: ExecutionRecord, context: ExecutionInputContext,
        loaded: LoadedModelContext | None = None,
    ) -> tuple[ExecutionRecord, ExecutionInputContext]:
        prepared = await self._context_archive.prepare(context,
            history_root=None if loaded is None else loaded.archive_root,
            history_digest=None if loaded is None else loaded.archive_digest)
        reference = prepared.reference
        current = execution
        while current.input_context is None:
            try:
                async def publish_context(transaction):
                    del transaction
                    await prepared.publish()
                    return await self._execution.executions.compare_and_swap(current.execution_id, tenant_id=self._tenant_id,
                        expected_revision=current.revision,
                        next_record=replace(current, input_context=reference, repository_instructions=None,
                                            revision=current.revision + 1, updated_at=datetime.now(timezone.utc)))
                committed = await self._execution.executions.state_store.mutate(publish_context)
                return committed, context
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                current = await self.load_execution(current.execution_id, tenant_id=self._tenant_id)
                if current is None or current.status is not ExecutionStatus.STARTED:
                    raise AIError(ErrorCode.EXECUTION_CANCELLED) from error
        accepted = await self.load_input_context(current.input_context)
        if accepted is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return current, accepted

    async def _read_stopped_admission(self, execution_id: str) -> ExecutionRecord | None:
        observed = await self.load_execution(execution_id, tenant_id=self._tenant_id)
        if observed is not None and observed.agent_run_seq == 0:
            if observed.status is ExecutionStatus.CANCELLING:
                raise asyncio.CancelledError
            if observed.status in {
                ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
            }:
                return observed
        return None

    async def _run(
        self,
        request: ExecutionRequest,
        original: ExecutionRecord,
        resume: _DeferredResume | None = None,
        *,
        producer_generation: int | None = None,
        initial_attempt: _InitialAttemptCapture | None = None,
    ) -> None:
        execution_id = original.execution_id
        handoff_scope = ExecutionHandoffScope()
        owner = self._turn_owners.get(execution_id)
        if self._handoff_requested or owner is not None and owner.stopping:
            handoff_scope.stop()
        self._handoff_scopes[execution_id] = handoff_scope
        execution_handoff_scope.set(handoff_scope)
        execution_tasks = self._execution_task_set(execution_id)
        checkpoint: ExecutionRecord | None = None
        agent_run_id: str | None = None
        recovery_history_agent_run_id: str | None = None
        metric_recorder = self._metric_recorder
        metric_id = uuid.uuid4().hex if metric_recorder is not None else None
        metric_started = monotonic_ns() if metric_id is not None else None
        metric_status = "FAILED"
        metric_error_code: str | None = None
        admission_claim: AgentAttemptClaim | None = None
        prepared_admission: AgentAttemptClaim | None = None
        activated: Observed[ExecutionRecord] | None = None
        try:
            if initial_attempt is None:
                current = await self.load_execution(
                    execution_id, tenant_id=self._tenant_id
                )
                if current is None:
                    raise AIError(ErrorCode.STORAGE_NOT_FOUND)
                checkpoint = current
            else:
                # Preparation has no producer authority. The attempt CAS checks
                # both captured revisions before any provider can be dispatched.
                current = initial_attempt.execution
                checkpoint = initial_attempt.execution
            if checkpoint is None or checkpoint.recovery is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if checkpoint.recovery_state is ExecutionRecoveryState.ADMITTED:
                admission_claim = AgentAttemptClaim(
                    execution_id=execution_id,
                    expected_execution_revision=current.revision,
                    expected_agent_run_seq=current.agent_run_seq,

                    expected_recovery_state=checkpoint.recovery_state,
                )
                if await self._can_prepare_initial_attempt(current, request.resolved_binding):
                    prepared_admission = admission_claim
                else:
                    try:
                        activated = (
                            await self._runtime_commands.commit_agent_attempt_checkpoint(
                                admission_claim,
                            )
                        )
                    except AIError as error:
                        if error.code is not ErrorCode.STORAGE_CONFLICT:
                            raise
                        stopped = await self._read_stopped_admission(execution_id)
                        if stopped is not None:
                            metric_status = stopped.status.value
                            metric_error_code = stopped.error_code
                            return
                        raise
                    current = activated.value
                    if owner is not None:
                        owner.observed = activated
                    checkpoint = current
                    producer_generation = current.producer_generation
                    _logger.info(
                        "agent attempt activated: execution=%s sequence=%s",
                        execution_id,
                        current.agent_run_seq,
                    )
            elif checkpoint.recovery_state is ExecutionRecoveryState.WAITING:
                if current.status is not ExecutionStatus.WAITING_DEFERRED:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                resume = await self._recovery_coordinator.reconcile_waiting_deferred(
                    current,
                    binding=request.resolved_binding,
                )
                if resume is None:
                    return
                current = resume.execution
                checkpoint = resume.execution
                producer_generation = resume.producer_generation
            elif checkpoint.recovery_state is not ExecutionRecoveryState.ACTIVE:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if (
                current.status is not ExecutionStatus.STARTED
                or (checkpoint.recovery.agent_run_id is None and prepared_admission is None)
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            binding = await self._execution_binding(current, request.resolved_binding)
            compiled_agent = binding.compiled_agent
            accepted_context = await self.load_input_context(current.input_context)
            initial_repository_instructions = (
                await self.load_repository_instructions(current.repository_instructions)
                if accepted_context is None else (
                    None if accepted_context.repository_instructions is None
                    else RepositoryInstructions.from_payload(accepted_context.repository_instructions)
                )
            )
            repository_overlay = await self.load_repository_instructions(
                checkpoint.recovery.repository_instruction_overlay
            )
            _validate_repository_instruction_frontier(checkpoint, repository_overlay)
            agent_run_id = (make_agent_run_id(
                namespace=self._namespace, tenant_id=self._tenant_id,
                execution_id=execution_id, agent_run_seq=current.agent_run_seq + 1,
            ) if prepared_admission is not None else checkpoint.recovery.agent_run_id)
            if producer_generation is None and resume is not None:
                producer_generation = resume.producer_generation
            if producer_generation is None and prepared_admission is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            agent_conversation_id = make_agent_conversation_id(
                namespace=self._namespace,
                tenant_id=self._tenant_id,
                execution_id=execution_id,
            )
            session = (initial_attempt.session if initial_attempt is not None
                and (current.session_id is None or initial_attempt.session is not None) else
                None if current.session_id is None else
                await self._load_session(current.session_id, tenant_id=self._tenant_id))
            if current.session_id is not None and session is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            session_metadata: Mapping[str, JsonValue] = {} if session is None else session.metadata
            initial_scopes = tuple(sorted({
                document.scope for document in (
                    () if initial_repository_instructions is None else initial_repository_instructions.documents
                )
            }))
            if session is not None and not set(initial_scopes).issubset(session.repository_instruction_scopes):
                # Keep scope knowledge outside admission CAS: concurrent starts
                # must still converge through the existing idempotency owner.
                await self.promote_pending_execution(execution_id)
                sessions = self._conversation.sessions
                session = await sessions.state_store.mutate(
                    lambda transaction: sessions.remember_repository_instruction_scopes_in_transaction(
                        transaction, session.session_id, tenant_id=self._tenant_id,
                        execution_id=execution_id, scopes=initial_scopes,
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
            source_replay_history = (
                None if resume is None else list(resume.history)
            )
            deferred_tool_results = None if resume is None else resume.results
            resumed_deferred_attempt = deferred_tool_results is not None
            output_retries_used = 0 if resume is None else resume.output_retries_used
            exact_recovery_context = checkpoint.recovery.source_agent_run_id is not None
            recovery_history_agent_run_id = checkpoint.recovery.source_agent_run_id
            if recovery_history_agent_run_id is not None:
                recovery_domain = (PersistenceDomain.CHECKPOINT if current.restart_checkpoints else
                                   PersistenceDomain.TRACE if current.durable_trace else PersistenceDomain.CONVERSATION)
                checkpoint_archive = self._get_run_store(recovery_domain)
                recovery_run = (await checkpoint_archive.get_run(run_id=recovery_history_agent_run_id)
                                if current.restart_checkpoints else
                                await checkpoint_archive.get_agent_run(agent_run_id=recovery_history_agent_run_id))
                recovery_history_checkpoint = (
                    await checkpoint_archive.latest_checkpoint(
                        agent_run_id=recovery_history_agent_run_id, include_interrupted=True,
                        **({"include_transcript": False} if current.restart_checkpoints else {}),
                    )
                )
                if recovery_history_checkpoint is None:
                    if recovery_run is not None:
                        raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
                    if (
                        self._tool_operations is not None
                        and await self._tool_operations.list_by_agent_run(
                            recovery_history_agent_run_id,
                            tenant_id=self._tenant_id, execution_id=execution_id,
                        )
                    ):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    # Attempt admission can commit before the first model checkpoint.
                    recovery_history_agent_run_id = None
                    exact_recovery_context = False
                elif self._tool_operations is not None:
                    unresolved = tuple(
                        operation
                        for operation in await self._tool_operations.list_by_agent_run(
                            recovery_history_agent_run_id,
                            tenant_id=self._tenant_id, execution_id=execution_id,
                        )
                        if operation.status
                        not in {
                            ToolOperationStatus.COMPLETED,
                            ToolOperationStatus.FAILED,
                        }
                    )
                    if unresolved:
                        try:
                            await self._reconcile_unresolved_tool_operations(
                                recovery_history_agent_run_id,
                                unresolved,
                                tenant_id=self._tenant_id,
                                execution=current, producer_generation=producer_generation,
                            )
                        except AIError as error:
                            if error.code is not ErrorCode.TOOL_EFFECT_UNKNOWN:
                                raise
                            persisted = await self.load_execution(
                                execution_id,
                                tenant_id=self._tenant_id,
                            )
                            if persisted is None:
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                            recovered = await self._check_reconcile_tool_effects(
                                persisted, producer_generation=producer_generation,
                            )
                            if recovered is not None:
                                if recovered.status is not ExecutionStatus.RECOVERY_REQUIRED:
                                    raise
                                metric_status = recovered.status.value
                                metric_error_code = recovered.error_code
                                return
            loaded_context = LoadedModelContext(())
            session_history_source = (
                current.lineage_kind is ExecutionLineageKind.SESSION_RESUME
                and history_id is not None
            )
            with runtime_object_source_reads():
                if source_replay_history is not None:
                    history = source_replay_history
                elif recovery_history_agent_run_id is not None:
                    output_retries_used = recovery_history_checkpoint.output_retries_used
                    if output_retries_used is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    context_messages = recovery_history_checkpoint.context_messages
                    if context_messages is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    loaded_context = LoadedModelContext(tuple(LoadedContextMessage(message, None) for message in context_messages))
                    history = list(loaded_context.model_messages())
                elif accepted_context is not None:
                    history = list(accepted_context.model_messages())
                    loaded_context = LoadedModelContext(tuple(LoadedContextMessage(message, None) for message in history))
                elif session_history_source:
                    loaded_context = await ConversationService.load_session_context(
                        execution=current, session=session, history_id=history_id,
                        tenant_id=self._tenant_id, run_store=self._run_store,
                        conversation_archive=self._run_stores[PersistenceDomain.CONVERSATION],
                    )
                    history = list(loaded_context.model_messages())
                else:
                    history = cast("list[ModelMessage]", await self._history(current))
            if resume is None and checkpoint.recovery.pending_tools is not None:
                resume = await self._recovery_coordinator.restore_resumed_deferred(
                    current, history=tuple(history),
                    producer_generation=producer_generation, binding=binding,
                    output_retries_used=output_retries_used,
                )
                deferred_tool_results = resume.results
                resumed_deferred_attempt = True
            session_history_start = (
                accepted_context.replace_history_system_prompt if accepted_context is not None
                else current.lineage_kind is ExecutionLineageKind.SESSION_RESUME and bool(history)
            )
            run_user_prompt = (
                None
                if resumed_deferred_attempt or recovery_history_agent_run_id is not None
                else request.user_prompt
            )

            async def sink(emission: "LiveDelta | DurableBoundary") -> None:
                if isinstance(emission, LiveDelta):
                    self._live_broker.publish(
                        ExecutionDelta(
                            current.execution_id,
                            emission.event_type,
                            emission.content,
                            position=emission.position,
                        )
                    )
                    return
                await self._append_event(current, emission.event_type, emission.payload,
                    producer_generation=producer_generation)

            subagent_refs = binding.binding_contract.subagents
            subagent_available = (
                current.parent_execution_id is None
                and self._subagent_dispatcher is not None
                and bool(subagent_refs)
            )
            memory = None
            selected_memory = MemoryCapability.select_tools(
                compiled_agent.tool_policy if current.context_imported
                else tuple(tool.id for tool in compiled_agent.selected_tools
                           if tool_class_from_metadata(tool.contract.get("metadata")) == "memory")
            )
            if current.memory_scope is not None and selected_memory:
                if self._memory_store_factory is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                memory = self._memory_store_factory(
                    self._tenant_id,
                    current.execution_id,
                    current.memory_scope or "default",
                )
            prepared_context = None
            context_write = None
            if accepted_context is None:
                memory_capture = None
                reason = None
                if memory is not None:
                    if isinstance(memory, RuntimeMemoryStore):
                        memory_capture = await memory.capture()
                    else:
                        reason = "memory_store_has_no_consistent_capture"
                try:
                    preload_skills = await self._executor.capture_preload_skills(binding)
                except AIError as error:
                    failed = await self._commit_failure(current, error, producer_generation=producer_generation)
                    metric_status = failed.status.value
                    metric_error_code = failed.error_code
                    return
                accepted_context = ExecutionInputContext.from_messages(history, session_metadata=session_metadata,
                    memory=memory_capture,
                    repository_instructions=None if initial_repository_instructions is None else initial_repository_instructions.to_payload(),
                    replace_history_system_prompt=session_history_start, unavailable_reason=reason,
                    preload_skills=preload_skills)
                proposed_context = accepted_context
                if prepared_admission is not None:
                    if self._pending_turns.get(execution_id, tenant_id=self._tenant_id) is not None:
                        prepared_context = await self._prepare_input_context(accepted_context)
                    else:
                        prepared = await self._context_archive.prepare(accepted_context, history_root=loaded_context.archive_root, history_digest=loaded_context.archive_digest)
                        prepared_context = prepared.reference
                        context_write = (prepared.archive, prepared.records)
                else:
                    current, accepted_context = await self._persist_input_context(
                        current, accepted_context, loaded_context,
                    )
                    if accepted_context is not proposed_context and accepted_context.digest != proposed_context.digest:
                        history = list(accepted_context.model_messages())
                        loaded_context = LoadedModelContext(tuple(LoadedContextMessage(message, None) for message in history))
                        session_history_start = accepted_context.replace_history_system_prompt
            if prepared_admission is not None:
                if prepared_context is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                try:
                    pending = self._pending_turns.get(execution_id, tenant_id=self._tenant_id)
                    if pending is None:
                        if context_write is None:
                            prepared = await self._context_archive.prepare(accepted_context, history_root=loaded_context.archive_root, history_digest=loaded_context.archive_digest)
                            prepared_context = prepared.reference
                            context_write = (prepared.archive, prepared.records)
                        activated = await self._runtime_commands.commit_prepared_agent_attempt_checkpoint(
                            prepared_admission, prepared_context, context_write=context_write,
                            observed=None if owner is None else owner.observed,
                        )
                    else:
                        async with pending.boundary:
                            if self._pending_turns.get(execution_id, tenant_id=self._tenant_id) is pending:
                                current = pending.activate(
                                    agent_run_id, prepared_context, now=datetime.now(timezone.utc),
                                )
                                pending.history_root = loaded_context.archive_root
                                pending.history_digest = loaded_context.archive_digest
                                if self._execution_buffer is not None:
                                    await self._execution_buffer.append(_ExecutionPayloadCapture(
                                        current, pending.reservation,
                                        audit_events=tuple(self._pending_audit_events.get(execution_id, ())),
                                    ))
                                assert self._pending_turns.coordination is not None and pending.coordination is not None
                                await self._pending_turns.coordination.publish(
                                    current, pending.coordination,
                                    request_identity_version=pending.reservation.idempotency.request_identity_version,
                                )
                            else:
                                if context_write is None:
                                    prepared = await self._context_archive.prepare(accepted_context, history_root=loaded_context.archive_root, history_digest=loaded_context.archive_digest)
                                    prepared_context = prepared.reference
                                    context_write = (prepared.archive, prepared.records)
                                activated = await self._runtime_commands.commit_prepared_agent_attempt_checkpoint(
                                    prepared_admission, prepared_context, context_write=context_write,
                            observed=None if owner is None else owner.observed,
                                )
                except AIError as error:
                    if error.code is not ErrorCode.STORAGE_CONFLICT:
                        raise
                    stopped = await self._read_stopped_admission(execution_id)
                    if stopped is not None:
                        metric_status = stopped.status.value
                        metric_error_code = stopped.error_code
                        return
                    raise
                if activated is not None:
                    current = activated.value
                    if owner is not None:
                        owner.observed = activated
                checkpoint = current
                producer_generation = current.producer_generation
                _logger.info(
                    "agent attempt activated: execution=%s sequence=%s",
                    execution_id,
                    current.agent_run_seq,
                )
            owner = self._turn_owners.get(execution_id)
            if owner is not None:
                owner.producer_generation = producer_generation
                owner.attempt = _InitialAttemptCapture(current, session)
            pending = self._pending_turns.get(execution_id, tenant_id=self._tenant_id)
            if pending is not None and pending.coordination is not None:
                pending.require_active(producer_generation, allow_cancelling=True)
                assert pending.execution.producer_claim_id is not None
                self._live_broker.admit_shared_producer(
                    execution_id, producer_generation, pending.execution.producer_claim_id,
                )
            if (self._live_broker.shared is not None
                    and self._pending_turns.get(execution_id, tenant_id=self._tenant_id) is None):
                shared_execution = await self._execution.executions.get(
                    execution_id, tenant_id=self._tenant_id,
                )
                if (shared_execution is not None
                        and shared_execution.producer_generation == producer_generation
                        and shared_execution.producer_claim_id is not None):
                    self._live_broker.admit_shared_producer(
                        execution_id, producer_generation, shared_execution.producer_claim_id,
                    )
            if self._execution_buffer is not None:
                self._execution_buffer.bind_producer(execution_id, producer_generation)
            if isinstance(self._run_store, RuntimeAgentRunStore):
                self._run_store.bind_execution_producer(
                    agent_run_id,
                    execution_id=execution_id,
                    producer_generation=producer_generation,
                    conversation=current.conversation,
                    restart_checkpoints=current.restart_checkpoints,
                    durable_trace=current.durable_trace,
                )
                if current.durable_trace and current.agent_run_seq > 1:
                    trace_archive = self._run_stores[PersistenceDomain.TRACE]
                    if isinstance(trace_archive, StateStepArchive):
                        # A fresh native run does not reopen its predecessor's
                        # Trace. Settle only requests fenced by this producer.
                        for sequence in range(1, current.agent_run_seq):
                            await trace_archive.interrupt_model_interactions(
                                agent_run_id=make_agent_run_id(
                                    namespace=self._namespace, tenant_id=self._tenant_id,
                                    execution_id=execution_id, agent_run_seq=sequence,
                                ),
                                execution_id=execution_id,
                                producer_generation=producer_generation,
                            )
                self._run_store.register_context_baseline(
                    agent_run_id, loaded_context,
                    source_run=(recovery_run if current.restart_checkpoints and recovery_history_agent_run_id is not None else None),
                )
                if owner is not None:
                    await self._bind_native_boundary(owner, current, agent_run_id)
            repository_boundary = _RepositoryInstructionBoundary(
                self._recovery_coordinator,
                current,
                initial_repository_instructions,
                repository_overlay,
                producer_generation=producer_generation,
                approved_frontiers=(None if resume is None else resume.repository_instruction_approvals),
            )
            if current.producer_generation != producer_generation or current.producer_claim_id is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            tool_repository = self._tool_operations
            pending_tools = self._pending_turns.get(execution_id, tenant_id=self._tenant_id)
            tool_operations = (
                RuntimeToolOperationBridge(
                    tool_repository,
                    self._checkpoint_objects,
                    namespace=self._namespace,
                    tenant_id=self._tenant_id,
                    execution_id=execution_id,
                    agent_run_id=agent_run_id,
                    binding_digest=current.binding_digest,
                    owner=current.producer_claim_id,
                    background_tasks=execution_tasks,
                    payload_policy=self._payload_policy,
                    recovery_agent_run_id=recovery_history_agent_run_id,
                    fresh_execution=original.agent_run_seq == 0 and current.agent_run_seq == 1,
                    producer_generation=producer_generation,
                    terminal_commands=self._runtime_commands,
                    budget_scope_id=current.budget_scope_id,
                    pending_tools=pending_tools,
                    before_effect=lambda: self.promote_pending_execution(execution_id),
                    require_active=lambda: self.require_tool_authority(execution_id, agent_run_id),
                    require_fresh_control=lambda: self.require_tool_authority(execution_id, agent_run_id, authoritative=True),
                )
                if tool_repository is not None
                else None
            )
            if (tool_operations is not None and owner is not None
                    and owner.tool_admission is not None and owner.tool_terminal is not None):
                tool_operations.prepare_boundaries(admission=owner.tool_admission, terminal=owner.tool_terminal,
                                                   unknown=owner.tool_unknown)

            if current.context_imported:
                session_metadata = accepted_context.session_metadata
                if memory is not None:
                    if not isinstance(memory, RuntimeMemoryStore):
                        raise AIError(ErrorCode.INPUT_CONTEXT_UNAVAILABLE, safe_details={"reason": "memory_store_has_no_isolated_capture"})
                    await memory.seed_capture(accepted_context.memory or {})
            original_input = current.stored_user_input
            if original_input.codec == "text" and original_input.payload.kind == "object":
                prompt = await self.restore_user_input(current)
                if not isinstance(prompt, str):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                original_input = replace(original_input, payload=StoredPayload.inline_text(prompt))
            public_context = AgentContext(
                app=self._app,
                principal=request.principal,
                namespace=self._namespace,
                session_id=current.session_id,
                execution_id=current.execution_id,
                session_metadata=session_metadata,
                memory_scope=current.memory_scope,
                correlation=current.correlation,
                deadline_at=None if request.budget is None else request.budget.deadline_at,
                root_execution_id=current.root_execution_id,
                parent_execution_id=current.parent_execution_id,
                agent_id=binding.compiled_agent.spec.id,
                agent_run_seq=current.agent_run_seq,
                input_context=accepted_context,
                original_input=stored_user_input_view(original_input),
            )
            plan_store = RuntimePlanStore(
                self._session_state_store
                if current.session_id is not None
                else self._execution_state_store,
                namespace=self._namespace,
                tenant_id=self._tenant_id,
                owner_kind="session" if current.session_id is not None else "execution",
                owner_id=current.session_id or current.execution_id,
            )
            deferred_capability_refs: dict[str, CapabilityRef] = {}
            try:
                result = await self._executor.execute(
                    _AgentRunScope(
                        binding=binding,
                        context=public_context,
                        workspace=self._workspace,
                        limits=self._limits,
                        execution_cwd=self._execution_cwd,
                        user_prompt=run_user_prompt,
                        initial_attachments=stored_input_attachment_views(
                            current.stored_user_input
                        ),
                        history=history,
                        output_retries_used=output_retries_used,
                        initial_context=loaded_context,
                        agent_conversation_id=agent_conversation_id,
                        run_store=self._run_store,
                        agent_run_id=agent_run_id,
                        agent_run_seq=current.agent_run_seq,
                        history_producer_generation=producer_generation,
                        history_id=history_id,
                        memory_store=memory if selected_memory else None,
                        memory_tool_names=selected_memory if memory is not None else (),
                        plan_store_resolver=lambda _ctx: plan_store,
                        mode=current.mode,
                        planning=current.planning,
                        thinking=current.thinking,
                        parent_agent_run_id=None,
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
                                parent_producer_generation=producer_generation,
                                root_execution_id=current.root_execution_id,
                                memory_scope=current.memory_scope,
                                principal=request.principal,
                                refs=subagent_refs,
                                binding=binding.binding_contract,
                                mode=current.mode,
                                require_frozen_bindings=self._execution_objects_durable,
                            )
                        ),
                        event_sink=sink,
                        usage_sink=lambda usage: self._capture_usage(
                            execution_id, usage
                        ),
                        tool_operations=tool_operations,
                        tool_response_ref=current.tool_response_ref,
                        tool_effects=current.tool_effects,
                        budget=(None if current.budget_scope_id is None else RunBudgetContext(
                            self._execution.budgets, current.budget_scope_id,
                            current.execution_id, agent_run_id,
                        )),
                        replace_history_system_prompt=(
                            session_history_start and not exact_recovery_context
                        ),
                        repository_instructions=initial_repository_instructions,
                        repository_instruction_boundary=repository_boundary,
                        deferred_tool_results=deferred_tool_results,
                        deferred_capability_refs=deferred_capability_refs,
                        deferred_approvals=(
                            () if checkpoint.recovery.pending_tools is None
                            else checkpoint.recovery.pending_tools.approvals
                        ),
                    )
                )
            except Exception as error:
                if isinstance(error, AIError) and error.code is ErrorCode.EXECUTION_CANCELLED:
                    raise asyncio.CancelledError from error
                current = await self.load_execution(
                    execution_id,
                    tenant_id=self._tenant_id,
                )
                if current is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                terminal = current.status in {
                    ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
                }
                capability_writes = (self._run_store.capability_writes
                                     if isinstance(self._run_store, RuntimeAgentRunStore) else None)
                rejected_capability_write = (
                    agent_run_id is not None and capability_writes is not None
                    and capability_writes.is_buffer_rejection(
                        agent_run_id, execution_id=execution_id, error=error,
                    )
                )
                rejected_execution_payload = (
                    self._execution_buffer is not None
                    and self._pending_turns.get(execution_id, tenant_id=self._tenant_id) is None
                    and await self._execution_buffer.settle_rejection(execution_id, error)
                )
                if _is_infrastructure_error(error) and not (
                    rejected_execution_payload or rejected_capability_write
                    or isinstance(error, AIError) and error.code is ErrorCode.STORAGE_CONFLICT and terminal
                ):
                    if (terminal or agent_run_id is None or producer_generation is None
                            or not await self._agent_run_lifecycle.settle_trace_buffer_failure(
                                agent_run_id, execution_id=execution_id,
                                producer_generation=producer_generation, error=error,
                            )):
                        raise
                if terminal and producer_generation is not None:
                    await self._discard_terminal_producer_tail(
                        current, agent_run_id=agent_run_id, producer_generation=producer_generation,
                    )
                    await self._publish_persisted_terminal_event(current)
                    metric_status = current.status.value
                    metric_error_code = current.error_code
                    return
                if current.status not in {
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                }:
                    try:
                        committed_failure = await self._commit_failure(
                            current,
                            error,
                            agent_run_id=agent_run_id,
                            producer_generation=producer_generation,
                        )
                        current = committed_failure
                    except asyncio.CancelledError:
                        raise
                    except Exception as commit_error:
                        try:
                            persisted = await self.load_execution(
                                execution_id,
                                tenant_id=self._tenant_id,
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
                if (isinstance(self._run_store, RuntimeAgentRunStore)
                        and current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}):
                    self._run_store.release_step_event_receipt(agent_run_id)
                metric_status = current.status.value
                metric_error_code = current.error_code
                if current.status is ExecutionStatus.RECOVERY_REQUIRED:
                    if (revoked := await self._finish_revoked_worker(
                        execution_id, agent_run_id=agent_run_id,
                        producer_generation=producer_generation, admission=original,
                    )) is not None:
                        metric_status = (
                            "SUPERSEDED" if producer_generation is None or revoked.producer_generation > producer_generation
                            else revoked.status.value
                        )
                        metric_error_code = None if metric_status == "SUPERSEDED" else revoked.error_code
                        return
                    _logger.warning(
                        "local execution requires recovery: execution=%s code=%s safe_error_details=%s",
                        execution_id,
                        current.error_code,
                        current.safe_error_details,
                        exc_info=True,
                    )
                else:
                    _logger.error(
                        "local execution failed: execution=%s code=%s safe_error_details=%s",
                        execution_id,
                        current.error_code,
                        current.safe_error_details,
                        exc_info=True,
                    )
                return
            if isinstance(result, DeferredToolRequests):
                if owner is not None:
                    owner.attempt = None
                if checkpoint is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self.promote_pending_execution(execution_id)
                try:
                    deferred = await self._recovery_coordinator.commit_deferred_pause(
                        current,
                        result,
                        agent_run_id=agent_run_id,
                        paused_at=datetime.now(timezone.utc),
                        capability_refs=deferred_capability_refs,
                        producer_generation=producer_generation,
                        repository_instruction_frontiers=repository_boundary.approval_frontiers(result.approvals),
                    )
                except AIError as error:
                    if error.code is ErrorCode.EXECUTION_CANCELLED:
                        raise asyncio.CancelledError from error
                    raise
                if deferred.status in {
                    ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
                }:
                    await self._discard_terminal_producer_tail(
                        deferred, agent_run_id=agent_run_id, producer_generation=producer_generation,
                    )
                    await self._publish_persisted_terminal_event(deferred)
                if isinstance(self._run_store, RuntimeAgentRunStore):
                    self._run_store.release_step_event_receipt(agent_run_id)
                metric_status = deferred.status.value
                metric_error_code = deferred.error_code
                return
            try:
                committed = await self._commit_success(
                    current,
                    binding,
                    result.output,
                    result.usage,
                    agent_run_id,
                    producer_generation=producer_generation,
                    tool_operations=tool_operations,
                )
            except asyncio.CancelledError:
                raise
            except Exception as terminal_error:
                if self._pending_turns.get(execution_id, tenant_id=self._tenant_id) is not None:
                    # Pending commits have no durable producer to supersede. Their
                    # own receipt decides whether to retain or release admission.
                    raise
                if (isinstance(terminal_error, AIError) and terminal_error.code is ErrorCode.STORAGE_CONFLICT
                        and (revoked := await self._finish_revoked_worker(
                            execution_id, agent_run_id=agent_run_id,
                            producer_generation=producer_generation, admission=original,
                        )) is not None):
                    metric_status = (
                        "SUPERSEDED" if producer_generation is None or revoked.producer_generation > producer_generation
                        else revoked.status.value
                    )
                    metric_error_code = None if metric_status == "SUPERSEDED" else revoked.error_code
                    return
                persisted = await self.load_execution(
                    execution_id,
                    tenant_id=self._tenant_id,
                )
                if persisted is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from terminal_error
                publish_readback = False
                if persisted.status in {
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                }:
                    committed = persisted
                    publish_readback = True
                else:
                    try:
                        if (self._execution_buffer is not None
                                and self._pending_turns.get(execution_id, tenant_id=self._tenant_id) is None):
                            await self._execution_buffer.settle_rejection(execution_id, terminal_error)
                        if producer_generation is not None:
                            await self._agent_run_lifecycle.settle_trace_buffer_failure(
                                agent_run_id, execution_id=execution_id,
                                producer_generation=producer_generation, error=terminal_error,
                            )
                        committed = await self._commit_failure(
                            persisted,
                            terminal_error,
                            agent_run_id=agent_run_id,
                            producer_generation=producer_generation,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as commit_error:
                        observed = await self.load_execution(
                            execution_id,
                            tenant_id=self._tenant_id,
                        )
                        if observed is None or observed.status not in {
                            ExecutionStatus.SUCCEEDED,
                            ExecutionStatus.FAILED,
                            ExecutionStatus.CANCELLED,
                        }:
                            raise _secondary_execution_error(
                                commit_error,
                                terminal_error,
                            ) from terminal_error
                        committed = observed
                        publish_readback = True
                if publish_readback:
                    if committed.conversation and committed.session_id is not None:
                        retained_result = await self._execution.executions.get_result(
                            execution_id, tenant_id=self._tenant_id,
                        )
                        if retained_result is None:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from terminal_error
                        try:
                            await self._runtime_commands.verify_conversation_turn(committed, retained_result)
                        except AIError as publication_error:
                            raise terminal_error from publication_error
                    await self._publish_persisted_terminal_event(committed)
            metric_status = (
                "CANCELLED"
                if committed.status is ExecutionStatus.CANCELLING
                else committed.status.value
            )
            metric_error_code = committed.error_code
            if committed.status in {
                ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
            } and producer_generation is not None:
                await self._discard_terminal_producer_tail(
                    committed, agent_run_id=agent_run_id, producer_generation=producer_generation,
                )
            if committed.status is ExecutionStatus.RECOVERY_REQUIRED:
                if (revoked := await self._finish_revoked_worker(
                    execution_id, agent_run_id=agent_run_id,
                    producer_generation=producer_generation, admission=original,
                )) is not None:
                    metric_status = (
                        "SUPERSEDED" if producer_generation is None or revoked.producer_generation > producer_generation
                        else revoked.status.value
                    )
                    metric_error_code = None if metric_status == "SUPERSEDED" else revoked.error_code
                    return
                _logger.warning(
                    "local execution requires recovery: execution=%s code=%s safe_error_details=%s",
                    execution_id,
                    committed.error_code,
                    committed.safe_error_details,
                )
            else:
                _logger.debug(
                    "local execution completed: execution=%s agent_run=%s",
                    execution_id,
                    agent_run_id,
                )
        except asyncio.CancelledError as error:
            cleanup_details: dict[str, JsonValue] = {}
            if (
                isinstance(error.__cause__, AIError)
                and error.__cause__.code is ErrorCode.SANDBOX_CLEANUP_FAILED
            ):
                cleanup_details["secondary_error_code"] = (
                    ErrorCode.SANDBOX_CLEANUP_FAILED.value
                )
            if (not cleanup_details and producer_generation is not None
                    and (revoked := await self._finish_revoked_worker(
                        execution_id, agent_run_id=agent_run_id,
                        producer_generation=producer_generation, admission=None,
                    )) is not None):
                metric_status = (
                    "SUPERSEDED" if producer_generation is None or revoked.producer_generation > producer_generation
                    else revoked.status.value
                )
                metric_error_code = None if metric_status == "SUPERSEDED" else revoked.error_code
                return
            current = await self.load_execution(
                execution_id,
                tenant_id=self._tenant_id,
            )
            if (
                producer_generation is None
                and current is not None
                and current.agent_run_seq > 0
            ):
                # An interrupted admission has not established this worker's
                # authority over whichever attempt won the durable claim.
                raise
            if current is not None and current.status is ExecutionStatus.FINALIZING:
                metric_status = "CANCELLED"
                raise
            if (handoff_scope.stopping and not cleanup_details
                    and current is not None and current.status is ExecutionStatus.STARTED):
                effects, active_claims = await self._reconcile_tool_effects(
                    execution_id, tenant_id=self._tenant_id,
                )
                if active_claims:
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED,
                        safe_details={"phase": "runtime_handoff", "active_tool_claims": active_claims})
                if effects:
                    current = await self._commit_recovery_required(
                        current, AIError(ErrorCode.TOOL_EFFECT_UNKNOWN), effects,
                        producer_generation=producer_generation,
                    )
                elif not current.restart_checkpoints:
                    current = await self._commit_failure(
                        current, AIError(ErrorCode.STORAGE_CONFLICT, safe_details={
                            "phase": "turn_owner_stopped", "restart_checkpoints": False,
                        }), agent_run_id=agent_run_id, producer_generation=producer_generation,
                    )
                if current.status is ExecutionStatus.STARTED:
                    self._agent_run_only_worker_exits.add(execution_id)
                    self._live_broker.handoff(execution_id)
                metric_status = current.status.value
                metric_error_code = current.error_code
                return
            if current is not None and current.status is ExecutionStatus.CANCELLING:
                self._worker_shutdown_requests.discard(execution_id)
                current = await self._commit_terminal(
                    current,
                    ExecutionStatus.CANCELLED,
                    None,
                    ErrorCode.EXECUTION_CANCELLED.value,
                    StopReason.CANCELLED,
                    agent_run_id=agent_run_id,
                    producer_generation=producer_generation,
                    safe_error_details=cleanup_details,
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
                    agent_run_id=agent_run_id,
                    producer_generation=producer_generation,
                    safe_error_details=cleanup_details,
                )
            if (
                current is not None
                and current.status is ExecutionStatus.RECOVERY_REQUIRED
            ):
                metric_status = current.status.value
                metric_error_code = current.error_code
            elif current is not None and current.status in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }:
                if producer_generation is not None and agent_run_id is not None:
                    await self._discard_terminal_producer_tail(
                        current, agent_run_id=agent_run_id, producer_generation=producer_generation,
                    )
                metric_status = current.status.value
                metric_error_code = current.error_code
            else:
                metric_status = "CANCELLED"
            raise
        except Exception as error:
            if (isinstance(error, AIError) and error.code is ErrorCode.STORAGE_CONFLICT
                    and (revoked := await self._finish_revoked_worker(
                        execution_id, agent_run_id=agent_run_id,
                        producer_generation=producer_generation, admission=original,
                    )) is not None):
                metric_status = (
                    "SUPERSEDED" if producer_generation is None or revoked.producer_generation > producer_generation
                    else revoked.status.value
                )
                metric_error_code = None if metric_status == "SUPERSEDED" else revoked.error_code
                return
            metric_status = "FAILED"
            metric_error_code = _execution_error_code(error).value
            _logger.exception(
                "local execution infrastructure failure: execution=%s",
                execution_id,
            )
            raise
        finally:
            self._handoff_scopes.pop(execution_id, None)
            await record_storage_operation(
                metric_recorder,
                observation_id=metric_id,
                started_at_ns=metric_started,
                source_namespace=self._namespace,
                tenant_id=self._tenant_id,
                execution_id=execution_id,
                session_id=original.session_id,
                correlation=original.correlation,
                status=metric_status,
                error_code=metric_error_code,
                domain="execution",
                target="runtime",
            )


    async def _history(self, execution: ExecutionRecord) -> list[object]:
        session = None if execution.session_id is None else await self._load_session(
            execution.session_id, tenant_id=self._tenant_id,
        )
        return await ConversationService.load_history(
            execution, session=session, namespace=self._namespace, tenant_id=self._tenant_id,
            run_store=self._run_store,
            execution_run_store=self._get_run_store(PersistenceDomain.TRACE),
            conversation_run_store=self._get_run_store(PersistenceDomain.CONVERSATION),
            checkpoint_archive=self._get_run_store(PersistenceDomain.CHECKPOINT),
            load_execution=lambda execution_id: self.load_execution(execution_id, tenant_id=self._tenant_id),
        )

    async def _append_event(
        self,
        execution: ExecutionRecord,
        event_type: ExecutionEventType,
        payload: JsonValue,
        *, producer_generation: int | None = None,
    ) -> None:
        normalized = normalize_json_value(payload)
        event_payload = normalized if isinstance(normalized, Mapping) else {"value": normalized}
        async with self._audit_lock(execution.execution_id):
            pending_events = self._pending_audit_events.setdefault(execution.execution_id, [])
            event = ExecutionEventAppend(event_type, event_payload)
            turn = self._pending_turns.get(execution.execution_id, tenant_id=self._tenant_id)
            if self._execution_buffer is not None:
                if producer_generation is not None:
                    self._execution_buffer.require_producer(execution.execution_id, producer_generation)
                previous = None
                if self._execution_buffer.has_scope(execution.execution_id):
                    previous, _ids = await self._execution_buffer.source_prefix(execution.execution_id)
                capture_execution = turn.execution if turn is not None else execution
                reservation = turn.reservation if turn is not None else None
                if previous is not None:
                    capture_execution = replace(
                        capture_execution, stored_user_input=previous.execution.stored_user_input,
                        repository_instructions=(None if previous.execution.input_context is not None
                            or capture_execution.input_context is not None else previous.execution.repository_instructions),
                        input_context=previous.execution.input_context or capture_execution.input_context,
                    )
                    reservation = previous.reservation
                    if previous.terminal is not None:
                        capture_execution = previous.execution
                await self._execution_buffer.append(_ExecutionPayloadCapture(
                    capture_execution, reservation,
                    terminal=None if previous is None else previous.terminal,
                    audit_events=(*(previous.audit_events if previous is not None else ()), event),
                    checkpoint_payload=None if previous is None else previous.checkpoint_payload,
                ))
            pending_events.append(event)
            self._live_broker.publish_event(
                execution.execution_id,
                event_type,
                event_payload,
                durable_seq=None,
            )
        _logger.debug(
            "execution audit event buffered: execution=%s type=%s pending=%s",
            execution.execution_id,
            str(event_type),
            len(self._pending_audit_events.get(execution.execution_id, ())),
        )

    async def _source_audit_prefix(self, execution_id: str) -> tuple[ExecutionEventAppend, ...]:
        if self._execution_buffer is None or not self._execution_buffer.has_scope(execution_id):
            return tuple(self._pending_audit_events.get(execution_id, ()))
        if self._execution_buffer.settled(execution_id):
            return ()
        captured, _ids = await self._execution_buffer.source_prefix(execution_id)
        start = self._execution_buffer.audit_position(execution_id)
        if captured is None:
            if start:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return ()
        if start > len(captured.audit_events):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return captured.audit_events[start:]

    def _confirm_source_audits(self, execution_id: str, count: int) -> None:
        if self._execution_buffer is not None and self._execution_buffer.has_scope(execution_id):
            self._execution_buffer.confirm_audits(
                execution_id, start=self._execution_buffer.audit_position(execution_id), count=count,
            )
        tail = self._pending_audit_events.get(execution_id)
        if tail is not None:
            del tail[:count]
            if not tail:
                self._pending_audit_events.pop(execution_id, None)

    async def _prepare_source_terminal_payload(
        self, current: ExecutionRecord, commit: ExecutionTerminalCommit, *,
        checkpoint_payload: bytes | None = None,
    ) -> tuple[ExecutionTerminalCommit, bytes | None, tuple[str, ...]]:
        buffer = self._execution_buffer
        if buffer is None:
            return commit, checkpoint_payload, ()
        previous = None
        ids: tuple[str, ...] = ()
        if buffer.has_scope(current.execution_id):
            previous, ids = await buffer.source_prefix(current.execution_id)
        if buffer.rejection_settled(current.execution_id):
            if commit.execution.status not in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return commit, None if previous is None else previous.checkpoint_payload, ids
        logical_execution = commit.execution
        if previous is not None:
            logical_execution = replace(
                logical_execution, stored_user_input=previous.execution.stored_user_input,
                repository_instructions=(None if previous.execution.input_context is not None
                    or logical_execution.input_context is not None else previous.execution.repository_instructions),
                input_context=previous.execution.input_context or logical_execution.input_context,
            )
        logical_commit = replace(commit, execution=logical_execution)
        if (previous is not None and previous.terminal is not None
                and previous.terminal.result.output is not None and commit.result.output is not None
                and previous.terminal.result.output.digest == commit.result.output.digest):
            logical_commit = replace(logical_commit,
                result=replace(logical_commit.result, output=previous.terminal.result.output))
        try:
            await buffer.append(_ExecutionPayloadCapture(
                logical_execution, None if previous is None else previous.reservation,
                terminal=logical_commit, audit_events=() if previous is None else previous.audit_events,
                checkpoint_payload=checkpoint_payload,
            ))
        except AIError as error:
            if (commit.execution.status not in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}
                    or not await buffer.settle_rejection(current.execution_id, error)):
                raise
            # Only definite rejection of this owner's candidate may settle the
            # necessary control outcome from its previously accepted prefix.
            return await self._prepare_source_terminal_payload(current, commit)
        selected, ids = await buffer.read(current.execution_id)
        if selected.terminal is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        # Source-owned admission fields may already be materialized (promotion).
        # Newly produced result bytes always come from the selected buffer entry.
        selected_commit = replace(selected.terminal, execution=replace(
            selected.execution, stored_user_input=current.stored_user_input,
            repository_instructions=current.repository_instructions, input_context=current.input_context,
        ))
        output = selected_commit.result.output
        if output is not None:
            output = await _materialize_execution_json(
                output, objects=self._execution_objects, keys=ObjectKeyFactory(self._namespace),
                policy=self._payload_policy, tenant_id=self._tenant_id,
            )
            selected_commit = replace(selected_commit, result=replace(selected_commit.result, output=output))
        return selected_commit, selected.checkpoint_payload, ids

    def _audit_lock(self, execution_id: str) -> asyncio.Lock:
        return self._pending_audit_locks.setdefault(execution_id, asyncio.Lock())

    def _confirm_committed_events(
        self,
        execution_id: str,
        *,
        pending_count: int,
        durable_seq: int,
    ) -> None:
        if pending_count:
            self._live_broker.confirm_events(
                execution_id,
                first_event_seq=durable_seq - pending_count,
                count=pending_count,
            )

    def _publish_terminal_event(
        self,
        execution_id: str,
        *,
        event_type: ExecutionEventType,
        payload: JsonValue,
        durable_seq: int,
    ) -> None:
        self._live_broker.publish_event(
            execution_id,
            event_type,
            payload,
            durable_seq=durable_seq,
        )

    async def _publish_persisted_terminal_event(
        self,
        execution: ExecutionRecord,
    ) -> None:
        if execution.status not in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        } or execution.event_seq < 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        page = await self._execution.events.list(
            execution.execution_id,
            tenant_id=self._tenant_id,
            after_event_seq=execution.event_seq - 1,
            limit=1,
        )
        if len(page.items) != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        event = page.items[0]
        expected_type = (
            ExecutionEventType.EXECUTION_SUCCEEDED
            if execution.status is ExecutionStatus.SUCCEEDED
            else ExecutionEventType.EXECUTION_CANCELLED
            if execution.status is ExecutionStatus.CANCELLED
            else ExecutionEventType.EXECUTION_FAILED
        )
        if (
            event.event_seq != execution.event_seq
            or event.event_type != expected_type
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._publish_terminal_event(
            execution.execution_id,
            event_type=expected_type,
            payload=dict(event.payload),
            durable_seq=event.event_seq,
        )
        self._live_broker.complete(execution.execution_id)

    def _get_run_store(self, runtime_domain: PersistenceDomain) -> AgentRunStore:
        return self._run_stores[runtime_domain]

    async def _discard_terminal_producer_tail(
        self, execution: ExecutionRecord, *, agent_run_id: str, producer_generation: int,
    ) -> None:
        await self._agent_run_lifecycle.discard_revoked_producer_staging(
            agent_run_id, execution_id=execution.execution_id, producer_generation=producer_generation,
            terminal_execution=execution,
            terminal_publication=(None if self._retained_publication is None else
                self._retained_publication(execution.execution_id, tenant_id=self._tenant_id)),
        )
        async with self._audit_lock(execution.execution_id):
            self._pending_audit_events.pop(execution.execution_id, None)
            if self._execution_buffer is not None:
                self._execution_buffer.discard(execution.execution_id)

    async def _finish_revoked_worker(
        self, execution_id: str, *, agent_run_id: str | None,
        producer_generation: int | None, admission: ExecutionRecord | None,
    ) -> ExecutionRecord | None:
        if self._pending_turns.get(execution_id, tenant_id=self._tenant_id) is not None:
            return None
        observed = await self._execution.executions.get_observed(
            execution_id, tenant_id=self._tenant_id,
        )
        current = None if observed is None else observed.value
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if producer_generation is None:
            if admission is None or admission.agent_run_seq != 0:
                return None
            checkpoint = await self._execution.executions.get_recovery(execution_id, tenant_id=self._tenant_id)
            if (checkpoint is None
                    or current.agent_run_seq <= admission.agent_run_seq
                    or current.producer_generation <= admission.revision):
                return None
        elif not (
            current.producer_generation > producer_generation
            or current.producer_generation == producer_generation
            and current.status is ExecutionStatus.RECOVERY_REQUIRED
        ):
            return None
        pending = tuple(
            task for task in self._execution_task_set(execution_id)
            if task is not asyncio.current_task() and not task.done()
        )
        if pending:
            await asyncio.gather(*(asyncio.shield(task) for task in pending))
        if producer_generation is not None and agent_run_id is not None:
            await self._agent_run_lifecycle.discard_revoked_producer_staging(
                agent_run_id, execution_id=execution_id, producer_generation=producer_generation,
            )
        async with self._audit_lock(execution_id):
            self._pending_audit_events.pop(execution_id, None)
            if self._execution_buffer is not None and producer_generation is not None:
                self._execution_buffer.retire(execution_id, producer_generation, abandon=True)
        self._agent_run_only_worker_exits.add(execution_id)
        self._live_broker.handoff(execution_id)
        _logger.info("revoked execution producer stopped: execution=%s generation=%s source_status=%s",
                     execution_id, producer_generation, current.status.value)
        return current

    async def _commit_success(
        self,
        execution: ExecutionRecord,
        binding: AgentBinding,
        output: JsonValue,
        usage: UsageMetrics,
        agent_run_id: str,
        *,
        producer_generation: int | None = None,
        tool_operations: RuntimeToolOperationBridge | None = None,
    ) -> ExecutionRecord:
        owner = self._turn_owners.get(execution.execution_id)
        capture = self._owned_attempt_capture(execution)
        observed = None if owner is None else owner.observed
        current = (observed.value if owner is not None and not owner.stopping
                   and owner.tool_admission is not None and owner.tool_terminal is not None
                   and producer_generation is not None and owner.producer_generation == producer_generation
                   and capture is not None and observed is not None
                   and observed.value == capture.execution else None)
        if current is None:
            current = await self.load_execution(execution.execution_id, tenant_id=self._tenant_id)
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if current.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            if producer_generation is not None:
                await self._discard_terminal_producer_tail(
                    current, agent_run_id=agent_run_id, producer_generation=producer_generation,
                )
            return current
        if current.status is ExecutionStatus.RECOVERY_REQUIRED:
            return current
        output_payload = StoredPayload.inline_json(output)
        return await self._commit_terminal(
            current,
            ExecutionStatus.SUCCEEDED,
            output_payload,
            None,
            StopReason.END_TURN,
            binding=binding,
            agent_run_id=agent_run_id,
            usage=usage,
            producer_generation=producer_generation,
            # The terminal CAS validates this confirmed boundary or source read.
            observed_execution=current,
            tool_operations=tool_operations,
        )

    async def _bind_native_boundary(
        self, owner: _OwnedExecutionTurn, execution: ExecutionRecord, agent_run_id: str,
    ) -> None:
        if not execution.restart_checkpoints or not isinstance(self._run_store, RuntimeAgentRunStore):
            return
        native = self._run_store.step_store
        state = self._execution.executions.state_store
        context = getattr(state.storage_group, "context", None)
        if (not isinstance(native, SqlStepStore) or not isinstance(context, SqlStorageContext)
                or native.storage_context.engine is not context.engine):
            return
        if self._pending_turns.get(execution.execution_id, tenant_id=self._tenant_id) is not None:
            return
        repository = self._execution.executions
        if owner.observed is None or owner.observed.value != execution:
            owner.observed = await repository.get_observed(execution.execution_id, tenant_id=self._tenant_id)
        if owner.observed is None or owner.observed.value != execution:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        captured_session = None if owner.attempt is None else owner.attempt.session
        generation, claim_id = execution.producer_generation, execution.producer_claim_id
        if generation is None or claim_id is None:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def verify_native(*, run, events, snapshot, receipt: StepWriteReceipt) -> bool:
            if run is not None:
                source = native.run_from_receipt(receipt)
                if source is None or await native.get_run(run_id=run.run_id) != source:
                    return False
            if events:
                proven = native.events_from_receipt(receipt)
                actual_events = (await native.list_events(run_id=events[0].run_id) if not proven else
                                 await native.list_events(run_id=events[0].run_id, started_receipt=receipt))
                expected_events = tuple(next((source for source in proven if source.idempotency_key == event.idempotency_key), event)
                                        for event in events)
                if any(event not in actual_events for event in expected_events):
                    return False
            if snapshot is not None:
                actual_snapshot = await native.latest_snapshot(run_id=snapshot.run_id, include_interrupted=True)
                if actual_snapshot != snapshot:
                    return False
            return True

        async def commit_boundary(*, run=None, events=(), snapshot=None) -> StepWriteReceipt | None:
            async with self._run_store.native_event_publication(agent_run_id):
                return await publish_boundary(run=run, events=events, snapshot=snapshot)

        async def publish_boundary(*, run=None, events=(), snapshot=None) -> StepWriteReceipt | None:
            if run is None and not events and snapshot is None:
                return
            terminal = any(isinstance(event, NativeLifecycleEvent) for event in events)

            def require_publication() -> None:
                if not terminal:
                    self._require_turn_dispatch(owner)
                elif self._turn_owners.get(execution.execution_id) is not owner:
                    raise asyncio.CancelledError

            managed = run is not None and self._run_store.has_pending_native_start(run.run_id)
            prepared = (await native.prepare_step(run=run, events=events, snapshot=snapshot, managed_start=True)
                        if managed else await native.prepare_step(run=run, events=events, snapshot=snapshot))
            async with owner.boundary:
                require_publication()
                if owner.observed is None or owner.attempt is None:
                    observed = await repository.get_observed(execution.execution_id, tenant_id=self._tenant_id)
                    if (observed is None or observed.value.producer_generation != generation
                            or observed.value.producer_claim_id != claim_id):
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                    owner.observed = observed
                observed = owner.observed
                next_record = observed.value
                written = None

                async def publish(group: StateGroupTransaction):
                    nonlocal written
                    transaction = group.transaction(state)
                    if not isinstance(transaction, SqlStateTransaction):
                        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                    receipt = await repository.commit_boundary_in_transaction(
                        transaction, observed, next_record=next_record,
                        expected_producer_generation=generation, expected_producer_claim_id=claim_id,
                        allow_cancelling=terminal,
                    )
                    source_receipt = (self._run_store.step_event_receipt(agent_run_id)
                                      if managed or terminal else None)
                    written = (await native.write_prepared_step(transaction, prepared, confirmed=source_receipt)
                               if source_receipt is not None else await native.write_prepared_step(transaction, prepared))
                    return receipt

                async def persist():
                    nonlocal observed, next_record, written
                    while True:
                        try:
                            written = None
                            return await state.storage_group.mutate((state,), publish)
                        except AIError as error:
                            actual = None
                            if error.code in {ErrorCode.STORAGE_COMMIT_UNKNOWN, ErrorCode.STORAGE_CONFLICT}:
                                actual = await repository.get_observed(execution.execution_id, tenant_id=self._tenant_id)
                            if error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
                                if (actual is not None and actual.value == next_record
                                        and actual.storage_revision == observed.storage_revision + 1
                                        and written is not None
                                        and await verify_native(run=run, events=events, snapshot=snapshot, receipt=written)):
                                    return actual
                            elif (error.code is ErrorCode.STORAGE_CONFLICT and written is None
                                    and actual is not None
                                    and actual.storage_revision > observed.storage_revision
                                    and (actual.value == observed.value or terminal
                                         and actual.value.status is ExecutionStatus.CANCELLING
                                         and actual.value.producer_generation == generation
                                         and actual.value.producer_claim_id == claim_id)):
                                # Cancellation stops new dispatch but must retain this
                                # producer's terminal source and interrupted checkpoint.
                                require_publication()
                                observed = actual
                                next_record = actual.value
                                owner.observed = actual
                                continue
                            owner.observed = None
                            if actual is not None and actual.value.status is ExecutionStatus.CANCELLING:
                                raise AIError(ErrorCode.EXECUTION_CANCELLED) from error
                            raise

                task = asyncio.create_task(persist(), name=f"ai-step-boundary-{execution.execution_id}")
                confirmed, cancelled = await self._await_checkpoint_task(
                    task, label="native step boundary", execution_id=execution.execution_id,
                )
                owner.observed = confirmed
                session = captured_session if owner.attempt is None else owner.attempt.session
                owner.attempt = _InitialAttemptCapture(confirmed.value, session)
                if written is None:
                    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
                native.confirm_prepared_step(written)
                source = native.run_from_receipt(written)
                if source is not None:
                    self._run_store.confirm_step_run(source, receipt=written)
                if cancelled is not None:
                    raise cancelled
                return written

        initial_observed = owner.observed

        async def event_writer(event):
            prepared = await native.prepare_step(events=(event,))
            async def write(group):
                await native.write_prepared_step(group.transaction(state), prepared)
            async def readback():
                return any(item == event for item in await native.list_events(run_id=event.run_id))
            return write, readback

        async def admit_tool(request, *, observed, event, known_new):
            write, readback = await event_writer(event)
            async with owner.boundary:
                self._require_turn_dispatch(owner)
                if owner.observed is None or owner.attempt is None:
                    owner.observed = await repository.get_observed(execution.execution_id, tenant_id=self._tenant_id)
                if owner.observed is None:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                receipt = await self._runtime_commands.commit_tool_admission_boundary(
                    request, execution=owner.observed, observed=observed,
                    write=write, readback_write=readback, known_new=known_new,
                )
                owner.observed = receipt.execution
                session = captured_session if owner.attempt is None else owner.attempt.session
                owner.attempt = _InitialAttemptCapture(receipt.execution.value, session)
                return receipt.operation

        async def settle_tool(operation_id, *, tenant_id, owner: str, fence, observed, event,
                              result_payload=None, error_code=None, error_payload=None):
            write, readback = await event_writer(event)
            async with turn_owner.boundary:
                # A known external result remains a fact after cancellation or
                # producer loss. Its own effect fence protects manual/new decisions.
                receipt = await self._runtime_commands.commit_tool_terminal_boundary(
                    operation_id, tenant_id=tenant_id, owner=owner, fence=fence,
                    execution=turn_owner.observed or initial_observed, observed=observed,
                    result_payload=result_payload, error_code=error_code, error_payload=error_payload,
                    write=write, readback_write=readback, stores=(state,),
                )
                return receipt.operation, False

        async def unknown_tool(operation_id, *, tenant_id, owner: str, fence, error_code, event):
            prepared = await native.prepare_step(events=(event,))
            written = None

            async def write(transaction):
                nonlocal written
                written = await native.write_prepared_step(transaction, prepared)

            result = await self._checkpoint.tools.mark_effect_unknown(
                operation_id, tenant_id=tenant_id, owner=owner, fence=fence,
                error_code=error_code, write=write,
            )
            if written is None:
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            native.confirm_prepared_step(written)
            return result

        turn_owner = owner
        owner.tool_admission = admit_tool
        owner.tool_terminal = settle_tool
        owner.tool_unknown = unknown_tool
        self._run_store.bind_step_boundary(agent_run_id, commit_boundary,
                                           lambda: self._require_turn_dispatch(owner))

    def _owned_attempt_capture(self, execution: ExecutionRecord) -> _InitialAttemptCapture | None:
        owner = self._turn_owners.get(execution.execution_id)
        capture = None if owner is None else owner.attempt
        if (capture is None or owner.producer_generation != execution.producer_generation
                or capture.execution.producer_generation != execution.producer_generation
                or capture.execution.producer_claim_id != execution.producer_claim_id
                or capture.execution.agent_run_seq != execution.agent_run_seq
                or capture.execution.recovery_state is not ExecutionRecoveryState.ACTIVE):
            return None
        return capture

    async def _expected_session_cursor(
        self,
        execution: ExecutionRecord,
        *,
        session: SessionRecord | None = None,
    ) -> ConversationCursor | None:
        history_id = None
        if execution.session_id is not None:
            if session is None:
                capture = self._owned_attempt_capture(execution)
                session = None if capture is None else capture.session
            if session is None:
                session = await self._load_session(
                    execution.session_id, tenant_id=self._tenant_id,
                )
            if session is not None and session.session_id != execution.session_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if session is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            history_id = session.history_id or (
                None
                if session.continuation is None
                else session.continuation.history_id
            )
        if execution.conversation_agent_run_id is not None:
            if (
                session is not None
                and session.continuation is not None
                and session.continuation.agent_run_id
                == execution.conversation_agent_run_id
            ):
                return session.continuation
            return ConversationCursor(
                execution.conversation_agent_run_id,
                history_id=history_id,
            )
        if execution.fork_base_execution_id is None:
            return None
        base = await self.load_execution(
            execution.fork_base_execution_id,
            tenant_id=self._tenant_id,
        )
        if base is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if base.agent_run_seq == 0:
            return None
        base_agent_run_id = make_agent_run_id(
            namespace=self._namespace, tenant_id=self._tenant_id,
            execution_id=base.execution_id, agent_run_seq=base.agent_run_seq,
        )
        if (session is not None and session.continuation is not None
                and session.continuation.agent_run_id == base_agent_run_id):
            return session.continuation
        return ConversationCursor(
            base_agent_run_id,
            history_id=history_id,
        )

    def _recovery_commands_for(self, execution_id: str) -> RuntimeRecoveryCommands:
        if self._tool_operations is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return RuntimeRecoveryCommands(
            self._execution.executions,
            self._execution.events,
            self._checkpoint.operations,
            self._tool_operations,
            execution_operations=self._execution.operations,
            background_tasks=self._execution_task_set(execution_id),
        )

    async def _commit_recovery_required_owned(
        self,
        execution: ExecutionRecord,
        error: AIError,
        effects: tuple[ExecutionRecoveryEffect, ...],
        *,
        expected_revision: int | None = None,
        producer_generation: int | None = None,
    ) -> ExecutionRecord:
        current = await self.load_execution(
            execution.execution_id,
            tenant_id=self._tenant_id,
        )
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if expected_revision is not None and current.revision != expected_revision:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
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
                    "agent_run_id": selected.agent_run_id,
                    "tool_call_id": selected.tool_call_id,
                    "tool_name": selected.tool_name,
                    "fence": selected.fence,
                }
            )
        else:
            details["unknown_effect_count"] = len(effects)

        if current.agent_run_seq > 0:
            await self._agent_run_lifecycle.flush_execution_projection(
                make_agent_run_id(
                    namespace=self._namespace,
                    tenant_id=self._tenant_id,
                    execution_id=current.execution_id,
                    agent_run_seq=current.agent_run_seq,
                ),
                execution_id=current.execution_id,
            )
        async with self._audit_lock(current.execution_id):
            payload_ids: tuple[str, ...] | None = None
            if (self._execution_buffer is not None and self._execution_buffer.has_scope(current.execution_id)
                    and not self._execution_buffer.settled(current.execution_id)):
                captured, payload_ids = await self._execution_buffer.source_prefix(current.execution_id)
                if captured is not None and (captured.terminal is not None or captured.checkpoint_payload is not None):
                    # This control boundary only covers admission and audits,
                    # not an uncommitted terminal/checkpoint candidate.
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            pending = await self._source_audit_prefix(current.execution_id)
            committed = await self._recovery_commands_for(
                current.execution_id
            ).commit_recovery_required(
                current,
                error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value,
                safe_error_details=details,
                audit_events=pending,
                producer_generation=producer_generation,
            )
            if payload_ids is not None:
                self._execution_buffer.confirm(current.execution_id, payload_ids)
            if pending:
                self._confirm_source_audits(current.execution_id, len(pending))
            if payload_ids is not None:
                # The executor has stopped (or cancellation has drained it).
                # Preserve effect recovery, while retiring only source-covered bytes.
                self._execution_buffer.retire(current.execution_id,
                    self._execution_buffer.producer_generation(current.execution_id))
            self._confirm_committed_events(
                current.execution_id,
                pending_count=len(pending),
                durable_seq=committed.event_seq,
            )
            self._live_broker.publish_event(
                current.execution_id,
                ExecutionEventType.EXECUTION_RECOVERY_REQUIRED,
                {
                    "error_code": ErrorCode.TOOL_EFFECT_UNKNOWN.value,
                    "safe_error_details": details,
                },
                durable_seq=committed.event_seq,
            )
            self._live_broker.complete(current.execution_id)
        _logger.error(
            "execution entered recovery-required state: execution=%s",
            current.execution_id,
        )
        return committed

    async def _commit_recovery_required(
        self,
        execution: ExecutionRecord,
        error: AIError,
        effects: tuple[ExecutionRecoveryEffect, ...],
        *,
        expected_revision: int | None = None,
        producer_generation: int | None = None,
    ) -> ExecutionRecord:
        task = asyncio.create_task(
            self._commit_recovery_required_owned(
                execution, error, effects, expected_revision=expected_revision,
                producer_generation=producer_generation,
            ),
            name=f"local-recovery-required-{execution.execution_id}",
        )
        committed, cancellation = await self._await_checkpoint_task(
            task,
            label="recovery-required",
            execution_id=execution.execution_id,
        )
        if cancellation is not None:
            raise cancellation
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
                agent_run_id=record.agent_run_id,
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

    async def _reconcile_cancelled_tool_claims(self, execution: ExecutionRecord) -> None:
        if self._tool_operations is None:
            return
        records = await self._tool_operations.list_by_execution(
            execution.execution_id, tenant_id=self._tenant_id,
        )
        for record in records:
            if record.status is ToolOperationStatus.CLAIMED:
                if (execution.status is not ExecutionStatus.CANCELLING
                        or execution.producer_generation < 1 or execution.producer_claim_id is None):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                await self._runtime_commands.reconcile_tool_turn_cancellation(
                    record.tool_operation_id, execution_id=execution.execution_id,
                    producer_generation=execution.producer_generation,
                    producer_claim_id=execution.producer_claim_id,
                )

    async def _reconcile_tool_effects(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> tuple[tuple[ExecutionRecoveryEffect, ...], int]:
        if self._tool_operations is None:
            return (), 0
        records = await self._tool_operations.list_by_execution(
            execution_id,
            tenant_id=tenant_id,
        )
        effects = tuple(
            ExecutionRecoveryEffect(
                operation_id=record.tool_operation_id,
                execution_id=record.execution_id,
                agent_run_id=record.agent_run_id,
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
        active_claims = sum(record.status is ToolOperationStatus.CLAIMED for record in records)
        return effects, active_claims

    async def _require_tool_effect_recovery(
        self,
        execution: ExecutionRecord,
        effects: tuple[ExecutionRecoveryEffect, ...],
        *,
        producer_generation: int | None = None,
    ) -> ExecutionRecord:
        details: dict[str, JsonValue] = {}
        if len(effects) == 1:
            details["operation_id"] = effects[0].operation_id
        return await self._commit_recovery_required(
            execution,
            AIError(
                ErrorCode.TOOL_EFFECT_UNKNOWN,
                safe_details=details,
            ),
            effects,
            producer_generation=producer_generation,
        )

    async def _check_reconcile_tool_effects(
        self,
        execution: ExecutionRecord,
        *,
        producer_generation: int | None = None,
        tool_operations: RuntimeToolOperationBridge | None = None,
    ) -> ExecutionRecord | None:
        if execution.status is ExecutionStatus.RECOVERY_REQUIRED:
            return execution
        if execution.status not in {
            ExecutionStatus.STARTED,
            ExecutionStatus.CANCELLING,
        }:
            return None
        if (execution.status is ExecutionStatus.STARTED
                and producer_generation == execution.producer_generation
                and tool_operations is not None
                and tool_operations.confirmed_terminal_operations(execution=execution) is not None):
            return None
        effects, active_claims = await self._reconcile_tool_effects(
            execution.execution_id,
            tenant_id=self._tenant_id,
        )
        if effects:
            return await self._require_tool_effect_recovery(
                execution, effects, producer_generation=producer_generation,
            )
        if active_claims:
            raise AIError(
                ErrorCode.STORAGE_RECOVERY_REQUIRED,
                safe_details={
                    "execution_id": execution.execution_id,
                    "phase": "tool_operations_unsettled",
                    "active_tool_claim_count": active_claims,
                },
            )
        return None

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

    async def recover_execution(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        recovery_operation: OperationLedgerInput | None = None,
    ) -> ExecutionRecord:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if recovery_operation is not None:
            task = asyncio.create_task(self._recover_keyed_execution(
                execution_id, expected_revision, recovery_operation,
            ), name=f"ai-execution-recover-{execution_id}")
            result, cancellation = await self._await_checkpoint_task(
                task, label="execution_recover",
            )
            if cancellation is not None:
                raise cancellation
            return result
        task = self._tasks.get(execution_id)
        if task is not None and not task.done():
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        execution, _ = await self._recovery_coordinator.recover_execution(
            execution_id, tenant_id=tenant_id, expected_revision=expected_revision,
        )
        return execution

    async def _recovery_receipt(
        self, operation: OperationLedgerInput,
    ) -> OperationLedgerRecord | None:
        receipt = await self._execution.operations.get(
            operation.operation_id, tenant_id=self._tenant_id,
        )
        if receipt is not None and not operation_identity_matches(receipt, operation):
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        return receipt

    async def _replay_recovery(
        self, receipt: OperationLedgerRecord,
    ) -> ExecutionRecord:
        if receipt.status in {OperationStatus.SUCCEEDED, OperationStatus.CANCELLED}:
            execution = await self.load_execution(receipt.resource_id, tenant_id=self._tenant_id)
            if execution is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            return execution
        if receipt.status is OperationStatus.FAILED:
            raise AIError(ErrorCode(receipt.error_code or ErrorCode.STORAGE_INTEGRITY_ERROR))
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED,
            operation_id=receipt.operation_id,
            safe_details={"phase": "execution_recover_in_progress"})

    async def _settle_recovery(
        self, receipt: OperationLedgerRecord, status: OperationStatus,
        *, error_code: str | None = None,
    ) -> None:
        settled = replace(receipt, status=status, error_code=error_code,
            updated_at=datetime.now(timezone.utc))
        try:
            await self._execution.operations.compare_and_swap(
                receipt.operation_id, tenant_id=self._tenant_id,
                expected_status=OperationStatus.RUNNING, next_record=settled,
            )
        except AIError as error:
            if error.code not in {ErrorCode.STORAGE_CONFLICT, ErrorCode.STORAGE_COMMIT_UNKNOWN}:
                raise
            current = await self._execution.operations.get(
                receipt.operation_id, tenant_id=self._tenant_id,
            )
            if current is None or current.status is not status or (
                current.result_ref != receipt.result_ref or current.error_code != error_code
            ):
                raise

    async def _finish_terminal_recovery(
        self, current: ExecutionRecord, receipt: OperationLedgerRecord,
    ) -> ExecutionRecord:
        checkpoint = await self.load_recovery_execution(
            current.execution_id, tenant_id=self._tenant_id,
        )
        if checkpoint is not None and checkpoint.recovery.terminal_handoff is not None:
            # A committed execution does not prove the prepared publication finished.
            current = await self._reconcile_handoff(checkpoint)
        if current.status is ExecutionStatus.FAILED:
            code = current.error_code or ErrorCode.INTERNAL_ERROR.value
            await self._settle_recovery(receipt, OperationStatus.FAILED, error_code=code)
            raise AIError(ErrorCode(code))
        await self._settle_recovery(receipt,
            OperationStatus.CANCELLED if current.status is ExecutionStatus.CANCELLED
            else OperationStatus.SUCCEEDED)
        return current

    async def _recover_keyed_execution(
        self, execution_id: str, expected_revision: int, operation: OperationLedgerInput,
    ) -> ExecutionRecord:
        receipt = await self._recovery_receipt(operation)
        if receipt is not None:
            current = await self.load_execution(execution_id, tenant_id=self._tenant_id)
            if (receipt.status is OperationStatus.RUNNING and receipt.result_ref is None
                    and current is not None and current.status in {
                        ExecutionStatus.FINALIZING, ExecutionStatus.SUCCEEDED,
                        ExecutionStatus.CANCELLED, ExecutionStatus.FAILED,
                    }):
                if current.status is not ExecutionStatus.FINALIZING:
                    return await self._finish_terminal_recovery(current, receipt)
                expected_revision = current.revision
            else:
                return await self._replay_recovery(receipt)
        if not self._accepting:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        worker = self._tasks.get(execution_id)
        if worker is not None and not worker.done():
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        try:
            result, launched = await self._recovery_coordinator.recover_execution(
                execution_id, tenant_id=self._tenant_id, expected_revision=expected_revision,
                recovery_operation=operation,
            )
        except AIError as error:
            receipt = await self._recovery_receipt(operation)
            if (receipt is not None and receipt.status is OperationStatus.RUNNING
                    and receipt.result_ref is None):
                current = await self.load_execution(execution_id, tenant_id=self._tenant_id)
                if current is not None and current.status in {
                    ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
                }:
                    return await self._finish_terminal_recovery(current, receipt)
                raise
            if receipt is not None and (
                receipt.result_ref != operation.result_ref
                or receipt.status is not OperationStatus.RUNNING
            ):
                return await self._replay_recovery(receipt)
            if receipt is not None and error.safe_details.get("phase") == "recovery_launch_rejected":
                await self._settle_recovery(receipt, OperationStatus.FAILED, error_code=error.code.value)
            raise
        receipt = await self._recovery_receipt(operation)
        if receipt is None:
            if not launched and result.status is ExecutionStatus.WAITING_DEFERRED:
                return result
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if launched or result.status is ExecutionStatus.SUCCEEDED:
            status = OperationStatus.SUCCEEDED
        elif result.status is ExecutionStatus.FAILED:
            code = result.error_code or ErrorCode.INTERNAL_ERROR.value
            await self._settle_recovery(receipt, OperationStatus.FAILED, error_code=code)
            raise AIError(ErrorCode(code))
        elif result.status is ExecutionStatus.CANCELLED:
            status = OperationStatus.CANCELLED
        else:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        await self._settle_recovery(receipt, status)
        return result

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
        cancel = []
        after_sequence = None
        while True:
            values = await self._execution.operations.list_pending(
                ResourceKind.EXECUTION, execution_id, tenant_id=tenant_id,
                limit=257, after_sequence=after_sequence,
                states=frozenset({OperationStatus.PENDING, OperationStatus.RUNNING,
                    OperationStatus.EFFECT_UNKNOWN}),
            )
            cancel.extend(value for value in values
                if value.operation_kind is OperationKind.EXECUTION_CANCEL)
            if len(cancel) > 256:
                raise AIError(ErrorCode.TOO_MANY_PENDING_OPERATIONS)
            if len(values) < 257:
                return tuple(cancel)
            after_sequence = values[-1].sequence

    async def _complete_recovered_cancel(
        self,
        execution: ExecutionRecord,
        operations: tuple[OperationLedgerRecord, ...],
        *,
        expected_revision: int | None = None,
        producer_generation: int | None = None,
    ) -> ExecutionRecord:
        cancelling = execution
        if cancelling.status is ExecutionStatus.STARTED:
            if not operations:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            operation = operations[0]
            cancelling = await self.commit_cancel_checkpoint(
                ExecutionCancelRequestCommit(
                    cancelling.execution_id,
                    cancelling.revision,
                    cancelling.event_seq,
                    operation.operation_id,
                    datetime.now(timezone.utc),
                ),
                expected_status=ExecutionStatus.STARTED,
            )
        elif cancelling.status is ExecutionStatus.RECOVERY_REQUIRED:
            cancelling = await self._recovery_commands_for(
                cancelling.execution_id
            ).commit_cancel_claim(cancelling)
        elif cancelling.status is not ExecutionStatus.CANCELLING:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if cancelling.recovery is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        terminal = await self._commit_terminal(
            cancelling,
            ExecutionStatus.CANCELLED,
            None,
            ErrorCode.EXECUTION_CANCELLED.value,
            StopReason.CANCELLED,
            agent_run_id=cancelling.recovery.agent_run_id,
            expected_revision=(cancelling.revision if expected_revision is not None else None),
            producer_generation=producer_generation,
        )
        self._worker_failures.pop(terminal.execution_id, None)
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
            tenant_id=self._tenant_id,
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
            OperationStatus.EFFECT_UNKNOWN,
        }:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        updated = OperationLedgerRecord(
            current.operation_id,
            self._tenant_id,
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
                tenant_id=self._tenant_id,
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
                tenant_id=self._tenant_id,
            )
            if (
                latest is not None
                and latest.status is OperationStatus.SUCCEEDED
                and latest.result_ref == execution.execution_id
                and latest.request_digest == current.request_digest
            ):
                return
            raise

    async def settle_denied_tool_result(
        self, execution: ExecutionRecord, source_agent_run_id: str,
        tool_call_id: str, decision: ApprovalRecord,
    ) -> None:
        if self._tool_operations is None:
            return
        operation = await self._tool_operations.get_by_call(
            source_agent_run_id, tool_call_id, tenant_id=self._tenant_id,
        )
        if operation is None:
            return
        if operation.execution_id != execution.execution_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        payload = StoredPayload.inline_bytes(canonical_json_bytes({
            "version": 1, "kind": "tool_call_failed",
            "message": decision.decision_message or "The tool call was denied.",
        }))
        await self._runtime_commands.commit_denied_tool_result(
            operation, decision, source_agent_run_id=source_agent_run_id, error_payload=payload,
        )

    async def settle_external_tool_result(
        self, execution: ExecutionRecord, source_agent_run_id: str,
        tool_call_id: str, supplied: ExternalCallRecord, result: JsonValue,
    ) -> None:
        if self._tool_operations is None:
            return
        operation = await self._tool_operations.get_by_call(
            source_agent_run_id, tool_call_id, tenant_id=self._tenant_id,
        )
        if operation is None:
            return
        if supplied.resolution_kind == "succeeded":
            if supplied.supplied_at is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            encoded = encode_model_messages((ModelRequest(parts=[ToolReturnPart(
                "runtime", result, tool_call_id=operation.tool_operation_id,
                timestamp=supplied.supplied_at,
            )]),))
            payload = await self._recovery_payload(execution, encoded)
            await self._runtime_commands.commit_external_tool_result(
                operation, supplied, source_agent_run_id=source_agent_run_id, result_payload=payload,
            )
        else:
            if not isinstance(result, str) or supplied.resolution_kind not in {"retry", "failed"}:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            retry = supplied.resolution_kind == "retry"
            payload = StoredPayload.inline_bytes(canonical_json_bytes({
                "version": 1, "kind": "tool_call_rejected" if retry else "tool_call_failed",
                "message": result,
            }))
            await self._runtime_commands.commit_external_tool_result(
                operation, supplied, source_agent_run_id=source_agent_run_id, error_payload=payload,
                error_code=(ErrorCode.TOOL_RETRY_REQUIRED.value if retry else ErrorCode.TOOL_EXECUTION_FAILED.value),
            )

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
        del execution
        encoded = json.dumps(
            {
                "version": 1,
                "kind": "tool_call_failed",
                "message": "Tool effect was confirmed failed during recovery.",
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return StoredPayload.inline_bytes(encoded)

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
                self._checkpoint_objects,
                ObjectKeyFactory(self._namespace),
                PersistenceDomain.CHECKPOINT,
                self._tenant_id,
                data,
            )
        )

    async def _store_repository_instructions(
        self,
        execution: ExecutionRecord,
        instructions: RepositoryInstructions,
    ) -> RuntimePayloadRef:
        payload = instructions.to_payload()
        inline = StoredPayload.inline_json(payload)
        if payload_fits_inline(inline, self._payload_policy):
            stored = inline
        else:
            reference = await put_runtime_object(
                self._checkpoint_objects,
                ObjectKeyFactory(self._namespace),
                PersistenceDomain.CHECKPOINT,
                self._tenant_id,
                canonical_json_bytes(payload),
            )
            stored = StoredPayload.object(reference)
        return RuntimePayloadRef(stored, PersistenceDomain.CHECKPOINT)

    async def commit_repository_instruction_barrier(
        self,
        execution: ExecutionRecord,
        overlay: RepositoryInstructions,
        barrier: RepositoryInstructionBarrier,
        *,
        producer_generation: int | None = None,
    ) -> ExecutionRecord:
        if execution.recovery is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        owner = self._turn_owners.get(execution.execution_id)
        if owner is not None:
            owner.attempt = None
        await self.promote_pending_execution(execution.execution_id)
        existing = tuple(
            item
            for item in execution.recovery.repository_instruction_barriers
            if item.agent_run_id == barrier.agent_run_id
            and item.tool_call_id == barrier.tool_call_id
        )
        if existing:
            if existing[0] != barrier:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            current = await self._execution.executions.get_recovery(
                execution.execution_id,
                tenant_id=self._tenant_id,
            )
            if current is None or current.recovery is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            current_overlay = await self.load_repository_instructions(
                current.recovery.repository_instruction_overlay
            )
            _validate_repository_instruction_frontier(current, current_overlay)
            if (
                current_overlay is None
                or not _repository_instructions_contain(current_overlay, overlay)
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return current
        if execution.session_id is not None:
            sessions = self._conversation.sessions
            # Discovery knowledge can outlive a failed barrier. Persist it first
            # so a committed execution pin never loses its Session scope memory,
            # including when conversation and recovery use different stores.
            await sessions.state_store.mutate(
                lambda transaction: sessions.remember_repository_instruction_scopes_in_transaction(
                    transaction, execution.session_id,
                    tenant_id=self._tenant_id, execution_id=execution.execution_id,
                    scopes=tuple(sorted({document.scope for document in overlay.documents})),
                )
            )
        overlay_reference = await self._store_repository_instructions(
            execution,
            overlay,
        )
        next_execution = replace(
            execution,
            recovery=replace(
                execution.recovery, repository_instruction_overlay=overlay_reference,
                repository_instruction_barriers=(*execution.recovery.repository_instruction_barriers, barrier),
            ),
            revision=execution.revision + 1, updated_at=datetime.now(timezone.utc),
        )
        try:
            if producer_generation is None:
                return await self._execution.executions.compare_and_swap(
                    execution.execution_id,
                    tenant_id=self._tenant_id,
                    expected_revision=execution.revision,
                    next_record=next_execution,
                )
            execution_store = self._execution.executions.state_store

            async def commit(group: StateGroupTransaction) -> ExecutionRecord:
                transaction = group.transaction(execution_store)
                await self._execution.executions.require_execution_producer_in_transaction(
                    transaction, execution.execution_id,
                    expected_producer_generation=producer_generation, allow_cancelling=False,
                )
                return await self._execution.executions.compare_and_swap_in_transaction(
                    transaction, execution.execution_id,
                    tenant_id=self._tenant_id, expected_revision=execution.revision,
                    next_record=next_execution,
                )

            return await execution_store.storage_group.mutate((execution_store,), commit)
        except AIError as error:
            if error.code is ErrorCode.EXECUTION_CANCELLED:
                raise
            if producer_generation is not None:
                control = await self._execution.executions.get(
                    execution.execution_id, tenant_id=self._tenant_id,
                )
                if control is not None and control.producer_generation > producer_generation:
                    raise AIError(ErrorCode.STORAGE_CONFLICT) from error
            current = await self._execution.executions.get_recovery(
                execution.execution_id,
                tenant_id=self._tenant_id,
            )
            if current is None or current.recovery is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            matching = tuple(
                item
                for item in current.recovery.repository_instruction_barriers
                if item.agent_run_id == barrier.agent_run_id
                and item.tool_call_id == barrier.tool_call_id
            )
            if matching:
                if matching[0] != barrier:
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT) from error
                current_overlay = await self.load_repository_instructions(
                    current.recovery.repository_instruction_overlay
                )
                _validate_repository_instruction_frontier(current, current_overlay)
                if (
                    current_overlay is None
                    or not _repository_instructions_contain(current_overlay, overlay)
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                return current
            raise

    async def _commit_failure(
        self,
        execution: ExecutionRecord,
        error: Exception,
        *,
        agent_run_id: str | None = None,
        producer_generation: int | None = None,
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
            agent_run_id=agent_run_id,
            producer_generation=producer_generation,
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
        agent_run_id: str | None = None,
        usage: UsageMetrics | None = None,
        safe_error_details: Mapping[str, JsonValue] | None = None,
        error_diagnostics: ErrorDiagnostics | None = None,
        expected_cursor: ConversationCursor | None = None,
        conversation_agent_run: AgentRunRecord | None = None,
        conversation_checkpoint: AgentRunCheckpoint | None = None,
        recovery_checkpoint: ExecutionRecord | None = None,
        producer_generation: int | None = None,
        expected_revision: int | None = None,
        observed_execution: ExecutionRecord | None = None,
        tool_operations: RuntimeToolOperationBridge | None = None,
    ) -> ExecutionRecord:
        current = observed_execution
        if current is None:
            current = await self.load_execution(
                execution.execution_id,
                tenant_id=self._tenant_id,
            )
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if expected_revision is not None and current.revision != expected_revision:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if current.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            return current
        if current.status is ExecutionStatus.RECOVERY_REQUIRED:
            return current
        pending = self._pending_turns.get(current.execution_id, tenant_id=self._tenant_id)
        if pending is not None:
            await self._pending_turns.settle(pending)
            coordinator = self._pending_turns.coordination
            if pending.tool_operations and (status is ExecutionStatus.CANCELLED or any(
                value.status not in {ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED}
                for value in pending.tool_operations.values()
            )):
                await self.promote_pending_execution(current.execution_id, allow_cancelled=True)
            if (status is ExecutionStatus.SUCCEEDED and coordinator is not None
                    and pending.coordination is not None
                    and await coordinator.promotion_requested(pending.coordination)):
                await self.promote_pending_execution(current.execution_id)
            async with pending.boundary:
                if self._pending_turns.get(current.execution_id, tenant_id=self._tenant_id) is pending:
                    return await self._commit_pending_terminal(
                        current, status, output, usage or self._captured_usage.get(current.execution_id, UsageMetrics()),
                        agent_run_id, error_code=error_code, stop_reason=stop_reason,
                        safe_error_details=safe_error_details, error_diagnostics=error_diagnostics,
                        producer_generation=producer_generation,
                    )
            current = await self.load_execution(current.execution_id, tenant_id=self._tenant_id)
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (self._execution_buffer is None and output is not None and output.kind == "inline"
                and not payload_fits_inline(output, self._payload_policy)):
            output = await _materialize_execution_json(
                output, objects=self._execution_objects, keys=ObjectKeyFactory(self._namespace),
                policy=self._payload_policy, tenant_id=self._tenant_id,
            )
            current = await self.load_execution(execution.execution_id, tenant_id=self._tenant_id)
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if expected_revision is not None and current.revision != expected_revision:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED,
                                  ExecutionStatus.CANCELLED, ExecutionStatus.RECOVERY_REQUIRED}:
                return current
        recovered = await self._check_reconcile_tool_effects(
            current, producer_generation=producer_generation,
            tool_operations=tool_operations,
        )
        if recovered is not None:
            return recovered
        if current.status is ExecutionStatus.CANCELLING:
            status = ExecutionStatus.CANCELLED
            output = None
            error_code = ErrorCode.EXECUTION_CANCELLED.value
            stop_reason = StopReason.CANCELLED
            error_diagnostics = None
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
            recovery_checkpoint = current if current.recovery is not None else None
            if recovery_checkpoint is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if (
                recovery_checkpoint.recovery.handoff_phase is RecoveryHandoffPhase.NONE
                and self._same_terminal_storage_group(
                    current,
                    status=status,
                    agent_run_id=agent_run_id,
                )
            ):
                try:
                    return await self._commit_same_group_recovery_terminal(
                        current,
                        recovery_checkpoint,
                        status=status,
                        output=output,
                        error_code=error_code,
                        stop_reason=stop_reason,
                        binding=binding,
                        agent_run_id=agent_run_id,
                        usage=captured_usage,
                        safe_error_details=safe_error_details,
                        error_diagnostics=error_diagnostics,
                        producer_generation=producer_generation,
                    )
                except AIError as error:
                    if (error.code is not ErrorCode.EXECUTION_RESULT_CONFLICT
                            or current.status is not ExecutionStatus.STARTED
                            or status is not ExecutionStatus.CANCELLED
                            or expected_revision is not None):
                        raise
                    observed = await self.load_execution(current.execution_id, tenant_id=self._tenant_id)
                    if observed is None or observed.status is not ExecutionStatus.CANCELLING:
                        raise
                    # The cancellation checkpoint can consume the audit prefix
                    # while the stopped producer prepares its terminal trace.
                    # Rebuild only after that source transition rejected the
                    # uncommitted terminal; an unknown receipt is never retried.
                    return await self._commit_terminal(
                        observed, status, output, error_code, stop_reason,
                        binding=binding, agent_run_id=agent_run_id, usage=captured_usage,
                        safe_error_details=safe_error_details, error_diagnostics=error_diagnostics,
                        producer_generation=producer_generation, observed_execution=observed,
                    )
            if recovery_checkpoint.recovery.handoff_phase is RecoveryHandoffPhase.NONE:
                if status is ExecutionStatus.SUCCEEDED and agent_run_id is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                captured_checkpoint = None
                if self._execution_buffer is not None:
                    checkpoint_payload = None
                    if (agent_run_id is not None and producer_generation is not None
                            and not self._execution_buffer.rejection_settled(current.execution_id)
                            and not self._agent_run_lifecycle.trace_buffer_failure_settled(
                                agent_run_id, execution_id=current.execution_id,
                                producer_generation=producer_generation)):
                        checkpoint_payload = await self._agent_run_lifecycle.capture_agent_run_checkpoint_payload(
                            agent_run_id, execution_id=current.execution_id,
                            producer_generation=producer_generation, include_interrupted=True,
                        )
                    candidate = ExecutionTerminalCommit(
                        current.revision, current.event_seq,
                        _terminal_record(current, status, now, error_code=error_code,
                            safe_error_details=safe_error_details, error_diagnostics=error_diagnostics),
                        ResultRecord(output, stop_reason, captured_usage, now),
                        ExecutionEventType.EXECUTION_SUCCEEDED if status is ExecutionStatus.SUCCEEDED
                        else ExecutionEventType.EXECUTION_CANCELLED if status is ExecutionStatus.CANCELLED
                        else ExecutionEventType.EXECUTION_FAILED,
                        {"agent_run_id": agent_run_id} if status is ExecutionStatus.SUCCEEDED
                        else _terminal_error_payload(error_code, safe_error_details or {}, error_diagnostics),
                        producer_generation=producer_generation,
                    )
                    async with self._audit_lock(current.execution_id):
                        selected, selected_checkpoint, _ids = await self._prepare_source_terminal_payload(
                            current, candidate, checkpoint_payload=checkpoint_payload,
                        )
                    output = selected.result.output
                    if selected_checkpoint is not None:
                        if agent_run_id is None or producer_generation is None:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        captured_checkpoint = await self._agent_run_lifecycle.decode_agent_run_checkpoint_payload(
                            selected_checkpoint, agent_run_id=agent_run_id, producer_generation=producer_generation,
                        )
                if agent_run_id is not None:
                    rejected_trace = (
                        status in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}
                        and producer_generation is not None
                        and self._agent_run_lifecycle.trace_buffer_failure_settled(
                            agent_run_id, execution_id=current.execution_id,
                            producer_generation=producer_generation,
                        )
                    )
                    rejected_payload = (self._execution_buffer is not None
                        and self._execution_buffer.rejection_settled(current.execution_id)
                        and captured_checkpoint is None)
                    if (not rejected_trace and not rejected_payload
                            and (current.conversation or current.restart_checkpoints or current.durable_trace)):
                        await self._agent_run_lifecycle.checkpoint_persistence.materialize_recovery_checkpoint(
                            agent_run_id=agent_run_id,
                            require_complete=status is ExecutionStatus.SUCCEEDED,
                            captured=captured_checkpoint,
                        )
                    if (current.conversation and status is ExecutionStatus.SUCCEEDED
                            and (current.session_id is not None
                                 or not current.restart_checkpoints and not current.durable_trace)):
                        # A split-route handoff references the independent Conversation owner.
                        # Publish it before PREPARED so a cold worker never needs staging.
                        await self._run_store.materialize_conversation(agent_run_id=agent_run_id)
                    if producer_generation is not None and not rejected_trace:
                        await self._agent_run_lifecycle.flush_execution_projection(
                            agent_run_id, execution_id=current.execution_id,
                            producer_generation=producer_generation,
                        )
                handoff_output = output
                source_domain = None
                if output is not None and output.kind == "object":
                    if self._execution_objects_durable:
                        source_domain = PersistenceDomain.EXECUTION
                    else:
                        reference = output.ref
                        if reference is None:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        content = await read_runtime_object(
                            self._execution_objects, reference
                        )
                        handoff_output = StoredPayload.object(
                            await put_runtime_object(
                                self._checkpoint_objects,
                                ObjectKeyFactory(self._namespace),
                                PersistenceDomain.CHECKPOINT,
                                self._tenant_id,
                                content,
                            )
                        )
                        source_domain = PersistenceDomain.CHECKPOINT
                conversation = None
                if (
                    current.conversation
                    and status is ExecutionStatus.SUCCEEDED
                    and current.session_id is not None
                    and agent_run_id is not None
                ):
                    session = await self._load_session(
                        current.session_id,
                        tenant_id=self._tenant_id,
                    )
                    if session is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    history_id = session.history_id or (
                        None if session.continuation is None else session.continuation.history_id
                    )
                    archive = self._get_run_store(PersistenceDomain.CONVERSATION)
                    if history_id is None or not isinstance(archive, StateStepArchive):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    message_count = await archive.committed_session_message_count(
                        history_id, agent_run_id=agent_run_id, tenant_id=self._tenant_id,
                    )
                    checkpoint_sequence = await archive.latest_checkpoint_sequence(agent_run_id=agent_run_id)
                    stored_checkpoint = await archive.checkpoint_at_sequence(
                        agent_run_id=agent_run_id, checkpoint_sequence=checkpoint_sequence,
                    )
                    conversation = RecoveryConversationIntent(
                        current.session_id,
                        await self._expected_session_cursor(current),
                        ConversationCursor(
                            agent_run_id, history_id=history_id, message_count=message_count,
                            checkpoint_sequence=checkpoint_sequence,
                            projection_digest=stored_checkpoint.projection_digest,
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
                            {"agent_run_id": agent_run_id}
                            if status is ExecutionStatus.SUCCEEDED
                            and agent_run_id is not None
                            else _terminal_error_payload(
                                error_code,
                                safe_error_details or {},
                                error_diagnostics,
                            )
                        ),
                        result_created_at=now,
                        error_diagnostics=error_diagnostics,
                    ),
                    agent_run_id or recovery_checkpoint.recovery.agent_run_id,
                    conversation,
                )
                prepared = replace(recovery_checkpoint, recovery=replace(recovery_checkpoint.recovery, handoff_phase=RecoveryHandoffPhase.PREPARED, terminal_handoff=handoff, pending_tools=None) , revision=recovery_checkpoint.revision + 1, updated_at=now)
                try:
                    if producer_generation is not None:
                        recovery_checkpoint = await self._runtime_commands.commit_recovery_handoff(
                            prepared,
                            expected_revision=recovery_checkpoint.revision,
                            producer_generation=producer_generation,
                        )
                    else:
                        recovery_checkpoint = await self._execution.executions.compare_and_swap(
                            current.execution_id,
                            tenant_id=self._tenant_id,
                            expected_revision=recovery_checkpoint.revision,
                            next_record=prepared,
                        )
                except AIError as error:
                    if error.code is not ErrorCode.STORAGE_CONFLICT:
                        raise
                    recovery_checkpoint = await self._execution.executions.get_recovery(
                        current.execution_id,
                        tenant_id=self._tenant_id,
                    )
                    if recovery_checkpoint is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                _logger.info(
                    "recovery handoff prepared: execution=%s agent_run=%s",
                    current.execution_id,
                    agent_run_id,
                )
            return await self._reconcile_handoff(recovery_checkpoint)
        await self.verify_terminal_projection(
            current,
            status,
            agent_run_id if status is ExecutionStatus.SUCCEEDED else None,
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
            current.event_seq,
            terminal,
            ResultRecord(
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
                {"agent_run_id": agent_run_id}
                if status is ExecutionStatus.SUCCEEDED and agent_run_id is not None
                else _terminal_error_payload(
                    error_code,
                    safe_error_details or {},
                    error_diagnostics,
                )
            ),
        )
        terminal_commit = replace(terminal_commit, producer_generation=producer_generation)

        async def publish_terminal(committed: ExecutionTerminalCommitResult) -> None:
            self._publish_terminal_event(
                current.execution_id,
                event_type=terminal_commit.terminal_event_type,
                payload=dict(terminal_commit.terminal_event_payload),
                durable_seq=committed.execution.event_seq,
            )

        committed = await self._commit_execution_terminal_checkpoint(
            current,
            terminal_commit,
            agent_run_id=agent_run_id,
            handoff=publish_terminal,
            expected_cursor=expected_cursor,
            conversation_agent_run=conversation_agent_run,
            conversation_checkpoint=conversation_checkpoint,
        )
        _logger.info(
            "execution terminal committed: execution=%s status=%s",
            current.execution_id,
            status.value,
        )
        return committed.execution

    async def _commit_pending_terminal(
        self, current: ExecutionRecord, status: ExecutionStatus, output: StoredPayload | None,
        usage: UsageMetrics, agent_run_id: str | None, *, error_code: str | None,
        stop_reason: StopReason, safe_error_details: Mapping[str, JsonValue] | None,
        error_diagnostics: ErrorDiagnostics | None, producer_generation: int | None,
    ) -> ExecutionRecord:
        turn = self._pending_turns.get(current.execution_id, tenant_id=self._tenant_id)
        if turn is None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        successful = status is ExecutionStatus.SUCCEEDED
        tool_operations = tuple(turn.tool_operations.values())
        run = checkpoint = None
        checkpoint_payload = None
        if successful:
            if output is None or agent_run_id is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            checkpoint = await self._run_store.conversation_checkpoint(agent_run_id=agent_run_id)
            if self._execution_buffer is None:
                run = await self._run_store.get_agent_run(agent_run_id=agent_run_id)
                if run is None or checkpoint is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            else:
                if producer_generation is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                checkpoint_payload = await self._agent_run_lifecycle.capture_agent_run_checkpoint_payload(
                    agent_run_id, execution_id=current.execution_id, producer_generation=producer_generation,
                )
        elif output is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        history_id = None if turn.session is None else turn.session.history_id or (
            None if turn.session.continuation is None else turn.session.continuation.history_id
        )
        await self._check_pending_lease(turn, allow_cancelled=not successful)
        now = datetime.now(timezone.utc)
        terminal = replace(
            _terminal_record(current, status, now, error_code=error_code,
                             safe_error_details=safe_error_details, error_diagnostics=error_diagnostics),
            conversation_agent_run_id=agent_run_id if successful and turn.session is not None else None,
        )
        commit = ExecutionTerminalCommit(
            current.revision, current.event_seq, terminal,
            ResultRecord(output, stop_reason, usage, now),
            (ExecutionEventType.EXECUTION_SUCCEEDED if successful else
             ExecutionEventType.EXECUTION_CANCELLED if status is ExecutionStatus.CANCELLED else
             ExecutionEventType.EXECUTION_FAILED),
            {"agent_run_id": agent_run_id} if successful else _terminal_error_payload(
                error_code, safe_error_details or {}, error_diagnostics,
            ),
            producer_generation=producer_generation,
        )
        reservation = turn.reservation
        async with self._audit_lock(current.execution_id):
            pending_audit = tuple(self._pending_audit_events.get(current.execution_id, ()))
            payload_ids: tuple[str, ...] = ()
            if self._execution_buffer is not None:
                await self._execution_buffer.append(_ExecutionPayloadCapture(
                    terminal, reservation, commit, pending_audit, checkpoint_payload,
                ))
                captured, payload_ids = await self._execution_buffer.read(current.execution_id)
                if captured.terminal is None or captured.reservation is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                commit, reservation, pending_audit = captured.terminal, captured.reservation, captured.audit_events
                if successful:
                    if captured.checkpoint_payload is None or agent_run_id is None or producer_generation is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    run, captured_checkpoint = await self._agent_run_lifecycle.decode_agent_run_checkpoint_payload(
                        captured.checkpoint_payload, agent_run_id=agent_run_id, producer_generation=producer_generation,
                    )
                    checkpoint = (None if captured_checkpoint is None else replace(captured_checkpoint,
                        context_messages=None if checkpoint is None else checkpoint.context_messages))
                    if checkpoint is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        context = await self.load_input_context(commit.execution.input_context)
        if context is None and successful:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        terminal = await _materialize_execution_payloads(
            commit.execution, input_materializer=self._input_materializer,
            objects=self._execution_objects, keys=ObjectKeyFactory(self._namespace),
            policy=self._payload_policy, tenant_id=self._tenant_id,
        )
        reservation = replace(reservation, execution=replace(
            reservation.execution, stored_user_input=terminal.stored_user_input,
            repository_instructions=terminal.repository_instructions,
        ))
        output = commit.result.output
        if output is not None:
            output = await _materialize_execution_json(
                output, objects=self._execution_objects, keys=ObjectKeyFactory(self._namespace),
                policy=self._payload_policy, tenant_id=self._tenant_id,
            )
        prepared_context = None if context is None else await self._context_archive.prepare(context,
            history_root=turn.history_root, history_digest=turn.history_digest)
        commit = replace(commit,
            execution=replace(terminal, input_context=None if prepared_context is None else prepared_context.reference),
            result=replace(commit.result, output=output),
        )
        plan: ExecutionTerminalSealPlan | None = None
        try:
            if current.durable_trace:
                async def prepare_trace() -> ExecutionTerminalSealPlan:
                    return await self._agent_run_lifecycle.prepare_execution_terminal_seal(
                        execution_id=current.execution_id,
                        agent_run_ids=() if agent_run_id is None else (agent_run_id,),
                        binding_digest=current.binding_digest,
                    )

                try:
                    plan = await prepare_trace()
                except Exception as trace_error:
                    if (successful or agent_run_id is None or producer_generation is None
                            or not await self._agent_run_lifecycle.settle_trace_buffer_failure(
                                agent_run_id, execution_id=current.execution_id,
                                producer_generation=producer_generation, error=trace_error,
                            )):
                        raise
                    plan = await prepare_trace()
            await self._check_pending_lease(turn, allow_cancelled=not successful)
            if self._execution_buffer is not None:
                self._execution_buffer.seal(current.execution_id, payload_ids)
            turn.begin_commit()
        except BaseException:
            if plan is not None:
                await self._agent_run_lifecycle.discard_execution_terminal_seal(plan)
            raise

        async def accept(committed: ExecutionTerminalCommitResult) -> bool:
            if committed.execution.execution_id != current.execution_id:
                # The canonical request belongs to another turn: this exact
                # source batch did not publish. Keep its local owner until the
                # stopped worker has released its claim and staging.
                if plan is not None:
                    await self._agent_run_lifecycle.discard_execution_terminal_seal(plan)
                if self._execution_buffer is not None:
                    self._execution_buffer.reject_source(current.execution_id, payload_ids)
                turn.committing = False
                turn.uncertain = False
                turn.reconcile = None
                failure = AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                self._pending_turns.fail(turn, failure)
                if not self.worker_installed(current.execution_id):
                    self._record_worker_failure(current.execution_id, failure)
                    cleanup = asyncio.create_task(self._cleanup_failed_pending_turn(turn))
                    self._track_checkpoint_task(cleanup, "pending_release")
                return False
            turn.confirm(committed)
            turn.uncertain = False
            turn.failure = None
            self._worker_failures.pop(current.execution_id, None)
            self._pending_turns.release(turn)
            if self._execution_buffer is not None:
                self._execution_buffer.confirm(current.execution_id, payload_ids)
            if plan is not None:
                try:
                    await self._agent_run_lifecycle.finalize_execution_terminal_seal(plan)
                except BaseException:
                    _logger.error("pending Trace finalization failed after source confirmation: execution=%s",
                                  current.execution_id, exc_info=True)
                    try:
                        await self._agent_run_lifecycle.reconcile_execution_terminal_seal(plan)
                    except BaseException:
                        _logger.error("pending Trace cleanup remains unsettled: execution=%s",
                                      current.execution_id, exc_info=True)
            async with self._audit_lock(current.execution_id):
                audit_tail = self._pending_audit_events.get(current.execution_id)
                if audit_tail is not None:
                    del audit_tail[:len(pending_audit)]
                    if not audit_tail:
                        self._pending_audit_events.pop(current.execution_id, None)
            if self._execution_buffer is not None:
                await self._execution_buffer.release(current.execution_id)
            self._confirm_committed_events(
                current.execution_id, pending_count=len(pending_audit) + 1,
                durable_seq=committed.execution.event_seq,
            )
            self._publish_terminal_event(
                current.execution_id, event_type=commit.terminal_event_type,
                payload=dict(commit.terminal_event_payload), durable_seq=committed.execution.event_seq,
            )
            await self._record_committed_terminal(committed, session_id=current.session_id)
            return True

        async def confirm() -> None:
            committed = await self._runtime_commands.read_pending_turn(
                reservation, expected_session=turn.session, tool_operations=tool_operations,
            )
            if committed is None:
                # Absence does not prove rollback of the original unknown attempt.
                raise turn.failure or AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            if (committed.execution.execution_id == current.execution_id
                    and producer_generation is not None and plan is None):
                await self._discard_terminal_producer_tail(
                    committed.execution, agent_run_id=agent_run_id,
                    producer_generation=producer_generation,
                )
            await accept(committed)

        turn.reconcile = lambda: self._reconcile_pending_boundary(turn, confirm)

        async def commit_owned() -> ExecutionTerminalCommitResult:
            committed = await self._runtime_commands.commit_pending_turn(
                reservation, commit, expected_session=turn.session,
                context_write=None if prepared_context is None else (prepared_context.archive, prepared_context.records),
                capability_run_id=agent_run_id,
                next_cursor=None if turn.session is None or not successful else ConversationCursor(agent_run_id, history_id=history_id),
                conversation_agent_run=(run if current.conversation and (turn.session is not None
                    or not current.restart_checkpoints and not current.durable_trace) else None),
                conversation_checkpoint=(checkpoint if current.conversation and (turn.session is not None
                    or not current.restart_checkpoints and not current.durable_trace) else None),
                execution_projections=() if plan is None else plan.projections,
                audit_events=pending_audit, tool_operations=tool_operations,
            )
            if not await accept(committed):
                assert turn.failure is not None
                raise turn.failure
            return committed

        task = asyncio.create_task(commit_owned(), name=f"ai-terminal-boundary-{current.execution_id}")
        committed, cancellation = await self._await_checkpoint_task(
            task, label="terminal", execution_id=current.execution_id,
        )
        if cancellation is not None:
            raise cancellation
        return committed.execution

    async def _reconcile_pending_boundary(
        self, turn: _PendingExecutionTurn, confirm: Callable[[], Awaitable[None]],
    ) -> None:
        async with turn.boundary:
            if not turn.uncertain:
                return

            async def confirm_owned() -> None:
                try:
                    await confirm()
                except Exception as error:
                    if not turn.uncertain:
                        raise
                    raise AIError(
                        ErrorCode.STORAGE_COMMIT_UNKNOWN,
                        safe_details={"phase": "pending_commit_readback"},
                    ) from error

            task = asyncio.create_task(
                confirm_owned(), name=f"ai-confirm-pending-{turn.execution.execution_id}",
            )
            _result, cancellation = await self._await_checkpoint_task(
                task, label="pending confirmation", execution_id=turn.execution.execution_id,
            )
            if cancellation is not None:
                raise cancellation

    def _same_terminal_storage_group(
        self,
        execution: ExecutionRecord,
        *,
        status: ExecutionStatus,
        agent_run_id: str | None,
    ) -> bool:
        stores = [
            self._execution.executions.state_store,
            self._execution.executions.state_store,
        ]
        if execution.session_id is not None:
            stores.append(self._conversation.sessions.state_store)
        if agent_run_id is not None:
            if execution.durable_trace:
                archive = self._run_stores[PersistenceDomain.TRACE]
                if isinstance(archive, StateStepArchive):
                    stores.append(archive.state_store)
            if (execution.conversation and status is ExecutionStatus.SUCCEEDED
                    and (execution.session_id is not None or not execution.restart_checkpoints and not execution.durable_trace)):
                archive = self._run_stores[PersistenceDomain.CONVERSATION]
                if not isinstance(archive, StateStepArchive):
                    return False
                stores.append(archive.state_store)
        return all(
            store.storage_group is stores[0].storage_group for store in stores[1:]
        )

    async def _commit_same_group_recovery_terminal(
        self,
        current: ExecutionRecord,
        checkpoint: ExecutionRecord,
        *,
        status: ExecutionStatus,
        output: StoredPayload | None,
        error_code: str | None,
        stop_reason: StopReason,
        binding: AgentBinding | None,
        agent_run_id: str | None,
        usage: UsageMetrics,
        safe_error_details: Mapping[str, JsonValue] | None,
        error_diagnostics: ErrorDiagnostics | None,
        producer_generation: int | None = None,
    ) -> ExecutionRecord:
        if current.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            return current
        await self.verify_terminal_projection(
            current,
            status,
            agent_run_id if status is ExecutionStatus.SUCCEEDED else None,
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
            current.event_seq,
            terminal,
            ResultRecord(
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
            {"agent_run_id": agent_run_id}
            if status is ExecutionStatus.SUCCEEDED and agent_run_id is not None
            else _terminal_error_payload(
                error_code,
                safe_error_details or {},
                error_diagnostics,
            ),
        )
        commit = replace(commit, producer_generation=producer_generation)
        commit = replace(commit, execution=replace(commit.execution, recovery=replace(
            current.recovery, handoff_phase=RecoveryHandoffPhase.COMPLETED,
            terminal_handoff=None, pending_tools=None,
        )))
        conversation_agent_run = None
        conversation_checkpoint = None
        expected_cursor = None
        if (current.conversation and status is ExecutionStatus.SUCCEEDED
                and (current.session_id is not None or not current.restart_checkpoints and not current.durable_trace)):
            conversation_agent_run = await self._run_store.get_agent_run(agent_run_id=agent_run_id or "")
            conversation_checkpoint = await self._run_store.conversation_checkpoint(
                agent_run_id=agent_run_id or ""
            )
            expected_cursor = (await self._expected_session_cursor(current) if current.session_id is not None else None)
            if conversation_agent_run is None or conversation_checkpoint is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        async def publish_terminal(committed: ExecutionTerminalCommitResult) -> None:
            self._publish_terminal_event(
                current.execution_id,
                event_type=commit.terminal_event_type,
                payload=dict(commit.terminal_event_payload),
                durable_seq=committed.execution.event_seq,
            )

        committed = await self._commit_execution_terminal_checkpoint(
            current,
            commit,
            agent_run_id=agent_run_id,
            handoff=publish_terminal,
            expected_cursor=expected_cursor,
            conversation_agent_run=conversation_agent_run,
            conversation_checkpoint=conversation_checkpoint,
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
        agent_run_id: str | None,
        handoff: Callable[[ExecutionTerminalCommitResult], Awaitable[ExecutionRecord | None]] | None = None,
        expected_cursor: ConversationCursor | None = None,
        conversation_agent_run: AgentRunRecord | None = None,
        conversation_checkpoint: AgentRunCheckpoint | None = None,
        conversation_message_range: tuple[int, int] | None = None,
    ) -> ExecutionTerminalCommitResult:
        async def commit_owned() -> ExecutionTerminalCommitResult:
            nonlocal commit, conversation_agent_run, conversation_checkpoint
            payload_ids: tuple[str, ...] = ()
            checkpoint_payload = None
            plan: ExecutionTerminalSealPlan | None = None
            durable_commit = False
            pending_count = 0
            selected_audit: tuple[ExecutionEventAppend, ...] = ()
            try:
                if current.restart_checkpoints and agent_run_id is not None:
                    await self._agent_run_lifecycle.checkpoint_persistence.materialize_recovery_checkpoint(
                        agent_run_id=agent_run_id,
                        require_complete=commit.execution.status is ExecutionStatus.SUCCEEDED,
                    )
                if (self._execution_buffer is not None and agent_run_id is not None
                        and commit.producer_generation is not None
                        and conversation_checkpoint is not None
                        and not self._execution_buffer.rejection_settled(current.execution_id)
                        and not self._agent_run_lifecycle.trace_buffer_failure_settled(
                            agent_run_id, execution_id=current.execution_id,
                            producer_generation=commit.producer_generation)):
                    checkpoint_payload = await self._agent_run_lifecycle.capture_agent_run_checkpoint_payload(
                        agent_run_id, execution_id=current.execution_id,
                        producer_generation=commit.producer_generation, include_interrupted=True,
                    )
                state_archive = current.durable_trace and isinstance(
                    self._run_stores[PersistenceDomain.TRACE],
                    StateStepArchive,
                )
                if state_archive:
                    if plan is None:
                        candidate_agent_run_ids = tuple(
                            make_agent_run_id(
                                namespace=self._namespace,
                                tenant_id=self._tenant_id,
                                execution_id=current.execution_id,
                                agent_run_seq=sequence,
                            )
                            for sequence in range(1, current.agent_run_seq + 1)
                        )
                        candidate_agent_run_ids = await self._existing_execution_agent_run_ids(
                            candidate_agent_run_ids
                        )
                        try:
                            plan = await self._agent_run_lifecycle.prepare_execution_terminal_seal(
                                execution_id=current.execution_id,
                                agent_run_ids=candidate_agent_run_ids,
                                binding_digest=current.binding_digest,
                            )
                        except Exception as error:
                            if (commit.execution.status not in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}
                                    or agent_run_id is None or commit.producer_generation is None
                                    or not await self._agent_run_lifecycle.settle_trace_buffer_failure(
                                        agent_run_id, execution_id=current.execution_id,
                                        producer_generation=commit.producer_generation, error=error,
                                    )):
                                raise
                            # Only a known rejected tail can be removed from this
                            # failure/cancellation plan. Keep its control intent and
                            # accepted prefix; unknown source outcomes are not retried.
                            plan = await self._agent_run_lifecycle.prepare_execution_terminal_seal(
                                execution_id=current.execution_id,
                                agent_run_ids=candidate_agent_run_ids,
                                binding_digest=current.binding_digest,
                            )
                if agent_run_id is not None and not state_archive:
                    await self._agent_run_lifecycle.flush_execution_projection(
                        agent_run_id,
                        execution_id=current.execution_id,
                    )
                async with self._audit_lock(current.execution_id):
                    commit, selected_checkpoint, payload_ids = await self._prepare_source_terminal_payload(
                        current, commit, checkpoint_payload=checkpoint_payload,
                    )
                    if self._execution_buffer is not None:
                        if selected_checkpoint is not None:
                            if agent_run_id is None or commit.producer_generation is None:
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                            selected_run, selected_record = await self._agent_run_lifecycle.decode_agent_run_checkpoint_payload(
                                selected_checkpoint, agent_run_id=agent_run_id,
                                producer_generation=commit.producer_generation,
                            )
                            if conversation_checkpoint is not None:
                                conversation_agent_run = selected_run
                                conversation_checkpoint = (None if selected_record is None else replace(selected_record,
                                    messages=conversation_checkpoint.messages,
                                    context_messages=conversation_checkpoint.context_messages,
                                    idempotency_key=conversation_checkpoint.idempotency_key))
                        elif self._execution_buffer.rejection_settled(current.execution_id):
                            conversation_agent_run = conversation_checkpoint = None
                        self._execution_buffer.seal(current.execution_id, payload_ids)
                    selected_audit = await self._source_audit_prefix(current.execution_id)
                    pending_count = len(selected_audit)
                    try:
                        committed = (
                            await self._commit_execution_terminal_checkpoint_locked_body(
                                current,
                                commit,
                                agent_run_id=agent_run_id,
                                expected_cursor=expected_cursor,
                                conversation_agent_run=conversation_agent_run,
                                conversation_checkpoint=conversation_checkpoint,
                                conversation_message_range=conversation_message_range,
                                                    terminal_plan=plan,
                            )
                        )
                    except AIError as source_error:
                        if (self._execution_buffer is not None and source_error.code in {
                                ErrorCode.STORAGE_CONFLICT, ErrorCode.EXECUTION_RESULT_CONFLICT}):
                            # A code alone is not a receipt: even post-commit
                            # local settlement may fail. Observe source first.
                            canonical = await self.load_execution(current.execution_id, tenant_id=self._tenant_id)
                            if canonical is None:
                                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from source_error
                            if canonical.status in {
                                    ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                                if canonical.status is commit.execution.status and canonical.result == commit.result:
                                    # Reuse full canonical/audit readback below;
                                    # never reopen a possibly committed batch.
                                    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from source_error
                                self._execution_buffer.reject_source(current.execution_id, payload_ids)
                                self._execution_buffer.discard(current.execution_id)
                            else:
                                self._execution_buffer.reject_source(current.execution_id, payload_ids)
                        raise
                    durable_commit = True
                    if self._execution_buffer is not None:
                        self._execution_buffer.confirm(current.execution_id, payload_ids)
                    self._confirm_source_audits(current.execution_id, pending_count)
                if self._execution_buffer is not None:
                    await self._execution_buffer.release(current.execution_id)
                try:
                    self._confirm_committed_events(
                        current.execution_id,
                        pending_count=pending_count,
                        durable_seq=committed.execution.event_seq,
                    )
                    if plan is not None:
                        await self._agent_run_lifecycle.finalize_execution_terminal_seal(
                            plan
                        )
                except BaseException:
                    _logger.error(
                        "local terminal finalization failed after durable commit: "
                        "execution=%s",
                        current.execution_id,
                        exc_info=True,
                    )
                    if plan is not None:
                        try:
                            await asyncio.shield(
                                self._agent_run_lifecycle.reconcile_execution_terminal_seal(
                                    plan
                                )
                            )
                        except BaseException:
                            _logger.error(
                                "local terminal seal reconciliation failed after "
                                "durable commit: execution=%s",
                                current.execution_id,
                                exc_info=True,
                            )
                return committed
            except BaseException as error:
                if (plan is not None or not current.durable_trace) and not durable_commit:
                    if (
                        isinstance(error, AIError)
                        and error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
                    ):
                        observed = await self.load_execution(
                            current.execution_id,
                            tenant_id=self._tenant_id,
                        )
                        if (
                            observed is not None
                            and observed.status
                            in {
                                ExecutionStatus.SUCCEEDED,
                                ExecutionStatus.FAILED,
                                ExecutionStatus.CANCELLED,
                            }
                        ):
                            if (
                                observed.status is not commit.execution.status
                                or observed.result != commit.result
                                or observed.error_code
                                != commit.execution.error_code
                                or observed.restart_checkpoints != current.restart_checkpoints
                                or observed.durable_trace != current.durable_trace
                                or observed.conversation != current.conversation
                            ):
                                raise AIError(
                                    ErrorCode.STORAGE_INTEGRITY_ERROR
                                ) from error
                            if not current.durable_trace:
                                if (observed.history_seal_digest is not None
                                        or commit.producer_generation is not None
                                        and observed.producer_generation != commit.producer_generation):
                                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                            try:
                                await self._runtime_commands.verify_conversation_turn(observed, commit.result)
                            except AIError as publication_error:
                                raise error from publication_error
                            if self._execution_buffer is not None:
                                expected_events = (*selected_audit, ExecutionEventAppend(
                                    commit.terminal_event_type, commit.terminal_event_payload))
                                if observed.event_seq != current.event_seq + len(expected_events):
                                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                                offset = current.event_seq
                                remaining = expected_events
                                while remaining:
                                    page = await self._execution.events.list(
                                        current.execution_id, tenant_id=self._tenant_id,
                                        after_event_seq=offset, limit=min(len(remaining), 100),
                                    )
                                    if not page.items or len(page.items) > len(remaining):
                                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                                    for actual, expected in zip(page.items, remaining, strict=False):
                                        offset += 1
                                        if (actual.event_seq != offset or actual.event_type != expected.event_type
                                                or actual.payload != expected.payload):
                                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                                    remaining = remaining[len(page.items):]
                            committed = ExecutionTerminalCommitResult(
                                observed,
                                commit.result,
                            )
                            durable_commit = True
                            if self._execution_buffer is not None:
                                self._execution_buffer.confirm(current.execution_id, payload_ids)
                                await self._execution_buffer.release(current.execution_id)
                            self._confirm_source_audits(current.execution_id, pending_count)
                            self._confirm_committed_events(
                                current.execution_id,
                                pending_count=pending_count,
                                durable_seq=observed.event_seq,
                            )
                            if plan is not None:
                                await asyncio.shield(
                                    self._agent_run_lifecycle.reconcile_execution_terminal_seal(
                                        plan
                                    )
                                )
                            return committed
                        if plan is not None:
                            await self._agent_run_lifecycle.discard_execution_terminal_seal(
                                plan
                            )
                    elif plan is not None:
                        await self._agent_run_lifecycle.discard_execution_terminal_seal(
                            plan
                        )
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
        if handoff is not None:
            completed = await handoff(committed)
            if completed is not None:
                if completed != committed.execution:
                    committed = ExecutionTerminalCommitResult(completed, committed.result)
        await self._record_committed_terminal(
            committed, session_id=current.session_id,
        )
        if cancellation is not None:
            raise cancellation
        return committed

    async def _existing_execution_agent_run_ids(
        self,
        candidate_agent_run_ids: Sequence[str],
    ) -> tuple[str, ...]:
        existing: list[str] = []
        execution_archive = self._run_stores[PersistenceDomain.TRACE]
        for agent_run_id in dict.fromkeys(candidate_agent_run_ids):
            staged = await self._run_store.get_agent_run(agent_run_id=agent_run_id)
            if staged is not None:
                existing.append(agent_run_id)
                continue
            archived = await execution_archive.get_agent_run(agent_run_id=agent_run_id)
            if archived is not None:
                existing.append(agent_run_id)
                continue
            if isinstance(execution_archive, StateStepArchive):
                head = await execution_archive.execution_history_head(agent_run_id)
                if head != (0, 0, 0, "empty"):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return tuple(existing)

    async def _commit_execution_terminal_checkpoint_locked_body(
        self,
        current: ExecutionRecord,
        commit: ExecutionTerminalCommit,
        *,
        agent_run_id: str | None,
        expected_cursor: ConversationCursor | None = None,
        conversation_agent_run: AgentRunRecord | None = None,
        conversation_checkpoint: AgentRunCheckpoint | None = None,
        conversation_message_range: tuple[int, int] | None = None,
        terminal_plan: ExecutionTerminalSealPlan | None,
    ) -> ExecutionTerminalCommitResult:
        pending_audit = await self._source_audit_prefix(current.execution_id)
        agent_run = None
        step_events: Sequence[StepEvent] = ()
        checkpoints: Sequence[AgentRunCheckpoint] = ()
        if terminal_plan is not None:
            execution_projections = terminal_plan.projections
            if execution_projections:
                agent_run = execution_projections[0].run
        else:
            execution_projections = ()
        next_cursor = None
        if current.conversation and agent_run_id is not None:
            history_id = None
            if current.session_id is not None:
                capture = self._owned_attempt_capture(current)
                session = (capture.session if capture is not None else
                    await self._load_session(current.session_id, tenant_id=self._tenant_id))
                if session is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                history_id = session.history_id or (
                    None
                    if session.continuation is None
                    else session.continuation.history_id
                )
            next_cursor = ConversationCursor(agent_run_id, history_id=history_id)
        owner = self._turn_owners.get(current.execution_id)
        committed = await self._runtime_commands.commit_terminal_checkpoint(
            commit,
            start_identity=(None if owner is None or owner.reservation.idempotency.status is not IdempotencyStatus.STARTED
                            else owner.reservation.idempotency),
            session_turn=None if owner is None else owner.session_turn,
            expected_execution=current,
            session_id=current.session_id,
            expected_cursor=expected_cursor,
            next_cursor=next_cursor,
            conversation_agent_run=conversation_agent_run,
            conversation_checkpoint=conversation_checkpoint,
            conversation_message_range=conversation_message_range,
            execution_run=agent_run,
            execution_events=step_events,
            execution_checkpoints=checkpoints,
            execution_projections=execution_projections,
            capability_run_id=agent_run_id,
            audit_events=pending_audit,
            background_tasks=self._execution_task_set(current.execution_id),
        )
        return committed

    def _capture_usage(self, execution_id: str, usage: UsageMetrics) -> None:
        self._captured_usage[execution_id] = usage
        _logger.debug(
            "execution usage captured: execution=%s requests=%s tool_calls=%s total_tokens=%s",
            execution_id,
            usage.model_requests,
            usage.tool_calls,
            usage.total_tokens,
        )

    async def verify_terminal_projection(
        self, execution: ExecutionRecord, status: ExecutionStatus, agent_run_id: "str | None"
    ) -> None:
        candidates = tuple(
            make_agent_run_id(
                namespace=self._namespace,
                tenant_id=self._tenant_id,
                execution_id=execution.execution_id,
                agent_run_seq=sequence,
            )
            for sequence in range(1, execution.agent_run_seq + 1)
        )
        await self._agent_run_lifecycle.verify_terminal_attempts(
            candidate_agent_run_ids=candidates,
            required_agent_run_id=agent_run_id
            if status is ExecutionStatus.SUCCEEDED and (
                execution.conversation or execution.restart_checkpoints or execution.durable_trace
            ) else None,
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
        event_seq=record.event_seq + 1,
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
        payload["error_diagnostics"] = error_diagnostics.to_payload()
    return payload


def _execution_error_code(error: Exception) -> ErrorCode:
    if isinstance(error, ValidationError):
        return ErrorCode.OUTPUT_VALIDATION_FAILED
    if isinstance(error, AIError):
        return error.code
    return ErrorCode.INTERNAL_ERROR


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
        ErrorCode.AGENT_BINDING_UNAVAILABLE,
        ErrorCode.EXECUTION_HISTORY_UNAVAILABLE,
        ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
        ErrorCode.SERVICE_NOT_READY,
    }


__all__ = ["LocalExecutionBackend"]
