#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution query API and the persistence-backed default service."""

import asyncio
import re
import uuid
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from typing import TYPE_CHECKING, Protocol

from linktools.core import environ

from ..execution import (TaskExecutionService, ExecutionReleaseCallback as _ExecutionReleaseCallback,
    LocalExecutionWaiter as _LocalExecutionWaiter, overlay_execution_correlation as _overlay_execution_correlation,
    consumed_terminal_query, observed_query as _observed_query)
from ..persistence.commands import StoredExecutionManagementAdapter


from ..execution import (ExecutionManager, ExecutionDriver, CancelEffectOutcome,
    terminal_error as _terminal_error,
    execution_recovery_error as _execution_recovery_error,
    operation_result as _operation_result,
    operation_failure as _operation_failure,
    operation_status as _operation_status,
    next_execution as _next_execution,
    terminal_idempotency_matches as _terminal_idempotency_matches,
    stable_idempotency_error as _stable_idempotency_error,
    stable_operation_error as _stable_operation_error,
)

from ..model import validate_model_inputs

from ..agent import (
    AgentBinding,
    AgentBindingContract,
    AgentCatalog,
    AgentCompiler,
    restore_output,
)
from ..core import (
    BudgetUsage,
    AuthorizationAction,
    AuthorizationPolicy,
    CorrelationData,
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionMode,
    ExecutionStatus,
    IdempotencyStatus,
    JsonValue,
    OperationKind,
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationStatus,
    Page,
    Principal,
    RUNTIME_OBJECT_STORE_ID,
    ResourceKind,
    ResourceRef,
    StopReason,
    UsageMetrics,
    WorkspaceFileInput,
    canonical_json_bytes,
    canonical_sha256,
    normalize_json_value,
    overlay_correlation,
    principal_identity_payload,
    validate_idempotency_key,
)
from ..core import (
    idempotency_key_digest as compute_idempotency_key_digest,
)
from ..asset import AssetVersionRef
from ..session import SessionAdmission
from ..spec import RepositoryInstructions
from ..errors import AIError, ErrorCode, ErrorDiagnostics
from ..task import TaskBindingContract, TaskEffectResolution
from ..storage import (
    ObjectRef,
    ObjectStore,
    PayloadPolicy,
    StoredPayload,
    TurnCoordination,
    payload_fits_inline,
)
from ._conversation_service import ConversationService
from ..inputs import ExecutionContextArchive, PreparedExecutionContext
from ..capability import ToolResponseFixture
from ._tool_response_fixture import _load_tool_response_fixture
from ..core import HandoffGate, HandoffState
from ..inputs import (
    ExecutionInputMaterializer,
    decode_user_content_payload,
    input_intent,
    validate_user_input,
)
from ..persistence.objects import ObjectKeyFactory, put_runtime_object, read_runtime_object
from ..execution import ExecutionRecoveryEffect, ResolveToolEffectRequest, ToolEffectResolutionResult
from ..execution import (
    CancelExecutionRequest,
    CancelExecutionResult,
    ExecutionHandle,
    ExecutionHistoryItem,
    ExecutionHistoryReader,
    ExecutionHistoryService,
    ExecutionRequest,
    ExecutionResult,
    ExecutionTraceItem,
    ExecutionView,
    ForkExecutionRequest,
    ListExecutionRequest,
    ModelInteractionItem,
    RetryExecutionRequest,
    TranscriptItem,
    UsageReadCutoff,
)
from ..execution import project_execution_view as _project_execution_view
from ..persistence import PersistenceDomain
from ._pending_turn import _PendingExecutionTurns
from ..persistence.coordination import PendingTurnCoordination
from ..persistence.repositories import ExecutionRepositoryImpl
from ..persistence.objects import runtime_object_source_reads
from ..execution import (
    ExecutionCancelRequestCommit,
    ExecutionRecord,
    ExecutionStartClaim,
    ExecutionStartReservation,
    ExecutionStartUnknownCommit,
    ExecutionTerminalCommit,
    ExecutionTerminalCommitResult,
    IdempotencyRecord,
    IdempotencyTerminalUpdate,
    OperationTerminalUpdate,
    ResultRecord,
    StoredUserInput,
)
from ..persistence.contracts import ExecutionRepositories, SessionRepository, TaskAdmissionRepository

if TYPE_CHECKING:
    from ..persistence.state import StateTransaction
    from ..execution import ExecutionRecoveryEffect, ResolveToolEffectRequest, ToolEffectResolutionResult
    from ..execution import RuntimePayloadRef
    from ..session import SessionRecord
    from ..spec import RepositoryInstructionResolver

_logger = environ.get_logger("ai.runtime.execution")


class _ExecutionTerminalVerifier(Protocol):
    async def __call__(
        self,
        execution: ExecutionRecord,
        status: ExecutionStatus,
        required_agent_run_id: "str | None",
    ) -> None: ...


class _SubagentCancellation(Protocol):
    async def cancel_children(
        self, parent_execution_id: str, principal: Principal
    ) -> None: ...


async def _missing_terminal_verifier(
    execution: ExecutionRecord,
    status: ExecutionStatus,
    required_agent_run_id: "str | None",
) -> None:
    del execution, status, required_agent_run_id
    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)


@dataclass
class _SessionLockEntry:
    lock: asyncio.Lock
    ref_count: int = 0


@dataclass(frozen=True, slots=True)
class ExecutionStartIdentity:
    scope: str
    idempotency_key_digest: str
    request_digest: str


@dataclass(frozen=True, slots=True)
class _ExecutionStartContext:
    request: ExecutionRequest
    canonical_files: tuple[str, ...]
    request_intent_digest: str
    stored_user_input: StoredUserInput | None = None


class ExecutionBackend(Protocol):
    async def prepare_start(
        self,
        request: ExecutionRequest,
        execution: ExecutionRecord,
        identity: ExecutionStartIdentity,
    ) -> ExecutionRecord: ...
    async def abort_start(self, execution: ExecutionRecord) -> None: ...
    async def launch(
        self, request: ExecutionRequest, execution: ExecutionRecord
    ) -> None: ...

    async def commit_cancel_checkpoint(
        self,
        commit: ExecutionCancelRequestCommit,
        *,
        expected_status: ExecutionStatus,
    ) -> ExecutionRecord: ...
    async def cancel(self, execution: ExecutionRecord, *, force: bool = False) -> "CancelEffectOutcome": ...
    def worker_failure(
        self, execution_id: str, *, tenant_id: str
    ) -> AIError | None: ...
    def worker_installed(self, execution_id: str) -> bool: ...
    def owns_execution(self, execution_id: str, *, tenant_id: str) -> bool: ...
    async def wait_terminal(self, execution_id: str, *, tenant_id: str) -> None: ...
    async def recovery_effects(
        self, execution_id: str, *, tenant_id: str
    ) -> "tuple[ExecutionRecoveryEffect, ...]": ...
    async def resolve_tool_effect(
        self,
        execution_id: str,
        request: "ResolveToolEffectRequest",
    ) -> "ToolEffectResolutionResult": ...
    async def recover_execution(
        self, execution_id: str, *, tenant_id: str, expected_revision: int,
        recovery_operation: OperationLedgerInput | None = None,
    ) -> ExecutionRecord: ...
    async def persist_cancel_intent(
        self,
        execution: ExecutionRecord,
        operation: OperationLedgerInput,
    ) -> OperationLedgerRecord: ...


_TerminalMetricDelivery = Callable[[], Awaitable[None]]


class _ExecutionRuntimePort(ExecutionBackend, Protocol):
    async def load_repository_instructions(
        self, reference: "RuntimePayloadRef | None",
    ) -> RepositoryInstructions | None: ...

    async def commit_terminal_checkpoint(
        self,
        commit: ExecutionTerminalCommit,
        *,
        session_id: str | None,
        expected_execution: ExecutionRecord | None = None,
        capture_metric: Callable[[_TerminalMetricDelivery], None] | None = None,
    ) -> ExecutionTerminalCommitResult: ...

    async def verify_terminal_projection(
        self,
        execution: ExecutionRecord,
        status: ExecutionStatus,
        required_agent_run_id: str | None,
    ) -> None: ...

    async def cancel_children(
        self,
        parent_execution_id: str,
        principal: Principal,
    ) -> None: ...


class _LiveExecutionStreamBroker(Protocol):
    def prepare_local_producer(self, execution_id: str) -> None: ...

    def abandon_prepared_local_producer(self, execution_id: str) -> None: ...


class _ExecutionRuntimeBridge:
    def __init__(self) -> None:
        self._port: _ExecutionRuntimePort | None = None
        self.pending_turns = _PendingExecutionTurns()
        self.pending_start: Callable[
            [ExecutionRequest, ExecutionStartReservation, "SessionRecord | None", TurnCoordination | None],
            Awaitable[str | None],
        ] | None = None
        self.pending_promote: Callable[[str], Awaitable[None]] | None = None
        self.pending_cancel_promote: Callable[[str], Awaitable[None]] | None = None

    def bind(self, runtime_port: _ExecutionRuntimePort) -> None:
        if self._port is not None:
            raise RuntimeError("execution runtime bridge is already bound")
        self._port = runtime_port

    @property
    def port(self) -> _ExecutionRuntimePort | None:
        return self._port


class _ExecutionManagementAdapter(StoredExecutionManagementAdapter):
    """Add Agent pending-turn coordination and live execution control."""

    def __init__(self, service: "DefaultExecutionService") -> None:
        self._service = service
        super().__init__(service._state, service._object_store,
            object_key_factory=service._object_key_factory, object_reader=service._object_reader)

    async def read_execution(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        return await self._service.load_execution(execution_id, tenant_id=tenant_id)

    @property
    def driver(self) -> ExecutionDriver | None:
        return self._service._runtime_bridge.port

    @property
    def has_pending_cancel_promotion(self) -> bool:
        return self._service._runtime_bridge.pending_cancel_promote is not None

    async def promote_pending_cancel(self, execution_id: str) -> None:
        promote = self._service._runtime_bridge.pending_cancel_promote
        if promote is not None:
            await promote(execution_id)

    async def read_header(self, execution_id: str, *, tenant_id: str) -> ResourceRef | None:
        service = self._service
        pending = service._runtime_bridge.pending_turns.get(execution_id, tenant_id=tenant_id)
        header = (ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id,
                              pending.execution.principal_id) if pending is not None
                  else await service._state.executions.get_header(execution_id, tenant_id=tenant_id))
        if header is None:
            record = await service._runtime_bridge.pending_turns.read_remote_execution(execution_id, tenant_id=tenant_id)
            if record is not None:
                header = ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id, record.principal_id)
            else:
                header = await service.load_source_header(execution_id, tenant_id=tenant_id)
        return header

    def has_local_pending(self, execution_id: str, *, tenant_id: str) -> bool:
        return self._service._runtime_bridge.pending_turns.get(execution_id, tenant_id=tenant_id) is not None

    async def read_remote_pending(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        return await self._service._runtime_bridge.pending_turns.read_remote_execution(execution_id, tenant_id=tenant_id)

    async def signal_remote_cancel(self, execution_id: str, *, tenant_id: str) -> None:
        await self._service._runtime_bridge.pending_turns.cancel_remote(execution_id, tenant_id=tenant_id)


class DefaultExecutionService(TaskExecutionService):
    """Apply execution authorization and state transitions before launching work."""

    def __init__(
        self,
        state: ExecutionRepositories,
        object_store: ObjectStore,
        authorization: AuthorizationPolicy,
        *,
        sessions: SessionRepository,
        catalog: AgentCatalog,
        compiler: AgentCompiler,
        runtime_bridge: _ExecutionRuntimeBridge,
        live_broker: _LiveExecutionStreamBroker,
        operation_ids: "Callable[[], str] | None" = None,
        history_reader: ExecutionHistoryReader,
        history_service: "ExecutionHistoryService | None" = None,
        release_terminal: _ExecutionReleaseCallback | None = None,
        instruction_resolver: "RepositoryInstructionResolver | None" = None,
        object_key_factory: "ObjectKeyFactory | None" = None,
        payload_policy: "PayloadPolicy | None" = None,
        input_materializer: "ExecutionInputMaterializer | None" = None,
        session_execution_ready: bool = True,
        task_admissions: TaskAdmissionRepository | None = None,
        tool_responses: ToolResponseFixture | None = None,
        object_reader: "Callable[[PersistenceDomain, ObjectRef], Awaitable[bytes]] | None" = None,
        restart_checkpoints: bool = False,
        durable_trace: bool = False,
        conversation: bool = False,
        terminal_commit: bool = False,
        context_archive: ExecutionContextArchive | None = None,
    ) -> None:
        if any(not isinstance(value, bool) for value in (
            restart_checkpoints, durable_trace, conversation, terminal_commit,
        )):
            raise TypeError("execution persistence guarantees must be bool")
        self._state = state
        self._object_store = object_store
        self._authorization = authorization
        self._object_key_factory = object_key_factory
        self._object_reader = object_reader
        self._sessions = sessions
        self._task_admissions = task_admissions
        self._operation_ids = operation_ids or (lambda: uuid.uuid4().hex)
        self._payload_policy = payload_policy
        super().__init__(
            state, authorization, port=_ExecutionManagementAdapter(self), sessions=sessions,
            task_admissions=task_admissions, operation_ids=operation_ids,
            release_terminal=release_terminal, payload_policy=payload_policy,
        )
        self._restart_checkpoints = restart_checkpoints
        self._durable_trace = durable_trace
        self._conversation = conversation
        self._terminal_commit = terminal_commit
        self._context_archive = context_archive
        self._tool_responses = tool_responses
        self._catalog = catalog
        self._compiler = compiler
        self._runtime_bridge = runtime_bridge
        self._live_broker = live_broker
        self._history_reader = history_reader
        self._history_service = history_service
        if not isinstance(session_execution_ready, bool):
            raise TypeError("session_execution_ready must be bool")
        if instruction_resolver is not None and (
            object_key_factory is None or payload_policy is None
        ):
            raise ValueError(
                "repository instruction resolver requires object key factory and payload policy"
            )
        self._instruction_resolver = instruction_resolver
        self._input_materializer = input_materializer
        self._session_execution_ready = session_execution_ready
        self._session_locks: dict[tuple[str, str], _SessionLockEntry] = {}
        self._session_locks_guard = asyncio.Lock()
        self._accepting = True
        self._admission_tasks: set[asyncio.Task[ExecutionHandle] | asyncio.Future[None]] = set()
        self._cancel_tasks: set[asyncio.Task[CancelExecutionResult]] = set()
        self._detached_cancel_finalizers: set[asyncio.Task[CancelExecutionResult]] = (
            set()
        )
        self._detached_cancel_failure: AIError | None = None
        self._management = self.manager

    @property
    def _backend(self) -> _ExecutionRuntimePort | None:
        return self._runtime_bridge.port

    @property
    def _terminal_committer(self) -> _ExecutionRuntimePort | None:
        return self._runtime_bridge.port

    @property
    def _terminal_verifier(self) -> _ExecutionTerminalVerifier:
        runtime_port = self._runtime_bridge.port
        if runtime_port is None:
            return _missing_terminal_verifier
        return runtime_port.verify_terminal_projection

    @property
    def local_waiter(self) -> _LocalExecutionWaiter | None:
        return self._runtime_bridge.port

    @property
    def _subagent_cancellation(self) -> _SubagentCancellation | None:
        return self._runtime_bridge.port

    def runtime_backend(self) -> ExecutionBackend:
        backend = self._runtime_bridge.port
        if backend is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return backend

    def abandon_prepared_stream(self, execution_id: str) -> None:
        self._live_broker.abandon_prepared_local_producer(execution_id)

    async def _materialize_repository_instructions(
        self,
        instructions: RepositoryInstructions,
        *,
        tenant_id: str,
    ) -> "RuntimePayloadRef":
        from ..execution import RuntimePayloadRef

        if self._object_key_factory is None or self._payload_policy is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        payload = instructions.to_payload()
        inline = StoredPayload.inline_json(payload)
        if payload_fits_inline(inline, self._payload_policy):
            stored = inline
        else:
            reference = await put_runtime_object(
                self._object_store,
                self._object_key_factory,
                PersistenceDomain.EXECUTION,
                tenant_id,
                canonical_json_bytes(payload),
            )
            stored = StoredPayload.object(reference)
        return RuntimePayloadRef(stored, PersistenceDomain.EXECUTION)

    def _validate_tool_response_mode(
        self, reference: AssetVersionRef | None, *, session_id: str | None,
    ) -> None:
        if reference is not None:
            if self._tool_responses is None:
                raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING,
                              safe_details={"kind": "tool_response_fixture_reader"})
            if session_id is not None:
                raise AIError(ErrorCode.BINDING_CONFLICT,
                              safe_details={"reason": "tool_response_fixture_session_unsupported"})
        elif self._tool_responses is not None:
            raise AIError(ErrorCode.BINDING_CONFLICT,
                          "a tool response fixture cannot replace an admitted live execution")

    async def _validate_tool_response_input(
        self, reference: AssetVersionRef | None, *, session_id: str | None,
    ) -> None:
        self._validate_tool_response_mode(reference, session_id=session_id)
        if reference is not None:
            assert self._tool_responses is not None
            await _load_tool_response_fixture(self._tool_responses, reference)

    async def _canonicalize_request(
        self,
        request: ExecutionRequest,
    ) -> _ExecutionStartContext:
        if self._input_materializer is None:
            if request.files:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            canonical_prompt = validate_user_input(request.user_prompt)
            if not isinstance(canonical_prompt, str) and any(
                isinstance(item, WorkspaceFileInput)
                for item in canonical_prompt
            ):
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            canonical_files = ()
        else:
            canonical_prompt = await self._input_materializer.canonicalize_input(
                request.user_prompt
            )
            canonical_files = await self._input_materializer.canonicalize_files(
                request.files
            )
        if self._input_materializer is None:
            intent = input_intent(canonical_prompt, canonical_files)
        else:
            intent = self._input_materializer.intent(
                canonical_prompt,
                canonical_files,
            )
        return _ExecutionStartContext(
            request=replace(
                request,
                user_prompt=canonical_prompt,
                files=canonical_files,
            ),
            canonical_files=canonical_files,
            request_intent_digest=intent.digest,
        )

    async def _freeze_input(
        self,
        context: _ExecutionStartContext,
        binding: AgentBinding,
    ) -> _ExecutionStartContext:
        request = context.request
        if self._input_materializer is None:
            canonical = validate_user_input(request.user_prompt)
            validate_model_inputs(binding.compiled_agent.model, canonical)
            if isinstance(canonical, str):
                stored = StoredUserInput(
                    "text",
                    StoredPayload.inline_text(canonical),
                )
            else:
                from ._input import _encode_user_content

                stored = StoredUserInput(
                    "user-content-v1",
                    StoredPayload.inline_json(_encode_user_content(canonical)),
                )
        else:
            canonical = await self._input_materializer.materialize(
                request.user_prompt,
                context.canonical_files,
            )
            validate_model_inputs(binding.compiled_agent.model, canonical)
            stored = await self._input_materializer.store(
                canonical,
                tenant_id=request.principal.tenant_id,
            )
        return replace(
            context,
            request=replace(request, user_prompt=canonical, files=()),
            stored_user_input=stored,
        )

    async def lookup_session_admission(
        self, session_id: str, *, principal: Principal, idempotency_key: str,
    ) -> SessionAdmission | None:
        """Read one session request without changing pending execution ownership."""
        key = compute_idempotency_key_digest(validate_idempotency_key(idempotency_key))
        scope = "session.resume"
        pending_turns = self._runtime_bridge.pending_turns
        pending = pending_turns.peek_request(
            tenant_id=principal.tenant_id, scope=scope, idempotency_key_digest=key,
        )
        identity = await self._state.idempotency.get(scope, key, tenant_id=principal.tenant_id)
        record = None
        if identity is None:
            if pending is not None:
                if pending.uncertain or pending.failure is not None:
                    raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
                record = pending.execution
            elif pending_turns.coordination is not None:
                record = await pending_turns.coordination.read_request_execution(
                    f"{scope}:{key}", tenant_id=principal.tenant_id,
                )
                if record is None:
                    # The owner may have committed and released between reads.
                    identity = await self._state.idempotency.get(scope, key, tenant_id=principal.tenant_id)
        if identity is not None:
            if (identity.resource_kind is not ResourceKind.EXECUTION
                    or identity.scope != scope or identity.idempotency_key_digest != key
                    or (pending is not None and pending.execution.execution_id != identity.resource_id)):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if identity.status in {IdempotencyStatus.RESERVED, IdempotencyStatus.START_UNKNOWN}:
                raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
            record = await self.load_source_execution(identity.resource_id, tenant_id=principal.tenant_id)
            if record is None:
                if identity.status in {IdempotencyStatus.FAILED, IdempotencyStatus.CANCELLED}:
                    fallback = (ErrorCode.EXECUTION_START_PERSISTENCE_FAILED
                                if identity.status is IdempotencyStatus.FAILED else ErrorCode.EXECUTION_CANCELLED)
                    raise _stable_idempotency_error(identity.error_code, fallback)
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if identity.resource_id != record.execution_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if record is None:
            return None
        await self._authorization.authorize(
            principal, AuthorizationAction.EXECUTION_READ,
            ResourceRef(ResourceKind.EXECUTION, record.execution_id, principal.tenant_id, record.principal_id),
        )
        if record.session_id != session_id:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        if record.started_at is None:
            if identity is not None and identity.status in {IdempotencyStatus.FAILED, IdempotencyStatus.CANCELLED}:
                fallback = (ErrorCode.EXECUTION_START_PERSISTENCE_FAILED
                            if identity.status is IdempotencyStatus.FAILED else ErrorCode.EXECUTION_CANCELLED)
                raise _stable_idempotency_error(identity.error_code, fallback)
            raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
        if identity is not None and record.status in {
            ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
        }:
            if identity.status is IdempotencyStatus.STARTED:
                # The first identity read may precede the atomic terminal commit.
                refreshed = await self._state.idempotency.get(scope, key, tenant_id=principal.tenant_id)
                if (refreshed is None or refreshed.request_digest != identity.request_digest
                        or refreshed.scope != scope or refreshed.idempotency_key_digest != key):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                identity = refreshed
            if not _terminal_idempotency_matches(identity, record):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if record.stored_user_input is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if self._input_materializer is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        prompt = await self._input_materializer.restore(record.stored_user_input)
        return SessionAdmission(record.execution_id, prompt)

    async def _request_for_execution(
        self,
        request: ExecutionRequest,
        execution: ExecutionRecord,
    ) -> ExecutionRequest:
        stored = execution.stored_user_input
        if stored is None:
            return replace(request, tool_response_ref=execution.tool_response_ref)
        if self._input_materializer is not None:
            prompt = await self._input_materializer.restore(stored)
        elif stored.codec == "text":
            prompt = stored.payload.decode()
        else:
            payload = stored.payload.decode()
            if not isinstance(payload, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            prompt = decode_user_content_payload(payload)
        return replace(request, user_prompt=prompt, files=(), tool_response_ref=execution.tool_response_ref)

    def _binding(
        self,
        binding_digest: str,
        binding_contract: "AgentBindingContract | None" = None,
        resolved_binding: AgentBinding | None = None,
    ) -> AgentBinding:
        if binding_contract is None:
            return self._catalog.binding(binding_digest)
        if binding_contract.binding_digest != binding_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        binding = resolved_binding if resolved_binding is not None else self._compiler.restore(binding_contract)
        if (
            binding.binding_digest != binding_digest
            or binding.binding_contract != binding_contract
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return binding

    def _validate_replayed_execution(
        self,
        execution: ExecutionRecord,
        binding: AgentBinding,
        request: ExecutionRequest,
        *,
        requires_task_invocation_capture: bool = False,
        budget_scope_id: str | None = None,
    ) -> None:
        expected_scope = budget_scope_id or (
            None if request.budget is None else "execution:" + execution.execution_id
        )
        if (
            execution.requires_task_invocation_capture is not requires_task_invocation_capture
            or execution.budget_scope_id != expected_scope
            or execution.tool_response_ref != request.tool_response_ref
            or execution.binding_digest != binding.binding_digest
            or execution.planning is not request.planning
            or execution.thinking is not request.thinking
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


    async def start(
        self,
        binding_digest: str,
        request: ExecutionRequest,
        *,
        dependency_hold_id: "str | None" = None,
        binding_contract: "AgentBindingContract | None" = None,
        resolved_binding: AgentBinding | None = None,
        requires_task_invocation_capture: bool = False,
        budget_scope_id: str | None = None,
    ) -> ExecutionHandle:
        if request.tool_response_ref is None:
            request = replace(request, tool_response_ref=None if self._tool_responses is None else self._tool_responses.ref)
        return await self._start(
            binding_digest,
            request,
            scope="execution.run",
            prepare_local_stream=True,
            dependency_hold_id=dependency_hold_id,
            requires_task_invocation_capture=requires_task_invocation_capture,
            budget_scope_id=budget_scope_id,
            binding_contract=binding_contract,
            resolved_binding=resolved_binding,
        )

    async def resolve_existing(
        self,
        binding_digest: str,
        request: ExecutionRequest,
        *,
        binding_contract: "AgentBindingContract | None" = None,
        requires_task_invocation_capture: bool = False,
        budget_scope_id: str | None = None,
    ) -> "ExecutionHandle | None":
        if request.tool_response_ref is None:
            request = replace(request, tool_response_ref=None if self._tool_responses is None else self._tool_responses.ref)
        if budget_scope_id is not None and (
            not isinstance(budget_scope_id, str) or not budget_scope_id.strip()
            or request.budget is not None
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not isinstance(requires_task_invocation_capture, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if re.fullmatch(r"[0-9a-f]{64}", binding_digest) is None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        binding = self._binding(binding_digest, binding_contract)
        context = await self._canonicalize_request(request)
        request = context.request
        if budget_scope_id is not None:
            await self.authorize_budget_scope(budget_scope_id, request.principal)
        scope = "execution.run"
        idempotency_key_digest = compute_idempotency_key_digest(request.idempotency_key)
        request_digest = _request_digest(
            request,
            binding_digest,
            session_id=None,
            previous_execution_id=None,
            fork_base_execution_id=None,
            parent_execution_id=None,
            root_execution_id=None,
            parent_invocation_id=None,
            lineage_kind=ExecutionLineageKind.RUN,
            request_intent_digest=context.request_intent_digest,
            requires_task_invocation_capture=requires_task_invocation_capture,
            budget_scope_id=budget_scope_id,
        )
        admitted = await self._management.resolve_admission(
            scope, idempotency_key_digest, request_digest, request.principal,
        )
        if admitted is None:
            return None
        existing, execution = admitted
        self._validate_replayed_execution(
            execution, binding, request,
            requires_task_invocation_capture=requires_task_invocation_capture,
            budget_scope_id=budget_scope_id,
        )
        if execution.budget_scope_id is not None:
            await self._state.budgets.read(execution.budget_scope_id)
        if existing.status is IdempotencyStatus.COMPLETED:
            if not _terminal_idempotency_matches(existing, execution):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return ExecutionHandle(execution.execution_id)
        if (
            existing.status is IdempotencyStatus.STARTED
            and execution.status
            in {
                ExecutionStatus.WAITING_DEFERRED,
                ExecutionStatus.FINALIZING,
                ExecutionStatus.RECOVERY_REQUIRED,
                ExecutionStatus.CANCELLING,
            }
        ):
            return ExecutionHandle(execution.execution_id)
        if (
            existing.status is IdempotencyStatus.STARTED
            and execution.status is ExecutionStatus.STARTED
        ):
            request = await self._request_for_execution(request, execution)
            await self._launch_started(
                request,
                execution,
                scope=scope,
                idempotency_key_digest=idempotency_key_digest,
                prepare_local_stream=True,
            )
            return ExecutionHandle(execution.execution_id)
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def start_for_session(
        self,
        agent_id: str,
        binding_digest: str,
        session_id: str,
        request: ExecutionRequest,
        *,
        binding_contract: "AgentBindingContract | None" = None,
        resolved_binding: AgentBinding | None = None,
        dependency_hold_id: "str | None" = None,
        requires_task_invocation_capture: bool = False,
        budget_scope_id: str | None = None,
    ) -> ExecutionHandle:
        if not self._session_execution_ready:
            raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING, "Session management is disabled")
        if request.tool_response_ref is None:
            request = replace(request, tool_response_ref=None if self._tool_responses is None else self._tool_responses.ref)
        self._validate_tool_response_mode(request.tool_response_ref, session_id=session_id)
        if not session_id.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return await self._start(
            binding_digest,
            request,
            session_id=session_id,
            session_agent_id=agent_id,
            scope="session.resume",
            prepare_local_stream=True,
            dependency_hold_id=dependency_hold_id,
            requires_task_invocation_capture=requires_task_invocation_capture,
            budget_scope_id=budget_scope_id,
            binding_contract=binding_contract,
            resolved_binding=resolved_binding,
        )

    async def start_subagent(
        self,
        binding_digest: str,
        request: ExecutionRequest,
        *,
        parent_execution_id: str,
        root_execution_id: str,
        parent_invocation_id: str,
        binding_contract: "AgentBindingContract | None" = None,
        resolved_binding: AgentBinding | None = None,
    ) -> ExecutionHandle:
        return await self._start(
            binding_digest,
            request,
            parent_execution_id=parent_execution_id,
            root_execution_id=root_execution_id,
            parent_invocation_id=parent_invocation_id,
            lineage_kind=ExecutionLineageKind.SUBAGENT,
            scope="execution.subagent",
            binding_contract=binding_contract,
            resolved_binding=resolved_binding,
        )

    async def replay_subagent(
        self,
        *,
        agent_id: str,
        user_prompt: str,
        files: tuple[str, ...] = (),
        principal: Principal,
        idempotency_key: str,
        memory_scope: "str | None",
        mode: ExecutionMode,
        parent_execution_id: str,
        root_execution_id: str,
        parent_invocation_id: str,
    ) -> "ExecutionHandle | None":
        """Replay an already-established child using its own durable binding."""
        idempotency_key_digest = compute_idempotency_key_digest(idempotency_key)
        existing = await self._state.idempotency.get(
            "execution.subagent",
            idempotency_key_digest,
            tenant_id=principal.tenant_id,
        )
        if existing is None:
            return None
        execution = await self.load_execution(
            existing.resource_id,
            tenant_id=principal.tenant_id,
        )
        if execution is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        binding = self._binding(execution.binding_digest, execution.binding)
        if binding.compiled_agent.spec.id != agent_id:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        if (
            execution.lineage_kind is not ExecutionLineageKind.SUBAGENT
            or execution.parent_execution_id != parent_execution_id
            or execution.root_execution_id != root_execution_id
            or execution.parent_invocation_id != parent_invocation_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        request = ExecutionRequest(
            user_prompt=user_prompt,
            principal=principal,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            mode=mode,
            planning=execution.planning,
            thinking=execution.thinking,
            correlation=execution.correlation,
            files=files,
            tool_response_ref=execution.tool_response_ref,
        )
        return await self.start_subagent(
            execution.binding_digest,
            request,
            parent_execution_id=parent_execution_id,
            root_execution_id=root_execution_id,
            parent_invocation_id=parent_invocation_id,
            binding_contract=execution.binding,
        )

    async def list_children(
        self, execution_id: str, *, principal: Principal
    ) -> tuple[ExecutionView, ...]:
        async with self.execution_consumer(execution_id, principal.tenant_id):
            pending = self._runtime_bridge.pending_turns.get(execution_id, tenant_id=principal.tenant_id)
            if pending is not None:
                await self.authorize_execution(execution_id, principal, AuthorizationAction.EXECUTION_READ)
                if self._runtime_bridge.pending_turns.get(execution_id, tenant_id=principal.tenant_id) is pending:
                    # Child admission first promotes this parent to source authority.
                    return ()
            repository = self._state.executions
            if type(repository) is ExecutionRepositoryImpl:
                header = await repository.get_header(
                    execution_id, tenant_id=principal.tenant_id
                )
                remote_pending = None
                if header is None:
                    header = await self.authorize_execution(
                        execution_id, principal, AuthorizationAction.EXECUTION_READ,
                    )
                    remote_pending = await self._runtime_bridge.pending_turns.read_remote_execution(
                        execution_id, tenant_id=principal.tenant_id,
                    )
                    if remote_pending is not None:
                        await self.authorize_loaded_execution(
                            remote_pending, principal, AuthorizationAction.EXECUTION_READ,
                            owner_principal_id=header.owner_principal_id,
                        )
                else:
                    await self._authorization.authorize(
                        principal, AuthorizationAction.EXECUTION_READ, header,
                    )

                async def read(
                    _transaction: "StateTransaction",
                ) -> tuple[ExecutionRecord | None, tuple[ExecutionView, ...]]:
                    current = await repository.get(
                        execution_id, tenant_id=principal.tenant_id,
                    )
                    if current is None:
                        if remote_pending is not None:
                            # A child cannot exist before its parent is promoted.
                            return None, ()
                        raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                    children = await repository.list_children(
                        execution_id, tenant_id=principal.tenant_id,
                    )
                    return current, tuple(_execution_view(child) for child in children)

                # Authorization may await arbitrary application work; only the
                # known metadata owner shares the post-authorization snapshot.
                current, children = await repository.state_store.read(read)
                if current is not None:
                    await self.authorize_loaded_execution(
                        current, principal, AuthorizationAction.EXECUTION_READ,
                        owner_principal_id=header.owner_principal_id,
                    )
                return children
            await self.load_authorized(
                execution_id, principal, AuthorizationAction.EXECUTION_READ
            )
            children = await self._state.executions.list_children(
                execution_id,
                tenant_id=principal.tenant_id,
            )
            return tuple(_execution_view(child) for child in children)

    async def _start(
        self,
        binding_digest: str,
        request: ExecutionRequest,
        *,
        session_id: "str | None" = None,
        session_agent_id: "str | None" = None,
        previous_execution_id: "str | None" = None,
        fork_base_execution_id: "str | None" = None,
        conversation_agent_run_id: "str | None" = None,
        parent_execution_id: "str | None" = None,
        root_execution_id: "str | None" = None,
        parent_invocation_id: "str | None" = None,
        lineage_kind: ExecutionLineageKind = ExecutionLineageKind.RUN,
        scope: str = "execution.run",
        prepare_local_stream: bool = False,
        dependency_hold_id: "str | None" = None,
        binding_contract: "AgentBindingContract | None" = None,
        resolved_binding: AgentBinding | None = None,
        requires_task_invocation_capture: bool = False,
        budget_scope_id: str | None = None,
    ) -> ExecutionHandle:
        metric_delivery: _TerminalMetricDelivery | None = None

        def capture_metric(delivery: _TerminalMetricDelivery) -> None:
            nonlocal metric_delivery
            metric_delivery = delivery

        # Track the caller-owned admission, without shielding its cancellation.
        admission_done = asyncio.get_running_loop().create_future()
        self._admission_tasks.add(admission_done)
        try:
            if session_id is None:
                return await self._start_unlocked(
                    binding_digest,
                    request,
                    session_id=session_id,
                    session_agent_id=session_agent_id,
                    previous_execution_id=previous_execution_id,
                    fork_base_execution_id=fork_base_execution_id,
                    conversation_agent_run_id=conversation_agent_run_id,
                    parent_execution_id=parent_execution_id,
                    root_execution_id=root_execution_id,
                    parent_invocation_id=parent_invocation_id,
                    lineage_kind=lineage_kind,
                    scope=scope,
                    prepare_local_stream=prepare_local_stream,
                    dependency_hold_id=dependency_hold_id,
                    requires_task_invocation_capture=requires_task_invocation_capture,
                    budget_scope_id=budget_scope_id,
                    binding_contract=binding_contract,
                    resolved_binding=resolved_binding,
                    capture_metric=capture_metric,
                )
            async with self._session_guard(request.principal.tenant_id, session_id):
                return await self._start_unlocked(
                    binding_digest,
                    request,
                    session_id=session_id,
                    session_agent_id=session_agent_id,
                    previous_execution_id=previous_execution_id,
                    fork_base_execution_id=fork_base_execution_id,
                    conversation_agent_run_id=conversation_agent_run_id,
                    parent_execution_id=parent_execution_id,
                    root_execution_id=root_execution_id,
                    parent_invocation_id=parent_invocation_id,
                    lineage_kind=lineage_kind,
                    scope=scope,
                    prepare_local_stream=prepare_local_stream,
                    dependency_hold_id=dependency_hold_id,
                    requires_task_invocation_capture=requires_task_invocation_capture,
                    budget_scope_id=budget_scope_id,
                    binding_contract=binding_contract,
                    resolved_binding=resolved_binding,
                    capture_metric=capture_metric,
                )
        finally:
            admission_done.set_result(None)
            self._admission_tasks.discard(admission_done)
            if metric_delivery is not None:
                await metric_delivery()

    async def _start_unlocked(
        self,
        binding_digest: str,
        request: ExecutionRequest,
        *,
        session_id: "str | None" = None,
        session_agent_id: "str | None" = None,
        previous_execution_id: "str | None" = None,
        fork_base_execution_id: "str | None" = None,
        conversation_agent_run_id: "str | None" = None,
        parent_execution_id: "str | None" = None,
        root_execution_id: "str | None" = None,
        parent_invocation_id: "str | None" = None,
        lineage_kind: ExecutionLineageKind = ExecutionLineageKind.RUN,
        scope: str = "execution.run",
        prepare_local_stream: bool = False,
        dependency_hold_id: "str | None" = None,
        binding_contract: "AgentBindingContract | None" = None,
        resolved_binding: AgentBinding | None = None,
        requires_task_invocation_capture: bool = False,
        budget_scope_id: str | None = None,
        capture_metric: Callable[[_TerminalMetricDelivery], None] | None = None,
    ) -> ExecutionHandle:
        if budget_scope_id is not None and (
            not isinstance(budget_scope_id, str) or not budget_scope_id.strip()
            or request.budget is not None
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not isinstance(requires_task_invocation_capture, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if re.fullmatch(r"[0-9a-f]{64}", binding_digest) is None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self._backend is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if request.idempotency_key is None:
            raise AIError(ErrorCode.IDEMPOTENCY_KEY_INVALID)
        context = await self._canonicalize_request(request)
        request = context.request
        binding = self._binding(binding_digest, binding_contract, resolved_binding)
        request = replace(request, resolved_binding=binding)
        context = replace(context, request=request)
        parent: ExecutionRecord | None = None
        if lineage_kind is ExecutionLineageKind.SUBAGENT:
            if parent_execution_id is None or not parent_invocation_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if self._runtime_bridge.pending_promote is not None:
                await self._runtime_bridge.pending_promote(parent_execution_id)
            parent = await self._state.executions.get(
                parent_execution_id,
                tenant_id=request.principal.tenant_id,
            )
            if parent is None or parent.root_execution_id != (
                root_execution_id or parent.root_execution_id
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if request.budget is not None:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            budget_scope_id = parent.budget_scope_id
            request = replace(request, correlation=parent.correlation, tool_response_ref=parent.tool_response_ref)
            context = replace(context, request=request)
        conversation_agent_run_id = conversation_agent_run_id
        session = None
        if session_id is not None and previous_execution_id is None:
            await self._runtime_bridge.pending_turns.settle_session(
                session_id, tenant_id=request.principal.tenant_id,
            )
            session = await self._load_session(
                session_id, tenant_id=request.principal.tenant_id
            )
            if session is None:
                raise AIError(ErrorCode.SESSION_NOT_FOUND)

            if session.history_quality == "unavailable":
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            if session.agent_id != session_agent_id:
                raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)
            if session.cwd is not None and (
                self._input_materializer is None
                or self._input_materializer.access is None
            ):
                raise AIError(
                    ErrorCode.REQUEST_FIELD_INVALID,
                    safe_details={"field": "cwd", "reason": "workspace_required"},
                )
            conversation_agent_run_id = (
                None
                if session.continuation is None
                else session.continuation.agent_run_id
            )
            fork_base_execution_id = None
            lineage_kind = ExecutionLineageKind.SESSION_RESUME
        execution_id = self._operation_ids()
        resource = ResourceRef(
            ResourceKind.EXECUTION, execution_id, request.principal.tenant_id
        )
        await self._authorization.authorize(
            request.principal, AuthorizationAction.EXECUTION_RUN, resource
        )
        self._validate_tool_response_mode(request.tool_response_ref, session_id=session_id)
        if budget_scope_id is not None:
            await self.authorize_budget_scope(
                budget_scope_id,
                request.principal,
                source_execution_id=(
                    previous_execution_id or fork_base_execution_id or parent_execution_id
                ),
            )
        idempotency_key_digest = compute_idempotency_key_digest(request.idempotency_key)
        request_digest = _request_digest(
            request,
            binding_digest,
            session_id=session_id,
            previous_execution_id=previous_execution_id,
            fork_base_execution_id=fork_base_execution_id,
            parent_execution_id=parent_execution_id,
            root_execution_id=root_execution_id,
            parent_invocation_id=parent_invocation_id,
            lineage_kind=lineage_kind,
            request_intent_digest=context.request_intent_digest,
            requires_task_invocation_capture=requires_task_invocation_capture,
            budget_scope_id=budget_scope_id,
        )
        pending_turn = await self._runtime_bridge.pending_turns.lookup(
            tenant_id=request.principal.tenant_id, scope=scope,
            idempotency_key_digest=idempotency_key_digest, request_digest=request_digest,
        )
        if pending_turn is not None:
            self._validate_replayed_execution(
                pending_turn.execution, binding, request,
                requires_task_invocation_capture=requires_task_invocation_capture,
                budget_scope_id=budget_scope_id,
            )
            return ExecutionHandle(pending_turn.execution.execution_id)
        existing = await self._state.idempotency.get(
            scope,
            idempotency_key_digest,
            tenant_id=request.principal.tenant_id,
        )
        if existing is not None:
            if existing.request_digest != request_digest:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if existing.status is IdempotencyStatus.START_UNKNOWN:
                raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
            if existing.status is IdempotencyStatus.RESERVED:
                pending = await self.load_execution(
                    existing.resource_id, tenant_id=request.principal.tenant_id
                )
                if pending is not None:
                    self._validate_replayed_execution(
                        pending, binding, request,
                        requires_task_invocation_capture=requires_task_invocation_capture,
                        budget_scope_id=budget_scope_id,
                    )
                if pending is not None and pending.status is ExecutionStatus.STARTED:
                    request = await self._request_for_execution(request, pending)
                    await self._launch_started(
                        request,
                        pending,
                        scope=scope,
                        idempotency_key_digest=idempotency_key_digest,
                        prepare_local_stream=prepare_local_stream,
                        dependency_hold_id=dependency_hold_id,
                    )
                    return ExecutionHandle(existing.resource_id)
                if (
                    pending is None
                    or pending.status is not ExecutionStatus.PENDING_START
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                request = await self._request_for_execution(request, pending)
                await self._prepare_and_launch(
                    request,
                    pending,
                    scope=scope,
                    idempotency_key_digest=idempotency_key_digest,
                    request_digest=request_digest,
                    prepare_local_stream=prepare_local_stream,
                    dependency_hold_id=dependency_hold_id,
                    capture_metric=capture_metric,
                )
                return ExecutionHandle(existing.resource_id)
            if existing.status in {
                IdempotencyStatus.FAILED,
                IdempotencyStatus.CANCELLED,
            }:
                terminal = await self.load_execution(
                    existing.resource_id,
                    tenant_id=request.principal.tenant_id,
                )
                if terminal is not None:
                    self._validate_replayed_execution(
                        terminal, binding, request,
                        requires_task_invocation_capture=requires_task_invocation_capture,
                        budget_scope_id=budget_scope_id,
                    )
                    if (
                        scope == "session.resume"
                        and terminal.started_at is not None
                    ):
                        if not _terminal_idempotency_matches(existing, terminal):
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        await self._acquire_start_dependency_hold(
                            terminal,
                            dependency_hold_id,
                        )
                        return ExecutionHandle(existing.resource_id)
                fallback = (
                    ErrorCode.EXECUTION_START_PERSISTENCE_FAILED
                    if existing.status is IdempotencyStatus.FAILED
                    else ErrorCode.EXECUTION_CANCELLED
                )
                raise _stable_idempotency_error(existing.error_code, fallback)
            started = await self.load_execution(
                existing.resource_id, tenant_id=request.principal.tenant_id
            )
            if started is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._validate_replayed_execution(
                started, binding, request,
                requires_task_invocation_capture=requires_task_invocation_capture,
                budget_scope_id=budget_scope_id,
            )
            if existing.status is IdempotencyStatus.STARTED and started.status in {
                ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
            }:
                # Terminal commit updates both records atomically, but these two
                # reads can straddle that commit. Refresh only this mixed view.
                refreshed = await self._state.idempotency.get(
                    scope, idempotency_key_digest, tenant_id=request.principal.tenant_id,
                )
                if (
                    refreshed is None
                    or refreshed.scope != existing.scope
                    or refreshed.idempotency_key_digest != existing.idempotency_key_digest
                    or refreshed.request_digest != existing.request_digest
                    or refreshed.resource_kind is not existing.resource_kind
                    or refreshed.resource_id != existing.resource_id
                    or not _terminal_idempotency_matches(refreshed, started)
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                existing = refreshed
                if existing.status in {IdempotencyStatus.FAILED, IdempotencyStatus.CANCELLED}:
                    if scope == "session.resume" and started.started_at is not None:
                        await self._acquire_start_dependency_hold(started, dependency_hold_id)
                        return ExecutionHandle(existing.resource_id)
                    fallback = (
                        ErrorCode.EXECUTION_START_PERSISTENCE_FAILED
                        if existing.status is IdempotencyStatus.FAILED
                        else ErrorCode.EXECUTION_CANCELLED
                    )
                    raise _stable_idempotency_error(existing.error_code, fallback)
            if existing.status is IdempotencyStatus.COMPLETED:
                if not _terminal_idempotency_matches(existing, started):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self._acquire_start_dependency_hold(
                    started,
                    dependency_hold_id,
                )
                return ExecutionHandle(existing.resource_id)
            if existing.status is IdempotencyStatus.STARTED and started.status in {
                ExecutionStatus.WAITING_DEFERRED,
                ExecutionStatus.FINALIZING,
                ExecutionStatus.RECOVERY_REQUIRED,
                ExecutionStatus.CANCELLING,
            }:
                await self._acquire_start_dependency_hold(
                    started,
                    dependency_hold_id,
                )
                return ExecutionHandle(existing.resource_id)
            if started.status is not ExecutionStatus.STARTED:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            request = await self._request_for_execution(request, started)
            await self._launch_started(
                request,
                started,
                scope=scope,
                idempotency_key_digest=idempotency_key_digest,
                prepare_local_stream=prepare_local_stream,
                dependency_hold_id=dependency_hold_id,
            )
            return ExecutionHandle(existing.resource_id)

        if session_id is not None and not self._session_execution_ready:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

        await self._validate_tool_response_input(request.tool_response_ref, session_id=session_id)
        input_instruction_paths = (
            *context.canonical_files,
            *(() if isinstance(request.user_prompt, str) else (
                item.path for item in request.user_prompt if isinstance(item, WorkspaceFileInput)
            )),
        )
        context = await self._freeze_input(context, binding)
        request = context.request

        input_context = None
        prepared_context = None
        repository_instructions = None
        if request.input_context is not None:
            if session_id is not None:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            if self._context_archive is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            prepared_context = await self._context_archive.prepare(request.input_context)
            input_context = prepared_context.reference
            if request.input_context.repository_instructions is not None:
                repository_instructions = await self._materialize_repository_instructions(
                    RepositoryInstructions.from_payload(request.input_context.repository_instructions), tenant_id=request.principal.tenant_id)
        elif self._instruction_resolver is not None:
            if lineage_kind is ExecutionLineageKind.SUBAGENT:
                if parent is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                repository_instructions = parent.repository_instructions
                if repository_instructions is None or input_instruction_paths:
                    documents = {}
                    if repository_instructions is not None:
                        port = self._runtime_bridge.port
                        if port is None:
                            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                        inherited = await port.load_repository_instructions(repository_instructions)
                        if inherited is None:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        documents.update((document.source, document) for document in inherited.documents)
                    for path in dict.fromkeys((
                        *((".",) if repository_instructions is None else ()),
                        *input_instruction_paths,
                    )):
                        resolved = await self._instruction_resolver.resolve(path, exclude_sources=frozenset(documents))
                        for document in resolved.documents:
                            documents.setdefault(document.source, document)
                    resolved_instructions = RepositoryInstructions(tuple(documents.values()))
                    self._instruction_resolver.validate(resolved_instructions)
                    repository_instructions = await self._materialize_repository_instructions(
                        resolved_instructions, tenant_id=request.principal.tenant_id,
                    )
            else:
                instruction_path: str = "."
                if session_id is not None:
                    if session is None:
                        session = await self._load_session(
                            session_id,
                            tenant_id=request.principal.tenant_id,
                        )
                        if session is None:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    instruction_path = "." if session.cwd is None else session.cwd
                instruction_paths = dict.fromkeys((
                    instruction_path,
                    *(() if session is None else session.repository_instruction_scopes),
                    *input_instruction_paths,
                ))
                documents = {}
                for path in instruction_paths:
                    resolved = await self._instruction_resolver.resolve(
                        path, exclude_sources=frozenset(documents),
                    )
                    for document in resolved.documents:
                        documents.setdefault(document.source, document)
                resolved_instructions = RepositoryInstructions(tuple(documents.values()))
                self._instruction_resolver.validate(resolved_instructions)
                repository_instructions = (
                    await self._materialize_repository_instructions(
                        resolved_instructions,
                        tenant_id=request.principal.tenant_id,
                    )
                )

        now = datetime.now(timezone.utc)
        execution = ExecutionRecord(
            execution_id=execution_id,
            session_id=session_id,
            parent_execution_id=parent_execution_id,
            root_execution_id=root_execution_id or execution_id,
            parent_invocation_id=parent_invocation_id,
            status=ExecutionStatus.PENDING_START,
            revision=0,
            event_seq=0,
            error_code=None,
            safe_error_details={},
            created_at=now,
            updated_at=now,
            previous_execution_id=previous_execution_id,
            fork_base_execution_id=fork_base_execution_id,
            lineage_kind=lineage_kind,
            agent_run_seq=0,
            memory_scope=("capture:" + execution_id) if request.input_context is not None else request.memory_scope,
            conversation_agent_run_id=conversation_agent_run_id,
            mode=request.mode,
            planning=request.planning,
            thinking=request.thinking,
            binding=binding.binding_contract,
            repository_instructions=repository_instructions,
            input_context=input_context,
            context_imported=request.input_context is not None,
            tool_response_ref=request.tool_response_ref,
            correlation=request.correlation,
            principal_id=request.principal.principal_id,
            principal_kind=request.principal.kind,
            stored_user_input=context.stored_user_input,
            requires_task_invocation_capture=requires_task_invocation_capture,
            budget_scope_id=budget_scope_id or (None if request.budget is None else "execution:" + execution_id),
            restart_checkpoints=(self._restart_checkpoints or requires_task_invocation_capture
                                 or parent is not None and parent.restart_checkpoints),
            durable_trace=(self._durable_trace or requires_task_invocation_capture
                           or parent is not None and parent.durable_trace),
            conversation=self._conversation,
        )
        proposed_reservation = ExecutionStartReservation(
                execution,
                IdempotencyRecord(
                    scope=scope,
                    idempotency_key_digest=idempotency_key_digest,
                    request_digest=request_digest,
                    resource_kind=ResourceKind.EXECUTION,
                    resource_id=execution_id,
                    status=IdempotencyStatus.RESERVED,
                    result_digest=None,
                    error_code=None,
                    created_at=now,
                    updated_at=now,
                ),
                budget=request.budget,
        )
        return await self._admit_execution_start(
            request, proposed_reservation, binding, session,
            prepared_context=prepared_context,
            prepare_local_stream=prepare_local_stream,
            dependency_hold_id=dependency_hold_id,
            requires_task_invocation_capture=requires_task_invocation_capture,
            budget_scope_id=budget_scope_id,
            capture_metric=capture_metric,
        )

    async def _admit_execution_start(
        self,
        request: ExecutionRequest,
        proposed_reservation: ExecutionStartReservation,
        binding: AgentBinding,
        session: "SessionRecord | None",
        *,
        prepared_context: PreparedExecutionContext | None = None,
        prepare_local_stream: bool,
        dependency_hold_id: str | None,
        requires_task_invocation_capture: bool,
        budget_scope_id: str | None,
        capture_metric: Callable[[_TerminalMetricDelivery], None] | None,
    ) -> ExecutionHandle:
        if not self._accepting:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        # Source-first and deferred writers must observe the same admission owner.
        coordinator = self._runtime_bridge.pending_turns.coordination
        execution = proposed_reservation.execution
        identity = proposed_reservation.idempotency

        async def admit() -> ExecutionHandle:
            if not self._accepting:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            coordination = None
            if coordinator is not None:
                coordination = await coordinator.claim(
                    session_key=(f"session:{execution.session_id}" if execution.session_id is not None else
                                 f"request:{identity.scope}:{identity.idempotency_key_digest}"),
                    request_key=f"{identity.scope}:{identity.idempotency_key_digest}",
                    request_digest=identity.request_digest,
                    execution_id=execution.execution_id,
                    owner_id=uuid.uuid4().hex,
                    lease_seconds=30,
                )
                if coordination is None:
                    raise AIError(ErrorCode.SESSION_BUSY)
                if coordination.lease is None:
                    return await self._join_coordinated_start(
                        coordinator, coordination, request, binding,
                        requires_task_invocation_capture=requires_task_invocation_capture,
                        budget_scope_id=budget_scope_id,
                    )
            transferred = False
            try:
                if (self._terminal_commit and execution.conversation
                        and dependency_hold_id is None
                        and self._runtime_bridge.pending_start is not None):
                    pending_id = await ConversationService.prepare_turn(
                        request, proposed_reservation, session, coordination,
                        start_pending=self._runtime_bridge.pending_start,
                        dependency_hold_id=dependency_hold_id,
                    )
                    if pending_id is not None:
                        transferred = True
                        return ExecutionHandle(pending_id)
                return await self._reserve_and_launch_start(
                    request, proposed_reservation, binding,
                    prepared_context=prepared_context,
                    prepare_local_stream=prepare_local_stream,
                    dependency_hold_id=dependency_hold_id,
                    requires_task_invocation_capture=requires_task_invocation_capture,
                    budget_scope_id=budget_scope_id,
                    capture_metric=capture_metric,
                )
            finally:
                if (coordinator is not None and coordination is not None
                        and coordination.lease is not None and not transferred):
                    try:
                        await coordinator.release(coordination.lease)
                    except Exception:
                        # Admission already has a source result or a known
                        # failure. Cleanup failure cannot change that outcome;
                        # the conditional claim expires on its own.
                        _logger.warning(
                            "execution admission claim cleanup failed: execution=%s",
                            execution.execution_id,
                            exc_info=environ.debug,
                        )

        if coordinator is None:
            return await admit()
        task = asyncio.create_task(admit(), name=f"execution-admission-{execution.execution_id}")
        self._admission_tasks.add(task)
        task.add_done_callback(self._admission_tasks.discard)
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                result = await asyncio.shield(task)
                break
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
                if task.done():
                    if task.cancelled():
                        raise AIError(
                            ErrorCode.STORAGE_COMMIT_UNKNOWN,
                            safe_details={"phase": "execution_admission"},
                        ) from error
                    result = task.result()
                    break
        if cancellation is not None:
            raise cancellation
        return result

    async def _join_coordinated_start(
        self,
        coordinator: PendingTurnCoordination,
        coordination: TurnCoordination,
        request: ExecutionRequest,
        binding: AgentBinding,
        *,
        requires_task_invocation_capture: bool,
        budget_scope_id: str | None,
    ) -> ExecutionHandle:
        pause = 0.05
        publication_deadline = asyncio.get_running_loop().time() + 3.0
        while True:
            record = await self.load_execution(
                coordination.execution_id, tenant_id=request.principal.tenant_id,
            )
            if record is not None:
                self._validate_replayed_execution(
                    record, binding, request,
                    requires_task_invocation_capture=requires_task_invocation_capture,
                    budget_scope_id=budget_scope_id,
                )
                return ExecutionHandle(record.execution_id)
            owner = await coordinator.read_owner(
                coordination.execution_id, tenant_id=request.principal.tenant_id,
            )
            if owner is None or owner.owner_id != coordination.owner_id:
                record = await self.load_source_execution(
                    coordination.execution_id, tenant_id=request.principal.tenant_id,
                )
                if record is not None:
                    self._validate_replayed_execution(
                        record, binding, request,
                        requires_task_invocation_capture=requires_task_invocation_capture,
                        budget_scope_id=budget_scope_id,
                    )
                    return ExecutionHandle(record.execution_id)
                raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
            if asyncio.get_running_loop().time() >= publication_deadline:
                raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
            await asyncio.sleep(pause)
            pause = min(1.0, pause * 2)

    async def _reserve_and_launch_start(
        self,
        request: ExecutionRequest,
        proposed_reservation: ExecutionStartReservation,
        binding: AgentBinding,
        *,
        prepared_context: PreparedExecutionContext | None = None,
        prepare_local_stream: bool,
        dependency_hold_id: str | None,
        requires_task_invocation_capture: bool,
        budget_scope_id: str | None,
        capture_metric: Callable[[_TerminalMetricDelivery], None] | None,
    ) -> ExecutionHandle:
        identity = proposed_reservation.idempotency
        scope = identity.scope
        idempotency_key_digest = identity.idempotency_key_digest
        request_digest = identity.request_digest
        if prepared_context is None:
            reservation = await self._state.executions.reserve_start(proposed_reservation)
        else:
            reservation = await self._state.executions.reserve_start(
                proposed_reservation,
                input_context_write=(prepared_context.archive, prepared_context.records),
            )
        if not reservation.created:
            if reservation.idempotency.request_digest != request_digest:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            self._validate_replayed_execution(
                reservation.execution, binding, request,
                requires_task_invocation_capture=requires_task_invocation_capture,
                budget_scope_id=budget_scope_id,
            )
            if (
                reservation.execution.status is ExecutionStatus.PENDING_START
                and reservation.idempotency.status is IdempotencyStatus.RESERVED
            ):
                request = await self._request_for_execution(
                    request, reservation.execution
                )
                await self._prepare_and_launch(
                    request,
                    reservation.execution,
                    scope=scope,
                    idempotency_key_digest=idempotency_key_digest,
                    request_digest=request_digest,
                    prepare_local_stream=prepare_local_stream,
                    dependency_hold_id=dependency_hold_id,
                    capture_metric=capture_metric,
                )
            elif (
                reservation.idempotency.status is IdempotencyStatus.STARTED
                and reservation.execution.status
                in {
                    ExecutionStatus.WAITING_DEFERRED,
                    ExecutionStatus.RECOVERY_REQUIRED,
                    ExecutionStatus.CANCELLING,
                }
            ):
                await self._acquire_start_dependency_hold(
                    reservation.execution,
                    dependency_hold_id,
                )
                return ExecutionHandle(reservation.execution.execution_id)
            elif (
                reservation.idempotency.status is IdempotencyStatus.STARTED
                and reservation.execution.status
                in {
                    ExecutionStatus.STARTED,
                    ExecutionStatus.FINALIZING,
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                }
            ):
                request = await self._request_for_execution(
                    request, reservation.execution
                )
                await self._launch_started(
                    request,
                    reservation.execution,
                    scope=scope,
                    idempotency_key_digest=idempotency_key_digest,
                    prepare_local_stream=prepare_local_stream,
                    dependency_hold_id=dependency_hold_id,
                )
            elif (
                reservation.execution.status is ExecutionStatus.START_UNKNOWN
                or reservation.idempotency.status is IdempotencyStatus.START_UNKNOWN
            ):
                raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
            else:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return ExecutionHandle(reservation.execution.execution_id)
        execution_id = reservation.execution.execution_id
        if not reservation.created:
            request = await self._request_for_execution(request, reservation.execution)
        await self._prepare_and_launch(
            request,
            reservation.execution,
            scope=scope,
            idempotency_key_digest=idempotency_key_digest,
            request_digest=request_digest,
            prepare_local_stream=prepare_local_stream,
            dependency_hold_id=dependency_hold_id,
            capture_metric=capture_metric,
        )
        _logger.info("execution started: execution=%s scope=%s", execution_id, scope)
        return ExecutionHandle(execution_id)

    @asynccontextmanager
    async def _session_guard(self, tenant_id: str, session_id: str):
        key = tenant_id, session_id
        async with self._session_locks_guard:
            entry = self._session_locks.get(key)
            if entry is None:
                entry = _SessionLockEntry(asyncio.Lock())
                self._session_locks[key] = entry
            entry.ref_count += 1
        acquired = False
        try:
            await entry.lock.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                entry.lock.release()
            async with self._session_locks_guard:
                entry.ref_count -= 1
                if entry.ref_count == 0 and self._session_locks.get(key) is entry:
                    self._session_locks.pop(key)

    async def _prepare_and_launch(
        self,
        request: ExecutionRequest,
        execution: ExecutionRecord,
        *,
        scope: str,
        idempotency_key_digest: str,
        request_digest: str,
        prepare_local_stream: bool = False,
        dependency_hold_id: "str | None" = None,
        capture_metric: Callable[[_TerminalMetricDelivery], None] | None = None,
    ) -> None:
        if self._backend is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if execution.budget_scope_id is not None:
            await self._state.budgets.read(execution.budget_scope_id)
        identity = ExecutionStartIdentity(scope, idempotency_key_digest, request_digest)
        try:
            started = await self._backend.prepare_start(
                request,
                execution,
                identity,
            )
        except AIError as error:
            if error.code in {ErrorCode.SESSION_BUSY, ErrorCode.SESSION_CONFLICT}:
                await self._reject_pending_start(
                    execution,
                    identity=identity,
                    error=error,
                    capture_metric=capture_metric,
                )
            raise
        if started is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._launch_started(
            request,
            started,
            scope=scope,
            idempotency_key_digest=idempotency_key_digest,
            prepare_local_stream=prepare_local_stream,
            dependency_hold_id=dependency_hold_id,
        )

    async def _acquire_start_dependency_hold(
        self,
        execution: ExecutionRecord,
        hold_id: "str | None",
    ) -> bool:
        if hold_id is None:
            return False
        await self.acquire_dependency_hold(
            execution.execution_id,
            tenant_id=self._state.executions.tenant_id,
            hold_id=hold_id,
        )
        return True

    async def _reject_pending_start(
        self,
        execution: ExecutionRecord,
        *,
        identity: ExecutionStartIdentity,
        error: AIError,
        capture_metric: Callable[[_TerminalMetricDelivery], None] | None = None,
    ) -> ExecutionRecord:
        if error.code not in {ErrorCode.SESSION_BUSY, ErrorCode.SESSION_CONFLICT}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        current = await self.load_execution(
            execution.execution_id,
            tenant_id=self._state.executions.tenant_id,
        )
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        idempotency = await self._state.idempotency.get(
            identity.scope,
            identity.idempotency_key_digest,
            tenant_id=self._state.executions.tenant_id,
        )
        if (
            idempotency is None
            or idempotency.scope != identity.scope
            or idempotency.idempotency_key_digest != identity.idempotency_key_digest
            or idempotency.resource_kind is not ResourceKind.EXECUTION
            or idempotency.resource_id != current.execution_id
            or idempotency.request_digest != identity.request_digest
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        metric_delivery: _TerminalMetricDelivery | None = None
        cancellation: asyncio.CancelledError | None = None

        def capture_terminal_metric(delivery: _TerminalMetricDelivery) -> None:
            nonlocal metric_delivery
            metric_delivery = delivery

        if current.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            if (
                current.status is not ExecutionStatus.FAILED
                or current.error_code != error.code.value
                or idempotency.status is not IdempotencyStatus.FAILED
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            terminal = current
        else:
            if current.status is not ExecutionStatus.PENDING_START:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if idempotency.status is not IdempotencyStatus.RESERVED:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            now = datetime.now(timezone.utc)
            diagnostics = error.diagnostics or ErrorDiagnostics.from_exception(error)
            terminal = _next_execution(
                current,
                ExecutionStatus.FAILED,
                now,
                error_code=error.code.value,
                safe_error_details=error.safe_details,
                error_diagnostics=diagnostics,
                terminal_event=True,
            )
            if self._terminal_committer is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            admitted_session_id = None
            if current.session_id is not None and self._sessions is not None:
                admitted_session = await self._sessions.get(current.session_id, tenant_id=self._state.executions.tenant_id)
                if admitted_session is not None and admitted_session.active_execution_id == current.execution_id:
                    admitted_session_id = current.session_id
            try:
                await self._terminal_committer.commit_terminal_checkpoint(
                    ExecutionTerminalCommit(
                        expected_revision=current.revision,
                        expected_event_seq=current.event_seq,
                        execution=terminal,
                        result=ResultRecord(
                            None,
                            StopReason.ERROR,
                            UsageMetrics(),
                            now,
                        ),
                        terminal_event_type=ExecutionEventType.EXECUTION_FAILED,
                        terminal_event_payload=_failed_event_payload(
                            error.code.value,
                            error.safe_details,
                            diagnostics,
                        ),
                        idempotency=IdempotencyTerminalUpdate(
                            idempotency.scope,
                            idempotency.idempotency_key_digest,
                            idempotency.status,
                            IdempotencyStatus.FAILED,
                            idempotency.request_digest,
                            None,
                            error.code.value,
                        ),
                    ),
                    session_id=admitted_session_id,
                    expected_execution=current,
                    capture_metric=capture_terminal_metric,
                )
            except asyncio.CancelledError as cancelled:
                cancellation = cancelled
                try:
                    observed = await self.load_execution(
                        current.execution_id,
                        tenant_id=self._state.executions.tenant_id,
                    )
                    observed_identity = await self._state.idempotency.get(
                        identity.scope,
                        identity.idempotency_key_digest,
                        tenant_id=self._state.executions.tenant_id,
                    )
                except Exception as read_error:
                    raise cancelled from read_error
                if (
                    observed is None
                    or observed.status is not ExecutionStatus.FAILED
                    or observed.error_code != error.code.value
                    or observed_identity is None
                    or observed_identity.status is not IdempotencyStatus.FAILED
                ):
                    raise
                terminal = observed
            except AIError as commit_error:
                if commit_error.code not in {
                    ErrorCode.STORAGE_CONFLICT,
                    ErrorCode.EXECUTION_RESULT_CONFLICT,
                }:
                    raise
                latest = await self.load_execution(
                    current.execution_id,
                    tenant_id=self._state.executions.tenant_id,
                )
                if (
                    latest is None
                    or latest.status is not ExecutionStatus.FAILED
                    or latest.error_code != error.code.value
                ):
                    raise
                terminal = latest
        try:
            if self._backend is not None:
                await self._backend.abort_start(terminal)
        except Exception:
            _logger.error(
                "pending start cleanup failed: execution=%s",
                terminal.execution_id,
                exc_info=environ.debug,
            )
        _logger.info(
            "pending start rejected: execution=%s code=%s",
            terminal.execution_id,
            error.code.value,
        )
        if metric_delivery is not None:
            if capture_metric is not None:
                capture_metric(metric_delivery)
            else:
                await metric_delivery()
        if cancellation is not None:
            raise cancellation
        return terminal

    async def _launch_started(
        self,
        request: ExecutionRequest,
        execution: ExecutionRecord,
        *,
        scope: str,
        idempotency_key_digest: str,
        prepare_local_stream: bool = False,
        dependency_hold_id: "str | None" = None,
    ) -> None:
        hold_acquired = await self._acquire_start_dependency_hold(
            execution,
            dependency_hold_id,
        )
        try:
            await self._launch_started_impl(
                request,
                execution,
                scope=scope,
                idempotency_key_digest=idempotency_key_digest,
                prepare_local_stream=prepare_local_stream,
            )
        except BaseException:
            if hold_acquired:
                await self.release_dependency_hold(
                    execution.execution_id,
                    tenant_id=self._state.executions.tenant_id,
                    hold_id=dependency_hold_id or "",
                )
            raise

    async def _launch_started_impl(
        self,
        request: ExecutionRequest,
        execution: ExecutionRecord,
        *,
        scope: str,
        idempotency_key_digest: str,
        prepare_local_stream: bool = False,
    ) -> None:
        if self._backend is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        launch_record = await self.load_execution(
            execution.execution_id,
            tenant_id=self._state.executions.tenant_id,
        )
        if launch_record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if launch_record.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.CANCELLING,
            ExecutionStatus.FINALIZING,
            ExecutionStatus.WAITING_DEFERRED,
            ExecutionStatus.RECOVERY_REQUIRED,
        }:
            return
        if launch_record.status is ExecutionStatus.START_UNKNOWN:
            raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
        if launch_record.status is not ExecutionStatus.STARTED:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if launch_record.budget_scope_id is not None:
            usage = await self._state.budgets.read(launch_record.budget_scope_id)
            request = replace(request, budget=usage.limits)
        if launch_record.agent_run_seq > 0:
            # A replay observes the admitted producer. Only explicit recovery
            # may obtain a new generation for an already claimed attempt.
            return
        prepared_local_stream = False
        try:
            if (
                prepare_local_stream
                and not self._backend.worker_installed(launch_record.execution_id)
            ):
                prepared_local_stream = True
                self._live_broker.prepare_local_producer(launch_record.execution_id)
            await self._backend.launch(request, launch_record)
            if (
                prepared_local_stream
                and not self._backend.worker_installed(launch_record.execution_id)
            ):
                self._live_broker.abandon_prepared_local_producer(
                    launch_record.execution_id
                )
        except asyncio.CancelledError:
            if (
                prepared_local_stream
                and not self._backend.worker_installed(launch_record.execution_id)
            ):
                self._live_broker.abandon_prepared_local_producer(
                    launch_record.execution_id
                )
            raise
        except BaseException as error:
            if self._backend.worker_installed(launch_record.execution_id):
                _logger.warning(
                    "execution launch raised after worker installation: execution=%s",
                    execution.execution_id,
                    exc_info=environ.debug,
                )
                return
            if prepared_local_stream:
                self._live_broker.abandon_prepared_local_producer(
                    launch_record.execution_id
                )
            current = await self.load_execution(
                execution.execution_id,
                tenant_id=self._state.executions.tenant_id,
            )
            if current is not None and current.status is ExecutionStatus.STARTED:
                identity = await self._state.idempotency.get(
                    scope,
                    idempotency_key_digest,
                    tenant_id=self._state.executions.tenant_id,
                )
                if identity is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self._state.executions.mark_start_unknown(
                    ExecutionStartUnknownCommit(
                        execution.execution_id,
                        current.revision,
                        current.event_seq,
                        scope,
                        idempotency_key_digest,
                        identity.request_digest,
                        datetime.now(timezone.utc),
                    )
                )
            _logger.error(
                "execution start outcome unknown: execution=%s", execution.execution_id
            )
            raise AIError(ErrorCode.EXECUTION_START_UNKNOWN) from error


    @_observed_query
    async def inspect(
        self, execution_id: str, *, principal: Principal
    ) -> ExecutionView:
        execution = await self.load_observed_authorized(
            execution_id, principal, AuthorizationAction.EXECUTION_READ
        )
        if (
            execution.status
            not in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }
            and self._backend is not None
        ):
            failure = self._backend.worker_failure(
                execution_id,
                tenant_id=principal.tenant_id,
            )
            if failure is not None:
                raise failure
        return _execution_view(execution)

    async def list(self, request: ListExecutionRequest) -> Page[ExecutionView]:
        if self._history_service is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await self._history_service.list(request)


    @consumed_terminal_query
    async def wait(
        self,
        execution_id: str,
        *,
        principal: Principal,
        timeout_seconds: "float | None" = None,
    ) -> ExecutionResult:
        if timeout_seconds is not None and timeout_seconds < 0:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        async def wait_once() -> ExecutionResult:
            execution = await self.load_authorized(
                execution_id,
                principal,
                AuthorizationAction.EXECUTION_READ,
            )
            self._live_broker.abandon_prepared_local_producer(execution_id)
            status = execution.status
            poll_backoff = 1.0
            while True:
                if status is ExecutionStatus.RECOVERY_REQUIRED:
                    raise _execution_recovery_error(execution)
                waiter = self.local_waiter
                owns_execution = waiter is not None and waiter.owns_execution(
                    execution_id,
                    tenant_id=principal.tenant_id,
                )
                if status in {
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                }:
                    if owns_execution:
                        _logger.debug(
                            "execution wait awaiting local terminal worker: execution=%s",
                            execution_id,
                        )
                        await waiter.wait_terminal(
                            execution_id,
                            tenant_id=principal.tenant_id,
                        )
                        execution = await self.load_authorized(
                            execution_id,
                            principal,
                            AuthorizationAction.EXECUTION_READ,
                        )
                        status = execution.status
                        continue
                    if self._backend is not None:
                        failure = self._backend.worker_failure(
                            execution_id,
                            tenant_id=principal.tenant_id,
                        )
                        if failure is not None:
                            raise failure
                    return await self.consume_result_from_execution(
                        execution, tenant_id=principal.tenant_id,
                    )
                if self._backend is not None:
                    failure = self._backend.worker_failure(
                        execution_id,
                        tenant_id=principal.tenant_id,
                    )
                    if failure is not None:
                        raise failure
                if owns_execution:
                    await waiter.wait_terminal(
                        execution_id,
                        tenant_id=principal.tenant_id,
                    )
                    poll_backoff = 1.0
                else:
                    await asyncio.sleep(poll_backoff)
                    poll_backoff = min(30.0, poll_backoff * 2)
                execution = await self.load_authorized(
                    execution_id,
                    principal,
                    AuthorizationAction.EXECUTION_READ,
                )
                if execution.status is not status:
                    poll_backoff = 1.0
                status = execution.status

        try:
            return await asyncio.wait_for(wait_once(), timeout_seconds)
        except asyncio.TimeoutError as error:
            raise AIError(ErrorCode.EXECUTION_WAIT_TIMEOUT) from error

    async def run(
        self,
        binding_digest: str,
        request: ExecutionRequest,
        *,
        timeout_seconds: "float | None" = None,
        binding_contract: "AgentBindingContract | None" = None,
    ) -> ExecutionResult:
        handle = await self.start(
            binding_digest,
            request,
            binding_contract=binding_contract,
        )
        return await self.wait(
            handle.execution_id,
            principal=request.principal,
            timeout_seconds=timeout_seconds,
        )

    async def retry(
        self, execution_id: str, request: RetryExecutionRequest
    ) -> ExecutionHandle:
        previous = await self.load_authorized(
            execution_id, request.principal, AuthorizationAction.EXECUTION_READ
        )
        if (
            previous.parent_execution_id is not None
            or not isinstance(previous.binding, AgentBindingContract)
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        binding_digest = previous.binding_digest
        self._binding(binding_digest, previous.binding)
        input_context = None
        if previous.context_imported:
            if previous.input_context is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if self._context_archive is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            input_context = await self._context_archive.load(previous.input_context)
        retry_request = ExecutionRequest(
            user_prompt=request.user_prompt,
            principal=request.principal,
            idempotency_key=request.idempotency_key,
            memory_scope=None if input_context is not None else previous.memory_scope,
            mode=previous.mode,
            planning=previous.planning,
            thinking=previous.thinking,
            correlation=_overlay_execution_correlation(
                previous.correlation, request.correlation
            ),
            files=request.files,
            input_context=input_context,
            tool_response_ref=previous.tool_response_ref,
        )
        return await self._start(
            binding_digest,
            retry_request,
            session_id=previous.session_id,
            parent_execution_id=previous.parent_execution_id,
            root_execution_id=previous.root_execution_id,
            scope="execution.retry",
            previous_execution_id=previous.execution_id,
            lineage_kind=ExecutionLineageKind.RETRY,
            fork_base_execution_id=previous.fork_base_execution_id,
            conversation_agent_run_id=previous.conversation_agent_run_id,
            binding_contract=previous.binding,
            budget_scope_id=previous.budget_scope_id,
        )

    async def fork(
        self, execution_id: str, request: ForkExecutionRequest
    ) -> ExecutionHandle:
        previous = await self.load_authorized(
            execution_id, request.principal, AuthorizationAction.EXECUTION_READ
        )
        if (
            previous.parent_execution_id is not None
            or not isinstance(previous.binding, AgentBindingContract)
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not previous.conversation:
            raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING)
        binding_digest = previous.binding_digest
        self._binding(binding_digest, previous.binding)
        fork_request = ExecutionRequest(
            user_prompt=request.user_prompt,
            principal=request.principal,
            idempotency_key=request.idempotency_key,
            memory_scope=previous.memory_scope,
            mode=previous.mode,
            planning=previous.planning,
            thinking=previous.thinking,
            correlation=_overlay_execution_correlation(
                previous.correlation, request.correlation
            ),
            files=request.files,
            budget=request.budget,
            tool_response_ref=previous.tool_response_ref,
        )
        return await self._start(
            binding_digest,
            fork_request,
            session_id=previous.session_id,
            parent_execution_id=previous.parent_execution_id,
            root_execution_id=previous.root_execution_id,
            scope="execution.fork",
            previous_execution_id=previous.execution_id,
            lineage_kind=ExecutionLineageKind.FORK,
            fork_base_execution_id=previous.execution_id,
            binding_contract=previous.binding,
            budget_scope_id=previous.budget_scope_id if request.budget is None else None,
        )

    async def cancel(
        self, execution_id: str, request: CancelExecutionRequest
    ) -> CancelExecutionResult:
        metric_delivery: _TerminalMetricDelivery | None = None

        def capture_metric(delivery: _TerminalMetricDelivery) -> None:
            nonlocal metric_delivery
            metric_delivery = delivery

        finalizer = asyncio.create_task(
            self._cancel_finalizer_with_handoff(
                execution_id, request, capture_metric=capture_metric,
            ),
            name=f"execution-cancel-finalizer-{request.principal.tenant_id}-{execution_id}",
        )
        self._cancel_tasks.add(finalizer)
        finalizer.add_done_callback(self._cancel_tasks.discard)
        try:
            result = await asyncio.shield(finalizer)
        except asyncio.CancelledError:
            if finalizer.done():
                result = finalizer.result()
            else:
                self._detach_cancel_finalizer(finalizer, execution_id)
                raise
        # Only durable settlement is runtime-owned; telemetry remains caller-owned.
        if metric_delivery is not None:
            await metric_delivery()
        return result

    async def _cancel_finalizer_with_handoff(
        self,
        execution_id: str,
        request: CancelExecutionRequest,
        *,
        capture_metric: Callable[[_TerminalMetricDelivery], None],
    ) -> CancelExecutionResult:
        async with self.execution_consumer(
            execution_id,
            request.principal.tenant_id,
        ):
            result = await self.cancel_authorized(
                execution_id, request, capture_metric=capture_metric,
            )
            await self.request_handoff_if_terminal(
                execution_id,
                request.principal.tenant_id,
            )
            return result

    def _detach_cancel_finalizer(
        self,
        task: asyncio.Task[CancelExecutionResult],
        execution_id: str,
    ) -> None:
        if task in self._detached_cancel_finalizers:
            return
        self._detached_cancel_finalizers.add(task)

        def consume(done: asyncio.Task[CancelExecutionResult]) -> None:
            try:
                done.result()
            except BaseException as error:  # noqa: BLE001
                _logger.warning(
                    "execution cancel finalizer failed after caller cancellation: "
                    "execution=%s error=%s",
                    execution_id,
                    type(error).__name__,
                )
                if self._detached_cancel_failure is None:
                    details = (
                        dict(error.safe_details) if isinstance(error, AIError) else {}
                    )
                    details.setdefault("phase", "execution_cancel_finalizer")
                    details.setdefault("execution_id", execution_id)
                    if isinstance(error, AIError):
                        self._detached_cancel_failure = AIError(
                            error.code,
                            category=error.category,
                            retryable=error.retryable,
                            operation_id=error.operation_id,
                            safe_details=details,
                            diagnostics=error.diagnostics,
                        )
                    else:
                        self._detached_cancel_failure = AIError(
                            ErrorCode.STORAGE_RECOVERY_REQUIRED,
                            safe_details=details,
                        )
            finally:
                self._detached_cancel_finalizers.discard(done)

        task.add_done_callback(consume)

    async def preflight_close(self) -> None:
        self._accepting = False
        while True:
            pending = tuple(task for task in self._admission_tasks if not task.done())
            if not pending:
                break
            await asyncio.gather(
                *(asyncio.shield(task) for task in pending),
                return_exceptions=True,
            )
        while True:
            pending = tuple(
                task for task in self._cancel_tasks | self._detached_cancel_finalizers if not task.done()
            )
            if not pending:
                break
            await asyncio.gather(
                *(asyncio.shield(task) for task in pending),
                return_exceptions=True,
            )
        failure = self._detached_cancel_failure
        if failure is not None:
            raise AIError(
                failure.code,
                category=failure.category,
                retryable=failure.retryable,
                operation_id=failure.operation_id,
                safe_details=dict(failure.safe_details),
                diagnostics=failure.diagnostics,
            )

    async def _cancel(self, execution_id: str, request: CancelExecutionRequest, *,
        capture_metric: Callable[[_TerminalMetricDelivery], None] | None = None,
    ) -> CancelExecutionResult:
        return await self._management.cancel_authorized(execution_id, request, capture_metric=capture_metric)

    async def cancel_authorized(self, execution_id: str, request: CancelExecutionRequest, *,
        capture_metric: Callable[[_TerminalMetricDelivery], None] | None = None,
    ) -> CancelExecutionResult:
        return await self._management.cancel_authorized(execution_id, request, capture_metric=capture_metric)

    async def recovery_effects(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> tuple[ExecutionRecoveryEffect, ...]:
        execution = await self.load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_RECOVER,
        )
        if execution.status is not ExecutionStatus.RECOVERY_REQUIRED:
            return ()
        return await self.runtime_backend().recovery_effects(
            execution_id,
            tenant_id=principal.tenant_id,
        )

    async def resolve_tool_effect(
        self,
        execution_id: str,
        request: ResolveToolEffectRequest,
    ) -> ToolEffectResolutionResult:
        execution = await self.load_authorized(
            execution_id,
            request.principal,
            AuthorizationAction.EXECUTION_RECOVER,
        )
        if execution.status is not ExecutionStatus.RECOVERY_REQUIRED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return await self.runtime_backend().resolve_tool_effect(
            execution_id,
            request,
        )

    async def recover(
        self,
        execution_id: str,
        *,
        principal: Principal,
        idempotency_key: str | None = None,
    ) -> ExecutionHandle:
        execution = await self.load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_RECOVER,
        )
        if idempotency_key is None:
            if execution.status not in {
                ExecutionStatus.RECOVERY_REQUIRED,
                ExecutionStatus.PENDING_START,
                ExecutionStatus.STARTED,
                ExecutionStatus.CANCELLING,
                ExecutionStatus.FINALIZING,
            }:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            await self.runtime_backend().recover_execution(
                execution_id,
                tenant_id=principal.tenant_id,
                expected_revision=execution.revision,
            )
        else:
            now = datetime.now(timezone.utc)
            operation = OperationLedgerInput(
                operation_id=compute_idempotency_key_digest(validate_idempotency_key(idempotency_key)),
                tenant_id=principal.tenant_id,
                resource_kind=ResourceKind.EXECUTION,
                resource_id=execution_id,
                execution_id=execution_id,
                operation_kind=OperationKind.EXECUTION_RECOVER,
                status=OperationStatus.RUNNING,
                request_digest=canonical_sha256({
                    "action": "execution.recover",
                    "principal": principal_identity_payload(principal),
                    "execution_id": execution_id,
                }),
                result_ref=uuid.uuid4().hex,
                result_digest=None,
                error_code=None,
                compactable=False,
                created_at=now,
                updated_at=now,
            )
            await self.runtime_backend().recover_execution(
                execution_id,
                tenant_id=principal.tenant_id,
                expected_revision=execution.revision,
                recovery_operation=operation,
            )
        return ExecutionHandle(execution_id)


    @_observed_query
    async def trace(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        step_index: int | None = None,
        tool_call_id: str | None = None,
    ) -> "Page[ExecutionTraceItem]":
        if self._history_service is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await self._history_service.trace(
            execution_id,
            principal=principal,
            cursor=cursor,
            limit=limit,
            agent_run_seq=agent_run_seq,
            model_request_seq=model_request_seq,
            step_index=step_index,
            tool_call_id=tool_call_id,
        )

    @_observed_query
    async def transcript(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        include_content: bool = False,
        limit: int = 100,
    ) -> Page[TranscriptItem]:
        if self._history_service is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await self._history_service.transcript(
            execution_id,
            principal=principal,
            cursor=cursor,
            include_content=include_content,
            limit=limit,
        )

    @_observed_query
    async def history(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        include_content: bool = False,
        limit: int = 100,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        step_index: int | None = None,
        tool_call_id: str | None = None,
        message_seq: int | None = None,
        part_index: int | None = None,
    ) -> "Page[ExecutionHistoryItem]":
        if self._history_service is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await self._history_service.history(
            execution_id,
            principal=principal,
            cursor=cursor,
            include_content=include_content,
            limit=limit,
            agent_run_seq=agent_run_seq,
            model_request_seq=model_request_seq,
            step_index=step_index,
            tool_call_id=tool_call_id,
            message_seq=message_seq,
            part_index=part_index,
        )

    @_observed_query
    async def model_interactions(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        include_content: bool = False,
        include_client_requests: bool = False,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        limit: int = 100,
        cutoffs: "tuple[UsageReadCutoff, ...] | None" = None,
    ) -> "Page[ModelInteractionItem]":
        if self._history_service is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await self._history_service.model_interactions(
            execution_id,
            principal=principal,
            cursor=cursor,
            include_content=include_content,
            **({"include_client_requests": include_client_requests} if include_client_requests is not False else {}),
            **({"agent_run_seq": agent_run_seq} if agent_run_seq is not None else {}),
            **({"model_request_seq": model_request_seq} if model_request_seq is not None else {}),
            limit=limit,
            cutoffs=cutoffs,
        )


    async def _load_session(self, session_id: str, *, tenant_id: str) -> "SessionRecord | None":
        return await self._sessions.state_store.read(
            lambda _transaction: self._sessions.get(session_id, tenant_id=tenant_id)
        )


    async def load_execution(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        pending = await self._runtime_bridge.pending_turns.read_execution(execution_id, tenant_id=tenant_id)
        if pending is not None:
            return pending
        record = await self.load_source_execution(execution_id, tenant_id=tenant_id)
        if record is not None:
            return record
        pending = await self._runtime_bridge.pending_turns.read_remote_execution(
            execution_id, tenant_id=tenant_id,
        )
        if pending is not None:
            return pending
        # Publication can disappear after the source read when its owner commits.
        return await self.load_source_execution(execution_id, tenant_id=tenant_id)


    async def load_observed_authorized(
        self, execution_id: str, principal: Principal, action: AuthorizationAction
    ) -> ExecutionRecord:
        header = await self.authorize_execution(execution_id, principal, action)
        record = await self._runtime_bridge.pending_turns.read_execution(
            execution_id, tenant_id=principal.tenant_id,
        )
        if record is None:
            record = await self._state.executions.get(
                execution_id, tenant_id=principal.tenant_id,
            )
        if record is None:
            record = await self._runtime_bridge.pending_turns.read_remote_execution(
                execution_id, tenant_id=principal.tenant_id,
            )
        if record is None:
            # Ordinary reads may retain a pre-commit miss after the owner releases.
            record = await self.load_source_execution(
                execution_id, tenant_id=principal.tenant_id,
            )
        if record is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self.authorize_loaded_execution(
            record, principal, action, owner_principal_id=header.owner_principal_id,
        )
        return record


    async def acquire_dependency_hold(
        self, execution_id: str, *, tenant_id: str, hold_id: str,
    ) -> bool:
        if self._runtime_bridge.pending_promote is not None:
            await self._runtime_bridge.pending_promote(execution_id)
        return await super().acquire_dependency_hold(execution_id, tenant_id=tenant_id, hold_id=hold_id)


def _execution_view(execution: ExecutionRecord) -> ExecutionView:
    return _project_execution_view(execution)


def _failed_event_payload(
    error_code: str,
    safe_error_details: Mapping[str, JsonValue],
    diagnostics: ErrorDiagnostics,
) -> dict[str, JsonValue]:
    return {
        "error_code": error_code,
        "safe_error_details": dict(safe_error_details),
        "error_diagnostics": diagnostics.to_payload(),
    }


def _request_digest(
    request: ExecutionRequest,
    binding_digest: str,
    *,
    session_id: str | None,
    previous_execution_id: str | None,
    fork_base_execution_id: str | None,
    parent_execution_id: str | None,
    root_execution_id: str | None,
    parent_invocation_id: str | None,
    lineage_kind: ExecutionLineageKind,
    request_intent_digest: str | None = None,
    requires_task_invocation_capture: bool = False,
    budget_scope_id: str | None = None,
) -> str:
    if request_intent_digest is None:
        user_prompt_identity = input_intent(
            validate_user_input(request.user_prompt),
            request.files,
        ).digest
    else:
        user_prompt_identity = request_intent_digest
    return canonical_sha256(
        {
            "input_intent": user_prompt_identity,
            **({"budget": request.budget.digest_payload()} if request.budget is not None else {}),
            **({"budget_scope_id": budget_scope_id} if budget_scope_id is not None else {}),
            **({"requires_task_invocation_capture": True} if requires_task_invocation_capture else {}),
            **({"input_context_digest": request.input_context.digest} if request.input_context is not None else {}),
            **({"tool_response_ref": request.tool_response_ref.to_payload()} if request.tool_response_ref is not None else {}),
            "binding_digest": binding_digest,
            "scope": session_id or "execution",
            "principal": principal_identity_payload(request.principal),
            "session_id": session_id,
            "previous_execution_id": previous_execution_id,
            "fork_base_execution_id": fork_base_execution_id,
            "parent_execution_id": parent_execution_id,
            "parent_invocation_id": parent_invocation_id,
            "root_identity": root_execution_id or "$self",
            "lineage_kind": lineage_kind.value,
            "memory_scope_digest": None
            if request.memory_scope is None
            else canonical_sha256(request.memory_scope),
            "mode": request.mode,
            "planning": request.planning,
            "thinking": request.thinking,
        }
    )


__all__ = ["CancelEffectOutcome", "DefaultExecutionService", "ExecutionBackend"]
