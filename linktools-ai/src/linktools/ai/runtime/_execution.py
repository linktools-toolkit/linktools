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
from ._context_archive import ExecutionContextArchive, PreparedExecutionContext
from ..capability import ToolResponseFixture
from ._tool_response_fixture import _load_tool_response_fixture
from ..core import HandoffGate, HandoffState
from ._input import (
    ExecutionInputMaterializer,
    decode_user_content_payload,
    input_intent,
    validate_user_input,
)
from ._object import RuntimeObjectKeyFactory, put_runtime_object, read_runtime_object
from .recovery import (
    ExecutionRecoveryEffect,
    ResolveToolEffectRequest,
    ToolEffectResolutionResult,
)
from .service_api import (
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
from .service_api import (
    project_execution_view as _project_execution_view,
)
from .state import RuntimeDomain
from ._pending_turn import _PendingExecutionTurns
from ._pending_coordination import _PendingTurnCoordination
from .state._execution_repositories import ExecutionRepositoryImpl
from .state._object_router import runtime_object_source_reads
from .state._contracts import (
    ExecutionCancelRequestCommit,
    ExecutionRecord,
    ExecutionStartClaim,
    ExecutionStartReservation,
    ExecutionStartUnknownCommit,
    ExecutionRepositories,
    ExecutionTerminalCommit,
    ExecutionTerminalCommitResult,
    IdempotencyRecord,
    IdempotencyTerminalUpdate,
    OperationTerminalUpdate,
    ResultRecord,
    SessionRepository,
    StoredUserInput,
    TaskAdmissionRepository,
)

if TYPE_CHECKING:
    from .state._store import StateTransaction
    from .recovery import (
        ExecutionRecoveryEffect,
        ResolveToolEffectRequest,
        ToolEffectResolutionResult,
    )
    from .state._contracts import RuntimePayloadRef, SessionRecord

_logger = environ.get_logger("ai.runtime.execution")


def _overlay_execution_correlation(
    base: Mapping[str, object],
    overlay: Mapping[str, object],
) -> CorrelationData:
    try:
        return overlay_correlation(base, overlay)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error


def consumed_terminal_query(method: "Callable[..., object]") -> "Callable[..., object]":
    @wraps(method)
    async def wrapped(
        self: "DefaultExecutionService",
        execution_id: str,
        *args: object,
        principal: Principal,
        **kwargs: object,
    ) -> object:
        async with self._execution_consumer(execution_id, principal.tenant_id):
            with runtime_object_source_reads():
                result = await method(
                    self, execution_id, *args, principal=principal, **kwargs
                )
            return result

    return wrapped


def _observed_query(method: "Callable[..., object]") -> "Callable[..., object]":
    @wraps(method)
    async def wrapped(
        self: "DefaultExecutionService",
        execution_id: str,
        *args: object,
        principal: Principal,
        **kwargs: object,
    ) -> object:
        async with self._execution_consumer(execution_id, principal.tenant_id):
            return await method(
                self, execution_id, *args, principal=principal, **kwargs
            )

    return wrapped


class _ExecutionReleaseCallback(Protocol):
    async def __call__(
        self, execution_id: str, *, tenant_id: str, terminal_commit: bool = False,
    ) -> bool: ...


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


class _LocalExecutionWaiter(Protocol):
    def owns_execution(self, execution_id: str, *, tenant_id: str) -> bool: ...

    async def wait_terminal(self, execution_id: str, *, tenant_id: str) -> None: ...


class _RepositoryInstructionBundle(Protocol):
    def to_payload(self) -> JsonValue: ...


class _RepositoryInstructionResolver(Protocol):
    async def resolve(self, path: str) -> _RepositoryInstructionBundle: ...


async def _no_release_terminal(
    execution_id: str, *, tenant_id: str, terminal_commit: bool = False,
) -> bool:
    del execution_id, tenant_id, terminal_commit
    return True


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


class _ExecutionManagementAdapter:
    """Bind execution management to Runtime's coordinated stores and driver."""

    def __init__(self, service: "DefaultExecutionService") -> None:
        self._service = service

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

    async def read_execution(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        return await self._service._load_execution(execution_id, tenant_id=tenant_id)

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
                header = await service._load_source_header(execution_id, tenant_id=tenant_id)
        return header

    async def read_source_header(self, execution_id: str, *, tenant_id: str) -> ResourceRef | None:
        return await self._service._load_source_header(execution_id, tenant_id=tenant_id)

    def has_local_pending(self, execution_id: str, *, tenant_id: str) -> bool:
        return self._service._runtime_bridge.pending_turns.get(execution_id, tenant_id=tenant_id) is not None

    async def read_source_execution(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        return await self._service._load_source_execution(execution_id, tenant_id=tenant_id)

    async def read_remote_pending(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        return await self._service._runtime_bridge.pending_turns.read_remote_execution(execution_id, tenant_id=tenant_id)

    async def signal_remote_cancel(self, execution_id: str, *, tenant_id: str) -> None:
        await self._service._runtime_bridge.pending_turns.cancel_remote(execution_id, tenant_id=tenant_id)

    async def get_operation(self, operation_id: str, *, tenant_id: str) -> OperationLedgerRecord | None:
        return await self._service._state.operations.get(operation_id, tenant_id=tenant_id)

    async def append_operation(self, operation: OperationLedgerInput) -> OperationLedgerRecord:
        return await self._service._state.operations.append(operation)

    async def compare_and_swap_operation(self, operation_id: str, *, tenant_id: str,
        expected_status: OperationStatus, next_record: OperationLedgerRecord) -> OperationLedgerRecord:
        return await self._service._state.operations.compare_and_swap(operation_id, tenant_id=tenant_id,
            expected_status=expected_status, next_record=next_record)

    async def list_idempotency(self, resource_kind: ResourceKind, resource_id: str,
                              *, tenant_id: str) -> tuple[IdempotencyRecord, ...]:
        return await self._service._state.idempotency.list_by_resource(resource_kind, resource_id, tenant_id=tenant_id)

    async def get_idempotency(self, scope: str, key_digest: str, *, tenant_id: str) -> IdempotencyRecord | None:
        return await self._service._state.idempotency.get(scope, key_digest, tenant_id=tenant_id)

    async def get_result(self, execution_id: str, *, tenant_id: str) -> ResultRecord | None:
        return await self._service._state.executions.get_result(execution_id, tenant_id=tenant_id)

    async def result_record(self, execution: ExecutionRecord, *, tenant_id: str) -> ResultRecord | None:
        repository = self._service._state.executions
        return (execution.result if type(repository) is ExecutionRepositoryImpl else
                await repository.get_result(execution.execution_id, tenant_id=tenant_id))

    async def read_output(self, reference: ObjectRef) -> bytes:
        service = self._service
        if service._object_reader is None:
            return await read_runtime_object(service._object_store, reference)
        if reference.store_id != RUNTIME_OBJECT_STORE_ID:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        return await service._object_reader(RuntimeDomain.EXECUTION, reference)


class DefaultExecutionService:
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
        instruction_resolver: "_RepositoryInstructionResolver | None" = None,
        object_key_factory: "RuntimeObjectKeyFactory | None" = None,
        payload_policy: "PayloadPolicy | None" = None,
        input_materializer: "ExecutionInputMaterializer | None" = None,
        session_execution_ready: bool = True,
        task_admissions: TaskAdmissionRepository | None = None,
        tool_responses: ToolResponseFixture | None = None,
        object_reader: "Callable[[RuntimeDomain, ObjectRef], Awaitable[bytes]] | None" = None,
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
        self._restart_checkpoints = restart_checkpoints
        self._durable_trace = durable_trace
        self._conversation = conversation
        self._terminal_commit = terminal_commit
        self._state = state
        self._context_archive = context_archive
        self._object_store = object_store
        self._object_reader = object_reader
        self._sessions = sessions
        self._task_admissions = task_admissions
        self._tool_responses = tool_responses
        self._catalog = catalog
        self._compiler = compiler
        self._authorization = authorization
        self._runtime_bridge = runtime_bridge
        self._live_broker = live_broker
        self._operation_ids = operation_ids or (lambda: uuid.uuid4().hex)
        self._history_reader = history_reader
        self._history_service = history_service
        self._release_terminal = release_terminal or _no_release_terminal
        if not isinstance(session_execution_ready, bool):
            raise TypeError("session_execution_ready must be bool")
        if instruction_resolver is not None and (
            object_key_factory is None or payload_policy is None
        ):
            raise ValueError(
                "repository instruction resolver requires object key factory and payload policy"
            )
        self._instruction_resolver = instruction_resolver
        self._object_key_factory = object_key_factory
        self._payload_policy = payload_policy
        self._input_materializer = input_materializer
        self._session_execution_ready = session_execution_ready
        self._session_locks: dict[tuple[str, str], _SessionLockEntry] = {}
        self._session_locks_guard = asyncio.Lock()
        self._handoff = HandoffGate[tuple[str, str], bool]()
        self._accepting = True
        self._admission_tasks: set[asyncio.Task[ExecutionHandle]] = set()
        self._cancel_tasks: set[asyncio.Task[CancelExecutionResult]] = set()
        self._detached_cancel_finalizers: set[asyncio.Task[CancelExecutionResult]] = (
            set()
        )
        self._detached_cancel_failure: AIError | None = None
        self._management = ExecutionManager(_ExecutionManagementAdapter(self), authorization)

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
    def _local_waiter(self) -> _LocalExecutionWaiter | None:
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
        instructions: _RepositoryInstructionBundle,
        *,
        tenant_id: str,
    ) -> "RuntimePayloadRef":
        from .state._contracts import RuntimePayloadRef

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
                RuntimeDomain.EXECUTION,
                tenant_id,
                canonical_json_bytes(payload),
            )
            stored = StoredPayload.object(reference)
        return RuntimePayloadRef(stored, RuntimeDomain.EXECUTION)

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
    ) -> AgentBinding:
        if binding_contract is None:
            return self._catalog.binding(binding_digest)
        if binding_contract.binding_digest != binding_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        binding = self._compiler.restore(binding_contract)
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

    async def acquire_dependency_hold(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        hold_id: str,
    ) -> bool:
        if self._runtime_bridge.pending_promote is not None:
            await self._runtime_bridge.pending_promote(execution_id)
        created = await self._state.executions.acquire_dependency_hold(
            execution_id,
            tenant_id=tenant_id,
            hold_id=hold_id,
        )
        try:
            await self._handoff.acquire_hold(
                (tenant_id, execution_id),
                hold_id,
            )
        except BaseException:
            if created:
                await self._state.executions.release_dependency_hold(
                    execution_id,
                    tenant_id=tenant_id,
                    hold_id=hold_id,
                )
            raise
        return created

    async def release_dependency_hold(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        hold_id: str,
    ) -> None:
        key = tenant_id, execution_id
        state, owner = await self._handoff.release_hold(key, hold_id)
        if owner and state is not None:
            await self._run_handoff_cleanup(execution_id, tenant_id, state)
        await self._state.executions.release_dependency_hold(
            execution_id,
            tenant_id=tenant_id,
            hold_id=hold_id,
        )
        await self._request_handoff_if_terminal(execution_id, tenant_id)

    async def request_terminal_handoff(
        self, execution_id: str, *, tenant_id: str, terminal_commit: bool = False,
    ) -> None:
        key = tenant_id, execution_id
        state, owner = await self._handoff.request_release(key, value=terminal_commit)
        if owner:
            await self._run_handoff_cleanup(execution_id, tenant_id, state)

    async def resume_terminal_handoff(
        self, execution_id: str, *, tenant_id: str,
    ) -> None:
        state = await self._handoff.claim_requested_release((tenant_id, execution_id))
        if state is not None:
            await self._run_handoff_cleanup(execution_id, tenant_id, state)

    async def _request_handoff_if_terminal(
        self, execution_id: str, tenant_id: str
    ) -> None:
        current = await self._load_execution(execution_id, tenant_id=tenant_id)
        if current is not None and current.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            await self.request_terminal_handoff(
                execution_id, tenant_id=tenant_id, terminal_commit=current.conversation,
            )

    async def _run_handoff_cleanup(
        self,
        execution_id: str,
        tenant_id: str,
        state: HandoffState[bool],
    ) -> None:
        key = tenant_id, execution_id
        waiter = self._local_waiter
        if waiter is not None and waiter.owns_execution(execution_id, tenant_id=tenant_id):
            await self._handoff.finish_release(key, state, succeeded=False)
            if not waiter.owns_execution(execution_id, tenant_id=tenant_id):
                await self.resume_terminal_handoff(execution_id, tenant_id=tenant_id)
            return
        try:
            if state.release_value:
                released = await self._release_terminal(
                    execution_id, tenant_id=tenant_id, terminal_commit=True,
                )
            else:
                released = await self._release_terminal(execution_id, tenant_id=tenant_id)
        except BaseException as error:
            await self._handoff.finish_release(key, state, succeeded=False)
            if not isinstance(error, Exception):
                raise
            _logger.error(
                "execution handoff cleanup failed: execution=%s",
                execution_id,
                exc_info=environ.debug,
            )
            return
        await self._handoff.finish_release(
            key,
            state,
            succeeded=released,
        )

    @asynccontextmanager
    async def _execution_consumer(self, execution_id: str, tenant_id: str):
        key = tenant_id, execution_id
        state = await self._handoff.enter(key)
        try:
            yield
        finally:
            owner = await self._handoff.leave(key, state)
            if owner:
                await self._run_handoff_cleanup(execution_id, tenant_id, state)

    async def start_task(
        self,
        binding: TaskBindingContract,
        *,
        principal: Principal,
        input: Mapping[str, JsonValue],
        idempotency_key: str,
        correlation: Mapping[str, str | int],
        requires_task_invocation_capture: bool = False,
        budget_scope_id: str | None = None,
    ) -> ExecutionHandle:
        if not isinstance(requires_task_invocation_capture, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not isinstance(binding, TaskBindingContract):
            raise TypeError("binding must be TaskBindingContract")
        if budget_scope_id is not None and (
            not isinstance(budget_scope_id, str) or not budget_scope_id.strip()
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        normalized_input = normalize_json_value(dict(input))
        if not isinstance(normalized_input, dict):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        normalized_correlation = _overlay_execution_correlation({}, correlation)
        execution_id = self._operation_ids()
        await self._authorization.authorize(
            principal,
            AuthorizationAction.EXECUTION_RUN,
            ResourceRef(ResourceKind.EXECUTION, execution_id, principal.tenant_id),
        )
        if budget_scope_id is not None:
            await self._authorize_budget_scope(budget_scope_id, principal)
        stored_input = StoredUserInput(
            "task-input-v1",
            StoredPayload.inline_json(normalized_input),
            {
                "version": 1,
                "kind": "task",
                "task_id": binding.id,
                "task_revision": binding.revision,
            },
        )
        scope = "execution.task"
        key_digest = compute_idempotency_key_digest(idempotency_key)
        request_digest = canonical_sha256(
            {
                "contract": "task-execution-request-v1",
                "principal": principal_identity_payload(principal),
                "binding_digest": binding.binding_digest,
                "input_digest": stored_input.digest,
                **({"budget_scope_id": budget_scope_id} if budget_scope_id is not None else {}),
                **({"requires_task_invocation_capture": True} if requires_task_invocation_capture else {}),
            }
        )
        now = datetime.now(timezone.utc)
        execution = ExecutionRecord(
            execution_id=execution_id,
            session_id=None,
            parent_execution_id=None,
            root_execution_id=execution_id,
            previous_execution_id=None,
            fork_base_execution_id=None,
            lineage_kind=ExecutionLineageKind.RUN,
            status=ExecutionStatus.PENDING_START,
            revision=0,
            event_seq=0,
            agent_run_seq=0,
            error_code=None,
            safe_error_details={},
            created_at=now,
            updated_at=now,
            mode=None,
            planning=None,
            thinking=None,
            binding=binding,
            principal_id=principal.principal_id,
            principal_kind=principal.kind,
            stored_user_input=stored_input,
            correlation=normalized_correlation,
            requires_task_invocation_capture=requires_task_invocation_capture,
            budget_scope_id=budget_scope_id,
            restart_checkpoints=True,
            durable_trace=True,
            conversation=False,
        )
        reservation = await self._state.executions.reserve_start(
            ExecutionStartReservation(
                execution,
                IdempotencyRecord(
                    scope=scope,
                    idempotency_key_digest=key_digest,
                    request_digest=request_digest,
                    resource_kind=ResourceKind.EXECUTION,
                    resource_id=execution_id,
                    status=IdempotencyStatus.RESERVED,
                    result_digest=None,
                    error_code=None,
                    created_at=now,
                    updated_at=now,
                ),
            )
        )
        current = reservation.execution
        if (
            current.requires_task_invocation_capture is not requires_task_invocation_capture
            or current.budget_scope_id != budget_scope_id
            or current.binding_digest != binding.binding_digest
            or current.binding != binding
            or current.stored_user_input.digest != stored_input.digest
            or current.principal_id != principal.principal_id
            or current.principal_kind != principal.kind
        ):
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        if reservation.idempotency.request_digest != request_digest:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        if current.status is ExecutionStatus.PENDING_START:
            current = await self._state.executions.claim_start(
                ExecutionStartClaim(
                    current.execution_id,
                    current.revision,
                    current.event_seq,
                    scope,
                    key_digest,
                    request_digest,
                    now,
                )
            )
        elif current.status is ExecutionStatus.START_UNKNOWN:
            raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
        elif current.status not in {
            ExecutionStatus.STARTED,
            ExecutionStatus.WAITING_RETRY,
            ExecutionStatus.WAITING_DEFERRED,
            ExecutionStatus.RECOVERY_REQUIRED,
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return ExecutionHandle(current.execution_id)

    async def claim_task_attempt(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionView:
        current = await self._load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_RUN,
        )
        if not isinstance(current.binding, TaskBindingContract):
            raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        if current.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.RECOVERY_REQUIRED,
        }:
            return _execution_view(current)
        now = datetime.now(timezone.utc)
        if current.status is ExecutionStatus.WAITING_RETRY:
            if current.task_next_attempt_at is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current.task_next_attempt_at > now:
                return _execution_view(current)
        elif current.status is not ExecutionStatus.STARTED:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if current.task_deadline_at is not None and current.task_deadline_at <= now:
            await self.fail_task(
                execution_id,
                principal=principal,
                error=AIError(
                    ErrorCode.EXECUTION_WAIT_TIMEOUT,
                    safe_details={"reason": "deadline_exceeded"},
                ),
                attempt=_execution_view(current),
            )
            current = await self._load_authorized(
                execution_id,
                principal,
                AuthorizationAction.EXECUTION_RUN,
            )
            return _execution_view(current)
        if current.task_attempt >= current.binding.max_attempts:
            await self.fail_task(
                execution_id,
                principal=principal,
                error=AIError(
                    ErrorCode.TASK_NODE_FAILED,
                    safe_details={"reason": "attempts_exhausted"},
                ),
                attempt=_execution_view(current),
            )
            current = await self._load_authorized(
                execution_id,
                principal,
                AuthorizationAction.EXECUTION_RUN,
            )
            return _execution_view(current)
        deadline = current.task_deadline_at
        if deadline is None and current.binding.timeout_seconds is not None:
            deadline = now + timedelta(seconds=current.binding.timeout_seconds)
        updated = await self._state.executions.transition_task_execution(
            execution_id,
            tenant_id=self._state.executions.tenant_id,
            expected_revision=current.revision,
            expected_event_seq=current.event_seq,
            expected_status=current.status,
            next_status=ExecutionStatus.STARTED,
            task_attempt=current.task_attempt + 1,
            task_deadline_at=deadline,
            task_next_attempt_at=None,
            error_code=None,
            safe_error_details={},
            event_type=ExecutionEventType.EXECUTION_RESUMED.value,
            payload={"attempt": current.task_attempt + 1},
            occurred_at=now,
        )
        return _execution_view(updated)

    async def schedule_task_retry(
        self,
        execution_id: str,
        *,
        principal: Principal,
        error_code: str,
        attempt: ExecutionView | None = None,
    ) -> ExecutionView:
        current = await self._load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_RUN,
        )
        if (
            not isinstance(current.binding, TaskBindingContract)
            or current.status is not ExecutionStatus.STARTED
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        _require_task_attempt(current, attempt)
        if current.task_attempt >= current.binding.max_attempts:
            await self.fail_task(
                execution_id,
                principal=principal,
                error=AIError(
                    ErrorCode.TASK_NODE_FAILED,
                    safe_details={"reason": "attempts_exhausted"},
                ),
                attempt=attempt,
            )
            return await self.inspect(execution_id, principal=principal)
        now = datetime.now(timezone.utc)
        retry_at = now + timedelta(seconds=current.binding.retry_delay_seconds)
        if current.task_deadline_at is not None and retry_at >= current.task_deadline_at:
            await self.fail_task(
                execution_id,
                principal=principal,
                error=AIError(
                    ErrorCode.EXECUTION_WAIT_TIMEOUT,
                    safe_details={"reason": "deadline_exceeded"},
                ),
                attempt=attempt,
            )
            return await self.inspect(execution_id, principal=principal)
        updated = await self._state.executions.transition_task_execution(
            execution_id,
            tenant_id=self._state.executions.tenant_id,
            expected_revision=current.revision,
            expected_event_seq=current.event_seq,
            expected_status=ExecutionStatus.STARTED,
            next_status=ExecutionStatus.WAITING_RETRY,
            task_attempt=current.task_attempt,
            task_deadline_at=current.task_deadline_at,
            task_next_attempt_at=retry_at,
            error_code=error_code,
            safe_error_details={},
            event_type=ExecutionEventType.EXECUTION_RETRY_SCHEDULED.value,
            payload={"attempt": current.task_attempt, "retry_at": retry_at.isoformat()},
            occurred_at=now,
        )
        return _execution_view(updated)

    async def require_task_recovery(
        self,
        execution_id: str,
        *,
        principal: Principal,
        error_code: str,
        attempt: ExecutionView | None = None,
    ) -> ExecutionView:
        current = await self._load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_RUN,
        )
        if not isinstance(current.binding, TaskBindingContract):
            raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        _require_task_attempt(current, attempt)
        if current.status is ExecutionStatus.RECOVERY_REQUIRED:
            return _execution_view(current)
        if current.status is not ExecutionStatus.STARTED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        now = datetime.now(timezone.utc)
        updated = await self._state.executions.transition_task_execution(
            execution_id,
            tenant_id=self._state.executions.tenant_id,
            expected_revision=current.revision,
            expected_event_seq=current.event_seq,
            expected_status=ExecutionStatus.STARTED,
            next_status=ExecutionStatus.RECOVERY_REQUIRED,
            task_attempt=current.task_attempt,
            task_deadline_at=current.task_deadline_at,
            task_next_attempt_at=None,
            error_code=error_code,
            safe_error_details={"effect": "unknown"},
            event_type=ExecutionEventType.EXECUTION_RECOVERY_REQUIRED.value,
            payload={"error_code": error_code, "safe_error_details": {"effect": "unknown"}},
            occurred_at=now,
        )
        return _execution_view(updated)

    async def defer_task_input(
        self,
        execution_id: str,
        *,
        principal: Principal,
        wait_id: str,
    ) -> ExecutionView:
        current = await self._load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_RUN,
        )
        if (
            not isinstance(current.binding, TaskBindingContract)
            or not isinstance(wait_id, str)
            or not wait_id.strip()
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if current.status is ExecutionStatus.WAITING_DEFERRED:
            return _execution_view(current)
        if current.status is not ExecutionStatus.STARTED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        now = datetime.now(timezone.utc)
        updated = await self._state.executions.transition_task_execution(
            execution_id,
            tenant_id=self._state.executions.tenant_id,
            expected_revision=current.revision,
            expected_event_seq=current.event_seq,
            expected_status=ExecutionStatus.STARTED,
            next_status=ExecutionStatus.WAITING_DEFERRED,
            task_attempt=current.task_attempt,
            task_deadline_at=current.task_deadline_at,
            task_next_attempt_at=None,
            error_code=None,
            safe_error_details={},
            event_type=ExecutionEventType.EXTERNAL_REQUESTED.value,
            payload={"wait_id": wait_id},
            occurred_at=now,
        )
        return _execution_view(updated)

    async def supply_task_input(
        self,
        execution_id: str,
        *,
        principal: Principal,
        value: JsonValue,
    ) -> ExecutionView:
        current = await self._load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_RUN,
        )
        if not isinstance(current.binding, TaskBindingContract):
            raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        normalized = normalize_json_value(value)
        try:
            _validate_task_binding_output(current.binding, normalized)
        except AIError as error:
            if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                raise
            raise AIError(
                ErrorCode.OUTPUT_CONTRACT_INVALID,
                retryable=False,
            ) from error
        value_digest = canonical_sha256(normalized)
        if current.status is ExecutionStatus.SUCCEEDED:
            result = await self.result(execution_id, principal=principal)
            if canonical_sha256(result.output) != value_digest:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return _execution_view(current)
        if current.status is not ExecutionStatus.WAITING_DEFERRED:
            raise AIError(ErrorCode.TASK_NOT_READY)
        try:
            await self._complete_task_output(
                current,
                principal=principal,
                output=normalized,
                terminal_event_payload={
                    "task_input": "supplied",
                    "task_input_value_digest": value_digest,
                },
            )
        except AIError as error:
            if error.code not in {ErrorCode.EXECUTION_RESULT_CONFLICT, ErrorCode.STORAGE_CONFLICT}:
                raise
            latest = await self.inspect(execution_id, principal=principal)
            if latest.status is not ExecutionStatus.SUCCEEDED:
                raise
            result = await self.result(execution_id, principal=principal)
            if canonical_sha256(result.output) != value_digest:
                raise
            return latest
        return await self.inspect(execution_id, principal=principal)

    async def resume_task_not_applied(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionView:
        current = await self._load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_RUN,
        )
        if (
            not isinstance(current.binding, TaskBindingContract)
            or current.status is not ExecutionStatus.RECOVERY_REQUIRED
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        now = datetime.now(timezone.utc)
        if (
            current.task_deadline_at is not None
            and current.task_deadline_at <= now
        ):
            await self.fail_task(
                execution_id,
                principal=principal,
                error=AIError(
                    ErrorCode.EXECUTION_WAIT_TIMEOUT,
                    safe_details={
                        "task_effect": "not_applied",
                        "reason": "deadline_exceeded",
                    },
                ),
            )
            return await self.inspect(execution_id, principal=principal)
        if current.task_attempt >= current.binding.max_attempts:
            await self.fail_task(
                execution_id,
                principal=principal,
                error=AIError(
                    ErrorCode.TASK_NODE_FAILED,
                    safe_details={
                        "task_effect": "not_applied",
                        "reason": "attempts_exhausted",
                    },
                ),
            )
            return await self.inspect(execution_id, principal=principal)
        updated = await self._state.executions.transition_task_execution(
            execution_id,
            tenant_id=self._state.executions.tenant_id,
            expected_revision=current.revision,
            expected_event_seq=current.event_seq,
            expected_status=ExecutionStatus.RECOVERY_REQUIRED,
            next_status=ExecutionStatus.WAITING_RETRY,
            task_attempt=current.task_attempt,
            task_deadline_at=current.task_deadline_at,
            task_next_attempt_at=now,
            error_code=None,
            safe_error_details={},
            event_type=ExecutionEventType.EXECUTION_RESUMED.value,
            payload={"reason": "task_effect_not_applied"},
            occurred_at=now,
        )
        return _execution_view(updated)

    async def resolve_task_effect(
        self,
        execution_id: str,
        *,
        principal: Principal,
        resolution: TaskEffectResolution,
    ) -> ExecutionView:
        if not isinstance(resolution, TaskEffectResolution):
            raise TypeError("resolution must be TaskEffectResolution")
        current = await self._load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_RUN,
        )
        binding = current.binding
        if (
            not isinstance(binding, TaskBindingContract)
            or binding.effect_policy != "non_replay_safe"
        ):
            raise AIError(ErrorCode.TASK_NOT_READY)

        applied_output = (
            None
            if resolution.kind != "applied"
            else normalize_json_value(resolution.value)
        )
        applied_digest = (
            None
            if resolution.kind != "applied"
            else canonical_sha256(applied_output)
        )

        if current.status is ExecutionStatus.WAITING_RETRY:
            if resolution.kind != "not_applied":
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return _execution_view(current)
        if current.status is ExecutionStatus.SUCCEEDED:
            if resolution.kind != "applied" or applied_digest is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            result = await self.result(execution_id, principal=principal)
            if canonical_sha256(result.output) != applied_digest:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return _execution_view(current)
        if current.status is ExecutionStatus.FAILED:
            if (
                resolution.kind != "applied"
                or applied_digest is None
                or current.error_code != ErrorCode.OUTPUT_CONTRACT_INVALID.value
                or current.safe_error_details.get("task_effect") != "applied"
                or current.safe_error_details.get("task_effect_value_digest")
                != applied_digest
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return _execution_view(current)
        if current.status is ExecutionStatus.CANCELLED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if current.status is not ExecutionStatus.RECOVERY_REQUIRED:
            raise AIError(ErrorCode.TASK_NOT_READY)
        if resolution.kind == "unknown":
            return _execution_view(current)

        if resolution.kind == "not_applied":
            now = datetime.now(timezone.utc)
            if (
                current.task_deadline_at is not None
                and current.task_deadline_at <= now
            ):
                await self.fail_task(
                    execution_id,
                    principal=principal,
                    error=AIError(
                        ErrorCode.EXECUTION_WAIT_TIMEOUT,
                        retryable=False,
                        safe_details={
                            "task_effect": "not_applied",
                            "reason": "deadline_exceeded",
                        },
                    ),
                )
                return await self.inspect(execution_id, principal=principal)
            if current.task_attempt >= binding.max_attempts:
                await self.fail_task(
                    execution_id,
                    principal=principal,
                    error=AIError(
                        ErrorCode.TASK_NODE_FAILED,
                        retryable=False,
                        safe_details={
                            "task_effect": "not_applied",
                            "reason": "attempts_exhausted",
                        },
                    ),
                )
                return await self.inspect(execution_id, principal=principal)
            return await self.resume_task_not_applied(
                execution_id,
                principal=principal,
            )

        if applied_digest is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            _validate_task_binding_output(binding, applied_output)
        except AIError as error:
            if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                raise
            await self.fail_task(
                execution_id,
                principal=principal,
                error=AIError(
                    ErrorCode.OUTPUT_CONTRACT_INVALID,
                    retryable=False,
                    safe_details={
                        "task_effect": "applied",
                        "task_effect_value_digest": applied_digest,
                    },
                ),
            )
            return await self.inspect(execution_id, principal=principal)

        await self._complete_task_output(
            current,
            principal=principal,
            output=applied_output,
            terminal_event_payload={
                "task_effect": "applied",
                "task_effect_value_digest": applied_digest,
            },
        )
        return await self.inspect(execution_id, principal=principal)

    async def cancel_task(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> CancelExecutionResult:
        current = await self._load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_RUN,
        )
        if not isinstance(current.binding, TaskBindingContract):
            raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        if current.status is ExecutionStatus.CANCELLED:
            return CancelExecutionResult(execution_id, True)
        if current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED}:
            return CancelExecutionResult(execution_id, False)
        if current.status is ExecutionStatus.RECOVERY_REQUIRED:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        if (
            current.binding.effect_policy == "non_replay_safe"
            and current.task_attempt > 0
            and current.status is ExecutionStatus.STARTED
        ):
            await self.require_task_recovery(
                execution_id,
                principal=principal,
                error_code=ErrorCode.TASK_EFFECT_UNKNOWN.value,
            )
            raise AIError(ErrorCode.TASK_EFFECT_UNKNOWN)
        now = datetime.now(timezone.utc)
        result = ResultRecord(
            None,
            StopReason.CANCELLED,
            UsageMetrics(),
            now,
        )
        terminal = replace(
            current,
            status=ExecutionStatus.CANCELLED,
            error_code=ErrorCode.EXECUTION_CANCELLED.value,
            safe_error_details={},
            error_diagnostics=None,
            task_next_attempt_at=None,
            updated_at=now,
        )
        idempotency = await self._task_terminal_idempotency(
            current,
            result,
            next_status=IdempotencyStatus.CANCELLED,
            error_code=ErrorCode.EXECUTION_CANCELLED.value,
        )
        await self._state.executions.commit_terminal(
            ExecutionTerminalCommit(
                current.revision,
                current.event_seq,
                terminal,
                result,
                ExecutionEventType.EXECUTION_CANCELLED,
                {},
                idempotency,
            )
        )
        return CancelExecutionResult(execution_id, True)

    async def complete_task(
        self,
        execution_id: str,
        *,
        principal: Principal,
        output: JsonValue,
        attempt: ExecutionView | None = None,
    ) -> ExecutionResult:
        current = await self._load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_RUN,
        )
        return await self._complete_task_output(
            current,
            principal=principal,
            output=output,
            attempt=attempt,
        )

    async def _complete_task_output(
        self,
        current: ExecutionRecord,
        *,
        principal: Principal,
        output: JsonValue,
        terminal_event_payload: "Mapping[str, JsonValue] | None" = None,
        attempt: ExecutionView | None = None,
    ) -> ExecutionResult:
        if not isinstance(current.binding, TaskBindingContract):
            raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        if attempt is not None and current.task_attempt != attempt.task_attempt:
            raise AIError(ErrorCode.TASK_FENCE_STALE)
        execution_id = current.execution_id
        if current.status is ExecutionStatus.SUCCEEDED:
            result = await self.result(execution_id, principal=principal)
            if canonical_sha256(result.output) != canonical_sha256(output):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return result
        if attempt is not None:
            _require_task_attempt(current, attempt)
            if current.status is not ExecutionStatus.STARTED:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
        if current.status not in {
            ExecutionStatus.STARTED,
            ExecutionStatus.WAITING_DEFERRED,
            ExecutionStatus.RECOVERY_REQUIRED,
        }:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        normalized = normalize_json_value(output)
        payload = await self._store_task_output(
            normalized,
            tenant_id=self._state.executions.tenant_id,
        )
        now = datetime.now(timezone.utc)
        result = ResultRecord(
            payload,
            StopReason.END_TURN,
            UsageMetrics(),
            now,
        )
        terminal = replace(
            current,
            status=ExecutionStatus.SUCCEEDED,
            error_code=None,
            safe_error_details={},
            error_diagnostics=None,
            task_next_attempt_at=None,
            updated_at=now,
        )
        idempotency = await self._task_terminal_idempotency(
            current,
            result,
            next_status=IdempotencyStatus.COMPLETED,
        )
        await self._state.executions.commit_terminal(
            ExecutionTerminalCommit(
                current.revision,
                current.event_seq,
                terminal,
                result,
                ExecutionEventType.EXECUTION_SUCCEEDED,
                {}
                if terminal_event_payload is None
                else dict(terminal_event_payload),
                idempotency,
            )
        )
        return await self.result(execution_id, principal=principal)

    async def fail_task(
        self,
        execution_id: str,
        *,
        principal: Principal,
        error: AIError,
        attempt: ExecutionView | None = None,
    ) -> ExecutionResult:
        current = await self._load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_RUN,
        )
        if not isinstance(current.binding, TaskBindingContract):
            raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        if attempt is not None and current.task_attempt != attempt.task_attempt:
            raise AIError(ErrorCode.TASK_FENCE_STALE)
        if current.status is ExecutionStatus.FAILED:
            result = await self.result(execution_id, principal=principal)
            if (
                current.error_code != error.code.value
                or dict(current.safe_error_details) != dict(error.safe_details)
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return result
        if attempt is not None:
            _require_task_attempt(current, attempt)
        if current.status not in {
            ExecutionStatus.STARTED,
            ExecutionStatus.WAITING_RETRY,
            ExecutionStatus.WAITING_DEFERRED,
            ExecutionStatus.RECOVERY_REQUIRED,
        }:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        now = datetime.now(timezone.utc)
        details = dict(error.safe_details)
        result = ResultRecord(
            None,
            StopReason.ERROR,
            UsageMetrics(),
            now,
        )
        terminal = replace(
            current,
            status=ExecutionStatus.FAILED,
            error_code=error.code.value,
            safe_error_details=details,
            error_diagnostics=error.diagnostics,
            task_next_attempt_at=None,
            updated_at=now,
        )
        idempotency = await self._task_terminal_idempotency(
            current,
            result,
            next_status=IdempotencyStatus.FAILED,
            error_code=error.code.value,
        )
        payload = {
            "error_code": error.code.value,
            "safe_error_details": details,
        }
        if error.diagnostics is not None:
            payload["error_diagnostics"] = {
                "exception_type": error.diagnostics.exception_type,
                "exception_message": error.diagnostics.exception_message,
                "cause_digest": error.diagnostics.cause_digest,
            }
        await self._state.executions.commit_terminal(
            ExecutionTerminalCommit(
                current.revision,
                current.event_seq,
                terminal,
                result,
                ExecutionEventType.EXECUTION_FAILED,
                payload,
                idempotency,
            )
        )
        return await self.result(execution_id, principal=principal)

    async def _store_task_output(
        self,
        output: JsonValue,
        *,
        tenant_id: str,
    ) -> StoredPayload:
        inline = StoredPayload.inline_json(output)
        if self._payload_policy is None or payload_fits_inline(inline, self._payload_policy):
            return inline
        if self._object_key_factory is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        reference = await put_runtime_object(
            self._object_store,
            self._object_key_factory,
            RuntimeDomain.EXECUTION,
            tenant_id,
            canonical_json_bytes(output),
        )
        return StoredPayload.object(reference)

    async def _task_terminal_idempotency(
        self,
        execution: ExecutionRecord,
        result: ResultRecord,
        *,
        next_status: IdempotencyStatus,
        error_code: str | None = None,
    ) -> IdempotencyTerminalUpdate:
        records = await self._state.idempotency.list_by_resource(
            ResourceKind.EXECUTION,
            execution.execution_id,
            tenant_id=self._state.executions.tenant_id,
        )
        if len(records) != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        record = records[0]
        if record.status is not IdempotencyStatus.STARTED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return IdempotencyTerminalUpdate(
            record.scope,
            record.idempotency_key_digest,
            IdempotencyStatus.STARTED,
            next_status,
            record.request_digest,
            None if result.output is None else result.output.digest,
            error_code,
        )

    async def start(
        self,
        binding_digest: str,
        request: ExecutionRequest,
        *,
        dependency_hold_id: "str | None" = None,
        binding_contract: "AgentBindingContract | None" = None,
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
            await self._authorize_budget_scope(budget_scope_id, request.principal)
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
        execution = await self._load_execution(
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
        async with self._execution_consumer(execution_id, principal.tenant_id):
            pending = self._runtime_bridge.pending_turns.get(execution_id, tenant_id=principal.tenant_id)
            if pending is not None:
                await self._authorize_execution(execution_id, principal, AuthorizationAction.EXECUTION_READ)
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
                    header = await self._authorize_execution(
                        execution_id, principal, AuthorizationAction.EXECUTION_READ,
                    )
                    remote_pending = await self._runtime_bridge.pending_turns.read_remote_execution(
                        execution_id, tenant_id=principal.tenant_id,
                    )
                    if remote_pending is not None:
                        await self._authorize_loaded_execution(
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
                    await self._authorize_loaded_execution(
                        current, principal, AuthorizationAction.EXECUTION_READ,
                        owner_principal_id=header.owner_principal_id,
                    )
                return children
            await self._load_authorized(
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
        requires_task_invocation_capture: bool = False,
        budget_scope_id: str | None = None,
    ) -> ExecutionHandle:
        metric_delivery: _TerminalMetricDelivery | None = None

        def capture_metric(delivery: _TerminalMetricDelivery) -> None:
            nonlocal metric_delivery
            metric_delivery = delivery

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
                    capture_metric=capture_metric,
                )
        finally:
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
        binding = self._binding(binding_digest, binding_contract)
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
            await self._authorize_budget_scope(
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
                pending = await self._load_execution(
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
                terminal = await self._load_execution(
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
            started = await self._load_execution(
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
                from ..spec import RepositoryInstructions
                repository_instructions = await self._materialize_repository_instructions(
                    RepositoryInstructions.from_payload(request.input_context.repository_instructions), tenant_id=request.principal.tenant_id)
        elif self._instruction_resolver is not None:
            if lineage_kind is ExecutionLineageKind.SUBAGENT:
                if parent is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                repository_instructions = parent.repository_instructions
                if repository_instructions is None:
                    resolved_instructions = await self._instruction_resolver.resolve(
                        "."
                    )
                    repository_instructions = (
                        await self._materialize_repository_instructions(
                            resolved_instructions,
                            tenant_id=request.principal.tenant_id,
                        )
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
                resolved_instructions = await self._instruction_resolver.resolve(
                    instruction_path
                )
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
    ) -> ExecutionHandle:
        if not self._accepting:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        # Source-first and deferred writers must observe the same admission owner.
        coordinator = self._runtime_bridge.pending_turns.coordination
        execution = proposed_reservation.execution
        identity = proposed_reservation.idempotency

        metric_delivery: _TerminalMetricDelivery | None = None

        def capture_owned_metric(delivery: _TerminalMetricDelivery) -> None:
            nonlocal metric_delivery
            metric_delivery = delivery

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
                    capture_metric=capture_owned_metric,
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
                if metric_delivery is not None:
                    await metric_delivery()

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
        coordinator: _PendingTurnCoordination,
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
            record = await self._load_execution(
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
                record = await self._load_source_execution(
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
        current = await self._load_execution(
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
                    observed = await self._load_execution(
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
                latest = await self._load_execution(
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
        launch_record = await self._load_execution(
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
            current = await self._load_execution(
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

    async def budget_usage(
        self, execution_id: str, *, principal: Principal,
    ) -> BudgetUsage | None:
        execution = await self._load_observed_authorized(
            execution_id, principal, AuthorizationAction.EXECUTION_READ,
        )
        if execution.budget_scope_id is None:
            return None
        return await self._state.budgets.read(execution.budget_scope_id)

    @_observed_query
    async def inspect(
        self, execution_id: str, *, principal: Principal
    ) -> ExecutionView:
        execution = await self._load_observed_authorized(
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
    async def result(
        self, execution_id: str, *, principal: Principal
    ) -> ExecutionResult:
        execution = await self._load_authorized(
            execution_id, principal, AuthorizationAction.EXECUTION_READ
        )
        return await self._consume_result_from_execution(execution, tenant_id=principal.tenant_id)

    async def _consume_result_from_execution(
        self, execution: ExecutionRecord, *, tenant_id: str
    ) -> ExecutionResult:
        result = await self._result_from_execution(execution, tenant_id=tenant_id)
        await self._require_retained_turn(execution)
        stored = execution.result
        if type(self._state.executions) is not ExecutionRepositoryImpl or (
            stored is not None and stored.output is not None and stored.output.kind != "inline"
        ):
            # Body and custom-owner reads may yield after the terminal snapshot.
            await self._request_handoff_if_terminal(execution.execution_id, tenant_id=tenant_id)
        else:
            await self.request_terminal_handoff(
                execution.execution_id, tenant_id=tenant_id, terminal_commit=execution.conversation,
            )
        return result

    async def _require_retained_turn(self, execution: ExecutionRecord) -> None:
        if (not execution.retention_closed and execution.conversation
                and execution.session_id is not None and self._sessions is not None):
            await self._sessions.state_store.read(
                lambda transaction: self._sessions.pending_turn_commit_in_transaction(
                    transaction, session_id=execution.session_id,
                    execution_id=execution.execution_id,
                    allow_unadmitted=execution.started_at is None,
                )
            )

    async def _result_from_execution(self, execution: ExecutionRecord, *, tenant_id: str) -> ExecutionResult:
        return await self._management.result_from_execution(execution, tenant_id=tenant_id)

    @consumed_terminal_query
    async def result_payload_size(
        self, execution_id: str, *, principal: Principal
    ) -> int:
        execution = await self._load_authorized(
            execution_id,
            principal,
            AuthorizationAction.EXECUTION_READ,
        )
        if execution.status is ExecutionStatus.RECOVERY_REQUIRED:
            raise _execution_recovery_error(execution)
        if execution.status is not ExecutionStatus.SUCCEEDED:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        repository = self._state.executions
        result = (execution.result if type(repository) is ExecutionRepositoryImpl else
                  await repository.get_result(execution_id, tenant_id=principal.tenant_id))
        if result is None or result.output is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._require_retained_turn(execution)
        if type(repository) is ExecutionRepositoryImpl:
            await self.request_terminal_handoff(
                execution_id, tenant_id=principal.tenant_id, terminal_commit=execution.conversation,
            )
        else:
            await self._request_handoff_if_terminal(execution_id, tenant_id=principal.tenant_id)
        return result.output.size

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
            execution = await self._load_authorized(
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
                waiter = self._local_waiter
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
                        execution = await self._load_authorized(
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
                    return await self._consume_result_from_execution(
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
                execution = await self._load_authorized(
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
        previous = await self._load_authorized(
            execution_id, request.principal, AuthorizationAction.EXECUTION_READ
        )
        if (
            previous.parent_execution_id is not None
            or not isinstance(previous.binding, AgentBindingContract)
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        binding_digest = previous.binding_digest
        binding = self._binding(binding_digest, previous.binding)
        if binding.binding_digest != binding_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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
        previous = await self._load_authorized(
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
        binding = self._binding(binding_digest, previous.binding)
        if binding.binding_digest != binding_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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
        async with self._execution_consumer(
            execution_id,
            request.principal.tenant_id,
        ):
            result = await self.cancel_authorized(
                execution_id, request, capture_metric=capture_metric,
            )
            await self._request_handoff_if_terminal(
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
        execution = await self._load_authorized(
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
        execution = await self._load_authorized(
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
        execution = await self._load_authorized(
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

    async def _authorize_budget_scope(
        self,
        scope_id: str,
        principal: Principal,
        *,
        source_execution_id: str | None = None,
    ) -> None:
        if source_execution_id is not None or scope_id.startswith("execution:"):
            source = await self._load_authorized(
                source_execution_id or scope_id[len("execution:"):],
                principal,
                AuthorizationAction.EXECUTION_RUN,
            )
            await self._authorization.authorize(
                principal,
                AuthorizationAction.EXECUTION_RUN,
                ResourceRef(
                    ResourceKind.EXECUTION,
                    source.execution_id,
                    principal.tenant_id,
                    source.principal_id,
                ),
            )
            if source.budget_scope_id != scope_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._state.budgets.read(scope_id)
            return
        if scope_id.startswith("graph:"):
            if self._task_admissions is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            graph_id = scope_id[len("graph:"):]
            admission = await self._task_admissions.get(
                graph_id, tenant_id=principal.tenant_id,
            )
            if admission is None:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            await self._authorization.authorize(
                principal,
                AuthorizationAction.TASK_RUN,
                ResourceRef(
                    ResourceKind.TASK_GRAPH,
                    admission.graph_id,
                    admission.principal.tenant_id,
                    admission.principal.principal_id,
                ),
            )
            if admission.graph_id != graph_id or admission.budget_scope_id != scope_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            usage = await self._state.budgets.read(scope_id)
            if usage.limits != admission.budget:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        raise AIError(ErrorCode.AUTHORIZATION_DENIED)

    async def _load_session(self, session_id: str, *, tenant_id: str) -> "SessionRecord | None":
        return await self._sessions.state_store.read(
            lambda _transaction: self._sessions.get(session_id, tenant_id=tenant_id)
        )

    async def _load_source_execution(
        self, execution_id: str, *, tenant_id: str,
    ) -> ExecutionRecord | None:
        repository = self._state.executions
        if type(repository) is ExecutionRepositoryImpl:
            return await repository.get_fresh(execution_id, tenant_id=tenant_id)
        return await repository.state_store.read(
            lambda _transaction: repository.get(execution_id, tenant_id=tenant_id)
        )

    async def _load_execution(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        pending = await self._runtime_bridge.pending_turns.read_execution(execution_id, tenant_id=tenant_id)
        if pending is not None:
            return pending
        record = await self._load_source_execution(execution_id, tenant_id=tenant_id)
        if record is not None:
            return record
        pending = await self._runtime_bridge.pending_turns.read_remote_execution(
            execution_id, tenant_id=tenant_id,
        )
        if pending is not None:
            return pending
        # Publication can disappear after the source read when its owner commits.
        return await self._load_source_execution(execution_id, tenant_id=tenant_id)

    async def _load_authorized(self, execution_id: str, principal: Principal,
                               action: AuthorizationAction) -> ExecutionRecord:
        return await self._management.load_authorized(execution_id, principal, action)

    async def _load_observed_authorized(
        self, execution_id: str, principal: Principal, action: AuthorizationAction
    ) -> ExecutionRecord:
        header = await self._authorize_execution(execution_id, principal, action)
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
            record = await self._load_source_execution(
                execution_id, tenant_id=principal.tenant_id,
            )
        if record is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorize_loaded_execution(
            record, principal, action, owner_principal_id=header.owner_principal_id,
        )
        return record

    async def _authorize_execution(
        self, execution_id: str, principal: Principal, action: AuthorizationAction
    ) -> ResourceRef:
        return await self._management.authorize_execution(execution_id, principal, action)


    async def _authorize_loaded_execution(self, execution: ExecutionRecord, principal: Principal,
        action: AuthorizationAction, *, owner_principal_id: str | None) -> None:
        await self._management.authorize_loaded_execution(execution, principal, action,
                                                         owner_principal_id=owner_principal_id)

    async def _load_source_header(
        self, execution_id: str, *, tenant_id: str,
    ) -> ResourceRef | None:
        repository = self._state.executions
        if type(repository) is ExecutionRepositoryImpl:
            return await repository.get_header_fresh(execution_id, tenant_id=tenant_id)
        return await repository.state_store.read(
            lambda _transaction: repository.get_header(execution_id, tenant_id=tenant_id)
        )

    async def load_authorized(
        self,
        execution_id: str,
        principal: Principal,
        action: AuthorizationAction,
    ) -> ExecutionRecord:
        return await self._load_authorized(execution_id, principal, action)


def _require_task_attempt(
    execution: ExecutionRecord,
    attempt: ExecutionView | None,
) -> None:
    if attempt is None:
        return
    if (
        attempt.execution_id != execution.execution_id
        or attempt.event_seq != execution.event_seq
        or attempt.task_attempt != execution.task_attempt
        or attempt.status is not execution.status
        or attempt.task_deadline_at != execution.task_deadline_at
        or attempt.task_next_attempt_at != execution.task_next_attempt_at
    ):
        raise AIError(ErrorCode.TASK_FENCE_STALE)


def _execution_view(execution: ExecutionRecord) -> ExecutionView:
    return _project_execution_view(execution)


def _validate_task_binding_output(
    binding: TaskBindingContract,
    output: JsonValue,
) -> None:
    contract = dict(binding.output_contract)
    if contract == {"kind": "json"}:
        return
    if set(contract) != {"mode", "schema"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        restored = restore_output(contract["mode"], contract["schema"])
    except AIError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    try:
        restored.validate_payload(output)
    except AIError as error:
        if error.code is ErrorCode.OUTPUT_VALIDATION_FAILED:
            raise AIError(
                ErrorCode.OUTPUT_CONTRACT_INVALID,
                retryable=False,
            ) from error
        raise


def _failed_event_payload(
    error_code: str,
    safe_error_details: Mapping[str, JsonValue],
    diagnostics: ErrorDiagnostics,
) -> dict[str, JsonValue]:
    return {
        "error_code": error_code,
        "safe_error_details": dict(safe_error_details),
        "error_diagnostics": {
            "exception_type": diagnostics.exception_type,
            "exception_message": diagnostics.exception_message,
            "cause_digest": diagnostics.cause_digest,
        },
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
