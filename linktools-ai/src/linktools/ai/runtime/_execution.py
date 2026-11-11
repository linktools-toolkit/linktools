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
from ..steps import NativeStepSource

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
    StoredPayload,
    TurnCoordination,
)
from ._conversation_service import ConversationService
from ._execution_payload import _materialize_execution_payloads
from ..inputs import ExecutionContextArchive
from ..capability import ToolResponseFixture
from ._tool_response_fixture import _load_tool_response_fixture
from ._feature_policy import _runtime_feature_policy
from ..core import HandoffGate, HandoffState
from ..inputs import (
    ExecutionInputMaterializer,
    decode_user_content_payload,
    encode_user_content,
    input_intent,
    validate_user_input,
)
from ..persistence.objects import ObjectKeyFactory, read_runtime_object
from ..execution import ExecutionRecoveryEffect, ResolveToolEffectRequest, ToolEffectResolutionResult
from ..execution import (
    CancelExecutionRequest,
    CancelExecutionResult,
    ExecutionHandle,
    ExecutionHistoryItem,
    ExecutionHistoryReader,
    ExecutionHistoryService,
    ExecutionRequest,
    ExecutionInputContext,
    RuntimePayloadRef,
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
    ExecutionStartReservationResult,
    ExecutionStartUnknownCommit,
    ExecutionTerminalCommit,
    ExecutionTerminalCommitResult,
    IdempotencyRecord,
    IdempotencyTerminalUpdate,
    OperationTerminalUpdate,
    ResultRecord,
    StoredUserInput,
)
from ..persistence.contracts import ExecutionRepositories, SessionRepository, TaskAdmissionRepository, ExecutionTerminalPublication

if TYPE_CHECKING:
    from ..persistence.state import StateTransaction
    from ..execution import ExecutionRecoveryEffect, ResolveToolEffectRequest, ToolEffectResolutionResult
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
    expected_session: "SessionRecord | None" = None
    request_identity_version: int = 1


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
    def preflight_handoff(self) -> None: ...

    async def validate_preload_skill_inputs(
        self, binding: AgentBinding, inputs: tuple[Mapping[str, JsonValue], ...],
    ) -> None: ...

    async def load_repository_instructions(
        self, reference: "RuntimePayloadRef | None",
    ) -> RepositoryInstructions | None: ...

    async def load_input_repository_instructions(
        self, reference: RuntimePayloadRef,
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
        self.prepare_reserved_start: Callable[
            [ExecutionRequest, ExecutionStartReservation, "SessionRecord | None"],
            Awaitable[ExecutionStartReservationResult | None],
        ] | None = None
        self.pending_start: Callable[
            [ExecutionRequest, ExecutionStartReservation, "SessionRecord | None", TurnCoordination | None],
            Awaitable[str | None],
        ] | None = None
        self.run_turn: Callable[
            [ExecutionStartReservation, Principal, Callable[[TurnCoordination | None], Awaitable[ExecutionHandle]]],
            Awaitable[ExecutionHandle],
        ] | None = None
        self.run_existing_turn: Callable[
            [ExecutionRecord, int | None, Callable[[], Awaitable[None]]], Awaitable[None],
        ] | None = None
        self.resume_subagent: Callable[[ExecutionRecord, int], Awaitable[None]] | None = None
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

    async def read_view(
        self, execution_id: str, *, tenant_id: str, observed: bool = False,
    ) -> tuple[ResourceRef, ExecutionView] | None:
        service = self._service
        pending = await service._runtime_bridge.pending_turns.read_execution(
            execution_id, tenant_id=tenant_id,
        )
        if pending is not None:
            return (ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id, pending.principal_id),
                    _execution_view(pending))
        current = (await service._state.executions.get_view(execution_id, tenant_id=tenant_id)
                   if observed else await self.read_source_view(execution_id, tenant_id=tenant_id))
        if current is not None:
            return current
        pending = await service._runtime_bridge.pending_turns.read_remote_execution(
            execution_id, tenant_id=tenant_id,
        )
        current = await self.read_source_view(execution_id, tenant_id=tenant_id)
        if current is not None:
            return current
        if pending is not None:
            return (ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id, pending.principal_id),
                    _execution_view(pending))
        return None

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
            header = await service.load_source_header(execution_id, tenant_id=tenant_id)
            if header is None and record is not None:
                header = ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id, record.principal_id)
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
        restore_binding: Callable[[AgentBindingContract], Awaitable[AgentBinding]] | None = None,
        native_step_source: Callable[[AgentBinding], Awaitable[NativeStepSource | None]] | None = None,
        runtime_bridge: _ExecutionRuntimeBridge,
        live_broker: _LiveExecutionStreamBroker,
        operation_ids: "Callable[[], str] | None" = None,
        history_reader: ExecutionHistoryReader,
        history_service: "ExecutionHistoryService | None" = None,
        release_terminal: _ExecutionReleaseCallback | None = None,
        instruction_resolver: "RepositoryInstructionResolver | None" = None,
        object_key_factory: "ObjectKeyFactory | None" = None,
        input_materializer: "ExecutionInputMaterializer | None" = None,
        task_admissions: TaskAdmissionRepository | None = None,
        object_reader: "Callable[[PersistenceDomain, ObjectRef], Awaitable[bytes]] | None" = None,
        context_archive: ExecutionContextArchive | None = None,
        retained_turn_confirmed: Callable[[ExecutionRecord], bool] | None = None,
        retained_publication: Callable[..., ExecutionTerminalPublication | None] | None = None,
    ) -> None:
        self._state = state
        self._retained_publication = retained_publication
        self._object_store = object_store
        self._authorization = authorization
        self._object_key_factory = object_key_factory
        self._object_reader = object_reader
        self._sessions = sessions
        self._task_admissions = task_admissions
        self._operation_ids = operation_ids or (lambda: uuid.uuid4().hex)
        super().__init__(
            state, authorization, port=_ExecutionManagementAdapter(self), sessions=sessions,
            task_admissions=task_admissions, operation_ids=operation_ids,
            release_terminal=release_terminal,
            retained_turn_confirmed=retained_turn_confirmed,
        )
        self._context_archive = context_archive
        self._catalog = catalog
        self._compiler = compiler
        self._restore_binding = restore_binding
        self._native_step_source = native_step_source
        self._runtime_bridge = runtime_bridge
        self._live_broker = live_broker
        self._history_reader = history_reader
        self._history_service = history_service
        self._instruction_resolver = instruction_resolver
        self._input_materializer = input_materializer
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

    def _capture_repository_instructions(
        self, instructions: RepositoryInstructions,
    ) -> RuntimePayloadRef:
        return RuntimePayloadRef(
            StoredPayload.inline_json(instructions.to_payload()), PersistenceDomain.EXECUTION,
        )

    def _validate_tool_response_mode(
        self, reference: AssetVersionRef | None, *, session_id: str | None, fixture: ToolResponseFixture | None,
    ) -> None:
        if reference is not None:
            if fixture is None:
                raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING,
                              safe_details={"kind": "tool_response_fixture_reader"})
            if session_id is not None:
                raise AIError(ErrorCode.BINDING_CONFLICT,
                              safe_details={"reason": "tool_response_fixture_session_unsupported"})
        elif fixture is not None:
            raise AIError(ErrorCode.BINDING_CONFLICT,
                          "a tool response fixture cannot replace an admitted live execution")

    async def _validate_tool_response_input(
        self, reference: AssetVersionRef | None, *, session_id: str | None, fixture: ToolResponseFixture | None,
    ) -> None:
        self._validate_tool_response_mode(reference, session_id=session_id, fixture=fixture)
        if reference is not None:
            assert fixture is not None
            await _load_tool_response_fixture(fixture, reference)

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

    async def _request_identity(
        self, scope: str, key_digest: str, *, tenant_id: str,
        include_remote: bool = False,
    ) -> tuple[int, IdempotencyRecord | None]:
        identity = await self._state.idempotency.get(scope, key_digest, tenant_id=tenant_id)
        if identity is not None:
            return identity.request_identity_version, identity
        pending = self._runtime_bridge.pending_turns
        local = pending.peek_request(tenant_id=tenant_id, scope=scope, idempotency_key_digest=key_digest)
        if local is not None:
            return local.reservation.idempotency.request_identity_version, None
        coordinator = pending.coordination
        if (include_remote and coordinator is not None and coordinator.supports_pending_views
                and coordinator.supports_request_lookup):
            version = await coordinator.read_request_identity_version(
                f"{scope}:{key_digest}", tenant_id=tenant_id,
            )
            if version is not None:
                return version, None
            # A live owner may have committed and released between source and owner reads.
            identity = await self._state.idempotency.get(scope, key_digest, tenant_id=tenant_id)
            if identity is not None:
                return identity.request_identity_version, identity
        return 2, None

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
                stored = StoredUserInput(
                    "user-content-v1",
                    StoredPayload.inline_json(encode_user_content(canonical)),
                )
        else:
            canonical = await self._input_materializer.materialize(
                request.user_prompt,
                context.canonical_files,
            )
            validate_model_inputs(binding.compiled_agent.model, canonical)
            stored = self._input_materializer.capture(canonical)
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
            identity = await self._management.refresh_terminal_identity(
                identity, record, tenant_id=principal.tenant_id,
            )
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
            return replace(request, tool_response_ref=execution.tool_response_ref, tool_effects=execution.tool_effects, session_context=execution.session_context)
        if self._input_materializer is not None:
            prompt = await self._input_materializer.restore(stored)
        elif stored.codec == "text":
            prompt = stored.payload.decode()
        else:
            payload = stored.payload.decode()
            if not isinstance(payload, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            prompt = decode_user_content_payload(payload)
        return replace(request, user_prompt=prompt, files=(), tool_response_ref=execution.tool_response_ref,
                       tool_effects=execution.tool_effects, session_context=execution.session_context)

    async def _binding(
        self,
        binding_digest: str,
        binding_contract: "AgentBindingContract | None" = None,
        resolved_binding: AgentBinding | None = None,
    ) -> AgentBinding:
        if binding_contract is None:
            return self._catalog.binding(binding_digest)
        if binding_contract.binding_digest != binding_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        binding = resolved_binding if resolved_binding is not None else (
            self._compiler.restore(binding_contract) if self._restore_binding is None
            else await self._restore_binding(binding_contract)
        )
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
            or execution.tool_effects != request.tool_effects
            or execution.session_context != request.session_context
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
        if budget_scope_id is not None and (
            not isinstance(budget_scope_id, str) or not budget_scope_id.strip()
            or request.budget is not None
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not isinstance(requires_task_invocation_capture, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if re.fullmatch(r"[0-9a-f]{64}", binding_digest) is None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        binding = await self._binding(binding_digest, binding_contract)
        fixture = _runtime_feature_policy(binding.compiled_agent.runtime_contributions).tool_responses
        if request.tool_response_ref is None and fixture is not None:
            request = replace(request, tool_response_ref=fixture.ref)
        context = await self._canonicalize_request(request)
        request = context.request
        if budget_scope_id is not None:
            await self.authorize_budget_scope(budget_scope_id, request.principal)
        scope = "execution.run"
        idempotency_key_digest = compute_idempotency_key_digest(request.idempotency_key)
        identity_version, existing = await self._request_identity(
            scope, idempotency_key_digest, tenant_id=request.principal.tenant_id,
        )
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
            request_identity_version=identity_version,
            requires_task_invocation_capture=requires_task_invocation_capture,
            budget_scope_id=budget_scope_id,
        )
        try:
            admitted = await self._management.resolve_admission(
                scope, idempotency_key_digest, request_digest, request.principal, identity=existing,
            )
        except AIError as error:
            if error.code is not ErrorCode.IDEMPOTENCY_CONFLICT or existing is not None:
                raise
            latest_version, existing = await self._request_identity(
                scope, idempotency_key_digest, tenant_id=request.principal.tenant_id,
            )
            if latest_version == identity_version:
                raise
            request_digest = _request_digest(
                request, binding_digest, session_id=None, previous_execution_id=None,
                fork_base_execution_id=None, parent_execution_id=None, root_execution_id=None,
                parent_invocation_id=None, lineage_kind=ExecutionLineageKind.RUN,
                request_intent_digest=context.request_intent_digest,
                request_identity_version=latest_version,
                requires_task_invocation_capture=requires_task_invocation_capture,
                budget_scope_id=budget_scope_id,
            )
            admitted = await self._management.resolve_admission(
                scope, idempotency_key_digest, request_digest, request.principal, identity=existing,
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
        session: "SessionRecord | None" = None,
        binding_contract: "AgentBindingContract | None" = None,
        resolved_binding: AgentBinding | None = None,
        dependency_hold_id: "str | None" = None,
        requires_task_invocation_capture: bool = False,
        budget_scope_id: str | None = None,
    ) -> ExecutionHandle:
        if not session_id.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return await self._start(
            binding_digest,
            request,
            session_id=session_id,
            session_agent_id=agent_id,
            session=session,
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
        parent_producer_generation: int,
        root_execution_id: str,
        parent_invocation_id: str,
        binding_contract: "AgentBindingContract | None" = None,
        resolved_binding: AgentBinding | None = None,
    ) -> ExecutionHandle:
        return await self._start(
            binding_digest,
            request,
            parent_execution_id=parent_execution_id,
            parent_producer_generation=parent_producer_generation,
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
        parent_producer_generation: int,
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
        binding = await self._binding(execution.binding_digest, execution.binding)
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
            tool_effects=execution.tool_effects,
        )
        handle = await self.start_subagent(
            execution.binding_digest,
            request,
            parent_execution_id=parent_execution_id,
            parent_producer_generation=parent_producer_generation,
            root_execution_id=root_execution_id,
            parent_invocation_id=parent_invocation_id,
            binding_contract=execution.binding,
        )
        if self._runtime_bridge.resume_subagent is not None:
            await self._runtime_bridge.resume_subagent(execution, parent_producer_generation)
        return handle

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

                # Policy callbacks complete before capturing the parent and children.
                current, children = await repository.get_view_with_children(
                    execution_id, tenant_id=principal.tenant_id,
                )
                if current is None and remote_pending is None:
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                if current is not None and (header.owner_principal_id is None
                        or current[0].owner_principal_id != header.owner_principal_id):
                    await self._authorization.authorize(
                        principal, AuthorizationAction.EXECUTION_READ, current[0],
                    )
                return children
            await self.manager.load_view_authorized(
                execution_id, principal, AuthorizationAction.EXECUTION_READ
            )
            return await self._state.executions.list_child_views(
                execution_id,
                tenant_id=principal.tenant_id,
            )

    async def _start(
        self,
        binding_digest: str,
        request: ExecutionRequest,
        *,
        session_id: "str | None" = None,
        session_agent_id: "str | None" = None,
        session: "SessionRecord | None" = None,
        previous_execution_id: "str | None" = None,
        fork_base_execution_id: "str | None" = None,
        conversation_agent_run_id: "str | None" = None,
        parent_execution_id: "str | None" = None,
        parent_producer_generation: int | None = None,
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
                    session=session,
                    previous_execution_id=previous_execution_id,
                    fork_base_execution_id=fork_base_execution_id,
                    conversation_agent_run_id=conversation_agent_run_id,
                    parent_execution_id=parent_execution_id,
                    parent_producer_generation=parent_producer_generation,
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
                    session=session,
                    previous_execution_id=previous_execution_id,
                    fork_base_execution_id=fork_base_execution_id,
                    conversation_agent_run_id=conversation_agent_run_id,
                    parent_execution_id=parent_execution_id,
                    parent_producer_generation=parent_producer_generation,
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
        session: "SessionRecord | None" = None,
        previous_execution_id: "str | None" = None,
        fork_base_execution_id: "str | None" = None,
        conversation_agent_run_id: "str | None" = None,
        parent_execution_id: "str | None" = None,
        parent_producer_generation: int | None = None,
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
        prepared_input: _ExecutionStartContext | None = None,
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
        context = (await self._canonicalize_request(request) if prepared_input is None else
                   replace(prepared_input, request=request))
        request = context.request
        binding = await self._binding(binding_digest, binding_contract, resolved_binding)
        policy = _runtime_feature_policy(binding.compiled_agent.runtime_contributions)
        fixture = policy.tool_responses
        if request.tool_response_ref is None and fixture is not None and lineage_kind is ExecutionLineageKind.RUN:
            request = replace(request, tool_response_ref=fixture.ref)
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
            request = replace(request, correlation=parent.correlation, tool_response_ref=parent.tool_response_ref,
                              tool_effects=parent.tool_effects)
            context = replace(context, request=request)
        conversation_agent_run_id = conversation_agent_run_id
        if session_id is not None and previous_execution_id is None:
            settled = await self._runtime_bridge.pending_turns.settle_session(
                session_id, tenant_id=request.principal.tenant_id,
            )
            if session is None or settled:
                current_session = await self._load_session(
                    session_id, tenant_id=request.principal.tenant_id
                )
                if (session is not None and current_session is not None
                        and current_session.owner_principal_id != session.owner_principal_id):
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                session = current_session
            if session is None:
                raise AIError(ErrorCode.SESSION_NOT_FOUND)
            if session.session_id != session_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

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
        if request.session_context != "all" and session_id is None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        self._validate_tool_response_mode(request.tool_response_ref, session_id=session_id, fixture=fixture)
        if budget_scope_id is not None:
            await self.authorize_budget_scope(
                budget_scope_id,
                request.principal,
                source_execution_id=(
                    previous_execution_id or fork_base_execution_id or parent_execution_id
                ),
            )
        idempotency_key_digest = compute_idempotency_key_digest(request.idempotency_key)
        identity_version, existing = await self._request_identity(
            scope, idempotency_key_digest, tenant_id=request.principal.tenant_id, include_remote=prepared_input is not None,
        )
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
            request_identity_version=identity_version,
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
        if existing is None:
            existing = await self._state.idempotency.get(
                scope,
                idempotency_key_digest,
                tenant_id=request.principal.tenant_id,
            )
        if existing is not None:
            if existing.request_identity_version != identity_version:
                identity_version = existing.request_identity_version
                request_digest = _request_digest(
                    request, binding_digest, session_id=session_id,
                    previous_execution_id=previous_execution_id, fork_base_execution_id=fork_base_execution_id,
                    parent_execution_id=parent_execution_id, root_execution_id=root_execution_id,
                    parent_invocation_id=parent_invocation_id, lineage_kind=lineage_kind,
                    request_intent_digest=context.request_intent_digest,
                    request_identity_version=identity_version,
                    requires_task_invocation_capture=requires_task_invocation_capture,
                    budget_scope_id=budget_scope_id,
                )
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
                        parent_producer_generation=parent_producer_generation,
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
                    request_identity_version=existing.request_identity_version,
                    prepare_local_stream=prepare_local_stream,
                    dependency_hold_id=dependency_hold_id,
                    capture_metric=capture_metric,
                    parent_producer_generation=parent_producer_generation,
                    expected_session=session,
                )
                return ExecutionHandle(existing.resource_id)
            started = await self.load_execution(
                existing.resource_id, tenant_id=request.principal.tenant_id
            )
            if started is not None:
                self._validate_replayed_execution(
                    started, binding, request,
                    requires_task_invocation_capture=requires_task_invocation_capture,
                    budget_scope_id=budget_scope_id,
                )
            if existing.status is IdempotencyStatus.STARTED and started is not None and started.status in {
                ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
            }:
                existing = await self._management.refresh_terminal_identity(
                    existing, started, tenant_id=request.principal.tenant_id,
                )
            replay = await self._replay_terminal_start(
                existing, started, scope=scope, dependency_hold_id=dependency_hold_id,
            )
            if replay is not None:
                return replay
            if started is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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
                parent_producer_generation=parent_producer_generation,
            )
            return ExecutionHandle(existing.resource_id)

        if session_id is not None and not policy.session.enabled:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

        await self._validate_tool_response_input(request.tool_response_ref, session_id=session_id, fixture=fixture)
        input_instruction_paths = (
            *context.canonical_files,
            *(() if isinstance(request.user_prompt, str) else (
                item.path for item in request.user_prompt if isinstance(item, WorkspaceFileInput)
            )),
        )
        original_request = request
        if prepared_input is None:
            context = await self._freeze_input(context, binding)
        else:
            context = replace(context, request=replace(
                request, user_prompt=prepared_input.request.user_prompt, files=(),
            ))
        request = context.request

        input_context = None
        repository_instructions = None
        if request.input_context is not None:
            if session_id is not None:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            if self._context_archive is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            port = self._runtime_bridge.port
            if port is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            await port.validate_preload_skill_inputs(binding, request.input_context.preload_skills)
            input_context = RuntimePayloadRef(
                StoredPayload.inline_json(request.input_context.to_payload()), PersistenceDomain.EXECUTION,
            )
            if request.input_context.repository_instructions is not None:
                RepositoryInstructions.from_payload(request.input_context.repository_instructions)
        elif self._instruction_resolver is not None:
            if lineage_kind is ExecutionLineageKind.SUBAGENT:
                if parent is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                repository_instructions = parent.repository_instructions
                if parent.input_context is not None:
                    port = self._runtime_bridge.port
                    if port is None:
                        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                    inherited = await port.load_input_repository_instructions(parent.input_context)
                    repository_instructions = (
                        None if inherited is None else self._capture_repository_instructions(inherited)
                    )
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
                    if resolved_instructions.documents:
                        repository_instructions = self._capture_repository_instructions(
                            resolved_instructions,
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
                if resolved_instructions.documents:
                    repository_instructions = self._capture_repository_instructions(
                        resolved_instructions,
                    )

        native_source = None
        if (policy.restart_checkpoints or parent is not None and parent.restart_checkpoints) and self._native_step_source is not None:
            native_source = await self._native_step_source(binding)
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
            tool_effects=request.tool_effects,
            session_context=request.session_context,
            correlation=request.correlation,
            principal_id=request.principal.principal_id,
            principal_kind=request.principal.kind,
            stored_user_input=context.stored_user_input,
            requires_task_invocation_capture=requires_task_invocation_capture,
            budget_scope_id=budget_scope_id or (None if request.budget is None else "execution:" + execution_id),
            native_step_source=native_source,
            restart_checkpoints=(policy.restart_checkpoints
                                 or parent is not None and parent.restart_checkpoints),
            durable_trace=(policy.durable_trace or requires_task_invocation_capture
                           or parent is not None and parent.durable_trace),
            conversation=policy.execution.retain_history,
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
                    request_identity_version=identity_version,
                ),
                budget=request.budget,
                parent_producer_generation=parent_producer_generation,
        )
        try:
            return await self._admit_execution_start(
                request, proposed_reservation, binding, session,
                prepare_local_stream=prepare_local_stream,
                dependency_hold_id=dependency_hold_id,
                requires_task_invocation_capture=requires_task_invocation_capture,
                budget_scope_id=budget_scope_id,
                capture_metric=capture_metric,
            )
        except AIError as error:
            if error.code is not ErrorCode.IDEMPOTENCY_CONFLICT or prepared_input is not None:
                raise
            winner_version, _ = await self._request_identity(
                scope, idempotency_key_digest, tenant_id=request.principal.tenant_id, include_remote=True,
            )
            if winner_version == 2:
                raise
            return await self._start_unlocked(
                binding_digest, original_request, session_id=session_id, session_agent_id=session_agent_id,
                session=session, previous_execution_id=previous_execution_id,
                fork_base_execution_id=fork_base_execution_id,
                conversation_agent_run_id=conversation_agent_run_id,
                parent_execution_id=parent_execution_id, parent_producer_generation=parent_producer_generation,
                root_execution_id=root_execution_id, parent_invocation_id=parent_invocation_id,
                lineage_kind=lineage_kind, scope=scope, prepare_local_stream=prepare_local_stream,
                dependency_hold_id=dependency_hold_id, binding_contract=binding_contract,
                resolved_binding=resolved_binding, requires_task_invocation_capture=requires_task_invocation_capture,
                budget_scope_id=budget_scope_id, capture_metric=capture_metric, prepared_input=context,
            )

    async def _replay_terminal_start(
        self,
        identity: IdempotencyRecord,
        execution: ExecutionRecord | None,
        *,
        scope: str,
        dependency_hold_id: str | None,
    ) -> ExecutionHandle | None:
        if identity.status in {IdempotencyStatus.FAILED, IdempotencyStatus.CANCELLED}:
            if scope != "session.resume" or execution is None or execution.started_at is None:
                fallback = (
                    ErrorCode.EXECUTION_START_PERSISTENCE_FAILED
                    if identity.status is IdempotencyStatus.FAILED
                    else ErrorCode.EXECUTION_CANCELLED
                )
                raise _stable_idempotency_error(identity.error_code, fallback)
        elif identity.status is not IdempotencyStatus.COMPLETED:
            return None
        if execution is None or not _terminal_idempotency_matches(identity, execution):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._acquire_start_dependency_hold(execution, dependency_hold_id)
        return ExecutionHandle(identity.resource_id)

    async def _admit_execution_start(
        self,
        request: ExecutionRequest,
        proposed_reservation: ExecutionStartReservation,
        binding: AgentBinding,
        session: "SessionRecord | None",
        *,
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
        policy = _runtime_feature_policy(binding.compiled_agent.runtime_contributions)

        async def admit(coordination: TurnCoordination | None) -> ExecutionHandle:
            if not self._accepting:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            if coordination is not None and coordination.lease is None:
                if coordinator is None:
                    raise RuntimeError("duplicate turn admission requires coordination")
                return await self._join_coordinated_start(
                    coordinator, coordination, request, binding,
                    requires_task_invocation_capture=requires_task_invocation_capture,
                    budget_scope_id=budget_scope_id,
                )
            if proposed_reservation.idempotency.request_identity_version != 2:
                identity = proposed_reservation.idempotency
                previous = await self._state.idempotency.get(
                    identity.scope, identity.idempotency_key_digest, tenant_id=request.principal.tenant_id,
                )
                if previous is None:
                    # The observed legacy owner disappeared; it cannot authorize a new legacy admission.
                    raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
            if (policy.execution.commit.mode == "terminal" and execution.conversation
                    and dependency_hold_id is None
                    and self._runtime_bridge.pending_start is not None):
                pending_id = await ConversationService.prepare_turn(
                    request, proposed_reservation, session, coordination,
                    start_pending=self._runtime_bridge.pending_start,
                    dependency_hold_id=dependency_hold_id,
                )
                if pending_id is not None:
                    return ExecutionHandle(pending_id)
            return await self._reserve_and_launch_start(
                request, proposed_reservation, binding, session,
                prepare_local_stream=prepare_local_stream,
                dependency_hold_id=dependency_hold_id,
                requires_task_invocation_capture=requires_task_invocation_capture,
                budget_scope_id=budget_scope_id,
                capture_metric=capture_metric,
            )

        run_turn = self._runtime_bridge.run_turn
        if run_turn is None:
            if coordinator is not None:
                raise RuntimeError("coordinated admission requires a turn lifecycle owner")
            return await admit(None)
        task = asyncio.create_task(
            run_turn(proposed_reservation, request.principal, admit),
            name=f"execution-admission-{execution.execution_id}",
        )
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
        session: "SessionRecord | None",
        *,
        prepare_local_stream: bool,
        dependency_hold_id: str | None,
        requires_task_invocation_capture: bool,
        budget_scope_id: str | None,
        capture_metric: Callable[[_TerminalMetricDelivery], None] | None,
    ) -> ExecutionHandle:
        proposed_reservation = replace(proposed_reservation, execution=await _materialize_execution_payloads(
            proposed_reservation.execution, input_materializer=self._input_materializer,
            tenant_id=request.principal.tenant_id,
        ))
        prepared_context = None
        captured_context = proposed_reservation.execution.input_context
        if captured_context is not None:
            if self._context_archive is None or captured_context.payload.kind != "inline":
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            prepared_context = await self._context_archive.prepare(
                ExecutionInputContext.from_payload(captured_context.payload.decode()),
            )
            proposed_reservation = replace(proposed_reservation, execution=replace(
                proposed_reservation.execution, input_context=prepared_context.reference,
            ))
        identity = proposed_reservation.idempotency
        scope = identity.scope
        idempotency_key_digest = identity.idempotency_key_digest
        request_digest = identity.request_digest
        reservation = None
        if prepared_context is None and self._runtime_bridge.prepare_reserved_start is not None:
            reservation = await self._runtime_bridge.prepare_reserved_start(request, proposed_reservation, session)
        if reservation is not None:
            pass
        elif prepared_context is None:
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
            replay = await self._replay_terminal_start(
                reservation.idempotency, reservation.execution,
                scope=scope, dependency_hold_id=dependency_hold_id,
            )
            if replay is not None:
                return replay
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
                    request_identity_version=reservation.idempotency.request_identity_version,
                    prepare_local_stream=prepare_local_stream,
                    dependency_hold_id=dependency_hold_id,
                    parent_producer_generation=proposed_reservation.parent_producer_generation,
                    capture_metric=capture_metric,
                    expected_session=session,
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
                    parent_producer_generation=proposed_reservation.parent_producer_generation,
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
        if reservation.execution.status is ExecutionStatus.STARTED:
            await self._launch_started(
                request, reservation.execution, scope=scope,
                idempotency_key_digest=idempotency_key_digest,
                prepare_local_stream=prepare_local_stream,
                dependency_hold_id=dependency_hold_id,
                parent_producer_generation=proposed_reservation.parent_producer_generation,
                confirmed_start=True,
            )
            return ExecutionHandle(execution_id)
        if not reservation.created:
            request = await self._request_for_execution(request, reservation.execution)
        await self._prepare_and_launch(
            request,
            reservation.execution,
            scope=scope,
            idempotency_key_digest=idempotency_key_digest,
            request_digest=request_digest,
            request_identity_version=reservation.idempotency.request_identity_version,
            prepare_local_stream=prepare_local_stream,
            dependency_hold_id=dependency_hold_id,
            capture_metric=capture_metric,
            parent_producer_generation=proposed_reservation.parent_producer_generation,
            expected_session=session,
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
        request_identity_version: int = 1,
        prepare_local_stream: bool = False,
        dependency_hold_id: "str | None" = None,
        capture_metric: Callable[[_TerminalMetricDelivery], None] | None = None,
        parent_producer_generation: int | None = None,
        expected_session: "SessionRecord | None" = None,
    ) -> None:
        async def prepare() -> None:
            if self._backend is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            launch_request = request
            if execution.budget_scope_id is not None:
                usage = await self._state.budgets.read(execution.budget_scope_id)
                launch_request = replace(request, budget=usage.limits)
            identity = ExecutionStartIdentity(
                scope, idempotency_key_digest, request_digest,
                expected_session, request_identity_version,
            )
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
                launch_request,
                started,
                scope=scope,
                idempotency_key_digest=idempotency_key_digest,
                prepare_local_stream=prepare_local_stream,
                dependency_hold_id=dependency_hold_id,
                parent_producer_generation=parent_producer_generation,
                confirmed_start=True,
            )

        if self._runtime_bridge.run_existing_turn is None:
            if (self._runtime_bridge.pending_turns.coordination is not None
                    and self._runtime_bridge.run_turn is None):
                raise RuntimeError("coordinated admission requires a turn lifecycle owner")
            await prepare()
        else:
            await self._runtime_bridge.run_existing_turn(execution, parent_producer_generation, prepare)

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
        parent_producer_generation: int | None = None,
        confirmed_start: bool = False,
    ) -> None:
        async def launch() -> None:
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
                    # The hold is now confirmed in the source. If the start
                    # capture predates it, refresh before claiming that revision.
                    confirmed_start=(confirmed_start and (
                        not hold_acquired
                        or dependency_hold_id in execution.dependency_hold_ids
                    )),
                )
            except BaseException:
                if hold_acquired:
                    await self.release_dependency_hold(
                        execution.execution_id,
                        tenant_id=self._state.executions.tenant_id,
                        hold_id=dependency_hold_id or "",
                    )
                raise

        if execution.status is ExecutionStatus.STARTED and execution.agent_run_seq == 0:
            if self._runtime_bridge.run_existing_turn is not None:
                await self._runtime_bridge.run_existing_turn(execution, parent_producer_generation, launch)
                return
            if (self._runtime_bridge.pending_turns.coordination is not None
                    and self._runtime_bridge.run_turn is None):
                raise RuntimeError("coordinated admission requires a turn lifecycle owner")
        await launch()

    async def _launch_started_impl(
        self,
        request: ExecutionRequest,
        execution: ExecutionRecord,
        *,
        scope: str,
        idempotency_key_digest: str,
        prepare_local_stream: bool = False,
        confirmed_start: bool = False,
    ) -> None:
        if self._backend is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        launch_record = execution if confirmed_start else await self.load_execution(
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
        if launch_record.budget_scope_id is not None and (not confirmed_start or request.budget is None):
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
        header = await self.authorize_execution(execution_id, principal, AuthorizationAction.EXECUTION_READ)
        execution = await self.manager.load_view_authorized(
            execution_id, principal, AuthorizationAction.EXECUTION_READ, authorized_header=header, observed=True,
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
        return execution

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

        async def load_confirmed_terminal() -> ExecutionRecord:
            publication = (None if self._retained_publication is None else
                self._retained_publication(execution_id, tenant_id=principal.tenant_id))
            if publication is None:
                return await self.load_authorized(execution_id, principal, AuthorizationAction.EXECUTION_READ)
            await self.authorize_loaded_execution(
                publication.execution, principal, AuthorizationAction.EXECUTION_READ, owner_principal_id=None,
            )
            return publication.execution

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
                        execution = await load_confirmed_terminal()
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
                execution = await load_confirmed_terminal()
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
        await self._binding(binding_digest, previous.binding)
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
            tool_effects=previous.tool_effects,
            session_context=previous.session_context,
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
        await self._binding(binding_digest, previous.binding)
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
            tool_effects=previous.tool_effects,
            session_context=previous.session_context,
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

    def preflight_handoff(self) -> None:
        if any(not task.done() for task in self._admission_tasks):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, retryable=True,
                safe_details={"phase": "runtime_handoff", "reason": "admission_in_progress"})
        backend = self._runtime_bridge.port
        if backend is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        backend.preflight_handoff()
        self._accepting = False

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
                ExecutionStatus.WAITING_DEFERRED,
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
        # Remote publication can outlive the source commit observed during this await.
        record = await self.load_source_execution(execution_id, tenant_id=tenant_id)
        return record if record is not None else pending


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
            pending = await self._runtime_bridge.pending_turns.read_remote_execution(
                execution_id, tenant_id=principal.tenant_id,
            )
            # Ordinary reads may retain a pre-commit miss after the owner releases.
            record = await self.load_source_execution(
                execution_id, tenant_id=principal.tenant_id,
            )
            if record is None:
                record = pending
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
    request_identity_version: int = 2,
    requires_task_invocation_capture: bool = False,
    budget_scope_id: str | None = None,
) -> str:
    if request_intent_digest is None or request_identity_version != 2:
        user_prompt_identity = input_intent(
            validate_user_input(request.user_prompt),
            request.files,
            identity_version=request_identity_version,
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
            **({"tool_effects": request.tool_effects} if request.tool_effects != "live" else {}),
            **({"session_context": request.session_context} if request.session_context != "all" else {}),
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
            "memory_scope": request.memory_scope,
            "mode": request.mode,
            "planning": request.planning,
            "thinking": request.thinking,
        }
    )


__all__ = ["CancelEffectOutcome", "DefaultExecutionService", "ExecutionBackend"]
