#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session lifecycle, authorization, admission reconciliation, and fork semantics."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
from typing import Generic, Protocol, TypeVar, overload

from linktools.core import environ
from ..agent import AgentBinding, AgentBindingContract
from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    CursorSigner,
    ExecutionStatus,
    HandoffGate,
    HandoffState,
    OperationKind,
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationStatus,
    Page,
    Principal,
    PrincipalKind,
    ResourceKind,
    ResourceRef,
    SessionStatus,
    canonical_sha256,
    idempotency_key_digest,
    validate_agent_id,
    validate_page_limit,
    encode_cursor as encode_runtime_cursor,
    decode_cursor as decode_runtime_cursor,
)
from ..errors import AIError, ErrorCode
from ._contracts import (
    SessionRecord,
    SessionRepository,
    ConversationCursor,
    CreateSessionRequest,
    ListSessionRequest,
    ResumeSessionRequest,
    ForkSessionRequest,
    UpdateSessionRequest,
    CloseSessionRequest,
    SessionView,
)

HandleT = TypeVar("HandleT")
HandleT_co = TypeVar("HandleT_co", covariant=True)
_logger = environ.get_logger("ai.session")
_SESSION_CURSOR_KIND = "SESSION"
_VIEW_ACTIVE_UNSET = object()


class SessionExecutionState(Protocol):
    @property
    def execution_id(self) -> str: ...
    @property
    def session_id(self) -> str | None: ...
    @property
    def parent_execution_id(self) -> str | None: ...
    @property
    def status(self) -> ExecutionStatus: ...
    @property
    def invalidates_session_history(self) -> bool: ...
    @property
    def conversation(self) -> bool: ...


class SessionExecutionReader(Protocol):
    async def get(
        self, execution_id: str, *, tenant_id: str
    ) -> SessionExecutionState | None: ...
    async def get_many(
        self, execution_ids: tuple[str, ...], *, tenant_id: str
    ) -> Mapping[str, SessionExecutionState]: ...


class SessionExecutionService(Protocol[HandleT_co]):
    async def start_for_session(
        self,
        agent_id: str,
        binding_digest: str,
        session_id: str,
        request: ResumeSessionRequest,
        *,
        binding_contract: AgentBindingContract | None = None,
        resolved_binding: AgentBinding | None = None,
        dependency_hold_id: str | None = None,
        requires_task_invocation_capture: bool = False,
        budget_scope_id: str | None = None,
    ) -> HandleT_co: ...
    async def cancel(
        self, execution_id: str, *, principal: Principal, idempotency_key: str
    ) -> None: ...
    async def wait(self, execution_id: str, *, principal: Principal) -> None: ...


class SessionOperations(Protocol):
    async def append(self, record: OperationLedgerInput) -> OperationLedgerRecord: ...
    async def get(
        self, operation_id: str, *, tenant_id: str
    ) -> OperationLedgerRecord | None: ...
    async def compare_and_swap(
        self,
        operation_id: str,
        *,
        tenant_id: str,
        expected_status: OperationStatus,
        next_record: OperationLedgerRecord,
    ) -> OperationLedgerRecord: ...


class SessionRepositories(Protocol):
    @property
    def sessions(self) -> SessionRepository: ...
    @property
    def operations(self) -> SessionOperations: ...


ExecutionStateT = TypeVar("ExecutionStateT", bound=SessionExecutionState, covariant=True)
AdmissionT = TypeVar("AdmissionT", bound=SessionExecutionState)


class SessionPendingExecutionLookup(Protocol[ExecutionStateT]):
    async def __call__(self, session_id: str, *, tenant_id: str) -> ExecutionStateT | None: ...


class SessionRecordReader(Protocol):
    async def __call__(self, session_id: str, *, tenant_id: str) -> SessionRecord | None: ...


class SessionViewObserver(Protocol):
    async def __call__(
        self, record: SessionRecord, principal: Principal, *, active: str | None,
    ) -> SessionView: ...


class SessionReleaseCallback(Protocol):
    async def __call__(
        self,
        session_id: str,
        *,
        tenant_id: str,
        continuation: ConversationCursor | None,
    ) -> None: ...


async def _no_release_terminal(
    session_id: str, *, tenant_id: str, continuation: ConversationCursor | None
) -> None:
    del session_id, tenant_id, continuation


class SessionManager(Generic[HandleT]):
    """Own session lifecycle; specialized adapters supply cross-domain presentation."""

    def __init__(
        self,
        conversation: SessionRepositories,
        executions: SessionExecutionReader,
        authorization: AuthorizationPolicy,
        execution: SessionExecutionService[HandleT],
        cursor_signer: CursorSigner,
        *,
        enabled: bool = True,
        release_terminal: SessionReleaseCallback | None = None,
        canonicalize_path: Callable[[str], Awaitable[str]] | None = None,
        pending_session_execution: SessionPendingExecutionLookup[SessionExecutionState] | None = None,
        read_session: SessionRecordReader | None = None,
        observe_view: SessionViewObserver | None = None,
    ) -> None:
        self._enabled = enabled
        self._conversation = conversation
        self._executions = executions
        self._authorization = authorization
        self._execution = execution
        self._cursor_signer = cursor_signer
        self._release_terminal = release_terminal or _no_release_terminal
        self._canonicalize_path = canonicalize_path
        self._pending_session_execution = pending_session_execution
        self._read_session = read_session or conversation.sessions.get
        self._observe_view = observe_view
        self._display_reads: set[asyncio.Future[None]] = set()
        self._display_close_started = False
        self._handoff = HandoffGate[tuple[str, str], ConversationCursor]()

    async def _observed_view(
        self, record: SessionRecord, principal: Principal, *, active: str | None
    ) -> SessionView:
        if self._observe_view is not None:
            return await self._observe_view(record, principal, active=active)
        return await self.project_view(record, principal, active=active)

    def require_available(self) -> None:
        if not self._enabled:
            raise AIError(
                ErrorCode.CAPABILITY_REQUIRED_MISSING, "Session management is disabled"
            )

    async def create(self, agent_id: str, request: CreateSessionRequest) -> SessionView:
        self.require_available()
        try:
            agent_id = validate_agent_id(agent_id)
        except AIError:
            raise
        except TypeError as error:
            raise AIError(ErrorCode.AGENT_ID_INVALID) from error
        if any(key.startswith("linktools.ai.") for key in request.metadata):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        cwd = await self._canonicalize_cwd(request.cwd)
        resource = ResourceRef(
            ResourceKind.SESSION,
            request.session_id,
            request.principal.tenant_id,
            request.principal.principal_id,
        )
        await self._authorization.authorize(
            request.principal, AuthorizationAction.SESSION_CREATE, resource
        )
        digest = canonical_sha256(
            {
                "action": "session.create",
                "tenant_id": request.principal.tenant_id,
                "principal_id": request.principal.principal_id,
                "session_id": request.session_id,
                "agent_id": agent_id,
                "cwd": cwd,
                "metadata": dict(request.metadata),
            }
        )
        now = datetime.now(timezone.utc)
        record = SessionRecord(
            session_id=request.session_id,
            owner_principal_id=request.principal.principal_id,
            status=SessionStatus.OPEN,
            revision=0,
            cwd=cwd,
            metadata=dict(request.metadata),
            created_at=now,
            updated_at=now,
            closed_at=None,
            active_execution_id=None,
            continuation=None,
            agent_id=agent_id,
        )
        operation = self._session_terminal_operation(
            request.idempotency_key,
            request.principal.tenant_id,
            request.session_id,
            OperationKind.SESSION_CREATE,
            digest,
            record.revision,
        )
        record, _ = await self._conversation.sessions.create_with_operation(
            record,
            operation=operation,
        )
        _logger.debug(
            "session created: session=%s tenant=%s",
            record.session_id,
            request.principal.tenant_id,
        )
        return await self.project_view(record, request.principal)

    async def get(self, session_id: str, *, principal: Principal) -> SessionView:
        async with self.consume_session(session_id, principal.tenant_id):
            record = await self.read_authorized(
                session_id, principal, AuthorizationAction.SESSION_READ
            )
            return await self._observed_view(
                record,
                principal,
                active=record.active_execution_id,
            )

    @staticmethod
    def _check_display_authority(record: SessionRecord, header: ResourceRef) -> None:
        # An authorization callback may have deleted/recreated the resource.
        if (
            header.kind is not ResourceKind.SESSION
            or record.session_id != header.id
            or record.owner_principal_id != header.owner_principal_id
        ):
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)

    async def list(self, request: ListSessionRequest) -> Page[SessionView]:
        self.require_available()
        await self._authorization.authorize(
            request.principal,
            AuthorizationAction.SESSION_READ,
            ResourceRef(ResourceKind.SESSION, "list", request.principal.tenant_id),
        )
        limit = validate_page_limit(request.limit)
        cursor, snapshot = _decode_session_cursor(
            request.cursor,
            request.principal.tenant_id,
            request.principal.principal_id,
            self._cursor_signer,
        )
        snapshot, page = await self._conversation.sessions.list_page(
            tenant_id=request.principal.tenant_id,
            owner_principal_id=request.principal.principal_id,
            cursor=cursor,
            limit=limit,
            snapshot=snapshot,
        )
        values = page.items
        active_ids = tuple(
            dict.fromkeys(
                record.active_execution_id
                for record in values
                if record.active_execution_id is not None
            )
        )
        active_by_id = (
            {}
            if not active_ids
            else await self._executions.get_many(
                active_ids,
                tenant_id=request.principal.tenant_id,
            )
        )
        views = tuple(
            await asyncio.gather(
                *(
                    self._observed_view(
                        record,
                        request.principal,
                        active=self._active_execution_id(
                            record,
                            None
                            if record.active_execution_id is None
                            else active_by_id.get(record.active_execution_id),
                        ),
                    )
                    for record in values
                )
            )
        )
        next_cursor = _make_cursor(
            snapshot,
            request.principal.tenant_id,
            request.principal.principal_id,
            page.next_cursor,
            self._cursor_signer,
        )
        return Page(views, next_cursor)

    async def reconcile(self, session_id: str, *, principal: Principal) -> SessionView:
        async with self.consume_session(session_id, principal.tenant_id):
            record = await self.read_authorized(
                session_id, principal, AuthorizationAction.SESSION_READ
            )
            record = await self.reconcile_admission(record)
            return await self.project_view(record, principal)

    async def resume(
        self,
        agent_id: str,
        binding_digest: str,
        session_id: str,
        request: ResumeSessionRequest,
        *,
        binding_contract: "AgentBindingContract | None" = None,
        resolved_binding: "AgentBinding | None" = None,
        dependency_hold_id: "str | None" = None,
        requires_task_invocation_capture: bool = False,
        budget_scope_id: str | None = None,
    ) -> HandleT:
        return await self._resume(
            agent_id,
            binding_digest,
            session_id,
            request,
            binding_contract=binding_contract, resolved_binding=resolved_binding,
            dependency_hold_id=dependency_hold_id,
            requires_task_invocation_capture=requires_task_invocation_capture,
            budget_scope_id=budget_scope_id,
        )

    async def _resume(
        self,
        agent_id: str,
        binding_digest: str,
        session_id: str,
        request: ResumeSessionRequest,
        *,
        binding_contract: "AgentBindingContract | None" = None,
        resolved_binding: "AgentBinding | None" = None,
        dependency_hold_id: "str | None" = None,
        requires_task_invocation_capture: bool = False,
        budget_scope_id: str | None = None,
    ) -> HandleT:
        async with self.consume_session(session_id, request.principal.tenant_id):
            record = await self.read_authorized(
                session_id, request.principal, AuthorizationAction.SESSION_READ
            )
            record = await self.reconcile_admission(record)
            if record.cwd is not None and self._canonicalize_path is None:
                raise AIError(
                    ErrorCode.REQUEST_FIELD_INVALID,
                    safe_details={"field": "cwd", "reason": "workspace_required"},
                )
            await self._authorization.authorize(
                request.principal,
                AuthorizationAction.EXECUTION_RUN,
                ResourceRef(
                    ResourceKind.EXECUTION,
                    session_id,
                    request.principal.tenant_id,
                ),
            )
            if record.agent_id != agent_id:
                raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)
            return await self._execution.start_for_session(
                agent_id,
                binding_digest,
                session_id,
                request,
                binding_contract=binding_contract, resolved_binding=resolved_binding,
                dependency_hold_id=dependency_hold_id,
                requires_task_invocation_capture=requires_task_invocation_capture,
                budget_scope_id=budget_scope_id,
            )

    async def fork(
        self, agent_id: str, session_id: str, request: ForkSessionRequest
    ) -> SessionView:
        async with self.consume_session(session_id, request.principal.tenant_id):
            source = await self.read_authorized(
                session_id, request.principal, AuthorizationAction.SESSION_READ
            )
            source = await self.reconcile_admission(source)
            if source.history_quality == "unavailable":
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            await self._authorization.authorize(
                request.principal,
                AuthorizationAction.SESSION_CREATE,
                ResourceRef(
                    ResourceKind.SESSION,
                    request.new_session_id,
                    request.principal.tenant_id,
                    request.principal.principal_id,
                ),
            )
            if source.agent_id != agent_id:
                raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)
            target_cwd = (
                source.cwd
                if request.cwd is None
                else await self._canonicalize_cwd(request.cwd)
            )
            if target_cwd is not None and self._canonicalize_path is None:
                raise AIError(
                    ErrorCode.REQUEST_FIELD_INVALID,
                    safe_details={"field": "cwd", "reason": "workspace_required"},
                )
            digest = canonical_sha256(
                {
                    "action": "session.fork",
                    "tenant_id": request.principal.tenant_id,
                    "principal_id": request.principal.principal_id,
                    "source": session_id,
                    "target": request.new_session_id,
                    "agent_id": source.agent_id,
                    "cwd": target_cwd,
                }
            )
            now = datetime.now(timezone.utc)
            target_metadata = dict(source.metadata)
            target = SessionRecord(
                session_id=request.new_session_id,
                owner_principal_id=source.owner_principal_id,
                status=SessionStatus.OPEN,
                revision=0,
                cwd=target_cwd,
                metadata=target_metadata,
                created_at=now,
                updated_at=now,
                closed_at=None,
                active_execution_id=None,
                continuation=source.continuation,
                agent_id=source.agent_id,
            )
            operation = self._session_terminal_operation(
                request.idempotency_key,
                request.principal.tenant_id,
                request.new_session_id,
                OperationKind.SESSION_FORK,
                digest,
                target.revision,
            )
            if source.history_id is None:
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            target, _ = await self._conversation.sessions.create_fork_with_operation(
                session_id,
                target,
                expected_source_revision=source.revision,
                operation=operation,
            )
            _logger.debug(
                "session forked: source=%s target=%s", session_id, target.session_id
            )
            return await self.project_view(target, request.principal)

    async def update(
        self, agent_id: str, session_id: str, request: UpdateSessionRequest
    ) -> SessionView:
        async with self.consume_session(session_id, request.principal.tenant_id):
            return await self._update(agent_id, session_id, request)

    async def _update(
        self, agent_id: str, session_id: str, request: UpdateSessionRequest
    ) -> SessionView:
        current = await self.read_authorized(
            session_id, request.principal, AuthorizationAction.SESSION_UPDATE
        )
        if current.agent_id != agent_id:
            raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)
        if any(key.startswith("linktools.ai.") for key in request.metadata):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        requested_cwd = (
            current.cwd
            if request.cwd is None
            else await self._canonicalize_cwd(request.cwd)
        )
        digest = canonical_sha256(
            {
                "action": "session.update",
                "tenant_id": request.principal.tenant_id,
                "principal_id": request.principal.principal_id,
                "session_id": session_id,
                "expected_revision": request.expected_revision,
                "metadata": request.metadata,
                "cwd": requested_cwd,
            }
        )
        now = datetime.now(timezone.utc)
        next_record = replace(
            current,
            revision=current.revision + 1,
            cwd=requested_cwd,
            metadata=dict(request.metadata),
            updated_at=now,
        )
        operation = self._session_terminal_operation(
            request.idempotency_key,
            request.principal.tenant_id,
            session_id,
            OperationKind.SESSION_UPDATE,
            digest,
            request.expected_revision + 1,
        )
        updated, _ = await self._conversation.sessions.compare_and_swap_with_operation(
            session_id,
            tenant_id=request.principal.tenant_id,
            expected_revision=request.expected_revision,
            next_record=next_record,
            operation=operation,
        )
        _logger.debug(
            "session updated: session=%s revision=%s", session_id, updated.revision
        )
        return await self.project_view(updated, request.principal)

    async def _canonicalize_cwd(self, value: str | None) -> str | None:
        if value is None:
            return None
        if self._canonicalize_path is None:
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"field": "cwd", "reason": "workspace_required"},
            )
        try:
            return await self._canonicalize_path(value)
        except AIError:
            raise
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error

    async def close(self, session_id: str, request: CloseSessionRequest) -> SessionView:
        async with self.consume_session(session_id, request.principal.tenant_id):
            return await self._close(session_id, request)

    async def _close(
        self, session_id: str, request: CloseSessionRequest
    ) -> SessionView:
        await self.read_authorized(
            session_id, request.principal, AuthorizationAction.SESSION_CLOSE
        )
        if self._pending_session_execution is not None:
            pending = await self._pending_session_execution(
                session_id, tenant_id=request.principal.tenant_id
            )
            if pending is not None:
                if not request.force:
                    raise AIError(ErrorCode.SESSION_ACTIVE_EXECUTIONS)
                await self._execution.cancel(
                    pending.execution_id,
                    principal=request.principal,
                    idempotency_key=f"{request.idempotency_key}/{pending.execution_id}",
                )
        digest = canonical_sha256(
            {
                "action": "session.close",
                "tenant_id": request.principal.tenant_id,
                "principal_id": request.principal.principal_id,
                "session_id": session_id,
                "force": request.force,
                "wait_timeout_seconds": request.wait_timeout_seconds,
            }
        )
        operation = await self._begin_close_operation(
            request.idempotency_key,
            request.principal.tenant_id,
            session_id,
            digest,
        )
        if operation.status is OperationStatus.SUCCEEDED:
            closed = await self._read_session(
                session_id, tenant_id=request.principal.tenant_id
            )
            if closed is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._validate_close_replay(
                operation,
                closed,
                session_id,
                tenant_id=request.principal.tenant_id,
            )
            view = await self.project_view(closed, request.principal)
            await self._request_session_release(
                session_id,
                request.principal.tenant_id,
                closed.continuation,
            )
            return view
        current = await self._read_session(
            session_id,
            tenant_id=request.principal.tenant_id,
        )
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        current = await self.reconcile_admission(current)
        if current.status is SessionStatus.CLOSED:
            await self._complete_close_operation(
                operation,
                request.principal.tenant_id,
                session_id,
                canonical_sha256(
                    {"session_id": session_id, "revision": current.revision}
                ),
            )
            view = await self.project_view(current, request.principal)
            await self._request_session_release(
                session_id,
                request.principal.tenant_id,
                current.continuation,
            )
            return view
        if current.status is SessionStatus.CLEANUP_REQUIRED and not request.force:
            raise AIError(ErrorCode.SESSION_CLEANUP_REQUIRED)

        if not request.force:
            if current.status is SessionStatus.OPEN:
                try:
                    current = await self._conversation.sessions.transition_status(
                        session_id,
                        tenant_id=request.principal.tenant_id,
                        expected=frozenset({SessionStatus.OPEN}),
                        next_status=SessionStatus.CLOSING,
                        require_no_active=True,
                    )
                except AIError as error:
                    if error.code not in {
                        ErrorCode.SESSION_ACTIVE_EXECUTIONS,
                        ErrorCode.SESSION_CONFLICT,
                    }:
                        raise
                    latest = await self._read_session(
                        session_id,
                        tenant_id=request.principal.tenant_id,
                    )
                    if latest is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if (
                        latest.status is SessionStatus.OPEN
                        and latest.active_execution_id is not None
                    ):
                        raise AIError(ErrorCode.SESSION_ACTIVE_EXECUTIONS)
                    if latest.status is SessionStatus.OPEN:
                        current = await self._conversation.sessions.transition_status(
                            session_id,
                            tenant_id=request.principal.tenant_id,
                            expected=frozenset({SessionStatus.OPEN}),
                            next_status=SessionStatus.CLOSING,
                            require_no_active=True,
                        )
                    elif latest.status is SessionStatus.CLEANUP_REQUIRED:
                        raise AIError(ErrorCode.SESSION_CLEANUP_REQUIRED)
                    else:
                        current = latest
            if current.status is SessionStatus.CLOSING:
                current = await self._close_idle_session(
                    session_id,
                    request.principal.tenant_id,
                    current,
                )
        else:
            if current.status is SessionStatus.OPEN:
                try:
                    current = await self._conversation.sessions.transition_status(
                        session_id,
                        tenant_id=request.principal.tenant_id,
                        expected=frozenset({SessionStatus.OPEN}),
                        next_status=SessionStatus.CLOSING,
                    )
                except AIError as error:
                    if error.code is ErrorCode.STORAGE_CONFLICT:
                        raise
                    if error.code is not ErrorCode.SESSION_CONFLICT:
                        raise
                    current = await self._read_session(
                        session_id,
                        tenant_id=request.principal.tenant_id,
                    )
                    if current is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if current.status is SessionStatus.OPEN:
                        raise
                    if current.status not in {
                        SessionStatus.CLOSING,
                        SessionStatus.CLEANUP_REQUIRED,
                        SessionStatus.CLOSED,
                    }:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            while True:
                current = await self._read_session(
                    session_id,
                    tenant_id=request.principal.tenant_id,
                )
                if current is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                current = await self.reconcile_admission(current)
                if current.active_execution_id is None:
                    break
                execution = await self._active_admitted_execution(current)
                if execution is None:
                    continue
                if execution.status is not ExecutionStatus.FINALIZING:
                    await self._execution.cancel(
                        execution.execution_id,
                        principal=request.principal,
                        idempotency_key=f"{request.idempotency_key}/{execution.execution_id}",
                    )
                try:
                    await asyncio.wait_for(
                        self._wait_for_no_active(
                            session_id,
                            request.principal.tenant_id,
                        ),
                        timeout=request.wait_timeout_seconds,
                    )
                except asyncio.TimeoutError as error:
                    current = await self._read_session(
                        session_id,
                        tenant_id=request.principal.tenant_id,
                    )
                    if current is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if current.status is SessionStatus.CLOSING:
                        await self._conversation.sessions.transition_status(
                            session_id,
                            tenant_id=request.principal.tenant_id,
                            expected=frozenset({SessionStatus.CLOSING}),
                            next_status=SessionStatus.CLEANUP_REQUIRED,
                        )
                    raise AIError(ErrorCode.SESSION_CLEANUP_REQUIRED) from error
                break
            current = await self._read_session(
                session_id,
                tenant_id=request.principal.tenant_id,
            )
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current.status in {
                SessionStatus.CLOSING,
                SessionStatus.CLEANUP_REQUIRED,
            }:
                current = await self._close_idle_session(
                    session_id,
                    request.principal.tenant_id,
                    current,
                )

        await self._complete_close_operation(
            operation,
            request.principal.tenant_id,
            session_id,
            canonical_sha256({"session_id": session_id, "revision": current.revision}),
        )
        _logger.debug(
            "session closed: session=%s revision=%s force=%s",
            session_id,
            current.revision,
            request.force,
        )
        view = await self.project_view(current, request.principal)
        await self._request_session_release(
            session_id,
            request.principal.tenant_id,
            current.continuation,
        )
        return view

    async def _close_idle_session(
        self,
        session_id: str,
        tenant_id: str,
        current: SessionRecord,
    ) -> SessionRecord:
        if current.active_execution_id is not None:
            raise AIError(ErrorCode.SESSION_ACTIVE_EXECUTIONS)
        try:
            return await self._conversation.sessions.transition_status(
                session_id,
                tenant_id=tenant_id,
                expected=frozenset({current.status}),
                next_status=SessionStatus.CLOSED,
                closed_at=datetime.now(timezone.utc),
                require_no_active=True,
            )
        except AIError as error:
            if error.code not in {
                ErrorCode.SESSION_ACTIVE_EXECUTIONS,
                ErrorCode.SESSION_CONFLICT,
            }:
                raise
            latest = await self._read_session(
                session_id,
                tenant_id=tenant_id,
            )
            if latest is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if latest.status is SessionStatus.CLOSED:
                return latest
            if latest.active_execution_id is not None:
                raise AIError(ErrorCode.SESSION_ACTIVE_EXECUTIONS)
            raise

    async def drain_display_reads(self) -> None:
        """Stop display-cache admission and settle requests before closing sources."""
        self._display_close_started = True
        while self._display_reads:
            await asyncio.wait(tuple(self._display_reads))

    @asynccontextmanager
    async def consume_session(
        self,
        session_id: str,
        tenant_id: str,
        *,
        display_read: bool = False,
    ) -> AsyncIterator[None]:
        """Keep session resources retained while a control or display reader uses them."""
        self.require_available()
        completed = None
        if display_read:
            if self._display_close_started:
                raise AIError(ErrorCode.STORAGE_CLOSED)
            completed = asyncio.get_running_loop().create_future()
            self._display_reads.add(completed)
        try:
            key = (tenant_id, session_id)
            state = await self._handoff.enter(key)
            try:
                yield
            finally:
                owner = await self._handoff.leave(key, state)
                if owner:
                    await self._run_session_release(session_id, tenant_id, state)
        finally:
            if completed is not None:
                completed.set_result(None)
                self._display_reads.discard(completed)

    async def _request_session_release(
        self, session_id: str, tenant_id: str, continuation: ConversationCursor | None
    ) -> None:
        key = (tenant_id, session_id)
        state, owner = await self._handoff.request_release(
            key,
            value=continuation,
            require_existing=True,
        )
        if owner:
            await self._run_session_release(session_id, tenant_id, state)

    async def _run_session_release(
        self,
        session_id: str,
        tenant_id: str,
        state: HandoffState[ConversationCursor],
    ) -> None:
        succeeded = False
        cleanup_error: BaseException | None = None
        try:
            await self._release_terminal(
                session_id,
                tenant_id=tenant_id,
                continuation=state.release_value,
            )
            succeeded = True
        except BaseException as error:
            cleanup_error = error
            if isinstance(error, Exception):
                _logger.error(
                    "session transient handoff cleanup failed: session=%s",
                    session_id,
                    exc_info=environ.debug,
                )
        finally:
            await self._handoff.finish_release(
                (tenant_id, session_id),
                state,
                succeeded=succeeded,
            )
        if cleanup_error is not None and not isinstance(cleanup_error, Exception):
            raise cleanup_error

    async def _wait_for_no_active(self, session_id: str, tenant_id: str) -> None:
        record = await self._read_session(session_id, tenant_id=tenant_id)
        if record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        execution = await self._active_admitted_execution(record)
        if execution is None:
            if record.active_execution_id is not None:
                await self.reconcile_admission(record)
            return
        await self._execution.wait(
            execution.execution_id,
            principal=Principal(
                record.owner_principal_id,
                tenant_id,
                PrincipalKind.LOCAL_TRUSTED.value,
            ),
        )
        current = await self._read_session(session_id, tenant_id=tenant_id)
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self.reconcile_admission(current)

    async def read_authorized(
        self, session_id: str, principal: Principal, action: AuthorizationAction
    ) -> SessionRecord:
        header = await self._conversation.sessions.get_header(
            session_id, tenant_id=principal.tenant_id
        )
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, action, header)
        record = await self._conversation.sessions.get(
            session_id, tenant_id=principal.tenant_id
        )
        if record is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        self._check_display_authority(record, header)
        return record

    @overload
    async def project_view(
        self,
        record: SessionRecord,
        principal: Principal,
        *,
        authorized_header: ResourceRef | None = None,
    ) -> SessionView: ...

    @overload
    async def project_view(
        self,
        record: SessionRecord,
        principal: Principal,
        *,
        active: str | None,
        authorized_header: ResourceRef | None = None,
    ) -> SessionView: ...

    async def project_view(
        self,
        record: SessionRecord,
        principal: Principal,
        *,
        active: object = _VIEW_ACTIVE_UNSET,
        authorized_header: ResourceRef | None = None,
    ) -> SessionView:
        """Project session metadata with validated admission and captured authority."""
        if authorized_header is not None:
            self._check_display_authority(record, authorized_header)
        if active is _VIEW_ACTIVE_UNSET:
            active_execution = await self._active_admitted_execution(record)
            active = None if active_execution is None else active_execution.execution_id
        if active is not None and not isinstance(active, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return SessionView(
            record.session_id,
            record.agent_id,
            record.status,
            record.revision,
            record.cwd,
            active,
            record.metadata,
            record.history_quality,
        )

    @staticmethod
    def _active_execution_id(
        record: SessionRecord,
        execution: "SessionExecutionState | None",
    ) -> "str | None":
        active = SessionManager.validate_admission(
            record,
            execution,
        )
        return None if active is None else active.execution_id

    @staticmethod
    def validate_admission(
        record: SessionRecord,
        execution: AdmissionT | None,
    ) -> AdmissionT | None:
        execution_id = record.active_execution_id
        if execution_id is None:
            if execution is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return None
        if (
            execution is None
            or execution.execution_id != execution_id
            or execution.session_id != record.session_id
            or execution.parent_execution_id is not None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            return None
        return execution

    async def _active_admitted_execution(
        self,
        record: SessionRecord,
    ) -> "SessionExecutionState | None":
        execution_id = record.active_execution_id
        if execution_id is None:
            return None
        execution = await self._executions.get(
            execution_id,
            tenant_id=self._conversation.sessions.tenant_id,
        )
        return self.validate_admission(record, execution)

    async def reconcile_admission(
        self, record: SessionRecord
    ) -> SessionRecord:
        execution_id = record.active_execution_id
        if execution_id is None:
            return record
        execution = await self._executions.get(
            execution_id,
            tenant_id=self._conversation.sessions.tenant_id,
        )
        if self.validate_admission(record, execution) is not None:
            return record
        await self._conversation.sessions.release_execution(
            record.session_id,
            tenant_id=self._conversation.sessions.tenant_id,
            execution_id=execution_id,
            history_unavailable=(
                execution is not None and execution.invalidates_session_history
            ),
            require_turn_commit=execution is not None and execution.conversation,
        )
        updated = await self._conversation.sessions.get(
            record.session_id,
            tenant_id=self._conversation.sessions.tenant_id,
        )
        if updated is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _logger.info(
            "released stale session admission: session=%s execution=%s",
            record.session_id,
            execution_id,
        )
        return updated

    def _session_terminal_operation(
        self,
        operation_id: str,
        tenant_id: str,
        session_id: str,
        operation_kind: OperationKind,
        request_digest: str,
        revision: int,
    ) -> OperationLedgerInput:
        now = datetime.now(timezone.utc)
        return OperationLedgerInput(
            idempotency_key_digest(operation_id),
            tenant_id,
            ResourceKind.SESSION,
            session_id,
            None,
            operation_kind,
            OperationStatus.SUCCEEDED,
            request_digest,
            session_id,
            canonical_sha256({"session_id": session_id, "revision": revision}),
            None,
            False,
            now,
            now,
        )

    async def _begin_close_operation(
        self,
        operation_id: str,
        tenant_id: str,
        session_id: str,
        request_digest: str,
    ) -> OperationLedgerRecord:
        operation_id = idempotency_key_digest(operation_id)
        now = datetime.now(timezone.utc)
        requested = OperationLedgerInput(
            operation_id,
            tenant_id,
            ResourceKind.SESSION,
            session_id,
            None,
            OperationKind.SESSION_CLOSE,
            OperationStatus.PENDING,
            request_digest,
            None,
            None,
            None,
            True,
            now,
            now,
        )
        try:
            existing = await self._conversation.operations.append(requested)
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            existing = await self._conversation.operations.get(
                operation_id,
                tenant_id=tenant_id,
            )
            if existing is None:
                raise
        if (
            existing.tenant_id != tenant_id
            or existing.resource_kind is not ResourceKind.SESSION
            or existing.resource_id != session_id
            or existing.execution_id is not None
            or existing.operation_kind is not OperationKind.SESSION_CLOSE
            or existing.request_digest != request_digest
            or not existing.compactable
        ):
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        if existing.status is OperationStatus.PENDING:
            if (
                existing.result_ref is not None
                or existing.result_digest is not None
                or existing.error_code is not None
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return existing
        if existing.status is OperationStatus.SUCCEEDED:
            if (
                existing.result_ref != session_id
                or existing.result_digest is None
                or existing.error_code is not None
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return existing
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _complete_close_operation(
        self,
        operation: OperationLedgerRecord,
        tenant_id: str,
        result_ref: str,
        result_digest: str,
    ) -> None:
        now = datetime.now(timezone.utc)
        completed = OperationLedgerRecord(
            operation.operation_id,
            tenant_id,
            operation.resource_kind,
            operation.resource_id,
            operation.execution_id,
            operation.operation_kind,
            OperationStatus.SUCCEEDED,
            operation.request_digest,
            result_ref,
            result_digest,
            None,
            operation.compactable,
            operation.sequence,
            operation.created_at,
            now,
        )
        try:
            await self._conversation.operations.compare_and_swap(
                operation.operation_id,
                tenant_id=tenant_id,
                expected_status=OperationStatus.PENDING,
                next_record=completed,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._conversation.operations.get(
                operation.operation_id, tenant_id=tenant_id
            )
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            self._validate_close_operation_identity(
                current,
                operation,
                tenant_id=tenant_id,
                session_id=result_ref,
            )
            if current.status is OperationStatus.SUCCEEDED:
                self._validate_succeeded_close_operation(
                    current,
                    tenant_id=tenant_id,
                    session_id=result_ref,
                    request_digest=operation.request_digest,
                    result_digest=result_digest,
                )
                return
            if current.status is OperationStatus.PENDING:
                if (
                    current.result_ref is not None
                    or current.result_digest is not None
                    or current.error_code is not None
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                raise
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    @staticmethod
    def _validate_close_operation_identity(
        current: OperationLedgerRecord,
        expected: OperationLedgerRecord,
        *,
        tenant_id: str,
        session_id: str,
    ) -> None:
        if (
            current.operation_id != expected.operation_id
            or current.tenant_id != expected.tenant_id
            or current.tenant_id != tenant_id
            or current.resource_kind is not expected.resource_kind
            or current.resource_kind is not ResourceKind.SESSION
            or expected.resource_kind is not ResourceKind.SESSION
            or current.resource_id != expected.resource_id
            or current.resource_id != session_id
            or expected.resource_id != session_id
            or current.execution_id != expected.execution_id
            or current.execution_id is not None
            or expected.execution_id is not None
            or current.operation_kind is not expected.operation_kind
            or current.operation_kind is not OperationKind.SESSION_CLOSE
            or expected.operation_kind is not OperationKind.SESSION_CLOSE
            or current.request_digest != expected.request_digest
            or current.compactable != expected.compactable
            or current.compactable is not True
            or current.sequence != expected.sequence
            or current.created_at != expected.created_at
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    @staticmethod
    def _validate_succeeded_close_operation(
        operation: OperationLedgerRecord,
        *,
        tenant_id: str,
        session_id: str,
        request_digest: str,
        result_digest: str,
    ) -> None:
        if (
            operation.tenant_id != tenant_id
            or operation.resource_kind is not ResourceKind.SESSION
            or operation.resource_id != session_id
            or operation.execution_id is not None
            or operation.operation_kind is not OperationKind.SESSION_CLOSE
            or operation.request_digest != request_digest
            or operation.status is not OperationStatus.SUCCEEDED
            or operation.result_ref != session_id
            or operation.result_digest != result_digest
            or operation.error_code is not None
            or not operation.compactable
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    @staticmethod
    def _validate_close_replay(
        operation: OperationLedgerRecord,
        session: SessionRecord,
        session_id: str,
        *,
        tenant_id: str,
    ) -> None:
        if (
            session.status is not SessionStatus.CLOSED
            or session.active_execution_id is not None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result_digest = canonical_sha256(
            {"session_id": session_id, "revision": session.revision}
        )
        SessionManager._validate_close_operation_identity(
            operation,
            operation,
            tenant_id=tenant_id,
            session_id=session_id,
        )
        SessionManager._validate_succeeded_close_operation(
            operation,
            tenant_id=tenant_id,
            session_id=session_id,
            request_digest=operation.request_digest,
            result_digest=result_digest,
        )


def _session_filter_digest(owner_principal_id: str) -> str:
    return canonical_sha256({"owner_principal_id": owner_principal_id})


def _make_cursor(
    snapshot: int,
    tenant_id: str,
    owner_principal_id: str,
    sort_key: "str | None",
    signer: CursorSigner,
) -> "str | None":
    if sort_key is None:
        return None
    return encode_runtime_cursor(
        signer,
        tenant_id=tenant_id,
        resource_kind=_SESSION_CURSOR_KIND,
        filter_digest=_session_filter_digest(owner_principal_id),
        position=sort_key,
        revision=snapshot,
    )


def _decode_session_cursor(
    cursor: "str | None", tenant_id: str, owner_principal_id: str, signer: CursorSigner
) -> "tuple[str | None, int | None]":
    if cursor is None:
        return None, None
    payload = decode_runtime_cursor(
        cursor,
        signer,
        tenant_id=tenant_id,
        resource_kind=_SESSION_CURSOR_KIND,
        filter_digest=_session_filter_digest(owner_principal_id),
    )
    return payload.position, payload.revision


__all__ = [
    "SessionManager",
    "SessionExecutionState",
    "SessionExecutionReader",
    "SessionExecutionService",
    "SessionRepositories",
    "SessionOperations",
    "SessionReleaseCallback",
    "SessionPendingExecutionLookup",
    "SessionRecordReader",
    "SessionViewObserver",
]
