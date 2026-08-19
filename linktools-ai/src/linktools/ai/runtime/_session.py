#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session query API and persistence-backed session service."""

import asyncio
import json
import time
from binascii import Error as Base64Error
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Protocol

from linktools.core import environ
from pydantic_ai_harness.step_persistence import ContinuableSnapshot

from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    CursorPayload,
    CursorSigner,
    ExecutionStatus,
    OperationKind,
    OperationStatus,
    Page,
    Principal,
    PrincipalKind,
    ResourceKind,
    ResourceRef,
    SessionStatus,
    canonical_sha256,
    idempotency_key_digest,
)
from ..errors import AIError, ErrorCode
from .service_api import (
    CancelExecutionRequest,
    CloseSessionRequest,
    CreateSessionRequest,
    ExecutionHandle,
    ExecutionRequest,
    ExecutionService,
    ForkSessionRequest,
    ListSessionRequest,
    LoadedSession,
    ResumeSessionRequest,
    SessionHistoryItem,
    SessionHistoryReader,
    SessionView,
    UpdateSessionRequest,
)
from .state._contracts import (
    ConversationCursor,
    ConversationHistoryParent,
    ConversationHistoryRecord,
    ConversationState,
    ExecutionRecord,
    ExecutionRepository,
    HistoryQuality,
    OperationLedgerInput,
    OperationLedgerRecord,
    SessionRecord,
)

_logger = environ.get_logger("ai.runtime.session")


class _SessionReleaseCallback(Protocol):
    async def __call__(self, session_id: str, *, tenant_id: str, continuation: ConversationCursor | None) -> None: ...


class _LaunchGate(Protocol):
    async def __call__(self, execution_id: str) -> None: ...


class _GatedExecutionService(Protocol):
    async def run_for_session(
        self,
        binding_digest: str,
        session_id: str,
        request: ExecutionRequest,
    ) -> ExecutionHandle: ...


class _SessionTranscriptStore(Protocol):
    async def has_canonical_transcript(self, *, run_id: str) -> bool: ...

    async def iter_messages(self, *, run_id: str) -> AsyncIterator[object]: ...

    async def iter_session_messages(
        self,
        history_id: str,
        *,
        tenant_id: str,
    ) -> AsyncIterator[object]: ...

    async def load_session_model_context(
        self,
        history_id: str,
        *,
        tenant_id: str,
        binding_digest: str | None = None,
    ) -> tuple[object, ...]: ...

    async def load_model_context(
        self,
        *,
        run_id: str,
        binding_digest: str,
    ) -> tuple[object, ...]: ...

    async def latest_snapshot(
        self,
        *,
        run_id: str,
        include_interrupted: bool = False,
    ) -> ContinuableSnapshot | None: ...

    async def _run_for_session_with_launch_gate(
        self,
        binding_digest: str,
        session_id: str,
        request: ExecutionRequest,
        gate: _LaunchGate,
    ) -> ExecutionHandle: ...


async def _no_release_terminal(session_id: str, *, tenant_id: str, continuation: ConversationCursor | None) -> None:
    del session_id, tenant_id, continuation


@dataclass
class _SessionHandoffState:
    active_consumers: int = 0
    release_requested: bool = False
    release_in_progress: bool = False
    continuation: ConversationCursor | None = None


class SessionQueryApi(Protocol):
    async def get(self, session_id: str, *, principal: Principal) -> SessionView: ...
    async def list(self, request: ListSessionRequest) -> "Page[SessionView]": ...
    async def history(
        self,
        session_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> "Page[SessionHistoryItem]": ...
    async def load(self, session_id: str, *, principal: Principal) -> LoadedSession: ...
    def iter_session_messages(
        self,
        session_id: str,
        *,
        principal: Principal,
    ) -> AsyncIterator[object]: ...
    async def load_model_context(
        self,
        session_id: str,
        *,
        principal: Principal,
    ) -> tuple[object, ...]: ...


class SessionApi(SessionQueryApi, Protocol):
    async def create(self, request: CreateSessionRequest) -> SessionView: ...
    async def resume(self, session_id: str, request: ResumeSessionRequest) -> ExecutionHandle: ...
    async def fork(self, session_id: str, request: ForkSessionRequest) -> SessionView: ...
    async def update(self, session_id: str, request: UpdateSessionRequest) -> SessionView: ...
    async def close(self, session_id: str, request: CloseSessionRequest) -> SessionView: ...


class DefaultSessionService:
    """Enforce session ownership, binding immutability, and revision CAS."""

    def __init__(
        self,
        conversation: ConversationState,
        executions: ExecutionRepository,
        authorization: AuthorizationPolicy,
        execution: ExecutionService,
        cursor_signer: CursorSigner,
        *,
        history_reader: SessionHistoryReader,
        transcript_store: "_SessionTranscriptStore | None" = None,
        release_terminal: _SessionReleaseCallback | None = None,
        gated_execution: "_GatedExecutionService | None" = None,
    ) -> None:
        self._conversation = conversation
        self._executions = executions
        self._authorization = authorization
        self._execution = execution
        self._cursor_signer = cursor_signer
        self._history_reader = history_reader
        self._transcript_store = transcript_store
        self._release_terminal = release_terminal or _no_release_terminal
        self._gated_execution = gated_execution
        self._handoff_states: dict[tuple[str, str], _SessionHandoffState] = {}
        self._handoff_condition = asyncio.Condition()

    async def create(self, binding_digest: str, request: CreateSessionRequest) -> SessionView:
        resource = ResourceRef(
            ResourceKind.SESSION, request.session_id, request.principal.tenant_id, request.principal.principal_id
        )
        await self._authorization.authorize(request.principal, AuthorizationAction.SESSION_CREATE, resource)
        digest = canonical_sha256(
            {
                "action": "session.create",
                "tenant_id": request.principal.tenant_id,
                "principal_id": request.principal.principal_id,
                "session_id": request.session_id,
                "binding": binding_digest,
                "cwd": request.cwd,
                "metadata": dict(request.metadata),
            }
        )
        now = datetime.now(timezone.utc)
        record = SessionRecord(
            session_id=request.session_id,
            tenant_id=request.principal.tenant_id,
            owner_principal_id=request.principal.principal_id,
            binding_digest=binding_digest,
            status=SessionStatus.OPEN,
            revision=0,
            resource_generation=0,
            cwd=request.cwd,
            metadata=dict(request.metadata),
            created_at=now,
            updated_at=now,
            closed_at=None,
            active_execution_id=None,
            continuation=None,
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
        _logger.debug("session created: session=%s tenant=%s", record.session_id, request.principal.tenant_id)
        return await self._view(record, request.principal)

    async def get(self, session_id: str, *, principal: Principal) -> SessionView:
        async with self._session_consumer(session_id, principal.tenant_id):
            record = await self._authorized(session_id, principal, AuthorizationAction.SESSION_READ)
            return await self._view(record, principal)

    async def history(
        self,
        session_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> "Page[SessionHistoryItem]":
        async with self._session_consumer(session_id, principal.tenant_id):
            record = await self._authorized(
                session_id,
                principal,
                AuthorizationAction.SESSION_READ,
            )
            record = await self._ensure_legacy_history(record)
            continuation = None if record.continuation is None else record.continuation.step_run_id
            continuation_history_id = (
                None
                if record.continuation is None
                else record.continuation.history_id or record.history_id
            )
            return await self._history_reader.history(
                session_id,
                tenant_id=principal.tenant_id,
                continuation_step_run_id=continuation,
                continuation_history_id=continuation_history_id,
                cursor=cursor,
                limit=limit,
            )

    async def _ensure_legacy_history(self, record: SessionRecord) -> SessionRecord:
        if (
            record.history_id is not None
            or record.continuation is None
            or self._transcript_store is None
        ):
            return record
        run_id = record.continuation.step_run_id
        snapshot = await self._transcript_store.latest_snapshot(
            run_id=run_id,
            include_interrupted=True,
        )
        if snapshot is None or snapshot.state != "complete":
            return record
        messages = [
            message
            async for message in self._transcript_store.iter_messages(run_id=run_id)
        ]
        history_id = canonical_sha256(
            {
                "kind": "conversation_history",
                "session_id": record.session_id,
                "tenant_id": record.tenant_id,
            }
        )
        history = ConversationHistoryRecord(
            history_id=history_id,
            session_id=record.session_id,
            tenant_id=record.tenant_id,
            parent=ConversationHistoryParent(
                history_id=run_id,
                through_message_count=len(messages),
            ),
            inherited_message_count=len(messages),
            local_message_count=0,
            history_quality=HistoryQuality.LEGACY_PARTIAL,
            revision=0,
        )
        try:
            return await self._conversation.sessions.ensure_history(
                record.session_id,
                tenant_id=record.tenant_id,
                expected_revision=record.revision,
                history=history,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._conversation.sessions.get(
                record.session_id,
                tenant_id=record.tenant_id,
            )
            if current is None or current.history_id != history_id:
                raise
            return current

    async def list(self, request: ListSessionRequest) -> Page[SessionView]:
        await self._authorization.authorize(
            request.principal,
            AuthorizationAction.SESSION_READ,
            ResourceRef(ResourceKind.SESSION, "list", request.principal.tenant_id),
        )
        if not 1 <= request.limit <= 200:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
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
            limit=request.limit,
            snapshot=snapshot,
        )
        values = page.items
        views = tuple(await asyncio.gather(*(self._view(record, request.principal) for record in values)))
        next_cursor = _make_cursor(
            snapshot,
            request.principal.tenant_id,
            request.principal.principal_id,
            page.next_cursor,
            self._cursor_signer,
        )
        return Page(views, next_cursor)

    async def load(self, session_id: str, *, principal: Principal) -> LoadedSession:
        async with self._session_consumer(session_id, principal.tenant_id):
            record = await self._authorized(session_id, principal, AuthorizationAction.SESSION_READ)
            record = await self._reconcile_terminal_admission(record)
            active_execution = await self._active_admitted_execution(record)
            active = () if active_execution is None else (active_execution.execution_id,)
            return LoadedSession(await self._view(record, principal), active)

    async def load_model_context(
        self,
        session_id: str,
        *,
        principal: Principal,
    ) -> tuple[object, ...]:
        async with self._session_consumer(session_id, principal.tenant_id):
            record = await self._authorized(session_id, principal, AuthorizationAction.SESSION_READ)
            record = await self._ensure_legacy_history(record)
            if self._transcript_store is None or record.continuation is None:
                return ()
            if record.history_id is not None:
                return await self._transcript_store.load_session_model_context(
                    record.history_id,
                    tenant_id=record.tenant_id,
                    binding_digest=record.binding_digest,
                )
            return await self._transcript_store.load_model_context(
                run_id=record.continuation.step_run_id,
                binding_digest=record.binding_digest,
            )

    async def _iter_session_messages(
        self,
        session_id: str,
        *,
        principal: Principal,
    ) -> AsyncIterator[object]:
        async with self._session_consumer(session_id, principal.tenant_id):
            record = await self._authorized(session_id, principal, AuthorizationAction.SESSION_READ)
            record = await self._ensure_legacy_history(record)
            if self._transcript_store is None or record.continuation is None:
                return
            if record.history_id is not None:
                async for message in self._transcript_store.iter_session_messages(
                    record.history_id,
                    tenant_id=record.tenant_id,
                ):
                    yield message
                return
            async for message in self._transcript_store.iter_messages(
                run_id=record.continuation.step_run_id,
            ):
                yield message

    def iter_session_messages(
        self,
        session_id: str,
        *,
        principal: Principal,
    ) -> AsyncIterator[object]:
        return self._iter_session_messages(session_id, principal=principal)

    async def resume(self, binding_digest: str, session_id: str, request: ResumeSessionRequest) -> ExecutionHandle:
        return await self._resume(binding_digest, session_id, request, launch_gate=None)

    async def _resume_with_launch_gate(
        self,
        binding_digest: str,
        session_id: str,
        request: ResumeSessionRequest,
        gate: _LaunchGate,
    ) -> ExecutionHandle:
        return await self._resume(binding_digest, session_id, request, launch_gate=gate)

    async def _resume(
        self,
        binding_digest: str,
        session_id: str,
        request: ResumeSessionRequest,
        *,
        launch_gate: _LaunchGate | None,
    ) -> ExecutionHandle:
        async with self._session_consumer(session_id, request.principal.tenant_id):
            record = await self._authorized(session_id, request.principal, AuthorizationAction.SESSION_READ)
            record = await self._reconcile_terminal_admission(record)
            record = await self._ensure_legacy_history(record)
            await self._authorization.authorize(
                request.principal,
                AuthorizationAction.EXECUTION_RUN,
                ResourceRef(ResourceKind.EXECUTION, session_id, request.principal.tenant_id),
            )
            if record.binding_digest != binding_digest:
                raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)
            execution_request = ExecutionRequest(
                user_prompt=request.user_prompt,
                principal=request.principal,
                idempotency_key=request.idempotency_key,
                memory_scope=request.memory_scope,
            )
            if self._gated_execution is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            if launch_gate is None:
                try:
                    return await self._gated_execution.run_for_session(
                        binding_digest,
                        session_id,
                        execution_request,
                    )
                except AttributeError as error:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY) from error
            try:
                return await self._gated_execution._run_for_session_with_launch_gate(
                    binding_digest,
                    session_id,
                    execution_request,
                    launch_gate,
                )
            except AttributeError as error:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY) from error

    async def fork(self, binding_digest: str, session_id: str, request: ForkSessionRequest) -> SessionView:
        async with self._session_consumer(session_id, request.principal.tenant_id):
            source = await self._authorized(session_id, request.principal, AuthorizationAction.SESSION_READ)
            source = await self._ensure_legacy_history(source)
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
            if source.binding_digest != binding_digest:
                raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)
            digest = canonical_sha256(
                {
                    "action": "session.fork",
                    "tenant_id": request.principal.tenant_id,
                    "principal_id": request.principal.principal_id,
                    "source": session_id,
                    "target": request.new_session_id,
                    "binding": source.binding_digest,
                    "cwd": request.cwd,
                }
            )
            now = datetime.now(timezone.utc)
            target = SessionRecord(
                session_id=request.new_session_id,
                tenant_id=source.tenant_id,
                owner_principal_id=source.owner_principal_id,
                binding_digest=source.binding_digest,
                status=SessionStatus.OPEN,
                revision=0,
                resource_generation=0,
                cwd=source.cwd if request.cwd is None else request.cwd,
                metadata=dict(source.metadata),
                created_at=now,
                updated_at=now,
                closed_at=None,
                active_execution_id=None,
                continuation=source.continuation,
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
            _logger.debug("session forked: source=%s target=%s", session_id, target.session_id)
            return await self._view(target, request.principal)

    async def update(self, binding_digest: str, session_id: str, request: UpdateSessionRequest) -> SessionView:
        async with self._session_consumer(session_id, request.principal.tenant_id):
            return await self._update(binding_digest, session_id, request)

    async def _update(self, binding_digest: str, session_id: str, request: UpdateSessionRequest) -> SessionView:
        current = await self._authorized(session_id, request.principal, AuthorizationAction.SESSION_UPDATE)
        if current.binding_digest != binding_digest:
            raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)
        if any(
            key.startswith("linktools.ai.") and current.metadata.get(key) != value
            for key, value in request.metadata.items()
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if any(key.startswith("linktools.ai.") for key in current.metadata if key not in request.metadata):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        digest = canonical_sha256(
            {
                "action": "session.update",
                "tenant_id": request.principal.tenant_id,
                "principal_id": request.principal.principal_id,
                "session_id": session_id,
                "expected_revision": request.expected_revision,
                "metadata": request.metadata,
                "cwd": request.cwd,
            }
        )
        requested_cwd = request.cwd if request.cwd is not None else current.cwd
        now = datetime.now(timezone.utc)
        next_record = replace(
            current,
            revision=current.revision + 1,
            resource_generation=current.resource_generation + 1,
            cwd=requested_cwd,
            metadata=request.metadata,
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
        _logger.debug("session updated: session=%s revision=%s", session_id, updated.revision)
        return await self._view(updated, request.principal)

    async def close(self, session_id: str, request: CloseSessionRequest) -> SessionView:
        async with self._session_consumer(session_id, request.principal.tenant_id):
            return await self._close(session_id, request)

    async def _close(self, session_id: str, request: CloseSessionRequest) -> SessionView:
        await self._authorized(session_id, request.principal, AuthorizationAction.SESSION_CLOSE)
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
            closed = await self._conversation.sessions.get(session_id, tenant_id=request.principal.tenant_id)
            if closed is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._validate_close_replay(operation, closed, session_id)
            view = await self._view(closed, request.principal)
            await self._request_session_release(
                session_id,
                request.principal.tenant_id,
                closed.continuation,
            )
            return view
        current = await self._conversation.sessions.get(
            session_id,
            tenant_id=request.principal.tenant_id,
        )
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        current = await self._reconcile_terminal_admission(current)
        if current.status is SessionStatus.CLOSED:
            await self._complete_close_operation(
                operation,
                request.principal.tenant_id,
                session_id,
                canonical_sha256({"session_id": session_id, "revision": current.revision}),
            )
            view = await self._view(current, request.principal)
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
                    latest = await self._conversation.sessions.get(
                        session_id,
                        tenant_id=request.principal.tenant_id,
                    )
                    if latest is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if latest.status is SessionStatus.OPEN and latest.active_execution_id is not None:
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
                    current = await self._conversation.sessions.get(
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
                current = await self._conversation.sessions.get(
                    session_id,
                    tenant_id=request.principal.tenant_id,
                )
                if current is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                current = await self._reconcile_terminal_admission(current)
                if current.active_execution_id is None:
                    break
                execution = await self._active_admitted_execution(current)
                if execution is None:
                    continue
                if execution.status is not ExecutionStatus.FINALIZING:
                    await self._execution.cancel(
                        execution.execution_id,
                        CancelExecutionRequest(
                            request.principal,
                            f"{request.idempotency_key}/{execution.execution_id}",
                            True,
                        ),
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
                    current = await self._conversation.sessions.get(
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
            current = await self._conversation.sessions.get(
                session_id,
                tenant_id=request.principal.tenant_id,
            )
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current.status in {SessionStatus.CLOSING, SessionStatus.CLEANUP_REQUIRED}:
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
        view = await self._view(current, request.principal)
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
            latest = await self._conversation.sessions.get(
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

    @asynccontextmanager
    async def _session_consumer(self, session_id: str, tenant_id: str):
        key = (tenant_id, session_id)
        async with self._handoff_condition:
            while True:
                state = self._handoff_states.get(key)
                if state is None:
                    state = _SessionHandoffState()
                    self._handoff_states[key] = state
                if not state.release_in_progress:
                    state.active_consumers += 1
                    break
                await self._handoff_condition.wait()
        cleanup_owner = False
        try:
            yield state
        finally:
            async with self._handoff_condition:
                state.active_consumers -= 1
                if state.active_consumers < 0:
                    raise RuntimeError("session consumer count became negative")
                if state.active_consumers == 0:
                    if state.release_requested and not state.release_in_progress:
                        state.release_in_progress = True
                        cleanup_owner = True
                    elif not state.release_requested and self._handoff_states.get(key) is state:
                        self._handoff_states.pop(key, None)
                self._handoff_condition.notify_all()
            if cleanup_owner:
                cleanup_succeeded = False
                try:
                    await self._release_terminal(session_id, tenant_id=tenant_id, continuation=state.continuation)
                    cleanup_succeeded = True
                except BaseException:
                    _logger.error(
                        "session transient handoff cleanup failed: session=%s", session_id, exc_info=environ.debug
                    )
                async with self._handoff_condition:
                    if self._handoff_states.get(key) is state:
                        if cleanup_succeeded and state.active_consumers == 0:
                            self._handoff_states.pop(key, None)
                        else:
                            state.release_in_progress = False
                            state.release_requested = True
                    self._handoff_condition.notify_all()

    async def _request_session_release(
        self, session_id: str, tenant_id: str, continuation: ConversationCursor | None
    ) -> None:
        key = (tenant_id, session_id)
        async with self._handoff_condition:
            state = self._handoff_states.get(key)
            if state is None:
                raise RuntimeError("session release requested without consumer")
            state.release_requested = True
            state.continuation = continuation
            self._handoff_condition.notify_all()

    async def _wait_for_no_active(self, session_id: str, tenant_id: str) -> None:
        record = await self._conversation.sessions.get(session_id, tenant_id=tenant_id)
        if record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        execution = await self._active_admitted_execution(record)
        if execution is None:
            if record.active_execution_id is not None:
                await self._reconcile_terminal_admission(record)
            return
        await self._execution.wait(
            execution.execution_id,
            principal=Principal(
                record.owner_principal_id,
                tenant_id,
                PrincipalKind.LOCAL_TRUSTED.value,
            ),
        )
        current = await self._conversation.sessions.get(session_id, tenant_id=tenant_id)
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._reconcile_terminal_admission(current)

    async def _authorized(self, session_id: str, principal: Principal, action: AuthorizationAction) -> SessionRecord:
        header = await self._conversation.sessions.get_header(session_id, tenant_id=principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, action, header)
        record = await self._conversation.sessions.get(session_id, tenant_id=principal.tenant_id)
        if record is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return record

    async def _view(self, record: SessionRecord, principal: Principal) -> SessionView:
        active_execution = await self._active_admitted_execution(record)
        active = () if active_execution is None else (active_execution.execution_id,)
        history_quality = record.history_quality
        if (
            history_quality == "complete"
            and record.history_id is None
            and record.continuation is not None
            and self._transcript_store is not None
            and not await self._transcript_store.has_canonical_transcript(
                run_id=record.continuation.step_run_id,
            )
        ):
            history_quality = "legacy_partial"
        return SessionView(
            record.session_id,
            record.binding_digest,
            record.status,
            record.revision,
            record.resource_generation,
            record.cwd,
            active,
            record.metadata,
            history_quality,
        )

    async def _active_admitted_execution(
        self,
        record: SessionRecord,
    ) -> "ExecutionRecord | None":
        execution_id = record.active_execution_id
        if execution_id is None:
            return None
        execution = await self._executions.get(execution_id, tenant_id=record.tenant_id)
        if (
            execution is None
            or execution.execution_id != execution_id
            or execution.tenant_id != record.tenant_id
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

    async def _reconcile_terminal_admission(self, record: SessionRecord) -> SessionRecord:
        execution_id = record.active_execution_id
        if execution_id is None:
            return record
        if await self._active_admitted_execution(record) is not None:
            return record
        await self._conversation.sessions.release_execution(
            record.session_id,
            tenant_id=record.tenant_id,
            execution_id=execution_id,
        )
        updated = await self._conversation.sessions.get(
            record.session_id,
            tenant_id=record.tenant_id,
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
            if existing.result_ref is not None or existing.result_digest is not None or existing.error_code is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return existing
        if existing.status is OperationStatus.SUCCEEDED:
            if existing.result_ref != session_id or existing.result_digest is None or existing.error_code is not None:
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
            current = await self._conversation.operations.get(operation.operation_id, tenant_id=tenant_id)
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
    ) -> None:
        if session.status is not SessionStatus.CLOSED or session.active_execution_id is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result_digest = canonical_sha256({"session_id": session_id, "revision": session.revision})
        DefaultSessionService._validate_close_operation_identity(
            operation,
            operation,
            tenant_id=session.tenant_id,
            session_id=session_id,
        )
        DefaultSessionService._validate_succeeded_close_operation(
            operation,
            tenant_id=session.tenant_id,
            session_id=session_id,
            request_digest=operation.request_digest,
            result_digest=result_digest,
        )


__all__ = ["DefaultSessionService", "SessionApi", "SessionQueryApi"]


def _make_cursor(
    snapshot: int, tenant_id: str, owner_principal_id: str, sort_key: "str | None", signer: CursorSigner
) -> "str | None":
    if sort_key is None:
        return None
    return signer.encode(
        CursorPayload(
            1,
            tenant_id,
            "SESSION",
            canonical_sha256({"owner_principal_id": owner_principal_id}),
            sort_key,
            snapshot,
            int(time.time()) + 3600,
        )
    )


def _decode_session_cursor(
    cursor: "str | None", tenant_id: str, owner_principal_id: str, signer: CursorSigner
) -> "tuple[str | None, int | None]":
    if cursor is None:
        return None, None
    try:
        payload = signer.decode(cursor)
        if (
            payload.cursor_version != 1
            or payload.tenant_id != tenant_id
            or payload.resource_kind != "SESSION"
            or payload.filter_digest != canonical_sha256({"owner_principal_id": owner_principal_id})
        ):
            raise ValueError("session cursor identity mismatch")
        if not payload.sort_key.strip():
            raise ValueError("session cursor sort key is empty")
        return payload.sort_key, payload.snapshot_or_store_revision
    except (Base64Error, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    except AIError as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
