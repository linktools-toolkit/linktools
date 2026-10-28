#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session query API and persistence-backed session service."""

import json
import math
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Protocol


from ..session import SessionPendingExecutionLookup, SessionReleaseCallback, SessionManager
from ..agent import AgentBinding, AgentBindingContract
from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    CursorSigner,
    ExecutionStatus,
    Page,
    Principal,
    ResourceKind,
    ResourceRef,
    SessionStatus,
    canonical_sha256,
    validate_page_limit,
)
from ..errors import AIError, ErrorCode
from ..storage import CacheBackend
from ..history import SessionDisplayCache
from ..history import (
    capture_session_refresh,
    project_captured_session_timeline,
    project_session_timeline,
)
from ..inputs import InputFileSource, stored_user_input_view
from ..execution import (
    CancelExecutionRequest,
    ExecutionHandle,
    ExecutionRequest,
    ExecutionService,
    ResumeSessionRequest,
    SessionDetail,
    SessionHistoryItem,
    SessionHistoryReader,
    SessionTurn,
    SessionView,
)
from ..persistence.objects import runtime_object_source_reads
from ..session import ConversationCursor, SessionRecord, SessionAdmission
from ..persistence.contracts import ConversationRepositories, ExecutionRepository
from ..execution import ExecutionRecord
from ..persistence.repositories import SessionRepositoryImpl
from ..persistence.repositories import ExecutionRepositoryImpl
from ..persistence.state import active_state_scope





class _SessionExecutionService(ExecutionService, Protocol):
    async def lookup_session_admission(
        self, session_id: str, *, principal: Principal, idempotency_key: str,
    ) -> SessionAdmission | None: ...

    async def start_for_session(
        self,
        agent_id: str,
        binding_digest: str,
        session_id: str,
        request: ExecutionRequest,
        *,
        binding_contract: "AgentBindingContract | None" = None,
        resolved_binding: "AgentBinding | None" = None,
        dependency_hold_id: "str | None" = None,
        requires_task_invocation_capture: bool = False,
        budget_scope_id: str | None = None,
    ) -> ExecutionHandle: ...


class _SessionTranscriptStore(Protocol):
    def iter_conversation_messages(
        self,
        *,
        history_id: str | None,
        agent_run_id: str,
        tenant_id: str,
    ) -> AsyncIterator[object]: ...

    def iter_conversation_message_range(
        self,
        *,
        history_id: str | None,
        agent_run_id: str,
        tenant_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[object]: ...

    async def load_conversation_model_context(
        self,
        *,
        history_id: str | None,
        agent_run_id: str,
        tenant_id: str,
        message_count: int | None = None,
    ) -> tuple[object, ...]: ...


class _SessionExecutionAdapter:
    def __init__(self, execution: _SessionExecutionService) -> None:
        self._execution = execution

    async def start_for_session(self, agent_id: str, binding_digest: str, session_id: str,
                                request: ResumeSessionRequest, *,
                                binding_contract: AgentBindingContract | None = None,
                                resolved_binding: AgentBinding | None = None,
                                dependency_hold_id: str | None = None,
                                requires_task_invocation_capture: bool = False,
                                budget_scope_id: str | None = None) -> ExecutionHandle:
        execution_request = ExecutionRequest(
            user_prompt=request.user_prompt, principal=request.principal,
            idempotency_key=request.idempotency_key, memory_scope=request.memory_scope,
            mode=request.mode, planning=request.planning, thinking=request.thinking,
            correlation=request.correlation, files=request.files, budget=request.budget,
        )
        return await self._execution.start_for_session(
            agent_id, binding_digest, session_id, execution_request,
            binding_contract=binding_contract, resolved_binding=resolved_binding, dependency_hold_id=dependency_hold_id,
            requires_task_invocation_capture=requires_task_invocation_capture,
            budget_scope_id=budget_scope_id,
        )

    async def cancel(self, execution_id: str, *, principal: Principal, idempotency_key: str) -> None:
        await self._execution.cancel(execution_id, CancelExecutionRequest(principal, idempotency_key, True))

    async def wait(self, execution_id: str, *, principal: Principal) -> None:
        await self._execution.wait(execution_id, principal=principal)


class RuntimeSessionService(SessionManager[ExecutionHandle]):
    """Enforce session ownership, Agent identity immutability, and revision CAS."""

    async def lookup_admission(
        self, session_id: str, *, principal: Principal, idempotency_key: str,
    ) -> SessionAdmission | None:
        self.require_available()
        await self.read_authorized(session_id, principal, AuthorizationAction.SESSION_READ)
        return await self._admission_execution.lookup_session_admission(
            session_id, principal=principal, idempotency_key=idempotency_key,
        )

    def __init__(
        self,
        conversation: ConversationRepositories,
        executions: ExecutionRepository,
        authorization: AuthorizationPolicy,
        execution: _SessionExecutionService,
        cursor_signer: CursorSigner,
        *,
        history_reader: SessionHistoryReader,
        enabled: bool = True,
        transcript_store: "_SessionTranscriptStore | None" = None,
        release_terminal: SessionReleaseCallback | None = None,
        workspace_access: InputFileSource | None = None,
        cache: CacheBackend | None = None,
        pending_session_execution: SessionPendingExecutionLookup[ExecutionRecord] | None = None,
    ) -> None:
        self._admission_execution = execution
        self._runtime_conversation = conversation
        self._runtime_executions = executions
        self._runtime_authorization = authorization
        self._runtime_cursor_signer = cursor_signer
        self._runtime_pending_session_execution = pending_session_execution
        super().__init__(
            conversation, executions, authorization, _SessionExecutionAdapter(execution), cursor_signer,
            enabled=enabled, release_terminal=release_terminal,
            canonicalize_path=None if workspace_access is None else workspace_access.canonicalize_path,
            pending_session_execution=pending_session_execution,
            read_session=self._read_session_record,
            observe_view=self._observe_session_view,
        )
        self._history_reader = history_reader
        self._transcript_store = transcript_store
        self._cache = cache if isinstance(cache, CacheBackend) else None
        self._display_cache = SessionDisplayCache(
            conversation, executions, transcript_store, cursor_signer, self._cache,
            now=lambda: time.time(),
        )

    async def history(
        self,
        session_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> "Page[SessionHistoryItem]":
        async with self.consume_session(session_id, principal.tenant_id):
            record = await self.read_authorized(
                session_id,
                principal,
                AuthorizationAction.SESSION_READ,
            )
            continuation = (
                None if record.continuation is None else record.continuation.agent_run_id
            )
            continuation_history_id = (
                None
                if record.continuation is None
                else record.continuation.history_id or record.history_id
            )
            return await self._history_reader.history(
                session_id,
                tenant_id=principal.tenant_id,
                continuation_agent_run_id=continuation,
                continuation_history_id=continuation_history_id,
                cursor=cursor,
                limit=limit,
            )

    async def timeline(
        self,
        session_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> Page[SessionTurn]:
        async with self.consume_session(session_id, principal.tenant_id):
            if self._transcript_store is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            return await project_session_timeline(
                self._runtime_conversation,
                self._runtime_executions,
                self._runtime_authorization,
                self._runtime_cursor_signer,
                self._transcript_store,
                session_id,
                principal=principal,
                cursor=cursor,
                limit=limit,
                pending_session_execution=(
                    self._pending_display_execution if self._runtime_pending_session_execution is not None else None
                ),
            )

    async def refresh(
        self,
        session_id: str,
        *,
        principal: Principal,
        cursor: str | None = None,
        limit: int = 100,
        max_age: float = 0,
    ) -> SessionDetail:
        """Read a complete display snapshot; zero age retains strong refresh behavior.

        Positive ages allow stale display and delayed detection of independently
        removed or damaged bodies. Admission changes and terminal active executions
        invalidate a cached display. Authorization is checked on every call;
        resource identity and membership use the store's bounded record freshness.
        Control operations never use this cache.
        """
        if isinstance(max_age, bool) or not isinstance(max_age, (int, float)) or not math.isfinite(max_age) or max_age < 0:
            raise ValueError("max_age must be finite and non-negative")
        started = time.time()
        cache_key = None
        async with self.consume_session(
            session_id, principal.tenant_id,
            display_read=max_age > 0 and self._cache is not None,
        ):
            header = await self._runtime_conversation.sessions.get_header(
                session_id, tenant_id=principal.tenant_id,
            )
            if header is None:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            await self._runtime_authorization.authorize(
                principal, AuthorizationAction.SESSION_READ, header,
            )
            pending = await self._pending_display_execution(header)
            cacheable_request = (pending is None or cursor is None) and max_age > 0 and active_state_scope() is None and self._display_cache.supported()
            if cacheable_request:
                try:
                    validate_page_limit(limit)
                    if cursor is not None:
                        coordinate = json.loads(self._runtime_cursor_signer.decode(cursor).position)
                        cacheable_request = coordinate[0] == session_id
                except (AIError, TypeError, ValueError, IndexError, KeyError):
                    cacheable_request = False
            if cacheable_request:
                cache_key = "session-display:v1:" + canonical_sha256({
                    "namespace": self._runtime_conversation.sessions.namespace,
                    "principal": [principal.tenant_id, principal.principal_id, principal.kind],
                    "session_id": session_id, "cursor": cursor, "limit": limit,
                })
            if cache_key is not None:
                cached = await self._display_cache.read(
                    cache_key, header, principal, cursor=cursor, max_age=max_age, pending=pending,
                )
                if cached is not None:
                    return cached
            state = await capture_session_refresh(
                self._runtime_conversation, self._runtime_executions, session_id,
                tenant_id=principal.tenant_id, cursor=cursor,
                cursor_signer=self._runtime_cursor_signer, limit=limit,
                reconcile=self.reconcile_admission,
                transcript_store=self._transcript_store,
                validate_active=self.validate_admission,
                pending_execution=pending,
                authorized_header=header if self._runtime_pending_session_execution is not None else None,
            )
            record = state.root
            session = await self.project_view(
                record, principal, active=None,
                authorized_header=header if cache_key is not None else None,
            )
            admitted = self.validate_admission(
                record,
                None if record.active_execution_id is None
                else state.executions.get(record.active_execution_id),
            )
            active = None if admitted is None else admitted.execution_id
            if state.pending_execution is not None:
                active = state.pending_execution.execution_id
            session = replace(session, active_execution_id=active)
            if self._transcript_store is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            with runtime_object_source_reads():
                timeline = await project_captured_session_timeline(
                    state, self._transcript_store, self._runtime_cursor_signer,
                    session_id=session_id, tenant_id=principal.tenant_id,
                    transcript_tenant_id=self._runtime_conversation.sessions.tenant_id,
                )
            detail = SessionDetail(session, timeline)
            if cache_key is not None and record.timeline_parent_session_id is None:
                await self._display_cache.write(cache_key, state, detail, started, max_age)
            return detail

    async def load_model_context(
        self,
        session_id: str,
        *,
        principal: Principal,
    ) -> tuple[object, ...]:
        async with self.consume_session(session_id, principal.tenant_id):
            record = await self.read_authorized(
                session_id, principal, AuthorizationAction.SESSION_READ
            )
            if record.history_quality == "unavailable":
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            if self._transcript_store is None or record.continuation is None:
                return ()
            history_id = record.continuation.history_id or record.history_id
            return await self._transcript_store.load_conversation_model_context(
                history_id=history_id,
                agent_run_id=record.continuation.agent_run_id,
                tenant_id=self._runtime_conversation.sessions.tenant_id,
                message_count=record.continuation.message_count,
            )

    async def _iter_session_messages(
        self,
        session_id: str,
        *,
        principal: Principal,
    ) -> AsyncIterator[object]:
        async with self.consume_session(session_id, principal.tenant_id):
            record = await self.read_authorized(
                session_id, principal, AuthorizationAction.SESSION_READ
            )
            if self._transcript_store is None or record.continuation is None:
                return
            history_id = record.continuation.history_id or record.history_id
            message_count = record.continuation.message_count
            if message_count is None:
                messages = self._transcript_store.iter_conversation_messages(
                    history_id=history_id,
                    agent_run_id=record.continuation.agent_run_id,
                    tenant_id=self._runtime_conversation.sessions.tenant_id,
                )
            else:
                messages = self._transcript_store.iter_conversation_message_range(
                    history_id=history_id,
                    agent_run_id=record.continuation.agent_run_id,
                    tenant_id=self._runtime_conversation.sessions.tenant_id,
                    start=0,
                    end=message_count,
                )
            async for message in messages:
                yield message

    def iter_session_messages(
        self,
        session_id: str,
        *,
        principal: Principal,
    ) -> AsyncIterator[object]:
        return self._iter_session_messages(session_id, principal=principal)















    async def _read_session_record(self, session_id: str, *, tenant_id: str) -> SessionRecord | None:
        repository = self._runtime_conversation.sessions
        return await repository.state_store.read(
            lambda _transaction: repository.get(session_id, tenant_id=tenant_id)
        )


    async def _pending_display_execution(self, header: ResourceRef) -> ExecutionRecord | None:
        if self._runtime_pending_session_execution is None:
            return None
        pending = await self._runtime_pending_session_execution(header.id, tenant_id=header.tenant_id)
        if (pending is None or pending.session_id != header.id
                or pending.principal_id != header.owner_principal_id
                or pending.parent_execution_id is not None
                or pending.result is not None
                or pending.status not in {ExecutionStatus.PENDING_START, ExecutionStatus.STARTED, ExecutionStatus.CANCELLING}):
            return None
        return pending

    async def _observe_session_view(
        self, record: SessionRecord, principal: Principal, *, active: str | None,
    ) -> SessionView:
        if record.active_execution_id is not None or record.status is not SessionStatus.OPEN:
            return await self.project_view(record, principal, active=active)
        header = ResourceRef(ResourceKind.SESSION, record.session_id, principal.tenant_id, record.owner_principal_id)
        pending = await self._pending_display_execution(header)
        if pending is None:
            return await self.project_view(record, principal, active=active)

        async def read() -> SessionView:
            # Observation may settle a turn. Re-read committed metadata before
            # using it; display reads retain the store's bounded freshness,
            # while control/admission reads stay authoritative.
            current = await self._runtime_conversation.sessions.get(record.session_id, tenant_id=principal.tenant_id)
            if current is None:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            view = await self.project_view(current, principal, authorized_header=header)
            if (current.status is SessionStatus.OPEN and current.active_execution_id is None
                    and pending.created_at >= current.created_at
                    and await self._runtime_executions.get(pending.execution_id, tenant_id=principal.tenant_id) is None):
                return replace(view, active_execution_id=pending.execution_id)
            return view

        sessions, executions = self._runtime_conversation.sessions, self._runtime_executions
        if (self._cache is not None and self._display_cache.supported()
                and active_state_scope() is None):
            return await read()
        if (type(sessions) is SessionRepositoryImpl and type(executions) is ExecutionRepositoryImpl
                and sessions.state_store.storage_group is executions.state_store.storage_group):
            return await sessions.state_store.storage_group.read_many(
                (sessions.state_store, executions.state_store), lambda _group: read(),
            )
        return await read()













__all__ = ["RuntimeSessionService"]
