#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only Runtime composition for persisted execution history."""

import heapq
from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime

from linktools.core import environ

from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    ExecutionLineageKind,
    ExecutionStatus,
    HmacCursorSigner,
    JsonValue,
    Page,
    Principal,
    ResourceKind,
    ResourceRef,
    SessionStatus,
    TenantAuthorizationPolicy,
    validate_page_limit,
    validate_persistence_namespace,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode, ErrorDiagnostics
from ._factory import _grant_key
from ._history import StepExecutionHistoryReader
from ._history_service import DefaultExecutionHistoryService
from .service_api import (
    ExecutionHistoryItem,
    ExecutionHistoryService,
    ExecutionTraceItem,
    ExecutionView,
    ListExecutionRequest,
    ModelInteractionItem,
    TranscriptItem,
)
from .state import RuntimeDomain, RuntimeState
from .state._contracts import (
    ExecutionRecord,
    ExecutionRepository,
    SessionRecord,
    SessionRepository,
)

_logger = environ.get_logger("ai.runtime.history")


@dataclass(frozen=True, slots=True)
class ExecutionInfo:
    """Execution metadata required by local diagnostics."""

    execution_id: str
    agent_id: str
    status: ExecutionStatus
    lineage_kind: ExecutionLineageKind
    parent_execution_id: str | None
    root_execution_id: str
    parent_invocation_id: str | None
    session_id: str | None
    created_at: datetime
    updated_at: datetime
    error_code: str | None
    safe_error_details: Mapping[str, JsonValue] = field(default_factory=dict)
    error_diagnostics: ErrorDiagnostics | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "safe_error_details", dict(self.safe_error_details))


@dataclass(frozen=True, slots=True)
class SessionInfo:
    """Session metadata required by local diagnostics."""

    session_id: str
    agent_id: str
    status: SessionStatus
    revision: int
    cwd: str | None
    active_execution_id: str | None
    history_quality: str
    created_at: datetime
    updated_at: datetime


def _project_session_info(record: SessionRecord) -> SessionInfo:
    return SessionInfo(
        session_id=record.session_id,
        agent_id=record.agent_id,
        status=record.status,
        revision=record.revision,
        cwd=record.cwd,
        active_execution_id=record.active_execution_id,
        history_quality=record.history_quality,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _project_execution_info(record: ExecutionRecord) -> ExecutionInfo:
    return ExecutionInfo(
        execution_id=record.execution_id,
        agent_id=record.agent_id,
        status=record.status,
        lineage_kind=record.lineage_kind,
        parent_execution_id=record.parent_execution_id,
        root_execution_id=record.root_execution_id,
        parent_invocation_id=record.parent_invocation_id,
        session_id=record.session_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
        error_code=record.error_code,
        safe_error_details=record.safe_error_details,
        error_diagnostics=record.error_diagnostics,
    )


class RuntimeHistory:
    """Stable read-only composition for persisted execution projections."""

    def __init__(
        self,
        service: ExecutionHistoryService,
        *,
        tenant_id: str,
        executions: "ExecutionRepository | None" = None,
        sessions: "SessionRepository | None" = None,
        authorization: "AuthorizationPolicy | None" = None,
    ) -> None:
        self._service = service
        self._tenant_id = tenant_id
        self._executions = executions
        self._sessions = sessions
        self._authorization = authorization

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    async def inspect_execution(
        self, execution_id: str, *, principal: Principal
    ) -> ExecutionInfo:
        return _project_execution_info(
            await self._authorized_record(execution_id, principal)
        )

    async def recent_executions(
        self,
        *,
        principal: Principal,
        limit: int = 20,
    ) -> tuple[ExecutionInfo, ...]:
        """Return the exact newest executions with O(limit) memory."""
        validate_page_limit(limit)
        executions, authorization = self._require_direct_reader()
        recent: list[tuple[datetime, str, ExecutionInfo]] = []
        cursor: str | None = None

        while True:
            page = await executions.list_candidates(
                tenant_id=principal.tenant_id,
                session_id=None,
                parent_execution_id=None,
                cursor=cursor,
                limit=1000,
            )
            if not page.items:
                break
            for candidate in page.items:
                record = candidate.record
                resource = ResourceRef(
                    ResourceKind.EXECUTION,
                    record.execution_id,
                    principal.tenant_id,
                )
                try:
                    await authorization.authorize(
                        principal,
                        AuthorizationAction.EXECUTION_READ,
                        resource,
                    )
                except AIError as error:
                    if error.code is ErrorCode.AUTHORIZATION_DENIED:
                        continue
                    raise
                info = _project_execution_info(record)
                entry = (record.created_at, record.execution_id, info)
                if len(recent) < limit:
                    heapq.heappush(recent, entry)
                elif entry[:2] > recent[0][:2]:
                    heapq.heapreplace(recent, entry)
            cursor = page.items[-1].cursor
            if not page.has_more:
                break

        recent.sort(key=lambda value: (value[0], value[1]), reverse=True)
        return tuple(value[2] for value in recent)

    async def list_executions(
        self, request: ListExecutionRequest
    ) -> Page[ExecutionView]:
        return await self._service.list(request)

    async def recent_sessions(
        self,
        *,
        principal: Principal,
        limit: int = 20,
    ) -> tuple[SessionInfo, ...]:
        validate_page_limit(limit)
        sessions, authorization = self._require_session_reader()
        recent: list[tuple[datetime, str, SessionInfo]] = []
        cursor: str | None = None
        snapshot: int | None = None
        while True:
            snapshot, page = await sessions.list_page(
                tenant_id=principal.tenant_id,
                owner_principal_id=principal.principal_id,
                cursor=cursor,
                limit=1000,
                snapshot=snapshot,
            )
            for record in page.items:
                resource = ResourceRef(
                    ResourceKind.SESSION,
                    record.session_id,
                    record.tenant_id,
                    record.owner_principal_id,
                )
                try:
                    await authorization.authorize(
                        principal,
                        AuthorizationAction.SESSION_READ,
                        resource,
                    )
                except AIError as error:
                    if error.code is ErrorCode.AUTHORIZATION_DENIED:
                        continue
                    raise
                info = _project_session_info(record)
                entry = (record.updated_at, record.session_id, info)
                if len(recent) < limit:
                    heapq.heappush(recent, entry)
                elif entry[:2] > recent[0][:2]:
                    heapq.heapreplace(recent, entry)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        recent.sort(key=lambda value: (value[0], value[1]), reverse=True)
        return tuple(value[2] for value in recent)

    async def inspect_session(
        self,
        session_id: str,
        *,
        principal: Principal,
    ) -> SessionInfo:
        sessions, authorization = self._require_session_reader()
        header = await sessions.get_header(
            session_id,
            tenant_id=principal.tenant_id,
        )
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await authorization.authorize(
            principal,
            AuthorizationAction.SESSION_READ,
            header,
        )
        record = await sessions.get(
            session_id,
            tenant_id=principal.tenant_id,
        )
        if record is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return _project_session_info(record)

    @classmethod
    def open(
        cls,
        namespace: str,
        *,
        state: RuntimeState,
        tenant_id: "str | None" = None,
        authorization: "AuthorizationPolicy | None" = None,
    ) -> AbstractAsyncContextManager["RuntimeHistory"]:
        return _open_runtime_history(
            namespace,
            state=state,
            tenant_id=tenant_id,
            authorization=authorization,
        )

    async def history(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> Page[ExecutionHistoryItem]:
        return await self._service.history(
            execution_id,
            principal=principal,
            cursor=cursor,
            limit=limit,
        )

    async def trace(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> Page[ExecutionTraceItem]:
        return await self._service.trace(
            execution_id,
            principal=principal,
            cursor=cursor,
            limit=limit,
        )

    async def transcript(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> Page[TranscriptItem]:
        return await self._service.transcript(
            execution_id,
            principal=principal,
            cursor=cursor,
            limit=limit,
        )

    async def model_interactions(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> Page[ModelInteractionItem]:
        return await self._service.model_interactions(
            execution_id,
            principal=principal,
            cursor=cursor,
            limit=limit,
        )

    def _require_session_reader(
        self,
    ) -> tuple[SessionRepository, AuthorizationPolicy]:
        if self._sessions is None or self._authorization is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._sessions, self._authorization

    def _require_direct_reader(
        self,
    ) -> tuple[ExecutionRepository, AuthorizationPolicy]:
        if self._executions is None or self._authorization is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._executions, self._authorization

    async def _authorized_record(
        self,
        execution_id: str,
        principal: Principal,
    ) -> ExecutionRecord:
        executions, authorization = self._require_direct_reader()
        header = await executions.get_header(
            execution_id,
            tenant_id=principal.tenant_id,
        )
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await authorization.authorize(
            principal,
            AuthorizationAction.EXECUTION_READ,
            header,
        )
        record = await executions.get(
            execution_id,
            tenant_id=principal.tenant_id,
        )
        if record is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return record


@asynccontextmanager
async def _open_runtime_history(
    namespace: str,
    *,
    state: RuntimeState,
    tenant_id: "str | None",
    authorization: "AuthorizationPolicy | None",
) -> AsyncIterator[RuntimeHistory]:
    resolved_namespace = validate_persistence_namespace(namespace)
    effective_tenant_id = (
        "default" if tenant_id is None else validate_tenant_id(tenant_id)
    )
    if not isinstance(state, RuntimeState):
        raise TypeError("state must be RuntimeState")
    selected_state = state
    initialized = False
    body_error: BaseException | None = None
    try:
        await selected_state.initialize(
            namespace=resolved_namespace,
            tenant_id=effective_tenant_id,
        )
        initialized = True
        if (
            selected_state.namespace != resolved_namespace
            or selected_state.tenant_id != effective_tenant_id
        ):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        reader = StepExecutionHistoryReader(
            namespace=resolved_namespace,
            executions=selected_state.execution.executions,
            store=selected_state.steps.read_store(RuntimeDomain.EXECUTION),
            cursor_signer=HmacCursorSigner(
                "execution-history",
                _grant_key(resolved_namespace),
            ),
        )
        effective_authorization = (
            TenantAuthorizationPolicy(effective_tenant_id)
            if authorization is None
            else authorization
        )
        service = DefaultExecutionHistoryService(
            selected_state.execution.executions,
            effective_authorization,
            reader,
            cursor_signer=HmacCursorSigner("execution", _grant_key(resolved_namespace)),
        )
        yield RuntimeHistory(
            service,
            tenant_id=effective_tenant_id,
            executions=selected_state.execution.executions,
            sessions=selected_state.conversation.sessions,
            authorization=effective_authorization,
        )
    except BaseException as error:
        body_error = error
        raise
    finally:
        if initialized:
            try:
                await selected_state.close()
            except BaseException as error:
                if body_error is None:
                    raise
                _log_secondary_cleanup("history.close", error)


def _log_secondary_cleanup(phase: str, error: BaseException) -> None:
    code = error.code.value if isinstance(error, AIError) else None
    _logger.error(
        "secondary cleanup failed: phase=%s code=%s exception_type=%s",
        phase,
        code,
        type(error).__name__,
    )


__all__ = ["ExecutionInfo", "RuntimeHistory", "SessionInfo"]
