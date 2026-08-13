#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session query API and persistence-backed session service."""

import asyncio
import json
import time
from binascii import Error as Base64Error
from dataclasses import replace
from datetime import datetime, timezone
from typing import Protocol

from linktools.core import environ

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
    ResourceKind,
    ResourceRef,
    SessionStatus,
    canonical_sha256,
    idempotency_key_hash,
)
from ..errors import AIError, ErrorCode
from ._persistence import (
    OperationLedgerInput,
    OperationLedgerRecord,
    RuntimeStores,
    SessionRecord,
)
from ._services import (
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
    SessionView,
    UpdateSessionRequest,
)

_logger = environ.get_logger("ai.runtime.session")


class SessionQueryApi(Protocol):
    async def get(self, session_id: str, *, principal: Principal) -> SessionView: ...
    async def list(self, request: ListSessionRequest) -> 'Page[SessionView]': ...
    async def load(self, session_id: str, *, principal: Principal) -> LoadedSession: ...


class SessionApi(SessionQueryApi, Protocol):
    async def create(self, request: CreateSessionRequest) -> SessionView: ...
    async def resume(self, session_id: str, request: ResumeSessionRequest) -> ExecutionHandle: ...
    async def fork(self, session_id: str, request: ForkSessionRequest) -> SessionView: ...
    async def update(self, session_id: str, request: UpdateSessionRequest) -> SessionView: ...
    async def close(self, session_id: str, request: CloseSessionRequest) -> SessionView: ...


class DefaultSessionService:
    """Enforce session ownership, binding immutability, and revision CAS."""

    def __init__(self, persistence: RuntimeStores, authorization: AuthorizationPolicy, execution: ExecutionService, cursor_signer: CursorSigner) -> None:
        self._persistence = persistence
        self._authorization = authorization
        self._execution = execution
        self._cursor_signer = cursor_signer

    async def create(self, binding_digest: str, request: CreateSessionRequest) -> SessionView:
        resource = ResourceRef(ResourceKind.SESSION, request.session_id, request.principal.tenant_id, request.principal.principal_id)
        await self._authorization.authorize(request.principal, AuthorizationAction.SESSION_CREATE, resource)
        digest = canonical_sha256({"action": "session.create", "tenant_id": request.principal.tenant_id, "principal_id": request.principal.principal_id, "session_id": request.session_id, "binding": binding_digest, "metadata": dict(request.metadata)})
        operation = await self._begin_operation(request.idempotency_key, request.principal.tenant_id, ResourceKind.SESSION, request.session_id, OperationKind.SESSION_CREATE, digest)
        if operation.result_ref:
            current = await self._persistence.conversation.sessions.get(operation.result_ref, tenant_id=request.principal.tenant_id)
            if current is not None:
                return await self._view(current, request.principal)
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
            continuation=None,
        )
        try:
            await self._persistence.conversation.sessions.create(record)
        except AIError as error:
            if error.code is not ErrorCode.SESSION_CONFLICT:
                raise
            current = await self._persistence.conversation.sessions.get(request.session_id, tenant_id=request.principal.tenant_id)
            if current is None or current.owner_principal_id != request.principal.principal_id or current.binding_digest != binding_digest:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            record = current
        await self._complete_operation(operation, request.principal.tenant_id, record.session_id, canonical_sha256({"session_id": record.session_id, "revision": record.revision}))
        _logger.debug("session created: session=%s tenant=%s", record.session_id, request.principal.tenant_id)
        return await self._view(record, request.principal)

    async def get(self, session_id: str, *, principal: Principal) -> SessionView:
        record = await self._authorized(session_id, principal, AuthorizationAction.SESSION_READ)
        return await self._view(record, principal)

    async def list(self, request: ListSessionRequest) -> Page[SessionView]:
        await self._authorization.authorize(request.principal, AuthorizationAction.SESSION_READ, ResourceRef(ResourceKind.SESSION, "list", request.principal.tenant_id))
        records = await self._persistence.conversation.sessions.list(tenant_id=request.principal.tenant_id, owner_principal_id=request.principal.principal_id)
        if not 1 <= request.limit <= 200:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        snapshot = _session_snapshot(records)
        start = _cursor_start(request.cursor, snapshot, request.principal.tenant_id, request.principal.principal_id, records, self._cursor_signer)
        values = records[start:start + request.limit]
        views = tuple(
            await asyncio.gather(*(self._view(record, request.principal) for record in values))
        )
        next_cursor = _make_cursor(snapshot, request.principal.tenant_id, request.principal.principal_id, views[-1].session_id if len(records) > start + request.limit and views else None, self._cursor_signer)
        return Page(views, next_cursor)

    async def load(self, session_id: str, *, principal: Principal) -> LoadedSession:
        record = await self._authorized(session_id, principal, AuthorizationAction.SESSION_READ)
        executions = await self._persistence.execution.executions.list_by_session(session_id, tenant_id=principal.tenant_id)
        active = tuple(sorted(item.execution_id for item in executions if item.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}))
        return LoadedSession(await self._view(record, principal), active)

    async def resume(self, binding_digest: str, session_id: str, request: ResumeSessionRequest) -> ExecutionHandle:
        record = await self._authorized(session_id, request.principal, AuthorizationAction.SESSION_READ)
        await self._authorization.authorize(request.principal, AuthorizationAction.EXECUTION_RUN, ResourceRef(ResourceKind.EXECUTION, session_id, request.principal.tenant_id))
        if record.status is not SessionStatus.OPEN:
            raise AIError(ErrorCode.SESSION_CONFLICT)
        if record.binding_digest != binding_digest:
            raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)
        return await self._execution.run_for_session(
            binding_digest,
            session_id,
            ExecutionRequest(prompt=request.prompt, principal=request.principal, idempotency_key=request.idempotency_key, memory_scope=request.memory_scope),
        )

    async def fork(self, binding_digest: str, session_id: str, request: ForkSessionRequest) -> SessionView:
        source = await self._authorized(session_id, request.principal, AuthorizationAction.SESSION_READ)
        await self._authorization.authorize(request.principal, AuthorizationAction.SESSION_CREATE, ResourceRef(ResourceKind.SESSION, request.new_session_id, request.principal.tenant_id, request.principal.principal_id))
        digest = canonical_sha256({"action": "session.fork", "tenant_id": request.principal.tenant_id, "principal_id": request.principal.principal_id, "source": session_id, "target": request.new_session_id, "binding": source.binding_digest})
        operation = await self._begin_operation(request.idempotency_key, request.principal.tenant_id, ResourceKind.SESSION, request.new_session_id, OperationKind.SESSION_FORK, digest)
        if operation.result_ref:
            current = await self._persistence.conversation.sessions.get(operation.result_ref, tenant_id=request.principal.tenant_id)
            if current is not None:
                return await self._view(current, request.principal)
        if source.binding_digest != binding_digest:
            raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)
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
            continuation=source.continuation,
        )
        try:
            await self._persistence.conversation.sessions.create(target)
        except AIError as error:
            if error.code is not ErrorCode.SESSION_CONFLICT:
                raise
            existing_target = await self._persistence.conversation.sessions.get(request.new_session_id, tenant_id=request.principal.tenant_id)
            if (
                existing_target is None
                or existing_target.tenant_id != target.tenant_id
                or existing_target.owner_principal_id != target.owner_principal_id
                or existing_target.binding_digest != target.binding_digest
                or existing_target.status is not SessionStatus.OPEN
                or existing_target.revision != 0
                or existing_target.resource_generation != 0
            ):
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT) from error
            target = existing_target
        await self._complete_operation(operation, request.principal.tenant_id, target.session_id, canonical_sha256({"session_id": target.session_id, "revision": target.revision}))
        _logger.debug("session forked: source=%s target=%s", session_id, target.session_id)
        return await self._view(target, request.principal)

    async def update(self, binding_digest: str, session_id: str, request: UpdateSessionRequest) -> SessionView:
        current = await self._authorized(session_id, request.principal, AuthorizationAction.SESSION_UPDATE)
        if current.binding_digest != binding_digest:
            raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)
        if any(key.startswith("linktools.ai.") and current.metadata.get(key) != value for key, value in request.metadata.items()):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if any(key.startswith("linktools.ai.") for key in current.metadata if key not in request.metadata):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        digest = canonical_sha256({"action": "session.update", "tenant_id": request.principal.tenant_id, "principal_id": request.principal.principal_id, "session_id": session_id, "expected_revision": request.expected_revision, "metadata": request.metadata, "cwd": request.cwd})
        operation = await self._begin_operation(request.idempotency_key, request.principal.tenant_id, ResourceKind.SESSION, session_id, OperationKind.SESSION_UPDATE, digest)
        if operation.result_ref:
            updated = await self._persistence.conversation.sessions.get(session_id, tenant_id=request.principal.tenant_id)
            if updated is not None:
                return await self._view(updated, request.principal)
        requested_cwd = request.cwd if request.cwd is not None else current.cwd
        if (
            operation.status is OperationStatus.PENDING
            and current.revision == request.expected_revision + 1
            and current.resource_generation == request.expected_revision + 1
            and current.status is SessionStatus.OPEN
            and current.metadata == request.metadata
            and current.cwd == requested_cwd
        ):
            await self._complete_operation(operation, request.principal.tenant_id, session_id, canonical_sha256({"session_id": session_id, "revision": current.revision}))
            _logger.debug("session update reconciled: session=%s revision=%s", session_id, current.revision)
            return await self._view(current, request.principal)
        if current.status is SessionStatus.CLEANUP_REQUIRED:
            raise AIError(ErrorCode.SESSION_CLEANUP_REQUIRED)
        if current.revision != request.expected_revision or current.status is not SessionStatus.OPEN:
            raise AIError(ErrorCode.SESSION_REVISION_CONFLICT)
        now = datetime.now(timezone.utc)
        next_record = replace(current, revision=current.revision + 1, resource_generation=current.resource_generation + 1, cwd=requested_cwd, metadata=request.metadata, updated_at=now)
        updated = await self._persistence.conversation.sessions.compare_and_swap(session_id, tenant_id=request.principal.tenant_id, expected_revision=request.expected_revision, next_record=next_record)
        await self._complete_operation(operation, request.principal.tenant_id, session_id, canonical_sha256({"session_id": session_id, "revision": updated.revision}))
        _logger.debug("session updated: session=%s revision=%s", session_id, updated.revision)
        return await self._view(updated, request.principal)

    async def close(self, session_id: str, request: CloseSessionRequest) -> SessionView:
        current = await self._authorized(session_id, request.principal, AuthorizationAction.SESSION_CLOSE)
        digest = canonical_sha256({"action": "session.close", "tenant_id": request.principal.tenant_id, "principal_id": request.principal.principal_id, "session_id": session_id, "force": request.force, "wait_timeout_seconds": request.wait_timeout_seconds})
        operation = await self._begin_operation(request.idempotency_key, request.principal.tenant_id, ResourceKind.SESSION, session_id, OperationKind.SESSION_CLOSE, digest)
        if operation.result_ref:
            closed = await self._persistence.conversation.sessions.get(session_id, tenant_id=request.principal.tenant_id)
            if closed is not None:
                return await self._view(closed, request.principal)
        executions = await self._active_executions(session_id, request.principal.tenant_id)
        active = executions
        if active and not request.force:
            raise AIError(ErrorCode.SESSION_ACTIVE_EXECUTIONS)
        if current.status is SessionStatus.CLOSED:
            await self._complete_operation(operation, request.principal.tenant_id, session_id, canonical_sha256({"session_id": session_id, "revision": current.revision}))
            return await self._view(current, request.principal)
        if current.status is SessionStatus.CLEANUP_REQUIRED and not request.force:
            raise AIError(ErrorCode.SESSION_CLEANUP_REQUIRED)
        if current.status is SessionStatus.OPEN:
            now = datetime.now(timezone.utc)
            current = await self._persistence.conversation.sessions.compare_and_swap(
                session_id,
                tenant_id=request.principal.tenant_id,
                expected_revision=current.revision,
                next_record=replace(current, status=SessionStatus.CLOSING, revision=current.revision + 1, updated_at=now),
            )
        if active and request.force:
            for execution in active:
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
                    self._wait_for_no_active(session_id, request.principal.tenant_id),
                    timeout=request.wait_timeout_seconds,
                )
            except asyncio.TimeoutError as error:
                now = datetime.now(timezone.utc)
                cleanup = replace(current, status=SessionStatus.CLEANUP_REQUIRED, revision=current.revision + 1, updated_at=now)
                await self._persistence.conversation.sessions.compare_and_swap(
                    session_id,
                    tenant_id=request.principal.tenant_id,
                    expected_revision=current.revision,
                    next_record=cleanup,
                )
                raise AIError(ErrorCode.SESSION_CLEANUP_REQUIRED) from error
        now = datetime.now(timezone.utc)
        closing = replace(current, status=SessionStatus.CLOSED, revision=current.revision + 1, updated_at=now, closed_at=now)
        updated = await self._persistence.conversation.sessions.compare_and_swap(session_id, tenant_id=request.principal.tenant_id, expected_revision=current.revision, next_record=closing)
        await self._complete_operation(operation, request.principal.tenant_id, session_id, canonical_sha256({"session_id": session_id, "revision": updated.revision}))
        _logger.debug("session closed: session=%s revision=%s force=%s", session_id, updated.revision, request.force)
        return await self._view(updated, request.principal)

    async def _active_executions(self, session_id: str, tenant_id: str) -> tuple:
        records = await self._persistence.execution.executions.list_by_session(session_id, tenant_id=tenant_id)
        return tuple(
            record
            for record in records
            if record.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}
        )

    async def _wait_for_no_active(self, session_id: str, tenant_id: str) -> None:
        while await self._active_executions(session_id, tenant_id):
            await asyncio.sleep(0.05)

    async def _authorized(self, session_id: str, principal: Principal, action: AuthorizationAction) -> SessionRecord:
        header = await self._persistence.conversation.sessions.get_header(session_id, tenant_id=principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, action, header)
        record = await self._persistence.conversation.sessions.get(session_id, tenant_id=principal.tenant_id)
        if record is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return record

    async def _view(self, record: SessionRecord, principal: Principal) -> SessionView:
        executions = await self._persistence.execution.executions.list_by_session(record.session_id, tenant_id=principal.tenant_id)
        active = tuple(sorted(item.execution_id for item in executions if item.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}))
        return SessionView(record.session_id, record.binding_digest, record.status, record.revision, record.resource_generation, record.cwd, active, record.metadata)

    async def _begin_operation(self, operation_id: str, tenant_id: str, resource_kind: ResourceKind, resource_id: str, kind: OperationKind, request_digest: str) -> OperationLedgerRecord:
        operation_id = idempotency_key_hash(operation_id)
        for _ in range(4):
            existing = await self._persistence.conversation.operations.get(operation_id, tenant_id=tenant_id)
            if existing is not None:
                if existing.request_digest != request_digest:
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                return existing
            now = datetime.now(timezone.utc)
            record = OperationLedgerInput(operation_id, tenant_id, resource_kind, resource_id, None, kind, OperationStatus.PENDING, request_digest, None, None, None, True, now, now)
            try:
                return await self._persistence.conversation.operations.append(record)
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def _complete_operation(self, operation: OperationLedgerRecord, tenant_id: str, result_ref: str, result_digest: str) -> None:
        now = datetime.now(timezone.utc)
        completed = OperationLedgerRecord(operation.operation_id, tenant_id, operation.resource_kind, operation.resource_id, operation.execution_id, operation.kind, OperationStatus.SUCCEEDED, operation.request_digest, result_ref, result_digest, None, operation.compactable, operation.sequence, operation.created_at, now)
        try:
            await self._persistence.conversation.operations.compare_and_swap(operation.operation_id, tenant_id=tenant_id, expected_status=OperationStatus.PENDING, next_record=completed)
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._persistence.conversation.operations.get(operation.operation_id, tenant_id=tenant_id)
            if current is None or current.status is not OperationStatus.SUCCEEDED or current.result_digest != result_digest:
                raise


__all__ = ["DefaultSessionService", "SessionApi", "SessionQueryApi"]


def _session_snapshot(records: tuple[SessionRecord, ...]) -> int:
    return int(canonical_sha256([(record.session_id, record.revision, record.resource_generation, record.status.value) for record in records])[:16], 16)


def _make_cursor(snapshot: int, tenant_id: str, owner_principal_id: str, session_id: "str | None", signer: CursorSigner) -> "str | None":
    if session_id is None:
        return None
    return signer.encode(CursorPayload(1, tenant_id, "SESSION", canonical_sha256({"owner_principal_id": owner_principal_id}), session_id, snapshot, int(time.time()) + 3600))


def _cursor_start(cursor: "str | None", snapshot: int, tenant_id: str, owner_principal_id: str, records: "tuple[SessionRecord, ...]", signer: CursorSigner) -> int:
    if cursor is None:
        return 0
    try:
        payload = signer.decode(cursor)
        if payload.cursor_version != 1 or payload.tenant_id != tenant_id or payload.resource_kind != "SESSION" or payload.filter_digest != canonical_sha256({"owner_principal_id": owner_principal_id}) or payload.snapshot_or_store_revision != snapshot:
            raise ValueError("session cursor identity mismatch")
        if not payload.sort_key.strip():
            raise ValueError("session cursor sort key is empty")
        return next((index + 1 for index, item in enumerate(records) if item.session_id == payload.sort_key), len(records))
    except (Base64Error, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    except AIError as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
