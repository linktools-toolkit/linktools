#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authorized read-only execution query and history service."""

import time

from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    CursorPayload,
    CursorSigner,
    Page,
    Principal,
    ResourceKind,
    ResourceRef,
    canonical_sha256,
    principal_identity_payload,
)
from ..errors import AIError, ErrorCode
from .service_api import (
    ExecutionHistoryItem,
    ExecutionHistoryReader,
    ExecutionTraceItem,
    ExecutionView,
    ListExecutionRequest,
    TranscriptItem,
    _project_execution_view,
)
from .state._contracts import ExecutionRecord, ExecutionRepository


class DefaultExecutionHistoryService:
    """Expose execution history projections through the Runtime auth boundary."""

    def __init__(
        self,
        executions: ExecutionRepository,
        authorization: AuthorizationPolicy,
        reader: ExecutionHistoryReader,
        cursor_signer: "CursorSigner | None" = None,
    ) -> None:
        self._executions = executions
        self._authorization = authorization
        self._reader = reader
        self._cursor_signer = cursor_signer

    async def inspect(
        self, execution_id: str, *, principal: Principal
    ) -> ExecutionView:
        return _project_execution_view(await self._authorize(execution_id, principal))

    async def list(self, request: ListExecutionRequest) -> Page[ExecutionView]:
        signer = self._cursor_signer
        if signer is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        filter_digest = _execution_filter_digest(request)
        repository_cursor = _decode_execution_cursor(
            request.cursor,
            tenant_id=request.principal.tenant_id,
            filter_digest=filter_digest,
            signer=signer,
        )
        views: list[ExecutionView] = []
        scanned = 0
        while scanned < 1000:
            page = await self._executions.list_candidates(
                tenant_id=request.principal.tenant_id,
                session_id=request.session_id,
                parent_execution_id=request.parent_execution_id,
                cursor=repository_cursor,
                limit=1000 - scanned,
            )
            if not page.items:
                break
            for index, candidate in enumerate(page.items):
                scanned += 1
                record = candidate.record
                if request.session_id is not None and (
                    record.session_id != request.session_id
                ):
                    continue
                if request.parent_execution_id is not None and (
                    record.parent_execution_id != request.parent_execution_id
                ):
                    continue
                if request.agent_id is not None and record.agent_id != request.agent_id:
                    continue
                resource = ResourceRef(
                    ResourceKind.EXECUTION,
                    record.execution_id,
                    request.principal.tenant_id,
                )
                try:
                    await self._authorization.authorize(
                        request.principal,
                        AuthorizationAction.EXECUTION_READ,
                        resource,
                    )
                except AIError as error:
                    if error.code is not ErrorCode.AUTHORIZATION_DENIED:
                        raise
                    continue
                views.append(_project_execution_view(record))
                if len(views) == request.limit:
                    has_more = page.has_more or index + 1 < len(page.items)
                    return Page(
                        tuple(views),
                        _encode_execution_cursor(
                            request.principal.tenant_id,
                            filter_digest,
                            candidate.cursor,
                            signer,
                        )
                        if has_more
                        else None,
                    )
            repository_cursor = page.items[-1].cursor
            if not page.has_more:
                break
        next_cursor = None
        if scanned >= 1000 and page.items and page.has_more:
            next_cursor = _encode_execution_cursor(
                request.principal.tenant_id,
                filter_digest,
                page.items[-1].cursor,
                signer,
            )
        return Page(tuple(views), next_cursor)

    async def trace(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> Page[ExecutionTraceItem]:
        record = await self._authorize(execution_id, principal)
        return await self._reader.trace(
            execution_id,
            tenant_id=record.tenant_id,
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
        record = await self._authorize(execution_id, principal)
        return await self._reader.transcript(
            execution_id,
            tenant_id=record.tenant_id,
            cursor=cursor,
            limit=limit,
        )

    async def history(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> Page[ExecutionHistoryItem]:
        record = await self._authorize(execution_id, principal)
        return await self._reader.history(
            execution_id,
            tenant_id=record.tenant_id,
            cursor=cursor,
            limit=limit,
        )

    async def _authorize(
        self,
        execution_id: str,
        principal: Principal,
    ) -> ExecutionRecord:
        tenant_id = principal.tenant_id
        header = await self._executions.get_header(
            execution_id,
            tenant_id=tenant_id,
        )
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(
            principal,
            AuthorizationAction.EXECUTION_READ,
            header,
        )
        record = await self._executions.get(
            execution_id,
            tenant_id=tenant_id,
        )
        if record is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return record


def _execution_filter_digest(request: ListExecutionRequest) -> str:
    return canonical_sha256(
        {
            "principal": principal_identity_payload(request.principal),
            "session_id": request.session_id,
            "agent_id": request.agent_id,
            "parent_execution_id": request.parent_execution_id,
        }
    )


def _encode_execution_cursor(
    tenant_id: str,
    filter_digest: str,
    repository_cursor: str,
    signer: CursorSigner,
) -> str:
    return signer.encode(
        CursorPayload(
            1,
            tenant_id,
            "EXECUTION",
            filter_digest,
            repository_cursor,
            0,
            int(time.time()) + 3600,
        )
    )


def _decode_execution_cursor(
    cursor: str | None,
    *,
    tenant_id: str,
    filter_digest: str,
    signer: CursorSigner,
) -> str | None:
    if cursor is None:
        return None
    try:
        payload = signer.decode(cursor)
    except AIError as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    if (
        payload.cursor_version != 1
        or payload.tenant_id != tenant_id
        or payload.resource_kind != "EXECUTION"
        or payload.filter_digest != filter_digest
        or payload.snapshot_or_store_revision != 0
        or not payload.sort_key
    ):
        raise AIError(ErrorCode.CURSOR_INVALID)
    return payload.sort_key


__all__ = ["DefaultExecutionHistoryService"]
