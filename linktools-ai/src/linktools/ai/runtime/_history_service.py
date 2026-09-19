#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authorized read-only execution query and history service."""

from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    CursorSigner,
    Page,
    Principal,
    ResourceKind,
    ResourceRef,
    canonical_sha256,
    principal_identity_payload,
)
from ..errors import AIError, ErrorCode
from ._cursor import decode_cursor as decode_runtime_cursor
from ._cursor import encode_cursor as encode_runtime_cursor
from .service_api import (
    ExecutionHistoryItem,
    ExecutionHistoryReader,
    ExecutionTraceItem,
    ExecutionView,
    ListExecutionRequest,
    ModelInteractionItem,
    TranscriptItem,
    project_execution_view,
)
from .state._contracts import ExecutionRecord, ExecutionRepository

_CURSOR_RESOURCE_KIND = "EXECUTION"


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
        return project_execution_view(await self._authorize(execution_id, principal))

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
                views.append(project_execution_view(record))
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
        include_content: bool = False,
        limit: int = 100,
    ) -> Page[ExecutionTraceItem]:
        record = await self._authorize(execution_id, principal)
        inner_cursor = self._decode_content_cursor(
            cursor,
            execution_id=execution_id,
            tenant_id=record.tenant_id,
            query_kind="trace",
            include_content=include_content,
        )
        page = await self._reader.trace(
            execution_id,
            tenant_id=record.tenant_id,
            cursor=inner_cursor,
            limit=limit,
        )
        return Page(
            page.items,
            self._encode_content_cursor(
                page.next_cursor,
                execution_id=execution_id,
                tenant_id=record.tenant_id,
                query_kind="trace",
                include_content=include_content,
            ),
        )

    async def transcript(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        include_content: bool = False,
        limit: int = 100,
    ) -> Page[TranscriptItem]:
        record = await self._authorize(execution_id, principal)
        inner_cursor = self._decode_content_cursor(
            cursor,
            execution_id=execution_id,
            tenant_id=record.tenant_id,
            query_kind="transcript",
            include_content=include_content,
        )
        page = await self._reader.transcript(
            execution_id,
            tenant_id=record.tenant_id,
            cursor=inner_cursor,
            limit=limit,
        )
        items = (
            page.items
            if include_content
            else tuple(
                TranscriptItem(
                    item.execution_id,
                    item.sequence,
                    None,
                    False,
                )
                for item in page.items
            )
        )
        return Page(
            items,
            self._encode_content_cursor(
                page.next_cursor,
                execution_id=execution_id,
                tenant_id=record.tenant_id,
                query_kind="transcript",
                include_content=include_content,
            ),
        )

    async def history(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        include_content: bool = False,
        limit: int = 100,
    ) -> Page[ExecutionHistoryItem]:
        record = await self._authorize(execution_id, principal)
        inner_cursor = self._decode_content_cursor(
            cursor,
            execution_id=execution_id,
            tenant_id=record.tenant_id,
            query_kind="history",
            include_content=include_content,
        )
        page = await self._reader.history(
            execution_id,
            tenant_id=record.tenant_id,
            cursor=inner_cursor,
            limit=limit,
        )
        items = (
            page.items
            if include_content
            else tuple(
                ExecutionHistoryItem(
                    item.execution_id,
                    item.sequence,
                    item.item_kind,
                    None,
                    item.tool_name,
                    item.tool_call_id,
                    False,
                )
                for item in page.items
            )
        )
        return Page(
            items,
            self._encode_content_cursor(
                page.next_cursor,
                execution_id=execution_id,
                tenant_id=record.tenant_id,
                query_kind="history",
                include_content=include_content,
            ),
        )

    async def model_interactions(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        include_content: bool = False,
        limit: int = 100,
    ) -> Page[ModelInteractionItem]:
        record = await self._authorize(execution_id, principal)
        inner_cursor = self._decode_content_cursor(
            cursor,
            execution_id=execution_id,
            tenant_id=record.tenant_id,
            query_kind="model_interactions",
            include_content=include_content,
        )
        page = await self._reader.model_interactions(
            execution_id,
            tenant_id=record.tenant_id,
            cursor=inner_cursor,
            limit=limit,
        )
        items = (
            page.items
            if include_content
            else tuple(
                ModelInteractionItem(
                    item.execution_id,
                    item.segment_sequence,
                    item.depth,
                    item.request_sequence,
                    item.purpose,
                    item.step_index,
                    item.output_retry_index,
                    item.model,
                    {},
                    None,
                    item.status,
                    item.error_code,
                    item.duration_ns,
                    item.usage,
                    False,
                )
                for item in page.items
            )
        )
        return Page(
            items,
            self._encode_content_cursor(
                page.next_cursor,
                execution_id=execution_id,
                tenant_id=record.tenant_id,
                query_kind="model_interactions",
                include_content=include_content,
            ),
        )

    def _content_filter_digest(
        self,
        execution_id: str,
        query_kind: str,
        include_content: bool,
    ) -> str:
        if not isinstance(include_content, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return canonical_sha256(
            {
                "execution_id": execution_id,
                "query_kind": query_kind,
                "include_content": include_content,
            }
        )

    def _decode_content_cursor(
        self,
        cursor: "str | None",
        *,
        execution_id: str,
        tenant_id: str,
        query_kind: str,
        include_content: bool,
    ) -> "str | None":
        if cursor is None:
            self._content_filter_digest(
                execution_id,
                query_kind,
                include_content,
            )
            return None
        signer = self._cursor_signer
        if signer is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        payload = decode_runtime_cursor(
            cursor,
            signer,
            tenant_id=tenant_id,
            resource_kind=f"EXECUTION_{query_kind.upper()}",
            filter_digest=self._content_filter_digest(
                execution_id,
                query_kind,
                include_content,
            ),
        )
        if payload.revision != 0:
            raise AIError(ErrorCode.CURSOR_INVALID)
        return payload.position

    def _encode_content_cursor(
        self,
        cursor: "str | None",
        *,
        execution_id: str,
        tenant_id: str,
        query_kind: str,
        include_content: bool,
    ) -> "str | None":
        if cursor is None:
            return None
        signer = self._cursor_signer
        if signer is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return encode_runtime_cursor(
            signer,
            tenant_id=tenant_id,
            resource_kind=f"EXECUTION_{query_kind.upper()}",
            filter_digest=self._content_filter_digest(
                execution_id,
                query_kind,
                include_content,
            ),
            position=cursor,
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
    return encode_runtime_cursor(
        signer,
        tenant_id=tenant_id,
        resource_kind=_CURSOR_RESOURCE_KIND,
        filter_digest=filter_digest,
        position=repository_cursor,
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
    payload = decode_runtime_cursor(
        cursor,
        signer,
        tenant_id=tenant_id,
        resource_kind=_CURSOR_RESOURCE_KIND,
        filter_digest=filter_digest,
    )
    if payload.revision != 0:
        raise AIError(ErrorCode.CURSOR_INVALID)
    return payload.position


__all__ = ["DefaultExecutionHistoryService"]
