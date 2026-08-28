#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authorized read-only execution history service."""

from ..core import AuthorizationAction, AuthorizationPolicy, Page, Principal
from ..errors import AIError, ErrorCode
from .service_api import (
    ExecutionHistoryItem,
    ExecutionHistoryReader,
    ExecutionTraceItem,
    TranscriptItem,
)
from .state import ExecutionRepository


class DefaultExecutionHistoryService:
    """Expose execution history projections through the Runtime auth boundary."""

    def __init__(
        self,
        executions: ExecutionRepository,
        authorization: AuthorizationPolicy,
        reader: ExecutionHistoryReader,
    ) -> None:
        self._executions = executions
        self._authorization = authorization
        self._reader = reader

    async def trace(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> Page[ExecutionTraceItem]:
        tenant_id = await self._authorize(execution_id, principal)
        return await self._reader.trace(
            execution_id,
            tenant_id=tenant_id,
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
        tenant_id = await self._authorize(execution_id, principal)
        return await self._reader.transcript(
            execution_id,
            tenant_id=tenant_id,
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
        tenant_id = await self._authorize(execution_id, principal)
        return await self._reader.history(
            execution_id,
            tenant_id=tenant_id,
            cursor=cursor,
            limit=limit,
        )

    async def _authorize(
        self,
        execution_id: str,
        principal: Principal,
    ) -> str:
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
        return record.tenant_id


__all__ = ["DefaultExecutionHistoryService"]
