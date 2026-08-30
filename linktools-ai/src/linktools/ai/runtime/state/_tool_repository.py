#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable ToolOperation convergence over optimistic StateStore CAS."""

import asyncio
from collections.abc import Awaitable, Callable

from ...errors import AIError, ErrorCode
from ...storage import StoredPayload
from .._tool import ToolOperationRecord
from ._contracts import ToolOperationAdmission
from ._repositories import ToolRepositoryImpl


class DurableToolRepositoryImpl(ToolRepositoryImpl):
    """Retry only raw storage-version races after the previous transaction exits."""

    async def admit(self, request: ToolOperationAdmission) -> ToolOperationRecord:
        base = super(DurableToolRepositoryImpl, self)
        return await self._retry_storage_conflict(lambda: base.admit(request))

    async def reserve(self, record: ToolOperationRecord) -> ToolOperationRecord:
        base = super(DurableToolRepositoryImpl, self)
        return await self._retry_storage_conflict(lambda: base.reserve(record))

    async def claim(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        lease_seconds: int,
    ) -> ToolOperationRecord:
        base = super(DurableToolRepositoryImpl, self)
        return await self._retry_storage_conflict(
            lambda: base.claim(
                tool_operation_id,
                tenant_id=tenant_id,
                owner=owner,
                lease_seconds=lease_seconds,
            )
        )

    async def complete_payload(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        result_payload: StoredPayload,
    ) -> ToolOperationRecord:
        base = super(DurableToolRepositoryImpl, self)
        return await self._retry_storage_conflict(
            lambda: base.complete_payload(
                tool_operation_id,
                tenant_id=tenant_id,
                owner=owner,
                fence=fence,
                result_payload=result_payload,
            )
        )

    async def fail(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        error_code: str,
    ) -> ToolOperationRecord:
        base = super(DurableToolRepositoryImpl, self)
        return await self._retry_storage_conflict(
            lambda: base.fail(
                tool_operation_id,
                tenant_id=tenant_id,
                owner=owner,
                fence=fence,
                error_code=error_code,
            )
        )

    async def fail_payload(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        error_code: str,
        error_payload: StoredPayload | None,
    ) -> ToolOperationRecord:
        base = super(DurableToolRepositoryImpl, self)
        return await self._retry_storage_conflict(
            lambda: base.fail_payload(
                tool_operation_id,
                tenant_id=tenant_id,
                owner=owner,
                fence=fence,
                error_code=error_code,
                error_payload=error_payload,
            )
        )

    async def mark_effect_unknown(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        error_code: str | None,
    ) -> ToolOperationRecord:
        base = super(DurableToolRepositoryImpl, self)
        return await self._retry_storage_conflict(
            lambda: base.mark_effect_unknown(
                tool_operation_id,
                tenant_id=tenant_id,
                owner=owner,
                fence=fence,
                error_code=error_code,
            )
        )

    async def _retry_storage_conflict(
        self,
        operation: Callable[[], Awaitable[ToolOperationRecord]],
    ) -> ToolOperationRecord:
        while True:
            try:
                return await operation()
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                await asyncio.sleep(0)


__all__ = ["DurableToolRepositoryImpl"]
