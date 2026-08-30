#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ToolOperation repository convergence for durable optimistic CAS races."""

import asyncio

from ...errors import AIError, ErrorCode
from ...storage import StoredPayload
from .._tool import ToolOperationRecord
from ._contracts import ToolOperationAdmission
from ._repositories import ToolRepositoryImpl


class DurableToolRepositoryImpl(ToolRepositoryImpl):
    """Retry standalone ToolOperation mutations only after a fresh transaction."""

    async def admit(self, request: ToolOperationAdmission) -> ToolOperationRecord:
        while True:
            try:
                return await super().admit(request)
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                await asyncio.sleep(0)

    async def reserve(self, record: ToolOperationRecord) -> ToolOperationRecord:
        while True:
            try:
                return await super().reserve(record)
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                await asyncio.sleep(0)

    async def claim(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        lease_seconds: int,
    ) -> ToolOperationRecord:
        while True:
            try:
                return await super().claim(
                    tool_operation_id,
                    tenant_id=tenant_id,
                    owner=owner,
                    lease_seconds=lease_seconds,
                )
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                await asyncio.sleep(0)

    async def complete_payload(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        result_payload: StoredPayload,
    ) -> ToolOperationRecord:
        while True:
            try:
                return await super().complete_payload(
                    tool_operation_id,
                    tenant_id=tenant_id,
                    owner=owner,
                    fence=fence,
                    result_payload=result_payload,
                )
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                await asyncio.sleep(0)

    async def fail(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        error_code: str,
    ) -> ToolOperationRecord:
        while True:
            try:
                return await super().fail(
                    tool_operation_id,
                    tenant_id=tenant_id,
                    owner=owner,
                    fence=fence,
                    error_code=error_code,
                )
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                await asyncio.sleep(0)

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
        while True:
            try:
                return await super().fail_payload(
                    tool_operation_id,
                    tenant_id=tenant_id,
                    owner=owner,
                    fence=fence,
                    error_code=error_code,
                    error_payload=error_payload,
                )
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                await asyncio.sleep(0)

    async def mark_effect_unknown(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        error_code: str | None,
    ) -> ToolOperationRecord:
        while True:
            try:
                return await super().mark_effect_unknown(
                    tool_operation_id,
                    tenant_id=tenant_id,
                    owner=owner,
                    fence=fence,
                    error_code=error_code,
                )
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                await asyncio.sleep(0)


__all__ = ["DurableToolRepositoryImpl"]
