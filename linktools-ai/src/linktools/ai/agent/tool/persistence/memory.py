#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Single-process tool state backend."""


import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from ....storage.coordination.lease import Lease, claim, renew
from ....storage.database import CoordinationScope
from ....errors import StorageConflictError
from ....json import normalize_json
from ..models import ToolOperationStatus

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ....json import JsonValue
    from ..models import ToolOperation

class LocalToolStateBackend:
    coordination_scope = CoordinationScope.PROCESS

    async def initialize_storage(self) -> None:
        return None

    def __init__(self) -> None:
        self._items: "dict[str, ToolOperation]" = {}
        # Serializes read-modify-write over operations. The lease helpers are
        # pure and the bodies below hold no interior await, so state is safe in
        # single-threaded asyncio today; the lock keeps that invariant
        # load-bearing rather than incidental, so a later refactor that adds
        # an await (e.g. an audit hook) cannot widen claim/complete/fail into
        # a real race.
        self._lock = asyncio.Lock()

    async def prepare(self, operation: "ToolOperation") -> "ToolOperation":
        async with self._lock:
            existing = self._items.get(operation.id)
            if existing is not None:
                if (
                    existing.idempotency_key != operation.idempotency_key
                    or existing.arguments_hash != operation.arguments_hash
                    or existing.binding_fingerprint != operation.binding_fingerprint
                    or existing.tenant_id != operation.tenant_id
                    or existing.execution_id != operation.execution_id
                    or existing.tool_call_id != operation.tool_call_id
                    or existing.tool_name != operation.tool_name
                ):
                    raise ValueError("tool operation idempotency conflict")
                return existing
            now = datetime.now(timezone.utc)
            self._items[operation.id] = replace(operation, created_at=operation.created_at or now, updated_at=now)
            return operation

    async def get(self, operation_id: str) -> "ToolOperation | None":
        return self._items.get(operation_id)

    async def claim(self, operation_id: str, *, owner: str, duration: timedelta = timedelta(minutes=5)) -> "ToolOperation":
        async with self._lock:
            item = self._items[operation_id]
            now = datetime.now(timezone.utc)
            if item.status is ToolOperationStatus.FAILED:
                raise StorageConflictError(
                    "failed tool operation cannot be claimed"
                )
            if (
                item.status is ToolOperationStatus.CLAIMED
                and item.lease.expires_at is not None
                and item.lease.expires_at <= now
                and not item.replay_safe
            ):
                indeterminate = replace(
                    item,
                    status=ToolOperationStatus.INDETERMINATE,
                    updated_at=now,
                )
                self._items[operation_id] = indeterminate
                return indeterminate
            claimed = replace(item, status=ToolOperationStatus.CLAIMED, lease=claim(item.lease, owner=owner, now=now, duration=duration), updated_at=now)
            self._items[operation_id] = claimed
            return claimed

    async def renew(self, operation_id: str, *, owner: str, fence: int, duration: timedelta = timedelta(minutes=5)) -> "ToolOperation":
        async with self._lock:
            item = self._items[operation_id]
            now = datetime.now(timezone.utc)
            updated = replace(item, lease=renew(item.lease, owner=owner, fence=fence, now=now, duration=duration), updated_at=now)
            self._items[operation_id] = updated
            return updated

    async def complete(self, operation_id: str, *, owner: str, fence: int, result: "JsonValue") -> "ToolOperation":
        # renew + terminal transition run as one atomic read-modify-write so a
        # concurrent claim/mark_indeterminate cannot observe (or clobber) the
        # half-renewed state between the two writes.
        async with self._lock:
            item = self._items[operation_id]
            now = datetime.now(timezone.utc)
            renew(item.lease, owner=owner, fence=fence, now=now, duration=timedelta(minutes=5))
            completed = replace(item, status=ToolOperationStatus.COMPLETED, result=normalize_json(result), lease=Lease(item.lease.owner, item.lease.fence, None), updated_at=now)
            self._items[operation_id] = completed
            return completed

    async def fail(self, operation_id: str, *, owner: str, fence: int, error: "JsonValue") -> "ToolOperation":
        async with self._lock:
            item = self._items[operation_id]
            now = datetime.now(timezone.utc)
            renew(item.lease, owner=owner, fence=fence, now=now, duration=timedelta(minutes=5))
            failed = replace(item, status=ToolOperationStatus.FAILED, error=normalize_json(error), lease=Lease(item.lease.owner, item.lease.fence, None), updated_at=now)
            self._items[operation_id] = failed
            return failed

    async def mark_indeterminate(self, operation_id: str, *, owner: str, fence: int, error: "JsonValue") -> "ToolOperation":
        async with self._lock:
            item = self._items[operation_id]
            now = datetime.now(timezone.utc)
            renew(item.lease, owner=owner, fence=fence, now=now, duration=timedelta(minutes=5))
            updated = replace(
                item,
                status=ToolOperationStatus.INDETERMINATE,
                error=normalize_json(error),
                lease=Lease(item.lease.owner, item.lease.fence, None),
                updated_at=now,
            )
            self._items[operation_id] = updated
            return updated

__all__ = ["LocalToolStateBackend"]
