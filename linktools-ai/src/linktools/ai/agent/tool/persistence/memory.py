"""Single-process tool state backend."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from ....storage.coordination.lease import Lease, claim, renew
from ....storage.database import CoordinationScope
from ....errors import StorageConflictError
from ....json import JsonValue, normalize_json
from ..models import ToolOperation, ToolOperationStatus


class LocalToolStateBackend:
    coordination_scope = CoordinationScope.PROCESS

    async def initialize_storage(self) -> None:
        return None

    def __init__(self) -> None:
        self._items: dict[str, ToolOperation] = {}

    async def prepare(self, operation: ToolOperation) -> ToolOperation:
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

    async def get(self, operation_id: str) -> ToolOperation | None:
        return self._items.get(operation_id)

    async def claim(self, operation_id: str, *, owner: str, duration: timedelta = timedelta(minutes=5)) -> ToolOperation:
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

    async def renew(self, operation_id: str, *, owner: str, fence: int, duration: timedelta = timedelta(minutes=5)) -> ToolOperation:
        item = self._items[operation_id]
        updated = replace(item, lease=renew(item.lease, owner=owner, fence=fence, now=datetime.now(timezone.utc), duration=duration), updated_at=datetime.now(timezone.utc))
        self._items[operation_id] = updated
        return updated

    async def complete(self, operation_id: str, *, owner: str, fence: int, result: JsonValue) -> ToolOperation:
        item = await self.renew(operation_id, owner=owner, fence=fence)
        completed = replace(item, status=ToolOperationStatus.COMPLETED, result=normalize_json(result), lease=Lease(item.lease.owner, item.lease.fence, None), updated_at=datetime.now(timezone.utc))
        self._items[operation_id] = completed
        return completed

    async def fail(self, operation_id: str, *, owner: str, fence: int, error: JsonValue) -> ToolOperation:
        item = await self.renew(operation_id, owner=owner, fence=fence)
        failed = replace(item, status=ToolOperationStatus.FAILED, error=normalize_json(error), lease=Lease(item.lease.owner, item.lease.fence, None), updated_at=datetime.now(timezone.utc))
        self._items[operation_id] = failed
        return failed

    async def mark_indeterminate(self, operation_id: str, *, owner: str, fence: int, error: JsonValue) -> ToolOperation:
        item = await self.renew(operation_id, owner=owner, fence=fence)
        updated = replace(
            item,
            status=ToolOperationStatus.INDETERMINATE,
            error=normalize_json(error),
            lease=Lease(item.lease.owner, item.lease.fence, None),
            updated_at=datetime.now(timezone.utc),
        )
        self._items[operation_id] = updated
        return updated

__all__ = ["LocalToolStateBackend"]
