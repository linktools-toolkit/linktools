"""Single-process tool state backend."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from ...storage.coordination.lease import Lease, claim, renew
from ...errors import StorageConflictError
from ..state import ToolOperation, ToolOperationStatus


class LocalToolStateBackend:
    def __init__(self) -> None:
        self._items: dict[str, ToolOperation] = {}

    async def prepare(self, operation: ToolOperation) -> ToolOperation:
        existing = self._items.get(operation.id)
        if existing is not None:
            if existing.idempotency_key != operation.idempotency_key:
                raise ValueError("tool operation idempotency conflict")
            return existing
        now = datetime.now(timezone.utc)
        self._items[operation.id] = replace(operation, created_at=operation.created_at or now, updated_at=now)
        return operation

    async def get(self, operation_id: str) -> ToolOperation | None:
        return self._items.get(operation_id)

    async def claim(self, operation_id: str, *, owner: str, duration: timedelta = timedelta(minutes=5)) -> ToolOperation:
        item = self._items[operation_id]
        claimed = replace(item, status=ToolOperationStatus.CLAIMED, lease=claim(item.lease, owner=owner, now=datetime.now(timezone.utc), duration=duration), updated_at=datetime.now(timezone.utc))
        self._items[operation_id] = claimed
        return claimed

    async def renew(self, operation_id: str, *, owner: str, fence: int, duration: timedelta = timedelta(minutes=5)) -> ToolOperation:
        item = self._items[operation_id]
        updated = replace(item, lease=renew(item.lease, owner=owner, fence=fence, now=datetime.now(timezone.utc), duration=duration), updated_at=datetime.now(timezone.utc))
        self._items[operation_id] = updated
        return updated

    async def complete(self, operation_id: str, *, owner: str, fence: int, result) -> ToolOperation:
        item = await self.renew(operation_id, owner=owner, fence=fence)
        completed = replace(item, status=ToolOperationStatus.COMPLETED, result=result, lease=Lease(item.lease.owner, item.lease.fence, None), updated_at=datetime.now(timezone.utc))
        self._items[operation_id] = completed
        return completed

    async def fail(self, operation_id: str, *, owner: str, fence: int, error) -> ToolOperation:
        item = await self.renew(operation_id, owner=owner, fence=fence)
        failed = replace(item, status=ToolOperationStatus.FAILED, error=error, lease=Lease(item.lease.owner, item.lease.fence, None), updated_at=datetime.now(timezone.utc))
        self._items[operation_id] = failed
        return failed

__all__ = ["LocalToolStateBackend"]
