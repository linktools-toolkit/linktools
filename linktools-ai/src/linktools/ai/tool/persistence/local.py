"""Single-process tool state backend."""

from dataclasses import replace

from ..state import ToolOperation, ToolOperationStatus


class LocalToolStateStore:
    def __init__(self) -> None:
        self._items: dict[str, ToolOperation] = {}

    async def prepare(self, operation: ToolOperation) -> ToolOperation:
        existing = self._items.get(operation.id)
        if existing is not None:
            if existing.idempotency_key != operation.idempotency_key:
                raise ValueError("tool operation idempotency conflict")
            return existing
        self._items[operation.id] = operation
        return operation

    async def get(self, operation_id: str) -> ToolOperation | None:
        return self._items.get(operation_id)

    async def claim(self, operation_id: str, *, owner: str) -> ToolOperation:
        item = self._items[operation_id]
        claimed = replace(item, status=ToolOperationStatus.CLAIMED, owner=owner, fence=item.fence + 1)
        self._items[operation_id] = claimed
        return claimed

    async def renew(self, operation_id: str, *, owner: str, fence: int) -> ToolOperation:
        item = self._items[operation_id]
        if item.owner != owner or item.fence != fence:
            raise ValueError("tool operation fence conflict")
        return item

    async def complete(self, operation_id: str, *, owner: str, fence: int, result) -> ToolOperation:
        item = await self.renew(operation_id, owner=owner, fence=fence)
        completed = replace(item, status=ToolOperationStatus.COMPLETED, result=result)
        self._items[operation_id] = completed
        return completed

    async def fail(self, operation_id: str, *, owner: str, fence: int, error) -> ToolOperation:
        item = await self.renew(operation_id, owner=owner, fence=fence)
        failed = replace(item, status=ToolOperationStatus.FAILED, error=error)
        self._items[operation_id] = failed
        return failed

__all__ = ["LocalToolStateStore"]
