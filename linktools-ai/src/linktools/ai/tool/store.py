"""Tool-state backend contract and composed Store."""

from datetime import timedelta
from typing import Any, Protocol

from .state import ToolOperation


class ToolStateBackend(Protocol):
    async def prepare(self, operation: ToolOperation) -> ToolOperation: ...
    async def get(self, operation_id: str) -> ToolOperation | None: ...
    async def claim(self, operation_id: str, *, owner: str, duration: timedelta = ...) -> ToolOperation: ...
    async def renew(self, operation_id: str, *, owner: str, fence: int, duration: timedelta = ...) -> ToolOperation: ...
    async def complete(self, operation_id: str, *, owner: str, fence: int, result: Any) -> ToolOperation: ...
    async def fail(self, operation_id: str, *, owner: str, fence: int, error: Any) -> ToolOperation: ...


class ToolStateStore:
    def __init__(self, backend: ToolStateBackend) -> None:
        self._backend = backend

    @property
    def backend(self) -> ToolStateBackend:
        return self._backend

    async def initialize_storage(self, *args: object) -> None:
        await self._backend.initialize_storage(*args)

    async def prepare(self, operation: ToolOperation) -> ToolOperation:
        return await self._backend.prepare(operation)

    async def get(self, operation_id: str) -> ToolOperation | None:
        return await self._backend.get(operation_id)

    async def claim(self, operation_id: str, *, owner: str, duration: timedelta = timedelta(minutes=5)) -> ToolOperation:
        return await self._backend.claim(operation_id, owner=owner, duration=duration)

    async def renew(self, operation_id: str, *, owner: str, fence: int, duration: timedelta = timedelta(minutes=5)) -> ToolOperation:
        return await self._backend.renew(operation_id, owner=owner, fence=fence, duration=duration)

    async def complete(self, operation_id: str, *, owner: str, fence: int, result: Any) -> ToolOperation:
        return await self._backend.complete(
            operation_id,
            owner=owner,
            fence=fence,
            result=result,
        )

    async def fail(self, operation_id: str, *, owner: str, fence: int, error: Any) -> ToolOperation:
        return await self._backend.fail(
            operation_id,
            owner=owner,
            fence=fence,
            error=error,
        )


__all__ = ["ToolStateBackend", "ToolStateStore"]
