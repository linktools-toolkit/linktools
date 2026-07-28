"""Tool execution store contract."""

from typing import Any, Protocol

from .state import ToolOperation


class ToolPort(Protocol):
    async def prepare(self, operation: ToolOperation) -> ToolOperation: ...
    async def get(self, operation_id: str) -> ToolOperation | None: ...
    async def claim(self, operation_id: str, *, owner: str) -> ToolOperation: ...
    async def renew(self, operation_id: str, *, owner: str, fence: int) -> ToolOperation: ...
    async def complete(self, operation_id: str, *, owner: str, fence: int, result: Any) -> ToolOperation: ...
    async def fail(self, operation_id: str, *, owner: str, fence: int, error: Any) -> ToolOperation: ...


class ToolStateStore:
    def __init__(self, backend: ToolPort) -> None:
        self.backend = backend

    def __getattr__(self, name: str):
        return getattr(self.backend, name)


__all__ = ["ToolPort", "ToolStateStore"]
