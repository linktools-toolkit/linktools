from typing import Protocol, runtime_checkable

from .index import MemoryIndexEvent


@runtime_checkable
class MemoryOutboxStore(Protocol):
    async def append(self, event: MemoryIndexEvent) -> None: ...
    async def claim(self, limit: int = 100) -> tuple[MemoryIndexEvent, ...]: ...
    async def ack(self, event: MemoryIndexEvent) -> None: ...
