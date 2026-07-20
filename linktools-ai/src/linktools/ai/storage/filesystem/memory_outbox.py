import asyncio
import json
from pathlib import Path

from ...memory.index import MemoryIndexEvent


class FilesystemMemoryOutboxStore:
    """Durable append-only outbox used to reconcile derived memory indexes."""

    def __init__(self, *, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def append(self, event: MemoryIndexEvent) -> None:
        async with self._lock:
            path = self._root / f"{event.memory_id}-{event.version}-{event.operation}.json"
            if not path.exists():
                path.write_text(json.dumps({"memory_id": event.memory_id, "version": event.version, "operation": event.operation}), encoding="utf-8")

    async def claim(self, limit: int = 100) -> tuple[MemoryIndexEvent, ...]:
        async with self._lock:
            events = []
            for path in sorted(self._root.glob("*.json"))[:limit]:
                raw = json.loads(path.read_text(encoding="utf-8"))
                events.append(MemoryIndexEvent(**raw))
            return tuple(events)

    async def ack(self, event: MemoryIndexEvent) -> None:
        async with self._lock:
            path = self._root / f"{event.memory_id}-{event.version}-{event.operation}.json"
            path.unlink(missing_ok=True)
