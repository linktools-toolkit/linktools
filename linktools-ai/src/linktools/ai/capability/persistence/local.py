"""Single-process capability persistence."""

import asyncio
import base64
from dataclasses import asdict
from pathlib import Path

from ...execution.codec import decode, encode
from ..entries import CapabilityEntry, CapabilityEntryChange, CapabilityEntryInfo
from ...storage.revision import SnapshotRequired


class LocalCapabilityStore:
    def __init__(self, root: str | Path = ".linktools") -> None:
        self.root = Path(root)
        self._lock = asyncio.Lock()

    @property
    def _directory(self) -> Path:
        return self.root / "capabilities"

    async def initialize_storage(self) -> None:
        await asyncio.to_thread(self._directory.mkdir, parents=True, exist_ok=True)

    def _path(self, path: str) -> Path:
        clean = path.strip("/")
        if not clean or ".." in clean.split("/"):
            raise ValueError("invalid capability path")
        return self._directory / (clean.replace("/", "__") + ".json")

    async def get(self, path: str) -> CapabilityEntry | None:
        info = await self.stat(path)
        if info is None:
            return None
        raw = await asyncio.to_thread(self._read, self._path(path))
        return CapabilityEntry(info, base64.b64decode(raw["content"]))

    async def stat(self, path: str) -> CapabilityEntryInfo | None:
        file = self._path(path)
        if not await asyncio.to_thread(file.exists):
            return None
        raw = await asyncio.to_thread(self._read, file)
        return CapabilityEntryInfo(**raw["info"])

    async def list_info(self, *, kind: str | None = None) -> tuple[CapabilityEntryInfo, ...]:
        await self.initialize_storage()
        files = await asyncio.to_thread(lambda: tuple(self._directory.glob("*.json")))
        values = [CapabilityEntryInfo(**(await asyncio.to_thread(self._read, file))["info"]) for file in files]
        return tuple(sorted((value for value in values if kind is None or value.kind == kind), key=lambda value: value.path))

    async def current_revision(self) -> int:
        file = self._directory / "revision.json"
        if not await asyncio.to_thread(file.exists):
            return 0
        return int((await asyncio.to_thread(self._read, file))["revision"])

    async def put(self, entry: CapabilityEntry) -> CapabilityEntry:
        async with self._lock:
            await self.initialize_storage()
            before = await self.current_revision()
            await asyncio.to_thread(self._write, self._path(entry.info.path), {"info": asdict(entry.info), "content": base64.b64encode(entry.content).decode()})
            await self._set_revision(before + 1)
            return entry

    async def delete(self, path: str) -> None:
        async with self._lock:
            file = self._path(path)
            await asyncio.to_thread(file.unlink, missing_ok=True)
            await self._set_revision((await self.current_revision()) + 1)

    async def reset(self, entries: tuple[CapabilityEntry, ...]) -> None:
        async with self._lock:
            await self.initialize_storage()
            files = await asyncio.to_thread(lambda: tuple(self._directory.glob("*.json")))
            for file in files:
                if file.name != "revision.json":
                    await asyncio.to_thread(file.unlink, missing_ok=True)
            for entry in entries:
                await asyncio.to_thread(self._write, self._path(entry.info.path), {"info": asdict(entry.info), "content": base64.b64encode(entry.content).decode()})
            await self._set_revision((await self.current_revision()) + 1)

    async def list_changes(self, *, after_revision: int, through_revision: int) -> tuple[CapabilityEntryChange, ...]:
        raise SnapshotRequired

    async def _set_revision(self, revision: int) -> None:
        await asyncio.to_thread(self._write, self._directory / "revision.json", {"revision": revision})

    @staticmethod
    def _read(path: Path) -> dict:
        return decode(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encode(value), encoding="utf-8")
