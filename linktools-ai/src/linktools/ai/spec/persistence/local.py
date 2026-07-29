"""Single-process spec persistence with manifest publication."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import asdict
from pathlib import Path

from ...storage.local.files import atomic_write_json, read_json
from ...storage.local.locks import KeyedLocks
from ...storage.local.paths import StoragePath, safe_child
from ..document import SpecDocument, SpecDocumentChange, SpecDocumentInfo
from ...storage.revision import SnapshotRequired


class LocalSpecBackend:
    def __init__(self, root: str | Path = ".linktools") -> None:
        self.root = Path(root)
        self._locks = KeyedLocks()

    @property
    def _directory(self) -> Path:
        return self.root / "spec"

    @property
    def _entries(self) -> Path:
        return self._directory / "entries"

    @property
    def _changes(self) -> Path:
        return self._directory / "changes"

    @property
    def _manifest(self) -> Path:
        return self._directory / "manifest.json"

    async def initialize_storage(self) -> None:
        await asyncio.to_thread(self._directory.mkdir, parents=True, exist_ok=True)

    def _path(self, path: str) -> Path:
        value = StoragePath.parse(path)
        return safe_child(self._entries, value) .with_suffix(".json")

    async def _exists(self, path: Path) -> bool:
        return await asyncio.to_thread(path.exists)

    async def _revision(self) -> int:
        if not await self._exists(self._manifest):
            return 0
        return int(dict(await asyncio.to_thread(read_json, self._manifest))["revision"])

    async def current_revision(self) -> int:
        return await self._revision()

    async def get(self, path: str) -> SpecDocument | None:
        file = self._path(path)
        if not await self._exists(file):
            return None
        raw = dict(await asyncio.to_thread(read_json, file))
        return SpecDocument(SpecDocumentInfo(**raw["info"]), base64.b64decode(raw["content"]))

    async def stat(self, path: str) -> SpecDocumentInfo | None:
        entry = await self.get(path)
        return None if entry is None else entry.info

    async def list_info(self, *, kind: str | None = None) -> tuple[SpecDocumentInfo, ...]:
        if not await self._exists(self._entries):
            return ()
        files = await asyncio.to_thread(lambda: tuple(self._entries.rglob("*.json")))
        values = []
        for file in files:
            raw = dict(await asyncio.to_thread(read_json, file))
            info = SpecDocumentInfo(**raw["info"])
            if kind is None or info.kind == kind:
                values.append(info)
        return tuple(sorted(values, key=lambda item: item.path))

    async def put(self, entry: SpecDocument) -> SpecDocument:
        async with self._locks.acquire(("spec", entry.info.path)):
            revision = await self._revision() + 1
            change = SpecDocumentChange(revision, entry.info.path, entry.info)
            await asyncio.to_thread(atomic_write_json, self._path(entry.info.path), {"info": asdict(entry.info), "content": base64.b64encode(entry.content).decode("ascii")})
            await self._publish(revision, (change,))
            return entry

    async def delete(self, path: str) -> None:
        async with self._locks.acquire(("spec", path)):
            old = await self.get(path)
            if old is None:
                return
            revision = await self._revision() + 1
            await asyncio.to_thread(self._path(path).unlink, missing_ok=True)
            await self._publish(revision, (SpecDocumentChange(revision, path, None),))

    async def reset(self, entries: tuple[SpecDocument, ...]) -> None:
        async with self._locks.acquire(("spec", "__reset__")):
            previous = {info.path: await self.get(info.path) for info in await self.list_info()}
            incoming = {entry.info.path: entry for entry in entries}
            changes: list[SpecDocumentChange] = []
            revision = await self._revision() + 1
            for path in sorted(set(previous) | set(incoming)):
                before, after = previous.get(path), incoming.get(path)
                if before is None and after is not None:
                    changes.append(SpecDocumentChange(revision, path, after.info))
                elif before is not None and after is None:
                    changes.append(SpecDocumentChange(revision, path, None))
                elif before is not None and after is not None and before.info != after.info:
                    changes.append(SpecDocumentChange(revision, path, after.info))
            files = await asyncio.to_thread(lambda: tuple(self._entries.rglob("*.json")) if self._entries.exists() else ())
            for file in files:
                await asyncio.to_thread(file.unlink, missing_ok=True)
            for entry in incoming.values():
                await asyncio.to_thread(atomic_write_json, self._path(entry.info.path), {"info": asdict(entry.info), "content": base64.b64encode(entry.content).decode("ascii")})
            await self._publish(revision, tuple(changes))

    async def _publish(self, revision: int, changes: tuple[SpecDocumentChange, ...]) -> None:
        await asyncio.to_thread(atomic_write_json, self._changes / f"{revision:020d}.json", [asdict(change) for change in changes])
        await asyncio.to_thread(atomic_write_json, self._manifest, {"revision": revision})

    async def list_changes(self, *, after_revision: int, through_revision: int) -> tuple[SpecDocumentChange, ...]:
        values: list[SpecDocumentChange] = []
        for revision in range(after_revision + 1, through_revision + 1):
            path = self._changes / f"{revision:020d}.json"
            if not await self._exists(path):
                raise SnapshotRequired
            raw_changes = await asyncio.to_thread(read_json, path)
            for raw in raw_changes:
                info = raw.get("info")
                values.append(SpecDocumentChange(revision, raw["path"], None if info is None else SpecDocumentInfo(**info)))
        return tuple(values)


__all__ = ["LocalSpecBackend"]
