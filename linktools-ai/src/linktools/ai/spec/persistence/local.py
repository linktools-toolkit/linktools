"""Single-process spec persistence over a manifest + immutable content objects.

Layout::

    <root>/
    ├── manifest.json      # { revision, entries: { path: { info, object_id } } }
    └── objects/<uuid>.bin # immutable content, named by object_id

The manifest is the single publication point: a mutation writes its new/changed
content object(s) first, then atomically replaces the manifest, then updates
in-memory state and best-effort cleans up unreferenced objects. A crash can
leave only orphan objects -- the manifest never references an unfinished object.

Metadata reloads read one manifest; content reads read one object. Single
process only: one ``asyncio.Lock`` serializes every mutation."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path

from ...errors import StorageCorruptionError
from ...storage.local.files import atomic_write_bytes, atomic_write_json, read_bytes, read_json
from ...storage.revision import (
    MetadataLoad,
    MetadataLoadMode,
    StorageChange,
    StorageMetadataBackend,
)
from ..document import SpecDocument, SpecDocumentInfo


class LocalSpecBackend(StorageMetadataBackend[int, str, SpecDocumentInfo]):
    def __init__(self, root: str | Path = ".linktools") -> None:
        self.root = Path(root)
        self._lock = asyncio.Lock()
        self._manifest: dict | None = None

    @property
    def _manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def _objects_dir(self) -> Path:
        return self.root / "objects"

    def _object_path(self, object_id: str) -> Path:
        return self._objects_dir / f"{object_id}.bin"

    async def initialize_storage(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._manifest_path.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(self._objects_dir.mkdir, parents=True, exist_ok=True)
            if self._manifest is None:
                if await asyncio.to_thread(self._manifest_path.exists):
                    self._manifest = await asyncio.to_thread(read_json, self._manifest_path)
                else:
                    self._manifest = self._empty_manifest()
                    await self._write_manifest()

    def _empty_manifest(self) -> dict:
        return {"revision": 0, "entries": {}}

    # ---- reader --------------------------------------------------------

    async def get(self, path: str) -> SpecDocument | None:
        await self._ensure_loaded()
        entry = self._manifest["entries"].get(path)
        if entry is None:
            return None
        return SpecDocument(
            _info(entry["info"]),
            await self._read_object(entry["object_id"]),
        )

    async def get_many(self, paths: tuple[str, ...]) -> dict[str, SpecDocument]:
        await self._ensure_loaded()
        result: dict[str, SpecDocument] = {}
        for path in paths:
            entry = self._manifest["entries"].get(path)
            if entry is not None:
                result[path] = SpecDocument(
                    _info(entry["info"]),
                    await self._read_object(entry["object_id"]),
                )
        return result

    async def stat(self, path: str) -> SpecDocumentInfo | None:
        await self._ensure_loaded()
        entry = self._manifest["entries"].get(path)
        return None if entry is None else _info(entry["info"])

    async def list_info(self, *, kind: str | None = None) -> tuple[SpecDocumentInfo, ...]:
        await self._ensure_loaded()
        infos = [_info(entry["info"]) for entry in self._manifest["entries"].values()]
        if kind is not None:
            infos = [info for info in infos if info.kind == kind]
        return tuple(sorted(infos, key=lambda item: item.path))

    # ---- metadata backend ---------------------------------------------

    async def load_metadata(
        self,
        after_revision: int | None,
    ) -> MetadataLoad[int, str, SpecDocumentInfo]:
        await self._ensure_loaded()
        head = self._manifest["revision"]
        # Local keeps no change log: any request that is not exactly at head
        # is a full REPLACE. An empty PATCH at the same revision is legal and
        # lets a caller confirm "nothing changed" with zero content reads.
        if after_revision == head:
            return MetadataLoad(head, MetadataLoadMode.PATCH, ())
        changes = tuple(
            StorageChange(path, _info(entry["info"]))
            for path, entry in sorted(self._manifest["entries"].items())
        )
        return MetadataLoad(head, MetadataLoadMode.REPLACE, changes)

    async def head_revision(self) -> int:
        # No change log to scan: the manifest's revision field is the head. A
        # cheap probe matching the value ``load_metadata`` would return.
        await self._ensure_loaded()
        return self._manifest["revision"]

    # ---- writer --------------------------------------------------------

    async def put(self, entry: SpecDocument) -> SpecDocument:
        entry.validate_etag()
        async with self._lock:
            await self._ensure_loaded()
            object_id = _object_id_for(entry.info)
            await self._write_object(object_id, entry.content)
            entries = dict(self._manifest["entries"])
            entries[entry.info.path] = {"info": asdict(entry.info), "object_id": object_id}
            await self._publish(entries)
        return entry

    async def delete(self, path: str) -> None:
        async with self._lock:
            await self._ensure_loaded()
            entries = dict(self._manifest["entries"])
            if path not in entries:
                return
            entries.pop(path)
            await self._publish(entries)

    async def reset(self, entries: tuple[SpecDocument, ...]) -> None:
        for entry in entries:
            entry.validate_etag()
        async with self._lock:
            await self._ensure_loaded()
            new: dict[str, dict] = {}
            for entry in entries:
                object_id = _object_id_for(entry.info)
                # _write_object is a no-op when the object already exists, so
                # unchanged content (same etag -> same object_id) reuses it for free.
                await self._write_object(object_id, entry.content)
                new[entry.info.path] = {"info": asdict(entry.info), "object_id": object_id}
            await self._publish(new)

    # ---- internals -----------------------------------------------------

    async def _ensure_loaded(self) -> None:
        if self._manifest is None:
            if await asyncio.to_thread(self._manifest_path.exists):
                self._manifest = await asyncio.to_thread(read_json, self._manifest_path)
            else:
                self._manifest = self._empty_manifest()

    async def _publish(self, entries: dict) -> None:
        previous_revision = self._manifest["revision"]
        previous_object_ids = {
            entry["object_id"] for entry in self._manifest["entries"].values()
        }
        manifest = {"revision": previous_revision + 1, "entries": entries}
        await self._write_manifest_with(manifest)
        self._manifest = manifest
        # Best-effort cleanup of objects no longer referenced after this publish.
        await self._cleanup_orphans(previous_object_ids, entries)

    async def _cleanup_orphans(self, previous_ids: set[str], entries: dict) -> None:
        kept = {entry["object_id"] for entry in entries.values()}
        orphan_paths = [self._object_path(oid) for oid in previous_ids - kept]

        def _unlink_all() -> None:
            for path in orphan_paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

        if orphan_paths:
            await asyncio.to_thread(_unlink_all)

    async def _write_object(self, object_id: str, content: bytes) -> None:
        path = self._object_path(object_id)
        if await asyncio.to_thread(path.exists):
            return
        await asyncio.to_thread(atomic_write_bytes, path, content)

    async def _read_object(self, object_id: str) -> bytes:
        path = self._object_path(object_id)
        try:
            return await asyncio.to_thread(read_bytes, path)
        except FileNotFoundError as exc:
            raise StorageCorruptionError(
                f"spec object {object_id} referenced by manifest is missing"
            ) from exc

    async def _write_manifest(self) -> None:
        await self._write_manifest_with(self._manifest)

    async def _write_manifest_with(self, manifest: dict) -> None:
        await asyncio.to_thread(atomic_write_json, self._manifest_path, manifest)


def _info(raw: dict) -> SpecDocumentInfo:
    return SpecDocumentInfo(
        raw["path"], raw["kind"], raw["version"], raw["etag"], raw.get("active", True)
    )


def _object_id_for(info: SpecDocumentInfo) -> str:
    # The etag is sha256(content); content-addressing the object by it makes
    # identical content share one object and lets reset reuse it.
    return info.etag


__all__ = ["LocalSpecBackend"]
