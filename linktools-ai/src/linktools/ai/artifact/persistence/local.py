"""Single-process content-addressed artifact persistence."""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ...storage.local.files import atomic_write_bytes, atomic_write_json, read_json
from ...storage.local.paths import Sha256Digest, StorageId, safe_child
from ..digest import ArtifactDigest
from ..models import ArtifactProvenance, ArtifactRecord, ArtifactRef

_BATCH_SIZE = 4 * 1024 * 1024


class LocalArtifactBackend:
    def __init__(self, root: str | Path = ".linktools") -> None:
        self.root = Path(root)

    async def initialize_storage(self) -> None:
        await asyncio.to_thread((self.root / "artifacts").mkdir, parents=True, exist_ok=True)

    def _record_path(self, artifact_id: str) -> Path:
        return safe_child(self.root, "artifacts", "records", StorageId.parse(artifact_id)).with_suffix(".json")

    def _blob_path(self, digest: str) -> Path:
        return safe_child(self.root, "artifacts", "blobs", Sha256Digest.parse(digest))

    async def put(self, *, record: ArtifactRecord, content) -> ArtifactRecord:
        StorageId.parse(record.ref.id)
        ArtifactDigest.parse(record.ref.sha256)
        temporary = await asyncio.to_thread(self._temporary_path)
        digest = hashlib.sha256()
        size = 0
        buffer = bytearray()
        try:
            async for chunk in content:
                if not isinstance(chunk, bytes):
                    raise TypeError("artifact content chunks must be bytes")
                digest.update(chunk)
                size += len(chunk)
                buffer.extend(chunk)
                if len(buffer) >= _BATCH_SIZE:
                    await asyncio.to_thread(self._append, temporary, bytes(buffer))
                    buffer.clear()
            if buffer:
                await asyncio.to_thread(self._append, temporary, bytes(buffer))
            if digest.hexdigest() != record.ref.sha256 or size != record.ref.size:
                raise ValueError("artifact digest or size mismatch")
            blob = self._blob_path(record.ref.sha256)
            await asyncio.to_thread(self._publish_blob, temporary, blob)
            await asyncio.to_thread(atomic_write_json, self._record_path(record.ref.id), asdict(record))
            return record
        finally:
            await asyncio.to_thread(Path.unlink, temporary, missing_ok=True)

    @staticmethod
    def _temporary_path() -> Path:
        fd, path = tempfile.mkstemp(prefix="linktools-artifact-")
        os.close(fd)
        return Path(path)

    @staticmethod
    def _append(path: Path, content: bytes) -> None:
        with path.open("ab") as stream:
            stream.write(content)

    @staticmethod
    def _publish_blob(temporary: Path, blob: Path) -> None:
        blob.parent.mkdir(parents=True, exist_ok=True)
        if blob.exists():
            temporary.unlink(missing_ok=True)
            return
        temporary.replace(blob)

    async def get_record(self, artifact_id: str) -> ArtifactRecord | None:
        path = self._record_path(artifact_id)
        if not await asyncio.to_thread(path.exists):
            return None
        raw = dict(await asyncio.to_thread(read_json, path))
        raw["ref"] = ArtifactRef(**raw["ref"])
        raw["provenance"]["parent_artifact_ids"] = tuple(raw["provenance"].get("parent_artifact_ids", ()))
        raw["provenance"] = ArtifactProvenance(**raw["provenance"])
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        return ArtifactRecord(**raw)

    async def open(self, artifact_id: str):
        record = await self.get_record(artifact_id)
        if record is None:
            return
        source = await asyncio.to_thread(open, self._blob_path(record.ref.sha256), "rb")
        try:
            while True:
                chunk = await asyncio.to_thread(source.read, _BATCH_SIZE)
                if not chunk:
                    break
                yield chunk
        finally:
            await asyncio.to_thread(source.close)

    async def delete(self, artifact_id: str) -> None:
        path = self._record_path(artifact_id)
        await asyncio.to_thread(path.unlink, missing_ok=True)

__all__ = ["LocalArtifactBackend"]
