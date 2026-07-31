#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Single-process content-addressed artifact persistence."""


import asyncio
import hashlib
from collections.abc import AsyncIterator
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from ...errors import ArtifactRecordConflictError
from ...storage.local.files import atomic_write_bytes, atomic_write_json, read_json
from ...storage.local.locks import KeyedLocks
from ...storage.local.paths import Sha256Digest, StorageId, safe_child
from ..digest import ArtifactDigest
from ..models import ArtifactBlobNotFoundError, ArtifactProvenance, ArtifactRecord, ArtifactRef

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterable

_BATCH_SIZE = 4 * 1024 * 1024


class LocalArtifactBackend:
    def __init__(self, root: "str | Path" = ".linktools") -> None:
        self.root = Path(root)
        self._temp = self.root / "temp"
        self._locks = KeyedLocks()

    async def initialize_storage(self) -> None:
        await asyncio.gather(
            asyncio.to_thread(
                self.root.mkdir,
                parents=True,
                exist_ok=True,
            ),
            asyncio.to_thread(self._temp.mkdir, parents=True, exist_ok=True),
        )

    def _record_path(self, artifact_id: str, tenant_id: str) -> Path:
        return safe_child(
            self.root,
            "records",
            StorageId.parse(tenant_id),
            StorageId.parse(artifact_id),
        ).with_suffix(".json")

    def _blob_path(self, digest: str) -> Path:
        return safe_child(self.root, "blobs", Sha256Digest.parse(digest))

    async def put(self, *, record: ArtifactRecord, content: "AsyncIterable[bytes]") -> ArtifactRecord:
        StorageId.parse(record.ref.id)
        StorageId.parse(record.tenant_id)
        ArtifactDigest.parse(record.ref.sha256)
        async with self._locks.acquire(
            ("artifact", f"{record.tenant_id}:{record.ref.id}")
        ):
            existing = await self.get_record(
                record.ref.id, tenant_id=record.tenant_id
            )
            if existing is not None:
                if not self._same_identity(existing, record):
                    raise ArtifactRecordConflictError(record.ref.id)
                return existing
            temporary = safe_child(self._temp, f"{uuid4().hex}.part")
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
                        await asyncio.to_thread(
                            self._append, temporary, bytes(buffer)
                        )
                        buffer.clear()
                if buffer:
                    await asyncio.to_thread(
                        self._append, temporary, bytes(buffer)
                    )
                if (
                    digest.hexdigest() != record.ref.sha256
                    or size != record.ref.size
                ):
                    raise ValueError("artifact digest or size mismatch")
                blob = self._blob_path(record.ref.sha256)
                await asyncio.to_thread(
                    self._publish_blob, temporary, blob
                )
                await asyncio.to_thread(
                    atomic_write_json,
                    self._record_path(record.ref.id, record.tenant_id),
                    asdict(record),
                )
                return record
            finally:
                await asyncio.to_thread(
                    Path.unlink, temporary, missing_ok=True
                )

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

    async def get_record(self, artifact_id: str, *, tenant_id: str) -> "ArtifactRecord | None":
        path = self._record_path(artifact_id, tenant_id)
        if not await asyncio.to_thread(path.exists):
            return None
        raw = dict(await asyncio.to_thread(read_json, path))
        raw["ref"] = ArtifactRef(**raw["ref"])
        raw["provenance"]["parent_artifact_ids"] = tuple(raw["provenance"].get("parent_artifact_ids", ()))
        raw["provenance"] = ArtifactProvenance(**raw["provenance"])
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        record = ArtifactRecord(**raw)
        return record if record.tenant_id == tenant_id else None

    async def open(self, artifact_id: str, *, tenant_id: str) -> "AsyncIterator[bytes]":
        record = await self.get_record(artifact_id, tenant_id=tenant_id)
        if record is None:
            raise ArtifactBlobNotFoundError(artifact_id)
        source = await asyncio.to_thread(open, self._blob_path(record.ref.sha256), "rb")
        try:
            while True:
                chunk = await asyncio.to_thread(source.read, _BATCH_SIZE)
                if not chunk:
                    break
                yield chunk
        finally:
            await asyncio.to_thread(source.close)

    async def delete(self, artifact_id: str, *, tenant_id: str) -> None:
        if await self.get_record(artifact_id, tenant_id=tenant_id) is None:
            return
        path = self._record_path(artifact_id, tenant_id)
        await asyncio.to_thread(path.unlink, missing_ok=True)

    @staticmethod
    def _same_identity(left: ArtifactRecord, right: ArtifactRecord) -> bool:
        return (
            left.tenant_id == right.tenant_id
            and left.ref.sha256 == right.ref.sha256
            and left.ref.media_type == right.ref.media_type
            and left.ref.size == right.ref.size
            and left.provenance == right.provenance
        )

__all__ = ["LocalArtifactBackend"]
