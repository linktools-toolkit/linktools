#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Content-addressed filesystem blob storage: ``<root>/blobs/<sha256>``.

Streams the source through 4 MiB batches (one thread-hop per batch, not per
chunk, bounding buffered memory to roughly that batch size), verifies the
digest/size against the caller-supplied :class:`ArtifactRef` before
publishing, and moves the staged temp file into place with ``os.replace`` so
a concurrent reader never observes a partially-written blob. A digest that
already has a blob is left untouched (content-addressed: identical bytes
never need rewriting).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from ...storage.local.paths import Sha256Digest, safe_child
from ..models import ArtifactBlobNotFoundError, ArtifactIntegrityError, ArtifactRef

_BATCH_SIZE = 4 * 1024 * 1024


class FilesystemArtifactBlobStore:
    def __init__(self, root: "str | Path") -> None:
        self.root = Path(root)

    async def initialize_storage(self) -> None:
        await asyncio.to_thread((self.root / "blobs").mkdir, parents=True, exist_ok=True)

    def _blob_path(self, digest: str) -> Path:
        return safe_child(self.root, "blobs", Sha256Digest.parse(digest))

    async def put(self, *, ref: ArtifactRef, content: AsyncIterator[bytes]) -> None:
        blob = self._blob_path(ref.sha256)
        if await asyncio.to_thread(blob.exists):
            return
        temporary = await asyncio.to_thread(self._temporary_path)
        digest = hashlib.sha256()
        size = 0
        buffer = bytearray()
        try:
            stream = await asyncio.to_thread(temporary.open, "wb")
            try:
                async for chunk in content:
                    if not isinstance(chunk, bytes):
                        raise TypeError("artifact content chunks must be bytes")
                    digest.update(chunk)
                    size += len(chunk)
                    buffer.extend(chunk)
                    if len(buffer) >= _BATCH_SIZE:
                        await asyncio.to_thread(stream.write, buffer)
                        buffer.clear()
                if buffer:
                    await asyncio.to_thread(stream.write, buffer)
            finally:
                await asyncio.to_thread(stream.close)
            if digest.hexdigest() != ref.sha256 or size != ref.size:
                raise ArtifactIntegrityError("artifact digest or size mismatch")
            await asyncio.to_thread(self._publish, temporary, blob)
        finally:
            await asyncio.to_thread(Path.unlink, temporary, missing_ok=True)

    async def open(self, digest: str) -> AsyncIterator[bytes]:
        path = self._blob_path(digest)
        if not await asyncio.to_thread(path.exists):
            raise ArtifactBlobNotFoundError(digest)
        source = await asyncio.to_thread(open, path, "rb")
        try:
            while True:
                chunk = await asyncio.to_thread(source.read, _BATCH_SIZE)
                if not chunk:
                    break
                yield chunk
        finally:
            await asyncio.to_thread(source.close)

    async def delete_orphan(self, digest: str) -> None:
        await asyncio.to_thread(self._blob_path(digest).unlink, missing_ok=True)

    @staticmethod
    def _temporary_path() -> Path:
        fd, path = tempfile.mkstemp(prefix="linktools-artifact-")
        os.close(fd)
        return Path(path)

    @staticmethod
    def _publish(temporary: Path, blob: Path) -> None:
        blob.parent.mkdir(parents=True, exist_ok=True)
        if blob.exists():
            temporary.unlink(missing_ok=True)
            return
        temporary.replace(blob)


__all__: "list[str]" = ["FilesystemArtifactBlobStore"]
