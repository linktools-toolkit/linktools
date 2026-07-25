#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FilesystemArtifactBlobStore: filesystem reference implementation of the
generic :class:`~linktools.ai.storage.blob.protocols.BlobStore` Protocol.

Content-addressed blobs live under ``blobs_root/<xx>/<sha256>`` (the two-hex-char
shard keeps any single directory small).

The write path ALWAYS consumes and verifies the source: it streams the source
into a same-dir temp file in fixed-size chunks (never holding the whole blob
resident), hashes incrementally, verifies the claimed digest and size, and
only then publishes. If a blob at that address already exists, the source is
still consumed and verified, the existing blob is re-hashed and size-checked,
and the existing BlobInfo is returned -- a put can never skip input validation
by claiming a digest that is already present.

Every blocking disk operation runs on a worker thread via the :mod:`._io`
helpers so a large artifact's I/O never blocks the event loop. ``digest`` is
the plain SHA-256 hex string; this module owns its own defense-in-depth
validation (a shard path is built directly from the argument, so a malformed
digest can never become a path) since there is no artifact-domain value
object at this layer to lean on."""

import hashlib
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING, AsyncIterator
from pathlib import Path

if TYPE_CHECKING:
    from ...observability.metrics import ObservabilityMetrics

from ...errors import StorageBlobIntegrityError, StorageBlobNotFoundError
from ..blob.protocols import BlobInfo
from . import _io

_CHUNK = 64 * 1024
_DIGEST_HEX_LEN = 64


def _validate_digest(digest: str) -> None:
    # sha256 hex: exactly 64 hex chars. Defended here so a malformed digest can
    # never become a shard path or be mistaken for a valid content address.
    if len(digest) != _DIGEST_HEX_LEN:
        raise ValueError(f"invalid digest length: {digest!r}")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(f"invalid digest (not hex): {digest!r}") from exc


class FilesystemArtifactBlobStore:
    """BlobStore backed by the local filesystem. Blobs are content-addressed
    by sha256 under ``blobs_root/<xx>/<sha>``. ``put_if_absent`` always
    consumes and verifies the source before deciding to publish a new blob or
    return an existing one; ``open`` streams the file back in bounded
    chunks."""

    def __init__(
        self,
        *,
        blobs_root: Path,
        chunk_size: int = _CHUNK,
        metrics: "ObservabilityMetrics | None" = None,
    ) -> None:
        self._root = Path(blobs_root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._chunk = chunk_size
        self._metrics = metrics

    def _path(self, digest: str) -> Path:
        _validate_digest(digest)
        return self._root / digest[:2] / digest

    def _fail(self, error: Exception, *, reason: str) -> None:
        if self._metrics is not None:
            self._metrics.counter(
                "artifact_blob_upload_failure_total", attributes={"reason": reason}
            )
        raise error

    async def put_if_absent(
        self, *, digest: str, source: AsyncIterator[bytes], size: "int | None"
    ) -> BlobInfo:
        final = self._path(digest)
        await _io.async_makedirs(final.parent)
        fd, tmp_name = await _io.async_mkstemp(
            directory=final.parent, prefix=f".{final.name}.", suffix=".tmp"
        )
        hasher = hashlib.sha256()
        written = 0
        tmp_path = Path(tmp_name)
        try:
            # Always consume the source: spool to a same-dir temp, hashing as we
            # go, then verify the claimed digest and size before publishing.
            with os.fdopen(fd, "wb") as f:
                async for chunk in source:
                    if not chunk:
                        continue
                    hasher.update(chunk)
                    written += len(chunk)
                    await _io.async_write_chunk(f, chunk)
                await _io.async_fsync_file(f)
            actual = hasher.hexdigest()
            if actual != digest:
                self._fail(
                    StorageBlobIntegrityError(
                        f"blob digest mismatch: claimed {digest[:12]}, actual {actual[:12]}"
                    ),
                    reason="digest_mismatch",
                )
            if size is not None and size != written:
                self._fail(
                    StorageBlobIntegrityError(
                        f"blob size mismatch: claimed {size}, actual {written}"
                    ),
                    reason="size_mismatch",
                )
            # Source verified. Publish a new blob, or reconcile with an existing
            # one without overwriting it.
            if await _io.async_stat_size(final) is None:
                await _io.async_replace(tmp_name, final)
                await _io.async_fsync_directory(final.parent)
                return BlobInfo(digest=digest, size=written, content_type=None)
            return await self._verify_existing(final, digest, declared_size=size)
        finally:
            # Drop the temp if the publish path did not rename it away.
            if await _io.async_exists(tmp_path):
                await _io.async_unlink(tmp_path)

    async def _verify_existing(
        self, final: Path, digest: str, *, declared_size: "int | None"
    ) -> BlobInfo:
        existing_digest = await _io.async_hash_file(final, chunk_size=self._chunk)
        if existing_digest != digest:
            self._fail(
                StorageBlobIntegrityError(
                    f"blob at sha256 {digest[:12]} is corrupt (actual {existing_digest[:12]}); "
                    f"refusing to record a reference to it"
                ),
                reason="corrupt",
            )
        existing_size = await _io.async_stat_size(final)
        if declared_size is not None and existing_size != declared_size:
            self._fail(
                StorageBlobIntegrityError(
                    f"existing blob size mismatch: claimed {declared_size}, actual {existing_size}"
                ),
                reason="size_mismatch",
            )
        return BlobInfo(digest=digest, size=existing_size, content_type=None)

    @asynccontextmanager
    async def open(self, *, digest: str):
        path = self._path(digest)
        if await _io.async_stat_size(path) is None:
            raise StorageBlobNotFoundError(f"blob for sha256 {digest[:12]} missing")
        f = await _io.async_open_read(path)
        try:
            chunk = self._chunk

            async def _chunks() -> AsyncIterator[bytes]:
                while True:
                    block = await _io.async_read_chunk(f, chunk)
                    if not block:
                        break
                    yield block

            yield _chunks()
        finally:
            await _io.async_close(f)

    async def stat(self, *, digest: str) -> "BlobInfo | None":
        path = self._path(digest)
        size = await _io.async_stat_size(path)
        if size is None:
            return None
        return BlobInfo(digest=digest, size=size, content_type=None)

    async def delete(self, *, digest: str) -> None:
        path = self._path(digest)
        await _io.async_unlink(path)

    async def iter_digests_with_mtime(self) -> "AsyncIterator[tuple[str, datetime]]":
        """Yield ``(digest, modified_at)`` for every stored blob, for orphan
        sweeping. Walks the two-hex-char shard dirs; the digest is the file
        name (the sha256 it was filed under)."""
        if not await _io.async_exists(self._root):
            return
        for shard in await _io.async_list_subdirs(self._root):
            for blob in await _io.async_list_files(shard):
                yield blob.name, await _io.async_mtime(blob)

__all__: "list[str]" = ["FilesystemArtifactBlobStore"]
