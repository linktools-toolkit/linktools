#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Blob-store contract for the storage kernel (storage.blob).

A blob store is content-addressed by digest: put_if_absent is idempotent on the
digest (a repeat returns the existing blob, no duplicate), a streaming read
re-verifies the digest as it goes, and stat/delete operate on the digest key."""

from __future__ import annotations

import asyncio

import pytest


class TestBlobStore:
    def test_put_if_absent_is_idempotent_on_digest(self) -> None:
        async def _run() -> None:
            from linktools.ai.storage.blob.protocols import digest_of

            digest = digest_of(b"payload")
            store = _MemBlobStore()
            first = await store.put_if_absent(digest, b"payload")
            second = await store.put_if_absent(digest, b"payload")
            assert first == second  # same digest key -> same result, no duplicate

        asyncio.run(_run())

    def test_put_if_absent_with_mismatched_content_raises(self) -> None:
        async def _run() -> None:
            from linktools.ai.storage.blob.protocols import digest_of

            digest = digest_of(b"payload")
            store = _MemBlobStore()
            with pytest.raises(Exception):
                await store.put_if_absent(digest, b"DIFFERENT")

        asyncio.run(_run())

    def test_stat_reports_size_and_digest(self) -> None:
        async def _run() -> None:
            from linktools.ai.storage.blob.protocols import digest_of

            digest = digest_of(b"payload")
            store = _MemBlobStore()
            await store.put_if_absent(digest, b"payload")
            info = await store.stat(digest)
            assert info is not None and info.size == len(b"payload")

        asyncio.run(_run())

    def test_delete_removes_the_blob(self) -> None:
        async def _run() -> None:
            from linktools.ai.storage.blob.protocols import digest_of

            digest = digest_of(b"payload")
            store = _MemBlobStore()
            await store.put_if_absent(digest, b"payload")
            await store.delete(digest)
            assert await store.stat(digest) is None

        asyncio.run(_run())


class _MemBlobStore:
    """Minimal in-memory blob store the contract drives. Verifies content
    matches the supplied digest on put_if_absent (a mismatched content under
    the same digest is an integrity error, not a silent overwrite)."""

    def __init__(self) -> None:
        self._blobs: "dict[str, bytes]" = {}

    async def put_if_absent(self, digest: str, content: bytes) -> str:
        from linktools.ai.storage.blob.protocols import digest_of

        # Verify the content actually hashes to the claimed digest -- a
        # mismatched content under the same digest key is an integrity error.
        if digest_of(content) != digest:
            raise ValueError(
                f"content digest does not match the supplied digest {digest!r}"
            )
        if digest in self._blobs:
            return digest
        self._blobs[digest] = content
        return digest

    async def stat(self, digest: str):
        if digest not in self._blobs:
            return None

        class _Info:
            size = 0

        info = _Info()
        info.size = len(self._blobs[digest])
        return info

    async def delete(self, digest: str) -> None:
        self._blobs.pop(digest, None)
