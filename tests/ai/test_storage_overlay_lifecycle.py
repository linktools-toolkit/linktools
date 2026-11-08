#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overlay admission and backend lifetime include mutations and receipts."""

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.asset import AssetInfo, AssetKey, AssetStore, InMemoryAssetBackend, SqlAssetBackend
from linktools.ai.core import JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_asset_database
from linktools.ai.storage import (
    InMemoryObjectStore,
    MetadataLoad,
    ObjectStat,
    PayloadPolicy,
    StorageBatchPartialError,
    StorageBatchResult,
    StorageChange,
    StorageDeleteResult,
    StorageEntryRevision,
    StorageEntryStatus,
    StorageLayer,
    StorageOperation,
    StorageOverlay,
    StoragePutResult,
    StorageResetResult,
    StorageRevision,
)

pytestmark = pytest.mark.asyncio
_KEY = AssetKey("text", "entry")
_CHANGES = (StorageChange(StorageOperation.PUT, _KEY, b"value", None),)


async def _invoke(overlay: StorageOverlay[AssetKey, bytes, AssetInfo], operation: str) -> object:
    if operation == "put":
        return await overlay.put(_KEY, b"value")
    if operation == "delete":
        return await overlay.delete(_KEY)
    if operation == "reset":
        return await overlay.reset(_KEY)
    if operation == "batch":
        return await overlay.apply_batch(_CHANGES)
    if operation == "replay":
        return await overlay.apply_batch(_CHANGES, idempotency_key="request", request_digest="a" * 64)
    if operation == "receipt":
        return await overlay.batch_result("request")
    if operation == "get":
        return await overlay.get(_KEY)
    raise AssertionError(operation)


class _PausedWriter(InMemoryAssetBackend):
    def __init__(self) -> None:
        super().__init__()
        self.blocked: str | None = None
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.active = False
        self.closed = False

    async def _pause(self, operation: str) -> None:
        if self.blocked != operation:
            return
        self.active = True
        self.entered.set()
        try:
            await self.release.wait()
            assert not self.closed
        finally:
            self.active = False

    async def put(
        self, key: AssetKey, value: bytes, *,
        expected_revision: StorageEntryRevision | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> StoragePutResult[AssetInfo]:
        await self._pause("put")
        return await super().put(key, value, expected_revision=expected_revision, metadata=metadata)

    async def delete(
        self, key: AssetKey, *, expected_revision: StorageEntryRevision | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> StorageDeleteResult[AssetKey]:
        await self._pause("delete")
        return await super().delete(key, expected_revision=expected_revision, metadata=metadata)

    async def reset(
        self, key: AssetKey, *, expected_revision: StorageEntryRevision | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> StorageResetResult[AssetKey]:
        await self._pause("reset")
        return await super().reset(key, expected_revision=expected_revision, metadata=metadata)

    async def apply_batch(
        self, changes: Sequence[StorageChange[AssetKey, bytes]], *,
        expected_revision: StorageRevision | None = None,
        idempotency_key: str | None = None, request_digest: str | None = None,
    ) -> StorageBatchResult[AssetInfo, AssetKey]:
        await self._pause("batch")
        return await super().apply_batch(
            changes, expected_revision=expected_revision,
            idempotency_key=idempotency_key, request_digest=request_digest,
        )

    async def batch_result(self, idempotency_key: str) -> StorageBatchResult[AssetInfo, AssetKey] | None:
        await self._pause("receipt")
        return await super().batch_result(idempotency_key)

    async def close(self) -> None:
        assert not self.active
        self.closed = True
        await super().close()


@pytest.mark.parametrize("operation,lower_writer", (
    ("put", False), ("delete", False), ("reset", False), ("batch", False),
    ("receipt", False), ("replay", False), ("put", True),
))
async def test_overlay_lifecycle_drains_admitted_write_or_receipt(operation: str, lower_writer: bool) -> None:
    writer = _PausedWriter()
    primary = InMemoryAssetBackend() if lower_writer else writer
    overlay = StorageOverlay(primary, writer=writer, layers=(StorageLayer("writer", writer),) if lower_writer else ())
    await overlay.initialize()
    await writer.apply_batch(_CHANGES, idempotency_key="request", request_digest="a" * 64)
    writer.blocked = "receipt" if operation == "replay" else operation
    pending = asyncio.create_task(_invoke(overlay, operation))
    closing: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(writer.entered.wait(), 2)
        writer.blocked = None
        closing = asyncio.create_task(overlay.close())
        await asyncio.sleep(0)
        with pytest.raises(AIError) as rejected:
            await _invoke(overlay, operation)
        assert rejected.value.code is ErrorCode.STORAGE_CLOSED
        assert not closing.done()
        assert not writer.closed
        writer.release.set()
        result = await pending
        if operation in {"receipt", "replay"}:
            assert result.idempotency_key == "request"
        await closing
        assert writer.closed
    finally:
        writer.release.set()
        await asyncio.gather(pending, *(closing,) if closing is not None else (), return_exceptions=True)
        await overlay.close()


async def test_overlay_lifecycle_rejects_all_operations_after_close_or_failed_initialize() -> None:
    class FailingInitialize(InMemoryAssetBackend):
        async def initialize(self) -> None:
            raise RuntimeError("initialize failed")

    for backend in (InMemoryAssetBackend(), FailingInitialize()):
        await backend.put(_KEY, b"existing")
        overlay = StorageOverlay(backend, writer=backend)
        if isinstance(backend, FailingInitialize):
            with pytest.raises(RuntimeError, match="initialize failed"):
                await overlay.initialize()
        else:
            await overlay.initialize()
            await overlay.close()
        for operation in ("put", "delete", "reset", "batch", "receipt", "get"):
            with pytest.raises(AIError) as rejected:
                await _invoke(overlay, operation)
            assert rejected.value.code is ErrorCode.STORAGE_CLOSED
        await overlay.close()


async def test_overlay_lifecycle_preserves_independently_ready_source_usage() -> None:
    backend = InMemoryAssetBackend()
    await backend.initialize()
    overlay = StorageOverlay(backend, writer=backend)
    try:
        await overlay.put(_KEY, b"value")
        assert await overlay.get(_KEY) == b"value"
    finally:
        await overlay.close()
        await backend.close()


class _SingleWriter:
    """A valid non-batch backend with an explicit pause between batch entries."""

    def __init__(self, failure: str | None) -> None:
        self.source = InMemoryAssetBackend()
        self.failure = failure
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

    async def initialize(self) -> None:
        await self.source.initialize()

    async def close(self) -> None:
        self.closed = True
        await self.source.close()

    async def get(self, key: AssetKey) -> bytes | None:
        assert not self.closed
        return await self.source.get(key)

    async def head_revision(self) -> StorageRevision:
        assert not self.closed
        return await self.source.head_revision()

    async def load_metadata(self, after_revision: StorageRevision | None) -> MetadataLoad[AssetKey, AssetInfo]:
        assert not self.closed
        return await self.source.load_metadata(after_revision)

    async def put(
        self, key: AssetKey, value: bytes, *,
        expected_revision: StorageEntryRevision | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> StoragePutResult[AssetInfo]:
        assert not self.closed
        if self.failure == "first":
            self.entered.set()
            await self.release.wait()
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
        result = await self.source.put(key, value, expected_revision=expected_revision, metadata=metadata)
        self.entered.set()
        await self.release.wait()
        assert not self.closed
        return result

    async def delete(
        self, key: AssetKey, *, expected_revision: StorageEntryRevision | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> StorageDeleteResult[AssetKey]:
        assert not self.closed
        if self.failure == "later":
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
        return await self.source.delete(key, expected_revision=expected_revision, metadata=metadata)

    async def reset(
        self, key: AssetKey, *, expected_revision: StorageEntryRevision | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> StorageResetResult[AssetKey]:
        assert not self.closed
        return await self.source.reset(key, expected_revision=expected_revision, metadata=metadata)


@pytest.mark.parametrize("failure", (None, "first", "later"))
async def test_overlay_lifecycle_keeps_admitted_fallback_batch_inside_its_lifetime(failure: str | None) -> None:
    writer = _SingleWriter(failure)
    deleted, reset = AssetKey("text", "deleted"), AssetKey("text", "reset")
    await writer.source.put(deleted, b"old")
    await writer.source.put(reset, b"old")
    overlay = StorageOverlay(writer, writer=writer)
    await overlay.initialize()
    revision = await overlay.current_revision() if failure is None else None
    batch = asyncio.create_task(overlay.apply_batch((
        StorageChange(StorageOperation.PUT, _KEY, b"new", None),
        StorageChange(StorageOperation.DELETE, deleted, None, None),
        StorageChange(StorageOperation.RESET, reset, None, None),
    ), expected_revision=revision))
    closing: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(writer.entered.wait(), 2)
        closing = asyncio.create_task(overlay.close())
        await asyncio.sleep(0)
        with pytest.raises(AIError) as rejected:
            await asyncio.wait_for(overlay.put(_KEY, b"late"), 2)
        assert rejected.value.code is ErrorCode.STORAGE_CLOSED
        writer.release.set()
        if failure is None:
            result = await batch
            assert not result.atomic
            assert len(result.results) == 3
            assert await writer.source.get(_KEY) == b"new"
            assert (await writer.source.stat(deleted)).status is StorageEntryStatus.DELETED
            assert (await writer.source.stat(reset)).status is StorageEntryStatus.RESET
        else:
            with pytest.raises(StorageBatchPartialError) as failed:
                await batch
            completed = 0 if failure == "first" else 1
            assert failed.value.failure.failed_index == completed
            assert len(failed.value.failure.completed) == completed
            assert failed.value.failure.error_code == ErrorCode.STORAGE_UNAVAILABLE.value
            assert failed.value.failure.store_revision == await writer.source.head_revision()
        await closing
    finally:
        writer.release.set()
        await asyncio.gather(batch, *(closing,) if closing is not None else (), return_exceptions=True)
        await overlay.close()


async def test_overlay_lifecycle_waits_for_write_cancellation_cleanup_and_cancelled_closer() -> None:
    class CancellingWriter(_PausedWriter):
        def __init__(self) -> None:
            super().__init__()
            self.cleaning = asyncio.Event()

        async def put(
            self, key: AssetKey, value: bytes, *,
            expected_revision: StorageEntryRevision | None = None,
            metadata: Mapping[str, JsonValue] | None = None,
        ) -> StoragePutResult[AssetInfo]:
            self.active = True
            self.entered.set()
            try:
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cleaning.set()
                    await self.release.wait()
                    assert not self.closed
                    raise
                return await super().put(key, value, expected_revision=expected_revision, metadata=metadata)
            finally:
                self.active = False

    writer = CancellingWriter()
    store = AssetStore(StorageOverlay(writer, writer=writer))
    await store.initialize()
    pending = asyncio.create_task(store.put(_KEY, b"value"))
    closing: asyncio.Task[None] | None = None
    repeated: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(writer.entered.wait(), 2)
        pending.cancel()
        await asyncio.wait_for(writer.cleaning.wait(), 2)
        closing = asyncio.create_task(store.close())
        await asyncio.sleep(0)
        closing.cancel()
        await asyncio.sleep(0)
        closing.cancel()
        repeated = asyncio.create_task(store.close())
        await asyncio.sleep(0)
        assert not closing.done()
        assert not repeated.done()
        assert not writer.closed
        writer.release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        with pytest.raises(asyncio.CancelledError):
            await closing
        await repeated
        assert writer.closed
    finally:
        writer.release.set()
        await asyncio.gather(*(task for task in (pending, closing, repeated) if task is not None), return_exceptions=True)
        await store.close()


async def test_overlay_lifecycle_failed_close_never_reopens_write_admission() -> None:
    class FailingClose(InMemoryAssetBackend):
        failed = False

        async def close(self) -> None:
            if not self.failed:
                self.failed = True
                raise OSError("close failed")
            await super().close()

    backend = FailingClose()
    overlay = StorageOverlay(backend, writer=backend)
    await overlay.initialize()
    await overlay.put(_KEY, b"value")
    with pytest.raises(OSError, match="close failed"):
        await overlay.close()
    with pytest.raises(AIError) as rejected:
        await asyncio.wait_for(overlay.put(_KEY, b"late"), 2)
    assert rejected.value.code is ErrorCode.STORAGE_CLOSED
    await overlay.close()
    await overlay.close()
    assert await backend.get(_KEY) == b"value"


async def test_overlay_lifecycle_sqlite_write_finishes_after_object_preparation(tmp_path: Path) -> None:
    class PausedObjects(InMemoryObjectStore):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def put(
            self, key: str, chunks: AsyncIterator[bytes], *, expected_size: int, expected_digest: str,
        ) -> ObjectStat:
            self.entered.set()
            await self.release.wait()
            return await super().put(key, chunks, expected_size=expected_size, expected_digest=expected_digest)

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'assets.sqlite'}")
    objects = PausedObjects()
    writer = SqlAssetBackend(engine, namespace="lifecycle", object_store=objects, payload_policy=PayloadPolicy(0))
    store = AssetStore(StorageOverlay(writer, writer=writer))
    pending: asyncio.Task[AssetInfo] | None = None
    closing: asyncio.Task[None] | None = None
    try:
        await provision_asset_database(engine)
        await store.initialize()
        pending = asyncio.create_task(store.put(_KEY, b"external object"))
        await asyncio.wait_for(objects.entered.wait(), 2)
        closing = asyncio.create_task(store.close())
        await asyncio.sleep(0)
        assert not closing.done()
        objects.release.set()
        info = await pending
        await closing
        assert info.size == len(b"external object")
        reader = AssetStore(StorageOverlay(SqlAssetBackend(engine, namespace="lifecycle", object_store=objects)))
        try:
            await reader.initialize()
            assert await reader.get(_KEY) == b"external object"
        finally:
            await reader.close()
    finally:
        objects.release.set()
        await asyncio.gather(*(task for task in (pending, closing) if task is not None), return_exceptions=True)
        await store.close()
        await engine.dispose()
