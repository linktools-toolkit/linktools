#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Archive object readers preserve stored-byte integrity and error ownership."""

import zlib
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence
from linktools.ai.persistence._archive import ContentArchive, PreparedArchive
from linktools.ai.storage import InMemoryObjectStore, ObjectRef, read_object

from .test_incremental_projection_boundaries import _storage
from .test_shared_content_archive import _archive, _publish


@pytest.mark.asyncio
@pytest.mark.parametrize("reader", ("default", "custom"))
@pytest.mark.parametrize("failure", (False, True))
async def test_archive_object_reader_preserves_bytes_and_source_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str, failure: bool,
) -> None:
    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="archive", tenant_id="tenant")
    source_error = OSError("object source unavailable")
    objects = InMemoryObjectStore()

    async def custom(reference: ObjectRef) -> bytes:
        return await read_object(objects, reference.key, expected_digest=reference.digest, expected_size=reference.size)

    async def unavailable(_key: str) -> AsyncIterator[bytes]:
        raise source_error
        yield b""  # pragma: no cover

    try:
        owner = await _archive(state)
        archive = ContentArchive(owner.store, owner.scope_id, namespace="archive", tenant_id="tenant",
                                 object_store=objects, object_reader=custom if reader == "custom" else None)
        raw = b"content" * 4000
        prepared = PreparedArchive()
        identities = await archive.prepare_content((("bytes", raw),), prepared)
        await _publish(archive, prepared)
        if failure:
            monkeypatch.setattr(objects, "open", unavailable)
            with pytest.raises(OSError) as caught:
                await archive.read_contents(identities, codec="bytes")
            assert caught.value is source_error
        else:
            assert await archive.read_contents(identities, codec="bytes") == (raw,)
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_custom_archive_reader_cannot_replace_stored_compression(tmp_path: Path) -> None:
    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="archive", tenant_id="tenant")
    objects = InMemoryObjectStore()
    raw = b"content" * 4000

    async def custom(reference: ObjectRef) -> bytes:
        original = await read_object(objects, reference.key, expected_digest=reference.digest, expected_size=reference.size)
        alternative = zlib.compress(raw, level=9)
        assert alternative != original and zlib.decompress(alternative) == raw
        return alternative

    try:
        owner = await _archive(state)
        archive = ContentArchive(owner.store, owner.scope_id, namespace="archive", tenant_id="tenant",
                                 object_store=objects, object_reader=custom)
        prepared = PreparedArchive()
        identities = await archive.prepare_content((("bytes", raw),), prepared)
        await _publish(archive, prepared)
        with pytest.raises(AIError) as caught:
            await archive.read_contents(identities, codec="bytes")
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()
