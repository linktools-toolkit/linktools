#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inline decode reuse never substitutes for current source observations."""

from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence
from linktools.ai.persistence._archive import ContentArchive, ContentRecord, PreparedArchive
from linktools.ai.persistence.state import StateTransaction
from .test_incremental_projection_boundaries import _storage
from .test_shared_content_archive import _archive, _publish


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_inline_decode_is_once_per_transaction_and_checks_changed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="archive", tenant_id="tenant")
    try:
        archive = await _archive(state)
        prepared = PreparedArchive()
        identities = await archive.prepare_content((("bytes", b"body"),), prepared)
        await _publish(archive, prepared)
        calls = []
        original = ContentArchive._read_content

        async def decode(owner: ContentArchive, content: ContentRecord) -> bytes:
            calls.append(content)
            return await original(owner, content)

        monkeypatch.setattr(ContentArchive, "_read_content", decode)
        async def read_twice(_transaction: StateTransaction) -> None:
            assert await archive.read_contents(identities, codec="bytes") == (b"body",)
            assert await archive.read_contents(identities, codec="bytes") == (b"body",)

        await archive.store.read(read_twice)
        assert len(calls) == 1
        await archive.store.read(read_twice)
        assert len(calls) == 2

        async def corrupt(transaction: StateTransaction) -> None:
            await read_twice(transaction)
            key, = archive.content_identity_keys(identities)
            current = await transaction.get_record(key)
            damaged = replace(current, payload_bytes=b"bad", storage_revision=current.storage_revision + 1)
            assert await transaction.replace_record(damaged, expected_storage_revision=current.storage_revision)
            with pytest.raises(AIError) as caught:
                await archive.read_contents(identities, codec="bytes")
            assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            assert await transaction.delete_record(key)
            with pytest.raises(AIError) as missing:
                await archive.read_contents(identities, codec="bytes")
            assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

        await archive.store.mutate(corrupt)
        assert len(calls) == 4  # new transaction plus corrupt bytes, never cached proof
    finally:
        await state.close()


def test_inline_content_size_rejects_bool_and_float() -> None:
    for size in (True, 1.0):
        with pytest.raises(AIError):
            ContentRecord("bytes", "0" * 64, size, "raw", body=b"x")
