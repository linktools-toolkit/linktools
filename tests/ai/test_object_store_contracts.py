#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend-neutral object input and immutable content contracts."""

import hashlib
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.runtime.state._codec import decode_domain, encode_domain
from linktools.ai.storage import (
    FilesystemObjectStore,
    InMemoryObjectStore,
    ObjectRef,
    ObjectStore,
    ObjectStoreInspection,
    SqlObjectStore,
    read_object,
)

_DIGEST = hashlib.sha256(b"x").hexdigest()


@pytest_asyncio.fixture(params=("memory", "filesystem", "sqlite"))
async def object_store(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> AsyncIterator[ObjectStore]:
    if request.param == "memory":
        yield InMemoryObjectStore()
    elif request.param == "filesystem":
        yield FilesystemObjectStore(tmp_path / "objects")
    else:
        engine = create_async_engine(
            URL.create("sqlite+aiosqlite", database=str(tmp_path / "objects.db"))
        )
        try:
            await provision_database(engine)
            yield SqlObjectStore(engine)
        finally:
            await engine.dispose()


async def _chunks(payload: bytes) -> AsyncIterator[bytes]:
    if payload:
        yield payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "size", "digest"),
    (
        pytest.param("payload", False, hashlib.sha256(b"").hexdigest(), id="false-size"),
        pytest.param("payload", True, _DIGEST, id="true-size"),
        pytest.param("payload", -1, _DIGEST, id="negative-size"),
        pytest.param("payload", 1.0, _DIGEST, id="float-size"),
        pytest.param("", 1, _DIGEST, id="empty-key"),
        pytest.param("nul\x00key", 1, _DIGEST, id="nul-key"),
        pytest.param(1, 1, _DIGEST, id="non-string-key"),
        pytest.param("payload", 1, "", id="empty-digest"),
        pytest.param("payload", 1, "g" * 64, id="non-hex-digest"),
        pytest.param("payload", 1, None, id="non-string-digest"),
    ),
)
async def test_put_rejects_invalid_metadata_before_consuming_chunks(
    object_store: ObjectStore,
    key: object,
    size: object,
    digest: object,
) -> None:
    consumed = False

    async def chunks() -> AsyncIterator[bytes]:
        nonlocal consumed
        consumed = True
        if size is not False:
            yield b"x"

    with pytest.raises(ValueError):
        await object_store.put(
            key, chunks(), expected_size=size, expected_digest=digest
        )
    assert consumed is False
    assert isinstance(object_store, ObjectStoreInspection)
    assert [value async for value in object_store.list_objects()] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", (b"", b"object content"), ids=("empty", "nonempty"))
async def test_put_roundtrips_and_preserves_immutable_content(
    object_store: ObjectStore,
    payload: bytes,
) -> None:
    digest = hashlib.sha256(payload).hexdigest()
    first = await object_store.put(
        "payload", _chunks(payload), expected_size=len(payload), expected_digest=digest
    )
    assert first.size == len(payload)
    assert type(first.size) is int
    assert await object_store.stat("payload") == first
    assert await read_object(
        object_store, "payload", expected_size=len(payload), expected_digest=digest
    ) == payload
    assert await object_store.put(
        "payload", _chunks(payload), expected_size=len(payload), expected_digest=digest
    ) == first

    different = payload + b"changed"
    with pytest.raises(AIError) as raised:
        await object_store.put(
            "payload",
            _chunks(different),
            expected_size=len(different),
            expected_digest=hashlib.sha256(different).hexdigest(),
        )
    assert raised.value.code is ErrorCode.STORAGE_CONFLICT
    assert await object_store.stat("payload") == first
    await object_store.validate_integrity()


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ("too-short", "too-long", "digest"))
async def test_put_integrity_failure_does_not_publish_an_object(
    object_store: ObjectStore,
    mismatch: str,
) -> None:
    payload = b"x"
    size = {"too-short": 2, "too-long": 0, "digest": 1}[mismatch]
    digest = "a" * 64 if mismatch == "digest" else hashlib.sha256(payload).hexdigest()
    with pytest.raises(AIError) as raised:
        await object_store.put(
            "payload", _chunks(payload), expected_size=size, expected_digest=digest
        )
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert await object_store.stat("payload") is None


@pytest.mark.parametrize("size", (False, True, -1, 1.0))
def test_object_ref_rejects_invalid_size(size: object) -> None:
    with pytest.raises(ValueError):
        ObjectRef("objects", "payload", "a" * 64, size)


@pytest.mark.parametrize("size", (0, 1))
def test_object_ref_roundtrips_through_domain_codec(size: int) -> None:
    reference = ObjectRef("objects", "payload", "a" * 64, size)
    assert decode_domain(encode_domain(reference), ObjectRef) == reference


@pytest.mark.parametrize("size", (False, True))
def test_object_ref_reader_rejects_boolean_size(size: bool) -> None:
    encoded = encode_domain(ObjectRef("objects", "payload", "a" * 64, 0))
    encoded["fields"]["size"] = size
    with pytest.raises(AIError) as raised:
        decode_domain(encoded, ObjectRef)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
