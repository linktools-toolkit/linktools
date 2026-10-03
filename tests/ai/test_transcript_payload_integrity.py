#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transcript ObjectStore integrity and stream ownership contracts."""

import asyncio
import hashlib
from collections.abc import AsyncIterator

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._history import TranscriptRepository
from linktools.ai.storage import ObjectRef, StoredPayload


class _Stream:
    def __init__(
        self,
        chunks: tuple[bytes, ...],
        *,
        block_read: bool = False,
        block_close: bool = False,
        read_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.chunks = iter(chunks)
        self.read_error = read_error
        self.close_error = close_error
        self.read_started = asyncio.Event()
        self.close_started = asyncio.Event()
        self.read_release = asyncio.Event()
        self.close_release = asyncio.Event()
        self.closed = False
        self.read_count = 0
        if not block_read:
            self.read_release.set()
        if not block_close:
            self.close_release.set()

    def __aiter__(self) -> "_Stream":
        return self

    async def __anext__(self) -> bytes:
        self.read_started.set()
        await self.read_release.wait()
        if self.read_error is not None:
            raise self.read_error
        try:
            value = next(self.chunks)
        except StopIteration:
            raise StopAsyncIteration from None
        self.read_count += 1
        return value

    async def aclose(self) -> None:
        self.close_started.set()
        await self.close_release.wait()
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class _Store:
    def __init__(self, stream: _Stream) -> None:
        self.stream = stream

    def open(self, key: str) -> AsyncIterator[bytes]:
        assert key == "transcript"
        return self.stream


def _reader(stream: _Stream) -> TranscriptRepository:
    return TranscriptRepository(
        object(),  # type: ignore[arg-type]
        object_store=_Store(stream),  # type: ignore[arg-type]
        namespace="history",
        tenant_id="tenant",
        runtime_domain=RuntimeDomain.EXECUTION,
    )


def _payload() -> StoredPayload:
    return StoredPayload.object(
        ObjectRef("runtime", "transcript", hashlib.sha256(b"abcd").hexdigest(), 4)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunks, valid, expected_reads",
    (
        ((b"ab", b"cd"), True, 2),
        ((b"ab", b"XY"), False, 2),
        ((b"ab",), False, 1),
        ((b"abcde", b"must not be read"), False, 1),
    ),
    ids=("valid", "digest", "truncated", "oversized"),
)
async def test_transcript_object_bytes_are_verified_and_closed(
    chunks: tuple[bytes, ...], valid: bool, expected_reads: int,
) -> None:
    stream = _Stream(chunks)
    request = _reader(stream).read_payload(_payload())
    if valid:
        assert await request == b"abcd"
    else:
        with pytest.raises(AIError) as raised:
            await request
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert stream.closed
    assert stream.read_count == expected_reads


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("none", "digest", "oversized", "read"))
async def test_transcript_object_close_failure_preserves_reader_precedence(failure: str) -> None:
    close_error = ValueError("close failed")
    read_error = OSError("read failed") if failure == "read" else None
    chunk = {"none": b"abcd", "digest": b"WXYZ", "oversized": b"abcde", "read": b"abcd"}[failure]
    stream = _Stream((chunk,), read_error=read_error, close_error=close_error)
    with pytest.raises(ValueError) as raised:
        await _reader(stream).read_payload(_payload())
    assert raised.value is close_error
    assert stream.closed
    if read_error is not None:
        assert raised.value.__context__ is read_error
    elif failure == "oversized":
        assert isinstance(raised.value.__context__, AIError)
        assert raised.value.__context__.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_transcript_object_read_failure_is_preserved_after_close() -> None:
    read_error = OSError("read failed")
    stream = _Stream((), read_error=read_error)
    with pytest.raises(OSError) as raised:
        await _reader(stream).read_payload(_payload())
    assert raised.value is read_error
    assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_during", ("read", "close"))
@pytest.mark.parametrize("close_fails", (False, True))
async def test_transcript_object_cancellation_waits_for_owned_close(
    cancel_during: str, close_fails: bool,
) -> None:
    close_error = ValueError("close failed") if close_fails else None
    stream = _Stream(
        (b"abcd",), block_read=cancel_during == "read", block_close=True,
        close_error=close_error,
    )
    task = asyncio.create_task(_reader(stream).read_payload(_payload()))
    try:
        if cancel_during == "read":
            await asyncio.wait_for(stream.read_started.wait(), timeout=2)
            task.cancel()
        await asyncio.wait_for(stream.close_started.wait(), timeout=2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert not stream.closed
        stream.close_release.set()
        with pytest.raises(ValueError if close_fails else asyncio.CancelledError) as raised:
            await asyncio.wait_for(task, timeout=2)
        if close_fails:
            assert raised.value is close_error
        assert stream.closed
    finally:
        stream.read_release.set()
        stream.close_release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
