#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded ordered transport for disposable, opaque stream records."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..errors import AIError, ErrorCode


@dataclass(frozen=True, slots=True)
class StreamBatch:
    """Records after a position, with explicit retention loss and end-of-stream.

    ``sealed`` is true only when this batch reaches the sealed tail, not when a
    limited batch still has unread records. A missing stream returns ``gap``.
    """

    records: tuple[tuple[int, bytes], ...]
    gap: bool
    sealed: bool


class StreamWriter(Protocol):
    """A producer handle that becomes invalid when its stream is evicted.

    Sequences start at one and advance without holes. Retained identical writes
    are idempotent; conflicting or already-trimmed retries raise STORAGE_CONFLICT.
    A seal appends its final record atomically and rejects new writes thereafter.
    """

    async def append(self, sequence: int, payload: bytes) -> None: ...

    async def seal(self, sequence: int, payload: bytes) -> None: ...


@runtime_checkable
class StreamBackend(Protocol):
    """Independent fan-out reads of bounded streams, without business authority.

    The caller supplies generation-scoped keys and must never reopen a key after
    its stream is lost. An unknown open outcome does not authorize a retry.
    Existing streams may be opened again only by their original producer.
    Backend clients are borrowed; consumers own and cancel their pending reads.
    """

    async def open_stream(self, key: bytes, producer: bytes) -> StreamWriter: ...

    async def read_stream(
        self, key: bytes, after: int, *, limit: int, wait_seconds: float,
    ) -> StreamBatch: ...


@dataclass(slots=True)
class _MemoryStream:
    producer: bytes
    token: object = field(default_factory=object)
    records: OrderedDict[int, bytes] = field(default_factory=OrderedDict)
    size: int = 0
    sequence: int = 0
    sealed: bool = False
    changed: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def charge(self) -> int:
        return max(1, self.size)

    def notify(self) -> None:
        self.changed.set()
        self.changed = asyncio.Event()


class InMemoryStreamBackend:
    """Shared, event-loop-local stream retention under one logical byte budget.

    ``max_bytes`` covers all streams and also bounds any one stream. Each record
    costs at least one byte; an empty stream costs one byte. This is a payload
    charge, not an RSS limit. Writes evict least-recently-written other streams,
    then trim the current stream's oldest records. Reads do not extend retention.
    Eviction wakes readers and invalidates existing writer handles. No client or
    background task is owned by this backend, and no TTL is imposed.
    """

    def __init__(self, *, max_bytes: int) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative integer")
        self.max_bytes = max_bytes
        self._streams: OrderedDict[bytes, _MemoryStream] = OrderedDict()
        self._size = 0
        self._lock = asyncio.Lock()

    async def open_stream(self, key: bytes, producer: bytes) -> StreamWriter:
        _require_bytes(key)
        _require_bytes(producer)
        async with self._lock:
            stream = self._streams.get(key)
            if stream is None:
                if self.max_bytes == 0:
                    raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
                stream = _MemoryStream(producer)
                self._streams[key] = stream
                self._size += stream.charge
                self._trim(key)
            elif stream.producer != producer:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return _MemoryStreamWriter(self, key, stream.token)

    async def read_stream(
        self, key: bytes, after: int, *, limit: int, wait_seconds: float,
    ) -> StreamBatch:
        _require_bytes(key)
        _require_integer(after, minimum=0)
        _require_integer(limit, minimum=1)
        if (
            isinstance(wait_seconds, bool)
            or not isinstance(wait_seconds, (int, float))
            or not math.isfinite(wait_seconds)
            or wait_seconds < 0
        ):
            raise ValueError("wait_seconds must be finite and non-negative")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_seconds
        observed: object | None = None
        while True:
            async with self._lock:
                stream = self._streams.get(key)
                if stream is None or observed is not None and stream.token is not observed:
                    return StreamBatch((), gap=True, sealed=False)
                observed = stream.token
                floor = next(iter(stream.records), stream.sequence + 1)
                start = max(after + 1, floor)
                records = tuple(
                    (sequence, stream.records[sequence])
                    for sequence in range(start, min(stream.sequence + 1, start + limit))
                )
                gap = after < floor - 1 or after > stream.sequence
                tail = records[-1][0] if records else after
                sealed = stream.sealed and tail >= stream.sequence
                remaining = deadline - loop.time()
                if records or gap or sealed or remaining <= 0:
                    return StreamBatch(records, gap=gap, sealed=sealed)
                changed = stream.changed
            try:
                await asyncio.wait_for(changed.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                pass

    async def _write(
        self, key: bytes, token: object, sequence: int, payload: bytes, *, seal: bool,
    ) -> None:
        _require_integer(sequence, minimum=1)
        _require_bytes(payload)
        async with self._lock:
            stream = self._streams.get(key)
            if stream is None or stream.token is not token:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if sequence <= stream.sequence:
                if (
                    stream.records.get(sequence) != payload
                    or seal != (stream.sealed and sequence == stream.sequence)
                ):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                return
            if stream.sealed or sequence != stream.sequence + 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            charge = max(1, len(payload))
            if charge > self.max_bytes:
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
            self._size -= stream.charge
            stream.records[sequence] = payload
            stream.size += charge
            stream.sequence = sequence
            stream.sealed = seal
            self._size += stream.charge
            self._streams.move_to_end(key)
            self._trim(key)
            stream.notify()

    def _trim(self, key: bytes) -> None:
        while self._size > self.max_bytes:
            oldest = next(iter(self._streams))
            if oldest != key:
                removed = self._streams.pop(oldest)
                self._size -= removed.charge
                removed.notify()
            else:
                stream = self._streams[key]
                _, payload = stream.records.popitem(last=False)
                charge = max(1, len(payload))
                stream.size -= charge
                self._size -= charge


class _MemoryStreamWriter:
    def __init__(self, backend: InMemoryStreamBackend, key: bytes, token: object) -> None:
        self._backend = backend
        self._key = key
        self._token = token

    async def append(self, sequence: int, payload: bytes) -> None:
        await self._backend._write(self._key, self._token, sequence, payload, seal=False)

    async def seal(self, sequence: int, payload: bytes) -> None:
        await self._backend._write(self._key, self._token, sequence, payload, seal=True)


def _require_bytes(value: bytes) -> None:
    if not isinstance(value, bytes):
        raise TypeError("stream identities and payloads must be bytes")


def _require_integer(value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"stream position or limit must be an integer >= {minimum}")


__all__ = ["StreamBackend", "StreamWriter", "StreamBatch", "InMemoryStreamBackend"]
