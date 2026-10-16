#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional bounded preview transport; durable execution state remains authoritative."""

import asyncio
import json
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from ..core import JsonValue, canonical_json_bytes
from ..errors import AIError, ErrorCode
from .service_api import ExecutionStreamEvent

if TYPE_CHECKING:
    from ..storage import StreamBackend

_MAX_BYTES = 1024 * 1024
_MAX_RECORDS = 256
_IO_TIMEOUT = 5.0
T = TypeVar("T")
_positions: ContextVar[Mapping[str, tuple[str, int]] | None] = ContextVar("shared_stream_positions", default=None)


@dataclass(frozen=True, slots=True)
class SharedStreamEvent(ExecutionStreamEvent):
    position: tuple[str, int] = ("", 0)


@contextmanager
def shared_stream_positions(values: Mapping[str, tuple[str, int]]) -> Iterator[None]:
    token = _positions.set(values)
    try:
        yield
    finally:
        _positions.reset(token)


def shared_stream_position(execution_id: str) -> tuple[str, int] | None:
    positions = _positions.get()
    return None if positions is None else positions.get(execution_id)


def shared_stream_requested() -> bool:
    return _positions.get() is not None


def stream_key(namespace: str, tenant_id: str, execution_id: str, generation: int, claim: str) -> bytes:
    return canonical_json_bytes(["execution-preview", namespace, tenant_id, execution_id, generation, claim])


def decode_stream_record(payload: bytes) -> dict[str, JsonValue]:
    try:
        value = json.loads(payload)
    except (ValueError, UnicodeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(value, dict) or value.get("kind") not in {"delta", "event", "seal", "gap"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


class _Producer:
    def __init__(self, backend: "StreamBackend", key: bytes, settled: Callable[["_Producer"], None]) -> None:
        self.backend = backend
        self.settled = settled
        self.key = key
        self.queue: deque[tuple[bytes, bool]] = deque()
        self.size = 0
        self.wakeup = asyncio.Event()
        self.finished = False
        self.failed = False
        self.operations: set[asyncio.Task[object]] = set()
        self.task = asyncio.create_task(self.run(), name="execution-preview-publisher")
        self.task.add_done_callback(lambda _: self.settled(self))

    def put(self, value: dict[str, JsonValue], *, seal: bool = False) -> None:
        if self.failed or self.finished:
            return
        if self.queue and not self.queue[-1][1]:
            previous = json.loads(self.queue[-1][0])
            if value.get("kind") == previous.get("kind") == "delta" and value.get("type") == previous.get("type"):
                value = dict(value, text=previous["text"] + value["text"],
                             truncated=previous["truncated"] or value["truncated"])
                old, _ = self.queue.pop()
                self.size -= len(old)
            elif value.get("kind") == previous.get("kind") == "event":
                old, _ = self.queue.pop()
                self.size -= len(old)
        payload = canonical_json_bytes(value)
        if len(self.queue) >= _MAX_RECORDS or self.size + len(payload) > _MAX_BYTES:
            self.queue.clear()
            self.size = 0
            payload = canonical_json_bytes({"kind": "gap"})
            seal = True
        self.queue.append((payload, seal))
        self.size += len(payload)
        self.finished = seal
        self.wakeup.set()

    def collect_operation(self, task: asyncio.Task[object]) -> None:
        self.operations.discard(task)
        if not task.cancelled():
            task.exception()
        self.settled(self)

    async def invoke(self, operation: Awaitable[T]) -> T:
        task = asyncio.create_task(operation)
        self.operations.add(task)
        task.add_done_callback(self.collect_operation)
        try:
            done, _ = await asyncio.wait((task,), timeout=_IO_TIMEOUT)
            if not done:
                task.cancel()
                raise TimeoutError("preview transport timed out")
            return task.result()
        except BaseException:
            if not task.done():
                task.cancel()
            raise

    async def run(self) -> None:
        try:
            # Admission owns this sole attempt. Unknown creation is never retried.
            writer = await self.invoke(self.backend.open(self.key, uuid.uuid4().bytes))
            sequence = 0
            while True:
                if not self.queue:
                    self.wakeup.clear()
                    await self.wakeup.wait()
                while self.queue:
                    payload, seal = self.queue.popleft()
                    self.size -= len(payload)
                    sequence += 1
                    await self.invoke(writer.seal(sequence, payload) if seal else writer.append(sequence, payload))
                    if seal:
                        return
        except asyncio.CancelledError:
            raise
        except Exception:
            # Preview failure cannot roll back or delay business persistence.
            self.failed = True
        finally:
            self.queue.clear()
            self.size = 0


class SharedStreamPublisher:
    def __init__(self, backend: "StreamBackend", namespace: str, tenant_id: str) -> None:
        self.backend = backend
        self.namespace = namespace
        self.tenant_id = tenant_id
        self.producers: dict[str, _Producer] = {}
        self.generations: dict[str, tuple[int, str]] = {}
        self.owned: set[_Producer] = set()
        self.closed = False

    def admit(self, execution_id: str, generation: int, claim: str) -> None:
        if self.closed or self.generations.get(execution_id) == (generation, claim):
            return
        self.generations[execution_id] = (generation, claim)
        previous = self.producers.get(execution_id)
        if previous is not None:
            previous.put({"kind": "gap"}, seal=True)
        producer = _Producer(
            self.backend, stream_key(self.namespace, self.tenant_id, execution_id, generation, claim), self.release,
        )
        self.producers[execution_id] = producer
        self.owned.add(producer)

    def release(self, producer: _Producer) -> None:
        if not producer.task.done() or producer.operations:
            return
        if not producer.task.cancelled():
            producer.task.exception()
        self.owned.discard(producer)
        for execution_id, current in tuple(self.producers.items()):
            if current is producer:
                del self.producers[execution_id]

    def delta(self, execution_id: str, event_type: str, content: str, truncated: bool) -> None:
        producer = self.producers.get(execution_id)
        if producer is not None:
            producer.put({"kind": "delta", "type": str(event_type), "text": content, "truncated": truncated})

    def event(self, execution_id: str, event_type: str, payload: JsonValue, durable_seq: int | None) -> None:
        producer = self.producers.get(execution_id)
        if producer is None or producer.finished or producer.failed:
            return
        if durable_seq is not None:
            producer.put({"kind": "event", "durable": durable_seq})

    def complete(self, execution_id: str) -> None:
        self.generations.pop(execution_id, None)
        producer = self.producers.get(execution_id)
        if producer is not None:
            producer.put({"kind": "seal"}, seal=True)

    def handoff(self, execution_id: str) -> None:
        self.generations.pop(execution_id, None)
        producer = self.producers.get(execution_id)
        if producer is not None:
            producer.put({"kind": "gap"}, seal=True)

    async def close(self) -> None:
        self.closed = True
        self.generations.clear()
        producers = tuple(self.owned)
        for producer in producers:
            producer.put({"kind": "gap"}, seal=True)
        tasks = {producer.task for producer in producers if not producer.task.done()}
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=_IO_TIMEOUT)
            for task in pending:
                task.cancel()
        operations = {task for producer in producers for task in producer.operations if not task.done()}
        for task in operations:
            task.cancel()
        unsettled = {task for task in (*tasks, *operations) if not task.done()}
        if unsettled:
            _, unsettled = await asyncio.wait(unsettled, timeout=_IO_TIMEOUT)
        if unsettled:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED, safe_details={"phase": "preview_close"})


__all__ = ["SharedStreamPublisher", "SharedStreamEvent", "stream_key", "decode_stream_record",
           "shared_stream_positions", "shared_stream_position", "shared_stream_requested"]
