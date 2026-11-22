#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional bounded preview transport; durable execution state remains authoritative."""

import asyncio
import json
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from ..core import JsonValue, canonical_json_bytes
from ..errors import AIError, ErrorCode
from ._service_api import ExecutionStreamEvent

if TYPE_CHECKING:
    from ..storage import StreamBackend

_MAX_BYTES = 1024 * 1024
_MAX_RECORDS = 256
_IO_TIMEOUT = 5.0
_OBSERVATION_INTERVAL = 2.0
OBSERVATION_TIMEOUT = 6.0
T = TypeVar("T")
_local_only: ContextVar[bool] = ContextVar("local_execution_stream_only", default=False)
_positions: ContextVar[Mapping[str, tuple[str, int]] | None] = ContextVar("shared_stream_positions", default=None)
_observation: ContextVar[Callable[[str, tuple[str, int], dict[str, JsonValue]], None] | None] = ContextVar(
    "shared_stream_observation", default=None,
)


@dataclass(frozen=True, slots=True)
class SharedStreamEvent(ExecutionStreamEvent):
    position: tuple[str, int] = ("", 0)


@contextmanager
def local_execution_stream() -> Iterator[None]:
    """Opt out of remote durable polling for this live observation."""
    token = _local_only.set(True)
    try:
        yield
    finally:
        _local_only.reset(token)


def local_execution_stream_requested() -> bool:
    return _local_only.get()


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


@contextmanager
def shared_stream_observation(
    observe: Callable[[str, tuple[str, int], dict[str, JsonValue]], None],
) -> Iterator[None]:
    previous = _observation.get()

    def combined(execution_id: str, position: tuple[str, int], record: dict[str, JsonValue]) -> None:
        if previous is not None:
            previous(execution_id, position, record)
        observe(execution_id, position, record)

    token = _observation.set(combined)
    try:
        yield
    finally:
        _observation.reset(token)


def deliver_shared_observation(execution_id: str, position: tuple[str, int], record: dict[str, JsonValue]) -> None:
    observe = _observation.get()
    if observe is not None:
        observe(execution_id, position, record)


def is_fresh_observation_coverage(record: Mapping[str, JsonValue]) -> bool:
    issued = record.get("published_at")
    now = time.time()
    return type(issued) in (int, float) and now - OBSERVATION_TIMEOUT < issued < now + OBSERVATION_TIMEOUT


def stream_key(namespace: str, tenant_id: str, execution_id: str, generation: int, claim: str) -> bytes:
    return canonical_json_bytes(["execution-preview", namespace, tenant_id, execution_id, generation, claim])


def decode_stream_record(payload: bytes) -> dict[str, JsonValue]:
    try:
        value = json.loads(payload)
    except (ValueError, UnicodeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(value, dict) or value.get("kind") not in ("delta", "event", "seal", "gap"):
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
        self.model_coverage = False
        self.tree_coverage = False
        self.durable_high_water = 0
        self.operations: set[asyncio.Task[object]] = set()
        self.task = asyncio.create_task(self.run(), name="execution-preview-publisher")
        self.task.add_done_callback(lambda _: self.settled(self))

    def put(self, value: dict[str, JsonValue], *, seal: bool = False) -> None:
        if self.failed or self.finished:
            return
        if self.queue and not self.queue[-1][1]:
            previous = json.loads(self.queue[-1][0])
            if (value.get("kind") == previous.get("kind") == "delta"
                    and value.get("type") == previous.get("type")
                    and value.get("position") == previous.get("position")
                    and not (value.get("position") or {}).get("part_start")):
                value = dict(value, text=previous["text"] + value["text"],
                             truncated=previous["truncated"] or value["truncated"])
                old, _ = self.queue.pop()
                self.size -= len(old)
            elif (value.get("kind") == previous.get("kind") == "event"
                  and "observation" not in value and "observation" not in previous):
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

    def coverage(self) -> dict[str, JsonValue]:
        return {"kind": "coverage", "model": self.model_coverage, "tree": self.tree_coverage,
                "published_at": time.time()}

    def observe(self, record: dict[str, JsonValue]) -> None:
        if self.durable_high_water:
            self.put({"kind": "event", "durable": self.durable_high_water, "observation": record})

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
            writer = await self.invoke(self.backend.open_stream(self.key, uuid.uuid4().bytes))
            sequence = 0
            loop = asyncio.get_running_loop()
            heartbeat = loop.time() + _OBSERVATION_INTERVAL
            while True:
                if not self.queue:
                    self.wakeup.clear()
                    if self.durable_high_water and (self.model_coverage or self.tree_coverage):
                        try:
                            await asyncio.wait_for(self.wakeup.wait(), timeout=max(0.0, heartbeat - loop.time()))
                        except TimeoutError:
                            pass
                    else:
                        await self.wakeup.wait()
                if self.queue:
                    payload, seal = self.queue.popleft()
                    self.size -= len(payload)
                    sequence += 1
                    await self.invoke(writer.seal(sequence, payload) if seal else writer.append(sequence, payload))
                    if seal:
                        return
                if (self.durable_high_water and (self.model_coverage or self.tree_coverage)
                        and not self.finished and loop.time() >= heartbeat):
                    self.observe(self.coverage())
                    heartbeat = loop.time() + _OBSERVATION_INTERVAL
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

    def admit(self, execution_id: str, generation: int, claim: str, *, durable_seq: int = 0) -> None:
        if self.closed:
            return
        if self.generations.get(execution_id) == (generation, claim):
            producer = self.producers.get(execution_id)
            if producer is not None and durable_seq > producer.durable_high_water:
                first = producer.durable_high_water == 0
                producer.durable_high_water = durable_seq
                if first and (producer.model_coverage or producer.tree_coverage):
                    producer.observe(producer.coverage())
            return
        self.generations[execution_id] = (generation, claim)
        previous = self.producers.get(execution_id)
        if previous is not None:
            previous.put({"kind": "gap"}, seal=True)
        producer = _Producer(
            self.backend, stream_key(self.namespace, self.tenant_id, execution_id, generation, claim), self.release,
        )
        producer.durable_high_water = durable_seq
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

    def delta(
        self, execution_id: str, event_type: str, content: str, truncated: bool,
        position: Mapping[str, JsonValue] | None = None,
    ) -> None:
        producer = self.producers.get(execution_id)
        if producer is not None:
            producer.put({"kind": "delta", "type": str(event_type), "text": content, "truncated": truncated,
                          **({"position": dict(position)} if position is not None else {})})

    def event(self, execution_id: str, event_type: str, payload: JsonValue, durable_seq: int | None) -> None:
        producer = self.producers.get(execution_id)
        if producer is None or producer.finished or producer.failed:
            return
        if durable_seq is not None:
            first = producer.durable_high_water == 0
            producer.durable_high_water = max(producer.durable_high_water, durable_seq)
            producer.put({"kind": "event", "durable": durable_seq})
            if first and (producer.model_coverage or producer.tree_coverage):
                producer.observe(producer.coverage())

    def cover_models(self, execution_id: str, generation: int, claim: str) -> None:
        producer = self.producers.get(execution_id)
        if self.generations.get(execution_id) == (generation, claim) and producer is not None:
            producer.model_coverage = True
            producer.observe(producer.coverage())

    def model(self, execution_id: str, generation: int, claim: str, item: JsonValue) -> None:
        producer = self.producers.get(execution_id)
        if (self.generations.get(execution_id) == (generation, claim)
                and producer is not None and producer.model_coverage):
            if item is None:
                producer.model_coverage = False
                producer.observe(producer.coverage())
            else:
                producer.observe({"kind": "model", "item": item})

    def cover_tree(self, execution_id: str, generation: int) -> None:
        producer = self.producers.get(execution_id)
        authority = self.generations.get(execution_id)
        if producer is not None and authority is not None and authority[0] == generation:
            producer.tree_coverage = True
            producer.observe(producer.coverage())

    def child(self, execution_id: str, generation: int, child_id: str | None) -> None:
        producer = self.producers.get(execution_id)
        authority = self.generations.get(execution_id)
        if (producer is None or authority is None or authority[0] != generation
                or not producer.tree_coverage):
            return
        if child_id is None:
            producer.tree_coverage = False
            producer.observe(producer.coverage())
        else:
            producer.observe({"kind": "child", "execution_id": child_id})

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
           "shared_stream_positions", "shared_stream_position", "shared_stream_requested",
           "local_execution_stream", "local_execution_stream_requested",
           "OBSERVATION_TIMEOUT", "shared_stream_observation", "deliver_shared_observation",
           "is_fresh_observation_coverage"]
