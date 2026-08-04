#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Process-local execution event contracts and sinks."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, TypeAlias

logger = logging.getLogger("linktools.ai.execution.live_events")


class ExecutionAlreadySubscribedError(RuntimeError):
    """Raised when an execution already has an active consumer."""


class LiveEventConsumerSlowError(RuntimeError):
    """Raised when a consumer cannot drain an execution queue in time."""


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    execution_id: str
    sequence: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True, slots=True)
class AssistantTextDelta(ExecutionEvent):
    text: str = ""


@dataclass(frozen=True, slots=True)
class AssistantThoughtDelta(ExecutionEvent):
    text: str = ""


@dataclass(frozen=True, slots=True)
class ToolCallStarted(ExecutionEvent):
    tool_call_id: str = ""
    tool_name: str = ""
    arguments: Any = None
    status: str = "pending"


@dataclass(frozen=True, slots=True)
class ToolCallProgress(ExecutionEvent):
    tool_call_id: str = ""
    tool_name: str = ""
    arguments: Any = None
    status: str = "in_progress"
    result: Any = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ToolCallCompleted(ExecutionEvent):
    tool_call_id: str = ""
    tool_name: str = ""
    arguments: Any = None
    status: str = "completed"
    result: Any = None


@dataclass(frozen=True, slots=True)
class ToolCallFailed(ExecutionEvent):
    tool_call_id: str = ""
    tool_name: str = ""
    arguments: Any = None
    status: str = "failed"
    error: str = ""


@dataclass(frozen=True, slots=True)
class PlanUpdated(ExecutionEvent):
    entries: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class UsageUpdated(ExecutionEvent):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_cost: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionPaused(ExecutionEvent):
    approval_id: str = ""
    tool_call_id: str = ""
    tool_name: str = ""
    approval_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionCompleted(ExecutionEvent):
    stop_reason: str = "end_turn"


@dataclass(frozen=True, slots=True)
class ExecutionFailed(ExecutionEvent):
    error_id: str = ""
    error_type: str = ""


@dataclass(frozen=True, slots=True)
class ExecutionCancelled(ExecutionEvent):
    reason: str | None = None


ExecutionTerminalEvent: TypeAlias = (
    ExecutionCompleted | ExecutionFailed | ExecutionCancelled
)


class ExecutionEventSubscription:
    def __init__(self, hub: "ExecutionEventHub", execution_id: str, queue: asyncio.Queue) -> None:
        self._hub = hub
        self.execution_id = execution_id
        self._queue = queue
        self._closed = False

    def __aiter__(self) -> "ExecutionEventSubscription":
        return self

    async def __anext__(self) -> ExecutionEvent:
        if self._closed:
            raise StopAsyncIteration
        try:
            event = await self._queue.get()
        except asyncio.CancelledError:
            self._closed = True
            await self._hub.consumer_failed(self.execution_id, self)
            raise
        if isinstance(event, (ExecutionCompleted, ExecutionFailed, ExecutionCancelled)):
            self._closed = True
        return event

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._hub.detach(self.execution_id, self)

    async def release(self) -> None:
        """Detach without treating the consumer as a failed execution."""
        if self._closed:
            return
        self._closed = True
        await self._hub.release(self.execution_id, self)


class ExecutionEventHub:
    """Route one ordered, bounded queue to the consumer of each execution."""

    def __init__(self, *, queue_size: int = 256, publish_timeout: float = 5.0) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if publish_timeout <= 0:
            raise ValueError("publish_timeout must be positive")
        self._queue_size = queue_size
        self._publish_timeout = publish_timeout
        self._subscriptions: dict[str, ExecutionEventSubscription] = {}
        self._queues: dict[str, asyncio.Queue] = {}
        self._sequences: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._cancel_callback: Any = None

    def set_cancel_callback(self, callback: Any) -> None:
        self._cancel_callback = callback

    @property
    def active_subscription_count(self) -> int:
        return len(self._subscriptions)

    async def subscribe(self, execution_id: str) -> ExecutionEventSubscription:
        async with self._lock:
            if execution_id in self._subscriptions:
                raise ExecutionAlreadySubscribedError(execution_id)
            queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
            subscription = ExecutionEventSubscription(self, execution_id, queue)
            self._subscriptions[execution_id] = subscription
            self._queues[execution_id] = queue
            self._sequences[execution_id] = 0
            logger.debug("execution event subscription attached execution=%s", execution_id)
            return subscription

    async def publish(self, execution_id: str, event: ExecutionEvent) -> None:
        subscription = self._subscriptions.get(execution_id)
        queue = self._queues.get(execution_id)
        if subscription is None or queue is None:
            return
        sequence = self._sequences[execution_id] + 1
        self._sequences[execution_id] = sequence
        if event.sequence != sequence or event.execution_id != execution_id:
            event = _with_event_identity(event, execution_id, sequence)
        try:
            await asyncio.wait_for(queue.put(event), self._publish_timeout)
            if isinstance(event, (ExecutionCompleted, ExecutionFailed, ExecutionCancelled)):
                await self._discard(execution_id, subscription)
        except asyncio.TimeoutError as exc:
            await self._discard(execution_id, subscription)
            logger.warning("execution event consumer slow execution=%s", execution_id)
            raise LiveEventConsumerSlowError(execution_id) from exc

    async def close(self, execution_id: str, terminal_event: ExecutionTerminalEvent) -> None:
        subscription = self._subscriptions.get(execution_id)
        queue = self._queues.get(execution_id)
        if subscription is None or queue is None:
            return
        await self.publish(execution_id, terminal_event)
        await self._discard(execution_id, subscription)

    async def detach(self, execution_id: str, subscription: ExecutionEventSubscription) -> None:
        if self._subscriptions.get(execution_id) is subscription:
            await self.consumer_failed(execution_id, subscription)

    async def release(self, execution_id: str, subscription: ExecutionEventSubscription) -> None:
        await self._discard(execution_id, subscription)

    async def consumer_failed(self, execution_id: str, subscription: ExecutionEventSubscription) -> None:
        if self._subscriptions.get(execution_id) is subscription and self._cancel_callback is not None:
            try:
                await self._cancel_callback(execution_id)
            except Exception:
                pass
        if self._subscriptions.get(execution_id) is subscription:
            await self._discard(execution_id, subscription)

    async def _discard(self, execution_id: str, subscription: ExecutionEventSubscription) -> None:
        async with self._lock:
            if self._subscriptions.get(execution_id) is subscription:
                self._subscriptions.pop(execution_id, None)
                self._queues.pop(execution_id, None)
                self._sequences.pop(execution_id, None)
                logger.debug("execution event subscription detached execution=%s", execution_id)


def _with_event_identity(event: ExecutionEvent, execution_id: str, sequence: int) -> ExecutionEvent:
    from dataclasses import replace

    return replace(event, execution_id=execution_id, sequence=sequence)


class RunLiveEventSink(Protocol):
    async def publish(self, event: Any) -> None: ...


class SecurityEventSink(Protocol):
    async def emit(self, event: Any) -> None: ...


class NoopRunLiveEventSink:
    async def publish(self, event: Any, *args: Any) -> None:
        return None


class NoopSecurityEventSink:
    async def emit(self, event: Any) -> None:
        return None


class StreamingRunLiveSink:
    """A :class:`RunLiveEventSink` that fans live engine events into an
    :class:`asyncio.Queue`, so a consumer can stream them as they are produced
    (rather than waiting for the run to finish).

    One sink instance is wired into the Runtime for its lifetime; each
    execution attaches its own queue and detaches it when the execution ends.
    Publishing before attach or after detach is a no-op."""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[Any]] = {}

    def attach(self, execution_id: str = "legacy") -> "asyncio.Queue[Any]":
        queue: "asyncio.Queue[Any]" = asyncio.Queue()
        self._queues[execution_id] = queue
        return queue

    def detach(self, execution_id: str = "legacy") -> None:
        self._queues.pop(execution_id, None)

    async def publish(self, event: Any, execution_id: str = "legacy") -> None:
        queue = self._queues.get(execution_id)
        if queue is None:
            return
        await queue.put(_typed_to_legacy(event))

    async def publish_execution(self, execution_id: str, event: Any) -> None:
        await self.publish(event, execution_id)


class SecurityEventSinkEmitter:
    def __init__(self, sink: SecurityEventSink) -> None:
        self._sink = sink

    async def emit_security(self, event: Any) -> None:
        await self._sink.emit(event)

    async def emit_observability(self, event: Any) -> None:
        await self._sink.emit(event)


async def publish_execution_event(
    sink: "RunLiveEventSink | ExecutionEventHub",
    execution_id: str,
    event: Any,
) -> None:
    """Publish through the execution-scoped hub or the legacy CLI sink."""
    if isinstance(sink, ExecutionEventHub):
        await sink.publish(execution_id, _legacy_event(execution_id, event))
        return
    publish_execution = getattr(sink, "publish_execution", None)
    if publish_execution is not None:
        await publish_execution(execution_id, event)
        return
    await sink.publish(event)


def _legacy_event(execution_id: str, event: Any) -> ExecutionEvent:
    if isinstance(event, ExecutionEvent):
        return event
    if not isinstance(event, dict):
        raise TypeError("live event must be a typed event or mapping")
    kind = event.get("type")
    if kind == "text":
        return AssistantTextDelta(execution_id=execution_id, text=str(event.get("text", "")))
    if kind in {"thinking", "thought"}:
        return AssistantThoughtDelta(execution_id=execution_id, text=str(event.get("text", "")))
    if kind == "tool":
        phase = event.get("phase")
        common = {
            "execution_id": execution_id,
            "tool_call_id": str(event.get("tool_call_id", "")),
            "tool_name": str(event.get("name", "")),
            "arguments": event.get("arguments"),
        }
        if phase == "start":
            return ToolCallStarted(**common)
        if phase == "end" and event.get("ok"):
            return ToolCallCompleted(**common, result=event.get("result"))
        if phase == "end":
            return ToolCallFailed(**common, error=str(event.get("error", "tool failed")))
        return ToolCallProgress(**common, result=event.get("result"), error=event.get("error"))
    raise TypeError(f"unsupported live event type: {kind!r}")


def _typed_to_legacy(event: Any) -> Any:
    if not isinstance(event, ExecutionEvent):
        return event
    if isinstance(event, AssistantTextDelta):
        return {"type": "text", "text": event.text}
    if isinstance(event, AssistantThoughtDelta):
        return {"type": "thinking", "text": event.text}
    if isinstance(event, ToolCallStarted):
        return {
            "type": "tool",
            "phase": "start",
            "name": event.tool_name,
            "tool_call_id": event.tool_call_id,
            "arguments": event.arguments,
        }
    if isinstance(event, ToolCallCompleted):
        return {
            "type": "tool",
            "phase": "end",
            "name": event.tool_name,
            "tool_call_id": event.tool_call_id,
            "arguments": event.arguments,
            "result": event.result,
            "ok": True,
        }
    if isinstance(event, ToolCallFailed):
        return {
            "type": "tool",
            "phase": "end",
            "name": event.tool_name,
            "tool_call_id": event.tool_call_id,
            "arguments": event.arguments,
            "error": event.error,
            "ok": False,
        }
    if isinstance(event, ToolCallProgress):
        return {
            "type": "tool",
            "phase": "progress",
            "name": event.tool_name,
            "tool_call_id": event.tool_call_id,
            "arguments": event.arguments,
            "result": event.result,
            "error": event.error,
        }
    if isinstance(event, ExecutionCompleted):
        return {"type": "completed", "stop_reason": event.stop_reason}
    if isinstance(event, ExecutionFailed):
        return {"type": "failed", "error_id": event.error_id}
    if isinstance(event, ExecutionCancelled):
        return {"type": "cancelled", "reason": event.reason}
    return event


__all__ = [
    "NoopRunLiveEventSink",
    "NoopSecurityEventSink",
    "RunLiveEventSink",
    "SecurityEventSink",
    "SecurityEventSinkEmitter",
    "StreamingRunLiveSink",
    "ExecutionAlreadySubscribedError",
    "LiveEventConsumerSlowError",
    "ExecutionEvent",
    "ExecutionEventHub",
    "ExecutionEventSubscription",
    "ExecutionTerminalEvent",
    "AssistantTextDelta",
    "AssistantThoughtDelta",
    "ToolCallStarted",
    "ToolCallProgress",
    "ToolCallCompleted",
    "ToolCallFailed",
    "PlanUpdated",
    "UsageUpdated",
    "ExecutionPaused",
    "ExecutionCompleted",
    "ExecutionFailed",
    "ExecutionCancelled",
    "publish_execution_event",
]
