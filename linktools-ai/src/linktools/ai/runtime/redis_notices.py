#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional Redis Pub/Sub carrier for Runtime turn notices."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Protocol

from linktools.core import environ

from ._turn_notices import TurnNotice, TurnNoticeScope

_logger = environ.get_logger(__name__)
_GAP = '{"type":"gap"}'
_GAP_HINT = object()
_MAX_NOTICE_BYTES = 2048


class _PubSub(Protocol):
    async def __aenter__(self) -> _PubSub: ...

    async def __aexit__(self, *exc: object) -> None: ...

    async def subscribe(self, *channels: str) -> None: ...

    def listen(self) -> AsyncIterator[dict[str, object]]: ...


class _Client(Protocol):
    def pubsub(self, *, ignore_subscribe_messages: bool) -> _PubSub: ...

    async def publish(self, channel: str, message: str) -> int: ...


class RedisTurnNoticeTransport:
    """Borrow a redis.asyncio-compatible client; one instance per bridge.

    Redis Pub/Sub is at-most-once. A reconnect ACK or delivered overflow hint
    asks consumers to reconcile, but a publisher crash/silent loss can still
    leave a subscriber unaware until its next authorized snapshot. The client
    and deployment ACL/isolation remain owned by the host.
    """

    def __init__(self, client: _Client, *, prefix: str, queue_size: int = 128) -> None:
        if not isinstance(prefix, str) or not prefix or len(prefix.encode("utf-8")) > 128:
            raise ValueError("Redis turn notice deployment prefix must be 1..128 UTF-8 bytes")
        if type(queue_size) is not int or not 1 <= queue_size <= 1024:
            raise ValueError("Redis turn notice queue size must be between 1 and 1024")
        self._client = client
        self._prefix = prefix
        self._queue_size = queue_size
        self._scope: TurnNoticeScope | None = None
        self._queue: asyncio.Queue[str] | None = None
        self._loss = False

    def offer(self, notice: TurnNotice) -> None:
        queue = self._queue
        if queue is None or notice.scope != self._scope:
            return
        payload = json.dumps({
            "type": "notice", "session_id": notice.session_id, "turn_id": notice.turn_id,
            "phase": notice.phase, "revision": notice.revision,
        }, separators=(",", ":"), ensure_ascii=False)
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            # A queued item already wakes the publisher; no extra task/queue.
            self._loss = True

    @asynccontextmanager
    async def subscribe(
        self, scope: TurnNoticeScope, *,
        on_notice: Callable[[TurnNotice], None],
        on_gap: Callable[[TurnNoticeScope], None],
    ) -> AsyncIterator[None]:
        if self._scope is not None:
            raise RuntimeError("Redis turn notice transport already subscribed")
        # Encode components rather than concatenating delimiters from user IDs.
        channel = "linktools:turn-notice:" + json.dumps(
            [self._prefix, scope.namespace, scope.tenant_id], separators=(",", ":"), ensure_ascii=False,
        )
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=self._queue_size)
        self._scope, self._queue = scope, queue
        ready = asyncio.Event()
        receiver = asyncio.create_task(self._receive(channel, scope, ready, on_notice, on_gap))
        sender = asyncio.create_task(self._send(channel, queue))
        try:
            await asyncio.wait_for(ready.wait(), timeout=10)
            yield
        finally:
            self._queue = None
            for task in (sender, receiver):
                task.cancel()
            try:
                await asyncio.gather(sender, receiver, return_exceptions=True)
            finally:
                self._scope = None
                self._loss = False

    async def _send(self, channel: str, queue: asyncio.Queue[str]) -> None:
        pending: str | None = None
        while True:
            if pending is None and not self._loss:
                pending = await queue.get()
            try:
                if self._loss:
                    self._loss = False
                    await self._client.publish(channel, _GAP)
                if pending is not None:
                    await self._client.publish(channel, pending)
                    pending = None
            except Exception as error:
                # The status of the last publish is unknown; don't replay it.
                pending = None
                self._loss = True
                _logger.warning("Redis turn notice publish failed: %s", type(error).__name__)
                await asyncio.sleep(1)

    async def _receive(
        self, channel: str, scope: TurnNoticeScope, ready: asyncio.Event,
        on_notice: Callable[[TurnNotice], None], on_gap: Callable[[TurnNoticeScope], None],
    ) -> None:
        channel_bytes = channel.encode("utf-8")
        while True:
            connected = False
            try:
                async with self._client.pubsub(ignore_subscribe_messages=False) as pubsub:
                    await pubsub.subscribe(channel)
                    async for message in pubsub.listen():
                        if message.get("channel") not in (channel, channel_bytes):
                            continue
                        kind = message.get("type")
                        if kind == "subscribe":
                            connected = True
                            if ready.is_set():
                                on_gap(scope)
                            else:
                                ready.set()
                        elif kind == "message" and connected:
                            payload = self._decode(message.get("data"), scope)
                            if payload is _GAP_HINT:
                                on_gap(scope)
                            elif isinstance(payload, TurnNotice):
                                on_notice(payload)
            except Exception as error:
                _logger.warning("Redis turn notice subscription failed: %s", type(error).__name__)
            if connected:
                on_gap(scope)
            await asyncio.sleep(1)

    @staticmethod
    def _decode(value: object, scope: TurnNoticeScope) -> object:
        if not isinstance(value, (bytes, str)) or len(value) > _MAX_NOTICE_BYTES:
            return None
        try:
            payload = json.loads(value)
            if not isinstance(payload, dict):
                return None
            if payload == {"type": "gap"}:
                return _GAP_HINT
            if set(payload) != {"type", "session_id", "turn_id", "phase", "revision"} or payload["type"] != "notice":
                return None
            return TurnNotice(scope, payload["session_id"], payload["turn_id"], payload["revision"], payload["phase"])
        except (TypeError, ValueError):
            return None


__all__ = ["RedisTurnNoticeTransport"]
