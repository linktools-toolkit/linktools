#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""NATS Live Delta adapter; it does not persist business events."""

from linktools.core import environ

logger = environ.get_logger("ai.outbound.nats.live")


class NatsLiveEventPublisher:
    def __init__(self, client: object, subject: str) -> None:
        self._client = client
        self._subject = subject

    async def publish(self, event: bytes) -> None:
        if len(event) > 32 * 1024:
            event = event[: 32 * 1024]
        try:
            await self._client.publish(self._subject, event)
        except Exception as error:
            logger.warning("live delta publish dropped error=%s", type(error).__name__)

    async def subscribe(self, execution_id: str) -> object:
        return await self._client.subscribe(f"{self._subject}.{execution_id}")


__all__ = ["NatsLiveEventPublisher"]
