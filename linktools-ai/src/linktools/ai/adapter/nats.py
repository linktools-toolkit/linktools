#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NATS event publisher protocol."""

from typing import Protocol


class NatsPublisher(Protocol):
    async def publish(self, subject: str, payload: bytes) -> None: ...


__all__ = ["NatsPublisher"]
