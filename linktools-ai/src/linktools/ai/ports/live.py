#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Best-effort live event protocol."""

from typing import Protocol


class LiveEventPublisher(Protocol):
    async def publish(self, event: object) -> None: ...
    async def subscribe(self, execution_id: str) -> object: ...


__all__ = ["LiveEventPublisher"]
