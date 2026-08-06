#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Durable event protocol."""

from typing import Protocol


class EventRepository(Protocol):
    async def append(self, event: object) -> object: ...
    async def list_after(self, execution_id: str, after_sequence: int, limit: int) -> object: ...


__all__ = ["EventRepository"]
