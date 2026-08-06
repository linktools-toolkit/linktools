#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable event query and stream API."""

from collections.abc import AsyncIterator
from typing import Protocol

from ..core import Page, Principal
from .services import ExecutionEvent, ExecutionStreamItem


class EventApi(Protocol):
    async def list(self, execution_id: str, *, principal: Principal, after_sequence: int = 0, limit: int = 100) -> 'Page[ExecutionEvent]': ...
    def stream(self, execution_id: str, *, principal: Principal, after_sequence: int = 0) -> 'AsyncIterator[ExecutionStreamItem]': ...


__all__ = ["EventApi"]
