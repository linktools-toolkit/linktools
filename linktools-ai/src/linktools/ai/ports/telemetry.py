#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Telemetry boundary."""

from contextlib import AbstractAsyncContextManager
from typing import Protocol


class TelemetrySink(Protocol):
    def span(self, name: str) -> "AbstractAsyncContextManager[object]": ...
    async def event(self, name: str, fields: "dict[str, object]") -> None: ...
    async def metric(self, name: str, value: float) -> None: ...


__all__ = ["TelemetrySink"]
