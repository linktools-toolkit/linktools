#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logfire telemetry sink protocol."""

from typing import Protocol


class LogfireSink(Protocol):
    def record(self, event: str, attributes: 'dict[str, str]') -> None: ...


__all__ = ["LogfireSink"]
