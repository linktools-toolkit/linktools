#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Call-local readiness handoff between the built-in tree and model observer."""

import asyncio
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._service_api import ExecutionView


class ExecutionTreeReady(asyncio.Event):
    def __init__(self, positions: Callable[[], Mapping[str, int]] | None = None) -> None:
        super().__init__()
        self.views: tuple[tuple["ExecutionView", int, int], ...] | None = None
        self.positions = positions
        self.sampled = asyncio.Event()
        self.snapshot: tuple[tuple["ExecutionView", int, int], ...] = ()
