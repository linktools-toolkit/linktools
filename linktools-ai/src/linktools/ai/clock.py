#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clock Protocol + SystemClock default.

Lives in a neutral module so any store or domain can depend on the time
abstraction without reaching across another domain for it."""

import asyncio
from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...

    async def sleep(self, seconds: float) -> None: ...


class SystemClock:
    """Default Clock backed by the wall clock."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


__all__: "list[str]" = ["Clock", "SystemClock"]
