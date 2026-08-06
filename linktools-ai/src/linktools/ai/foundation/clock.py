#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Explicit time dependency used at application boundaries."""

from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    """Provide an aware UTC timestamp."""

    def now(self) -> datetime:
        """Return the current UTC time."""


class SystemClock:
    """Clock backed by the system UTC clock."""

    def now(self) -> datetime:
        """Return the current UTC time."""
        return datetime.now(timezone.utc)
