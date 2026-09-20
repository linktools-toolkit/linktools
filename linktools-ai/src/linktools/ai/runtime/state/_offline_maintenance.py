#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline Runtime exclusivity contract used by snapshot operations."""

from contextlib import AbstractAsyncContextManager
from typing import Protocol


class OfflineExclusiveStorage(Protocol):
    def offline_exclusivity(self) -> AbstractAsyncContextManager[None]: ...


__all__ = ["OfflineExclusiveStorage"]
