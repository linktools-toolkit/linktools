#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supported independent capabilities of an injected cache."""

from typing import TypeAlias

from ._cache import CacheBackend
from ._stream import StreamBackend
from ._turn import TurnCoordinationBackend
from ._write_buffer import WriteBufferBackend

CacheCapabilities: TypeAlias = CacheBackend | WriteBufferBackend | StreamBackend | TurnCoordinationBackend

__all__ = ["CacheCapabilities"]
