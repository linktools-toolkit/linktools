#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility helpers for the supported Python runtime floor."""

try:
    from enum import StrEnum as StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        """Python 3.10 equivalent of :class:`enum.StrEnum`."""

        def __str__(self) -> str:
            return str.__str__(self)

        @staticmethod
        def _generate_next_value_(
            name: str,
            start: int,
            count: int,
            last_values: list[object],
        ) -> str:
            return name.lower()


__all__ = ["StrEnum"]
