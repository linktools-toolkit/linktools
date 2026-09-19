#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure Runtime snapshot admission limits."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SnapshotLimits:
    max_entries: int
    max_bytes: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_entries, bool)
            or not isinstance(self.max_entries, int)
            or self.max_entries < 1
            or isinstance(self.max_bytes, bool)
            or not isinstance(self.max_bytes, int)
            or self.max_bytes < 1
        ):
            raise ValueError("snapshot limits must be positive integers")


__all__ = ["SnapshotLimits"]
