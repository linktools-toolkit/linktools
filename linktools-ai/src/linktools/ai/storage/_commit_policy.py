#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable preferences for optional source-write deferral."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class CommitPolicy:
    """Write timing preferences; owner safety boundaries always take precedence.

    Batch and terminal preferences require suitable borrowed cache resources.
    Missing resources select the synchronous path rather than an implicit queue.
    """

    mode: Literal["sync", "batch", "terminal"]
    batch_size: int | None = None
    flush_interval: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"sync", "batch", "terminal"}:
            raise ValueError("unsupported commit policy")
        if self.mode != "batch":
            if self.batch_size is not None or self.flush_interval is not None:
                raise ValueError("only batch commit accepts batch settings")
            return
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if (isinstance(self.flush_interval, bool)
                or not isinstance(self.flush_interval, (int, float))
                or not math.isfinite(self.flush_interval) or self.flush_interval <= 0):
            raise ValueError("flush_interval must be finite and positive")
        object.__setattr__(self, "flush_interval", float(self.flush_interval))

    @classmethod
    def sync(cls) -> CommitPolicy:
        """Await source writes at their owning semantic boundaries."""
        return cls("sync")

    @classmethod
    def terminal(cls) -> CommitPolicy:
        """Prefer eligible turn publication at the final atomic boundary."""
        return cls("terminal")

    @classmethod
    def batch(cls, *, batch_size: int, flush_interval: float) -> CommitPolicy:
        """Batch optional observations using a configured write cache."""
        return cls("batch", batch_size, flush_interval)


__all__ = ["CommitPolicy"]
