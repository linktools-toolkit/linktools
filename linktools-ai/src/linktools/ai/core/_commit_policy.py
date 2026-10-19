#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable preferences for optional source-write deferral."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from ._write_buffer import WriteBufferBackend


@dataclass(frozen=True, slots=True)
class CommitPolicy:
    """Write timing preferences; owner safety boundaries always take precedence.

    Batch borrows its write buffer without owning its lifetime. Buffer identity
    is separate from policy equality and must be included when grouping writers.
    Missing batch buffers select synchronous writes rather than an implicit queue.
    Terminal defers optional publication to the owner's final boundary; execution
    admission eligibility still depends on its coordination and safety rules.
    """

    mode: Literal["sync", "batch", "terminal"]
    batch_size: int | None = None
    flush_interval: float | None = None
    buffer: WriteBufferBackend | None = field(default=None, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in {"sync", "batch", "terminal"}:
            raise ValueError("unsupported commit policy")
        if self.buffer is not None and not all(
            callable(getattr(self.buffer, name, None)) for name in ("append", "read_batch", "ack")
        ):
            raise TypeError("buffer must provide append, read_batch and ack")
        if self.mode != "batch":
            if self.batch_size is not None or self.flush_interval is not None or self.buffer is not None:
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
        """Prefer publication at the owner's final atomic boundary."""
        return cls("terminal")

    @classmethod
    def batch(
        cls, *, batch_size: int, flush_interval: float,
        buffer: WriteBufferBackend | None = None,
    ) -> CommitPolicy:
        """Batch optional observations using a configured write cache."""
        return cls("batch", batch_size, flush_interval, buffer)


__all__ = ["CommitPolicy"]
