#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable preferences for optional source-write deferral."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Literal

from ._write_buffer import WriteBufferBackend


@dataclass(frozen=True, slots=True)
class CommitPolicy:
    """Write timing preferences; owner safety boundaries always take precedence.

    Batch borrows its write buffer without owning its lifetime. Buffer identity
    is separate from policy equality and must be included when grouping writers.
    Missing batch buffers select synchronous writes rather than an implicit queue.
    Batch timing may be omitted while authoring; each owner resolves its defaults
    once before retaining the policy or admitting writes.
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
        if self.batch_size is not None and (
            isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if self.flush_interval is not None:
            if (isinstance(self.flush_interval, bool)
                    or not isinstance(self.flush_interval, (int, float))
                    or not math.isfinite(self.flush_interval) or self.flush_interval <= 0):
                raise ValueError("flush_interval must be finite and positive")
            object.__setattr__(self, "flush_interval", float(self.flush_interval))

    def resolve_batch(self, *, batch_size: int, flush_interval: float) -> CommitPolicy:
        """Fill omitted batch timing from one owner's defaults; explicit values win.

        Sync and terminal policies are unchanged. Owners retain the returned
        complete policy; Runtime does not choose defaults for custom capabilities.
        """
        if self.mode != "batch":
            return self
        resolved = replace(
            self,
            batch_size=self.batch_size if self.batch_size is not None else batch_size,
            flush_interval=self.flush_interval if self.flush_interval is not None else flush_interval,
        )
        if resolved.batch_size is None or resolved.flush_interval is None:
            raise ValueError("owner batch defaults must provide batch_size and flush_interval")
        return resolved

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
        cls, *, batch_size: int | None = None, flush_interval: float | None = None,
        buffer: WriteBufferBackend | None = None,
    ) -> CommitPolicy:
        """Batch optional observations, inheriting omitted timing from their owner."""
        return cls("batch", batch_size, flush_interval, buffer)


__all__ = ["CommitPolicy"]
