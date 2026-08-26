#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable execution usage accounting values."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UsageMetrics:
    model_requests: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __post_init__(self) -> None:
        values = (
            self.model_requests,
            self.tool_calls,
            self.input_tokens,
            self.output_tokens,
            self.cache_read_tokens,
            self.cache_write_tokens,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
            raise ValueError("usage metrics must be non-negative integers")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


__all__ = ["UsageMetrics"]
