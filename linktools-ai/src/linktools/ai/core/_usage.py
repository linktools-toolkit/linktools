#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable execution usage accounting values."""

from dataclasses import dataclass

MODEL_USAGE_INPUT_METADATA_KEY = "linktools.ai.model_usage.input_tokens"
MODEL_USAGE_OUTPUT_METADATA_KEY = "linktools.ai.model_usage.output_tokens"
MODEL_USAGE_CACHE_READ_METADATA_KEY = "linktools.ai.model_usage.cache_read_tokens"
MODEL_USAGE_CACHE_WRITE_METADATA_KEY = "linktools.ai.model_usage.cache_write_tokens"
MODEL_USAGE_METADATA_KEYS = frozenset(
    {
        MODEL_USAGE_INPUT_METADATA_KEY,
        MODEL_USAGE_OUTPUT_METADATA_KEY,
        MODEL_USAGE_CACHE_READ_METADATA_KEY,
        MODEL_USAGE_CACHE_WRITE_METADATA_KEY,
    }
)


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


__all__ = [
    "MODEL_USAGE_CACHE_READ_METADATA_KEY",
    "MODEL_USAGE_CACHE_WRITE_METADATA_KEY",
    "MODEL_USAGE_INPUT_METADATA_KEY",
    "MODEL_USAGE_METADATA_KEYS",
    "MODEL_USAGE_OUTPUT_METADATA_KEY",
    "UsageMetrics",
]
