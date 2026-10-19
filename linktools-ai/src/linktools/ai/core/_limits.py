#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution resource limits and observation scheduling defaults."""

from dataclasses import dataclass

DEFAULT_TRACE_BATCH_SIZE = 128
DEFAULT_TRACE_FLUSH_INTERVAL = 3.0


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptLimits:
    max_binary_input_parts: int = 32
    max_binary_input_bytes: int = 64 * 1024 * 1024
    max_file_output_bytes: int = 256 * 1024

    def __post_init__(self) -> None:
        values = (
            self.max_binary_input_parts,
            self.max_binary_input_bytes,
            self.max_file_output_bytes,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in values
        ):
            raise ValueError("prompt limits must be positive integers")


__all__ = ["PromptLimits", "DEFAULT_TRACE_BATCH_SIZE", "DEFAULT_TRACE_FLUSH_INTERVAL"]
