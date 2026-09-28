#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt and model-context resource limits."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptLimits:
    max_preloaded_skill_bytes: int = 256 * 1024
    max_binary_input_parts: int = 32
    max_binary_input_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        values = (
            self.max_preloaded_skill_bytes,
            self.max_binary_input_parts,
            self.max_binary_input_bytes,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in values
        ):
            raise ValueError("prompt limits must be positive integers")


__all__ = ["PromptLimits"]
