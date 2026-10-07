#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Process-local transcript visibility before checkpoint materialization."""

from dataclasses import dataclass

from pydantic_ai.messages import ModelMessage


@dataclass(frozen=True, slots=True)
class StagedTranscript:
    messages: tuple[ModelMessage, ...]
    pending: ModelMessage | None = None
    pending_keys: tuple[str, ...] = ()


__all__ = ["StagedTranscript"]
