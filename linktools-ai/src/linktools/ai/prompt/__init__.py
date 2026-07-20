#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai.prompt: context-window policies + model-prompt template
composition (PromptBuilder). The prompt domain owns template composition and
does not read the filesystem -- callers hand in already-fetched sections and
PromptBuilder only composes (plan §4.2)."""

from .builder import PromptBuilder
from .window import (
    NoopWindowPolicy,
    RecentWindowPolicy,
    SessionWindowPolicy,
)

__all__ = [
    "PromptBuilder",
    "SessionWindowPolicy",
    "NoopWindowPolicy",
    "RecentWindowPolicy",
]
