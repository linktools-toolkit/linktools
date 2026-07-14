#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai.prompt: context-window policies."""

from .window import (
    NoopWindowPolicy,
    RecentWindowPolicy,
    SessionWindowPolicy,
)

__all__ = [
    "SessionWindowPolicy",
    "NoopWindowPolicy",
    "RecentWindowPolicy",
]
