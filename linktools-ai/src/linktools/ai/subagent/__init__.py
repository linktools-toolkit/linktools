#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai.subagent: tree-style delegation. Distinct from Swarm --
a subagent call is one parent -> one named child -> synchronous result."""

from .models import SubagentResult, SubagentStatus
from .provider import SubagentProvider
from .runner import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_DEPTH,
    DEFAULT_TIMEOUT_SECONDS,
    SubagentExecutor,
    _CURRENT_DEPTH,
    current_depth,
    enforce_depth,
)
from .toolset import build_subagent_toolset

__all__ = [
    "SubagentResult", "SubagentStatus",
    "SubagentExecutor", "enforce_depth", "current_depth", "_CURRENT_DEPTH",
    "build_subagent_toolset", "SubagentProvider",
    "DEFAULT_MAX_DEPTH", "DEFAULT_MAX_CONCURRENCY", "DEFAULT_TIMEOUT_SECONDS",
]
