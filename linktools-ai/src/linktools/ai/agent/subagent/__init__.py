#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai.agent.subagent: tree-style delegation. Distinct from Swarm --
a subagent call is one parent -> one named child -> synchronous result."""

from .models import SubagentResult, SubagentStatus
from .provider import (
    AgentBackedSubagentProvider,
    SubagentAgentProvider,
    SubagentProvider,
)
from .runner import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_DEPTH,
    DEFAULT_TIMEOUT_SECONDS,
    SubagentExecutorProtocol,
    current_depth,
    enforce_depth,
)
from .toolset import build_subagent_toolset

__all__ = [
    "SubagentResult",
    "SubagentStatus",
    "SubagentExecutorProtocol",
    "enforce_depth",
    "current_depth",
    "build_subagent_toolset",
    "SubagentProvider",
    "SubagentAgentProvider",
    "AgentBackedSubagentProvider",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_TIMEOUT_SECONDS",
]
