#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Subagent execution contract. A subagent run creates a child
session (parent_id = parent session) and a child run recording parent_run_id /
root_run_id, then executes the resolved AgentSpec and returns a SubagentResult.

The Runtime supplies the real executor (it owns the AgentRunner + Storage); the
SubagentProvider accepts any executor implementing this Protocol so the
authorization + depth/concurrency/timeout gates are testable in isolation."""

from typing import Any, Protocol, runtime_checkable

from ..agent.spec import AgentSpec
from ..package.scope import PackageScope
from ..run.identity import ParentRunIdentity
from .models import SubagentResult

import contextvars

DEFAULT_MAX_DEPTH = 3
DEFAULT_MAX_CONCURRENCY = 1
DEFAULT_TIMEOUT_SECONDS = 120

# Tracks the current delegation depth across a call chain. The top-level run
# sees 0; each subagent executor increments it for the child run it drives and
# resets it when the child returns, so depth accounting holds across multiple
# hops and parallel calls without a schema change to RunRecord/RunContext.

_CURRENT_DEPTH: "contextvars.ContextVar[int]" = contextvars.ContextVar(
    "linktools_subagent_depth",
    default=0,
)


def current_depth() -> int:
    """The depth of the run currently executing (0 for a top-level agent)."""
    return _CURRENT_DEPTH.get()


@runtime_checkable
class SubagentExecutor(Protocol):
    """Executes a resolved child AgentSpec under a parent run. Implementations
    create the child session + run, enforce timeout, and return the result.
    ``parent`` is the single ParentRunIdentity every spawner (subagent
    toolset, package entrypoint toolset) builds identically -- a downstream
    implementation written strictly to this Protocol must not fail on a call
    it did not anticipate; there is exactly one call shape."""

    async def execute(
        self,
        *,
        agent_spec: AgentSpec,
        task: str,
        context: "dict[str, Any] | None",
        parent: ParentRunIdentity,
        scope: "PackageScope | None",
        timeout_seconds: "float | None",
    ) -> SubagentResult: ...


def enforce_depth(current_depth: int, max_depth: int) -> int:
    """Return the child depth, raising SubagentDepthExceededError if the
    delegation would exceed ``max_depth``. ``current_depth`` is the parent's
    depth (0 for a top-level agent)."""
    from ..errors import SubagentDepthExceededError

    child_depth = current_depth + 1
    if child_depth > max_depth:
        raise SubagentDepthExceededError(
            f"subagent delegation depth {child_depth} exceeds max_depth {max_depth}",
            depth=child_depth,
            max_depth=max_depth,
        )
    return child_depth
