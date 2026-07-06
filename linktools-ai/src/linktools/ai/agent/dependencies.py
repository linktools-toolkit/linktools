#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentDependencies: the per-Run context passed to pydantic-ai capabilities
via dependency injection. Replaces the prior mutable per-Run field pattern.

Phase 1 of the review-doc refactoring: ``CompiledAgent`` is compiled once and
reused across many real Runs. Previously, ``PolicyCapability`` and
``MiddlewareCapability`` each carried a mutable per-Run ToolContext field that
``AgentRunner`` set/cleared around each ``agent.pydantic_agent.run()`` call --
a data race waiting to happen whenever two Runs shared one ``CompiledAgent``.

``AgentDependencies`` is the per-Run object pydantic-ai threads through every
capability hook as ``ctx.deps``. The runner constructs one per Run and passes
it via ``deps=`` at call time; capabilities read ``ctx.deps.tool_context``.
No mutable shared state, no set/clear lifecycle, safe for concurrent reuse.

Minimal for Phase 1 -- just ``tool_context``. The full review-doc shape
(execution, workspace, session, memory) arrives in later phases."""

from dataclasses import dataclass

from ..policy.rule import ToolContext


@dataclass(frozen=True, slots=True)
class AgentDependencies:
    tool_context: ToolContext
