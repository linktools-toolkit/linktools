#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentDependencies: the per-Run context passed to pydantic-ai capabilities
via dependency injection. Replaces the prior mutable per-Run field pattern.

``CompiledAgent`` is compiled once and
reused across many real Runs. Previously, ``PolicyCapability`` and
``MiddlewareCapability`` each carried a mutable per-Run ToolContext field that
``AgentRunner`` set/clear around each ``agent.pydantic_agent.run()`` call --
a data race waiting to happen whenever two Runs shared one ``CompiledAgent``.

``AgentDependencies`` is the per-Run object pydantic-ai threads through every
capability hook as ``ctx.deps``. The runner constructs one per Run and passes
it via ``deps=`` at call time; capabilities read ``ctx.deps.tool_context``.
No mutable shared state, no set/clear lifecycle, safe for concurrent reuse.

``execution`` carries the per-Run ``ExecutionBackend`` the
runner uses to construct the builtin file/terminal toolset at execution time
(via ``agent.iter(prompt, toolsets=[...])``). ``None`` (default) means the run
exposes no builtin tools -- a conversational-only agent. Decoupling the backend
from ``AgentCompiler`` keeps the compiler stateless (no filesystem surface)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

from ..policy.rule import ToolContext

if TYPE_CHECKING:
    from ..execution.protocols import ExecutionBackend
    from ..tool.models import ToolDescriptor


@dataclass(frozen=True, slots=True)
class AgentDependencies:
    tool_context: ToolContext
    execution: "ExecutionBackend | None" = None
    # Per-run tool-name -> ToolDescriptor lookup, populated once the
    # CapabilityAssembler has resolved this run's tool contributions. Lets
    # PolicyCapability (the global before-every-tool-call hook) classify a
    # call by category/risk/mutating instead of only by tool name -- None
    # (default) when no assembler ran, preserving the prior name-only path.
    descriptor_lookup: "Mapping[str, ToolDescriptor] | None" = None
