#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CompiledAgent: the stateless output of AgentCompiler.compile(). Reusable
across many Runs -- no Session, no Run, no Checkpoint, no Workspace, and no
mutable per-Run fields anywhere on its
capabilities. policy_capability/middleware_capability are the SAME instances
already inside pydantic_agent's capabilities=[...] list; the per-Run
ToolContext reaches them via pydantic-ai dependency injection
(``deps=AgentDependencies(...)`` -> ``ctx.deps.tool_context``), so one
CompiledAgent is safe to share across concurrent Runs."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai import Agent as PydanticAgent

from ..model.registry import ModelBundle
from ..tool.pydantic import PolicyCapability
from .spec import AgentSpec

if TYPE_CHECKING:
    from ..middleware.capability import MiddlewareCapability


@dataclass(frozen=True, slots=True)
class CompiledAgent:
    spec: AgentSpec
    pydantic_agent: PydanticAgent
    model_bundle: ModelBundle
    policy_capability: PolicyCapability
    middleware_capability: "MiddlewareCapability | None" = None
