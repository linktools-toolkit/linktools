#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CompiledAgent: the stateless output of AgentCompiler.compile(). Reusable
across many Runs -- no Session, no Run, no Checkpoint, no Workspace.
policy_capability/middleware_capability are the SAME instances already inside
pydantic_agent's capabilities=[...] list -- AgentRunner sets their
current_context per-Run rather than reaching into pydantic-ai internals."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai import Agent as PydanticAgent

from ..core.model_runtime import ModelBundle
from ..tool.capability import PolicyCapability
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
