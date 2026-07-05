#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CompiledAgent: the stateless output of AgentCompiler.compile(). Reusable
across many Runs -- no Session, no Run, no Checkpoint, no Workspace.
policy_capability is the SAME PolicyCapability instance already inside
pydantic_agent's capabilities=[...] list -- AgentRunner sets its
current_context per-Run rather than reaching into pydantic-ai internals."""

from dataclasses import dataclass

from pydantic_ai import Agent as PydanticAgent

from ..core.model_runtime import ModelBundle
from ..tool.capability import PolicyCapability
from .spec import AgentSpec


@dataclass(frozen=True, slots=True)
class CompiledAgent:
    spec: AgentSpec
    pydantic_agent: PydanticAgent
    model_bundle: ModelBundle
    policy_capability: PolicyCapability
