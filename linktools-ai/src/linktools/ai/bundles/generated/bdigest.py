#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Deterministic generated Bundle registry input."""

from typing import Final

from pydantic_ai import Agent

from ...agent.deps import AgentDeps

try:
    from pydantic_ai.durable_exec.temporal import TemporalDurability
except ImportError:
    TemporalDurability = None

BUNDLE_ID: "Final[str]" = "lt.generated.empty"
AGENT_NAME: "Final[str]" = "lt.generated.empty"
TOOLSET_IDS: "Final[tuple[str, ...]]" = ()
MODEL_NAME: "Final[str]" = "test"
if TemporalDurability is None:
    agent: "Agent[AgentDeps, str]" = Agent(MODEL_NAME, name=AGENT_NAME, deps_type=AgentDeps, output_type=str)
else:
    agent = Agent(
        MODEL_NAME,
        name=AGENT_NAME,
        deps_type=AgentDeps,
        output_type=str,
        capabilities=[TemporalDurability(name=AGENT_NAME, deps_type=AgentDeps)],
    )


__all__ = ["AGENT_NAME", "BUNDLE_ID", "MODEL_NAME", "TOOLSET_IDS", "agent"]
