#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai.agent: the agent domain's public model (spec §18.2). AgentSpec
is the declaration surface; the compiler/runner that turn it into Runs live in
their submodules (``agent.compiler``, ``agent.runner``). A run's result is
``RunResult`` (from ``linktools.ai.run``), not an agent-local type."""

from .spec import AgentSpec

__all__ = ["AgentSpec"]
