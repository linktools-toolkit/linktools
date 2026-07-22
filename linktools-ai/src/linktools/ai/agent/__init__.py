#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai.agent: the agent domain's public model. AgentSpec
is the declaration surface, with PromptSpec/ToolRef as its component value
types; the compiler/runner that turn it into Runs live in their submodules
(``agent.compiler``, ``agent.engine``). A run's result is ``RunResult``
(from ``linktools.ai.run``), not an agent-local type."""

from .spec import AgentSpec, PromptSpec, ToolRef

__all__ = ["AgentSpec", "PromptSpec", "ToolRef"]
