#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SubagentSpec / SubagentRegistry: tree-delegation subagent definitions."""

from dataclasses import dataclass

from ..core.registry import AgentSpec, MarkdownAgentRegistry


@dataclass(slots=True)
class SubagentSpec(AgentSpec):
    pass


class SubagentRegistry(MarkdownAgentRegistry[SubagentSpec]):
    _kind: str = "subagent"
    _spec_cls: "type[SubagentSpec]" = SubagentSpec
